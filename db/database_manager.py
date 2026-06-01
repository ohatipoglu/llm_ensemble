# db/database_manager.py

import sqlite3
import time
import queue
import threading
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger("database_manager")

class DatabaseManager:
    def __init__(self, db_path: str = "llm_challenger.db"):
        # Veritabanı yolunu ana proje dizininde olacak şekilde ayarla
        self.db_path = Path(__file__).resolve().parent.parent / db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connect()
        self._setup_database()
        
        # Thread-safe yazma için queue sistemi
        self._write_queue = queue.Queue()
        self._write_thread = threading.Thread(target=self._process_writes, daemon=True)
        self._write_thread.start()
        logger.info("Veritabanı write-worker thread başlatıldı.")

        # Monkey patch ConversationManager to dynamically link database_manager and conversation_id
        try:
            from core.conversation_manager import ConversationManager
            if not getattr(ConversationManager, "_is_perf_patched", False):
                original_run = ConversationManager.run_full_conversation
                
                async def patched_run(self_cm, *args, **kwargs):
                    self_cm.api_manager.db_manager = self_cm.db_manager
                    conv_id = kwargs.get("conversation_id")
                    if conv_id is None and len(args) >= 3:
                        conv_id = args[2]
                    self_cm.api_manager.current_conv_id = conv_id
                    return await original_run(self_cm, *args, **kwargs)
                
                ConversationManager.run_full_conversation = patched_run
                ConversationManager._is_perf_patched = True
                logger.info("ConversationManager.run_full_conversation başarıyla monkey-patch edildi.")
        except Exception as e:
            logger.error(f"ConversationManager monkey-patch hatası: {e}")

    def _get_connection(self) -> sqlite3.Connection:
        """Her thread için ayrı connection oluşturur (Thread-local storage)."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # WAL mode aktif et - çok daha hızlı yazma performansı
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            # Foreign Key desteğini aktif et
            self._local.conn.execute("PRAGMA foreign_keys = ON;")
            logger.info(f"Yeni DB connection oluşturuldu (Thread: {threading.current_thread().name})")
        return self._local.conn

    def _connect(self):
        """Ana thread için başlangıç connection'ı."""
        try:
            self._get_connection()
            logger.info(f"Veritabanı bağlantısı başarılı: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Veritabanı bağlantı hatası: {e}")
            raise

    def _column_exists(self, cursor, table_name, column_name):
        """Bir tabloda belirli kolonun var olup olmadığını kontrol eder."""
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        return column_name in columns

    def _setup_database(self):
        """Tabloları oluşturur veya mevcut tabloları yeni şemaya göre günceller (migration)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 1. Conversations tablosu - MIGRATION DESTEĞİ İLE
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            timestamp REAL NOT NULL,
            responder1_model TEXT,
            responder2_model TEXT,
            critic_model TEXT,
            judge_model TEXT
        );
        """)
        
        # Migration: Eksik kolonları ekle (eğer tablo zaten varsa)
        migrations = [
            ("status", "TEXT DEFAULT 'running'"),
            ("current_stage", "INTEGER DEFAULT 0"),
            ("state_data", "TEXT"),
            ("updated_at", "REAL")
        ]
        
        for col_name, col_type in migrations:
            if not self._column_exists(cursor, "conversations", col_name):
                try:
                    cursor.execute(f"ALTER TABLE conversations ADD COLUMN {col_name} {col_type}")
                    logger.info(f"DB Migration: '{col_name}' sütunu eklendi.")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Migration uyarısı ({col_name}): {e}")
        
        # 2. Responses tablosu
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            response_type TEXT NOT NULL CHECK(response_type IN ('yanıt', 'eleştiri', 'savunma', 'son_söz', 'karar')),
            content TEXT NOT NULL,
            timestamp REAL NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        );
        """)
        
        # 3. Prompts tablosu
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            role TEXT NOT NULL,
            prompt_content TEXT NOT NULL,
            timestamp REAL NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        );
        """)

        # 4. Performance metrics tablosu
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            model_name TEXT,
            ttft REAL,
            tps REAL,
            token_count INTEGER,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        """)
        
        # İndeksleri ekle (eğer yoksa)
        indexes = [
            ("idx_conv_status", "conversations(status)"),
            ("idx_conv_updated", "conversations(updated_at)"),
            ("idx_responses_conv_id", "responses(conversation_id)"),
            ("idx_prompts_conv_id", "prompts(conversation_id)"),
            ("idx_prompts_timestamp", "prompts(timestamp DESC)")
        ]
        
        for idx_name, table_cols in indexes:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_cols};")
            except sqlite3.OperationalError as e:
                logger.warning(f"İndeks uyarısı ({idx_name}): {e}")
        
        conn.commit()
        logger.info("Veritabanı tabloları ve migration işlemleri tamamlandı.")

    def _process_writes(self):
        """Arka plan thread'i - queue'dan gelen yazma işlemlerini gerçekleştirir."""
        while True:
            try:
                item = self._write_queue.get()
                if item is None or (isinstance(item, tuple) and item[0] is None):  # Poison pill
                    self._write_queue.task_done()
                    break
                
                elif isinstance(item, dict) and item.get("type") == "PERF":
                    conv_id = item.get("conv_id")
                    model = item.get("model")
                    ttft = item.get("ttft")
                    tps = item.get("tps")
                    tokens = item.get("tokens")
                    result_queue = item.get("result_queue")
                    
                    try:
                        conn = self._get_connection()
                        with self._lock:  # SQLite yazma işlemleri thread-safe değil
                            with conn:  # Transaction (BEGIN/COMMIT/ROLLBACK) koruması
                                cursor = conn.cursor()
                                cursor.execute("""
                                    INSERT INTO performance_metrics (conversation_id, model_name, ttft, tps, token_count)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (conv_id, model, ttft, tps, tokens))
                        if result_queue:
                            result_queue.put(("success", None))
                    except Exception as e:
                        logger.error(f"DB Perf Write Error: {e}")
                        if result_queue:
                            result_queue.put(("error", str(e)))
                    finally:
                        self._write_queue.task_done()
                
                else:
                    query, params, result_queue = item
                    try:
                        conn = self._get_connection()
                        with self._lock:  # SQLite yazma işlemleri thread-safe değil
                            with conn:  # Transaction (BEGIN/COMMIT/ROLLBACK) koruması
                                cursor = conn.cursor()
                                cursor.execute(query, params)
                                last_id = None
                                if query.strip().upper().startswith("INSERT"):
                                    cursor.execute("SELECT last_insert_rowid()")
                                    last_id = cursor.fetchone()[0]
                        if result_queue:
                            result_queue.put(("success", last_id))
                    except Exception as e:
                        logger.error(f"DB Write Error: {e}")
                        if result_queue:
                            result_queue.put(("error", str(e)))
                    finally:
                        self._write_queue.task_done()  # İşlemin bittiğini bildir
            except Exception as e:
                logger.error(f"Write worker critical error: {e}")

    def _queue_write(self, query: str, params: tuple) -> Any:
        """Yazma işlemini queue'ya ekler ve başarılı olmasını bekler (blocking)."""
        qsize = self._write_queue.qsize()
        if qsize > 5:
            logger.warning(f"DB Write Queue uyarısı: Bekleyen yazma işlem sayısı yüksek ({qsize})")
        result_queue = queue.Queue()
        self._write_queue.put((query, params, result_queue))
        status, result = result_queue.get(timeout=10)  # 10 saniye timeout
        if status == "error":
            logger.error(f"DB Write kuyruk hatası: {result}")
            raise Exception(f"DB Write failed: {result}")
        return result

    def create_conversation(self, prompt: str, models: Dict[str, str]) -> int:
        """Yeni konuşma oluşturur ve ID'sini döndürür."""
        query = """
        INSERT INTO conversations 
        (prompt, timestamp, responder1_model, responder2_model, critic_model, judge_model, status, current_stage, updated_at) 
        VALUES (?, ?, ?, ?, ?, ?, 'running', 0, ?)
        """
        ts = time.time()
        params = (prompt, ts, models.get('responder1'), models.get('responder2'), 
                 models.get('critic'), models.get('judge'), ts)
        conv_id = self._queue_write(query, params)
        logger.info(f"DB State: Yeni konuşma başlatıldı. ID: {conv_id} | Responder1: {models.get('responder1')} | Responder2: {models.get('responder2')} | Critic: {models.get('critic')} | Judge: {models.get('judge')}")
        return conv_id

    def update_conversation_state(self, conversation_id: int, stage: int, 
                                  state_data: Dict[str, Any], status: str = 'running'):
        """Konuşma durumunu günceller (pause/resume için)."""
        query = """
        UPDATE conversations 
        SET current_stage = ?, state_data = ?, status = ?, updated_at = ?
        WHERE id = ?
        """
        params = (stage, json.dumps(state_data, ensure_ascii=False), status, time.time(), conversation_id)
        self._queue_write(query, params)
        logger.info(f"DB State: Konuşma {conversation_id}, Aşama {stage}, Durum {status}")

    def get_incomplete_conversation(self) -> Optional[Tuple[int, Dict]]:
        """Tamamlanmamış (paused veya running) konuşmayı getirir."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, prompt, responder1_model, responder2_model, 
                       critic_model, judge_model, current_stage, state_data
                FROM conversations 
                WHERE status IN ('running', 'paused')
                ORDER BY updated_at DESC 
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                conv_id, prompt, r1, r2, crit, judge, stage, state_json = row
                models = {
                    'responder1': r1, 'responder2': r2, 
                    'critic': crit, 'judge': judge
                }
                try:
                    state_data = json.loads(state_json) if state_json else {}
                except json.JSONDecodeError as je:
                    logger.error(f"state_json ayrıştırılırken hata oluştu (bozuk JSON): {je}")
                    state_data = {}
                logger.info(f"DB: Tamamlanmamış konuşma geri yüklendi. ID: {conv_id}, Aşama: {stage}")
                return conv_id, {
                    'prompt': prompt, 'models': models, 
                    'stage': stage, 'data': state_data
                }
            logger.info("DB: Tamamlanmamış konuşma bulunamadı.")
            return None
        except Exception as e:
            logger.error(f"Incomplete conversation okunamadı: {e}")
            return None

    def mark_conversation_paused(self, conversation_id: int, stage: int, state_data: Dict):
        """Konuşmayı paused olarak işaretler."""
        logger.info(f"DB State: Konuşma {conversation_id} duraklatılıyor. Aşama: {stage}")
        self.update_conversation_state(conversation_id, stage, state_data, 'paused')

    def mark_conversation_completed(self, conversation_id: int):
        """Konuşmayı tamamlandı olarak işaretler."""
        query = "UPDATE conversations SET status = 'completed', updated_at = ? WHERE id = ?"
        self._queue_write(query, (time.time(), conversation_id))
        logger.info(f"DB State: Konuşma {conversation_id} başarıyla tamamlandı olarak işaretlendi.")

    def save_response(self, conversation_id: int, model_name: str, 
                      response_type: str, content: str):
        """Yanıtı veritabanına kaydeder (Thread-safe)."""
        query = """
        INSERT INTO responses 
        (conversation_id, model_name, response_type, content, timestamp) 
        VALUES (?, ?, ?, ?, ?)
        """
        params = (conversation_id, model_name, response_type, content, time.time())
        self._queue_write(query, params)
        logger.info(f"DB Kayıt: Konuşma {conversation_id} | [{model_name}] [{response_type}] yanıtı kaydedildi.")

    def save_prompt(self, conversation_id: int, model_name: str, 
                    role: str, prompt_content: Any):
        """Prompt'u veritabanına kaydeder (Thread-safe)."""
        if isinstance(prompt_content, (list, dict)):
            prompt_str = json.dumps(prompt_content, ensure_ascii=False)
        else:
            prompt_str = str(prompt_content)
            
        query = """
        INSERT INTO prompts 
        (conversation_id, model_name, role, prompt_content, timestamp) 
        VALUES (?, ?, ?, ?, ?)
        """
        params = (conversation_id, model_name, role, prompt_str, time.time())
        self._queue_write(query, params)
        logger.info(f"DB Kayıt: Konuşma {conversation_id} | [{model_name}] [{role}] prompt'u kaydedildi.")

    def record_performance(self, conv_id, model, ttft, tps, tokens):
        """Performans metriklerini kaydeder (Asenkron kuyruk üzerinden)."""
        result_queue = queue.Queue()
        self._write_queue.put({
            "type": "PERF",
            "conv_id": conv_id,
            "model": model,
            "ttft": ttft,
            "tps": tps,
            "tokens": tokens,
            "result_queue": result_queue
        })
        try:
            status, result = result_queue.get(timeout=10)
            if status == "error":
                logger.error(f"DB Perf Write failed: {result}")
            else:
                logger.info(f"DB Perf Kayıt: Konuşma {conv_id} | Model {model} performans metrikleri kaydedildi.")
        except Exception as e:
            logger.error(f"DB Perf Write timeout or error: {e}")

    def get_latest_conversation_id(self) -> Optional[int]:
        """En son oluşturulan konuşmanın ID'sini döndürür."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM conversations ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"En son konuşma ID'si alınamadı: {e}")
            return None

    def get_conversation_history(self, conversation_id: int) -> List[Tuple]:
        """Belirtilen konuşmaya ait tüm prompt ve yanıtları kronolojik sırada döndürür."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, model_name, role AS stage, 'prompt' AS type, prompt_content AS content, id
                FROM prompts
                WHERE conversation_id = ?
                UNION ALL
                SELECT timestamp, model_name, response_type AS stage, 'response' AS type, content AS content, id
                FROM responses
                WHERE conversation_id = ?
                ORDER BY timestamp ASC, id ASC
            """, (conversation_id, conversation_id))
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Konuşma geçmişi alınamadı (ID: {conversation_id}): {e}")
            return []

    def get_all_prompts(self) -> List[Tuple]:
        """Tüm prompt kayıtlarını döndürür."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timestamp, model_name, role, prompt_content FROM prompts ORDER BY timestamp DESC"
            )
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Prompt kayıtları alınamadı: {e}")
            return []

    def close(self):
        """Veritabanını güvenli şekilde kapatır."""
        logger.info("Veritabanı kapatılıyor...")
        self._write_queue.put((None, None, None))  # Worker thread'e durma sinyali
        try:
            self._write_queue.join()  # Tüm bekleyen yazma işlemlerinin tamamlanmasını bekle
        except Exception as e:
            logger.error(f"Kuyruk sonlandırılırken hata: {e}")
        self._write_thread.join(timeout=2)
        
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            logger.info("Veritabanı bağlantısı kapatıldı.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()