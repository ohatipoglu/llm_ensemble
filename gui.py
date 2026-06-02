# gui.py

import sys
import asyncio
import markdown
import datetime
import logging
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QComboBox, QPushButton, QLabel, QTextEdit, QTabWidget,
    QMessageBox, QSplitter, QDialog, QTableWidget, QTableWidgetItem, 
    QHeaderView, QProgressBar, QCheckBox, QTextBrowser,
    QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtGui import QFont, QFontDatabase, QTextCursor, QColor, QBrush

from api.api_manager import APIManager
from db.database_manager import DatabaseManager
from core.conversation_manager import ConversationManager, Stage

logger = logging.getLogger("gui")

class StreamRedirector(QObject):
    """Terminal çıktısını arayüze yönlendirir."""
    textWritten = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.buffer = ""

    def write(self, text):
        if text == '\n':
            if self.buffer.strip():
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                self.textWritten.emit(f"[{timestamp}] {self.buffer}\n")
            else:
                self.textWritten.emit('\n')
            self.buffer = ""
        else:
            self.buffer += text
    
    def flush(self):
        if self.buffer:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.textWritten.emit(f"[{timestamp}] {self.buffer}\n")
            self.buffer = ""
class ModelLoaderWorker(QThread):
    """Ollama model listesini arka planda asenkron yükler."""
    modelsLoaded = pyqtSignal(list)
    errorOccurred = pyqtSignal(str)

    def __init__(self, api_manager: APIManager):
        super().__init__()
        self.api_manager = api_manager

    def run(self):
        try:
            models = self.api_manager.get_model_list()
            self.modelsLoaded.emit(models)
        except Exception as e:
            self.errorOccurred.emit(str(e))

class Worker(QThread):
    """Backend işlemini ayrı thread'de çalıştırır."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    chunk_received = pyqtSignal(str, str)  # (model_key, chunk_text)
    paused = pyqtSignal()  # Duraklatıldığında
    
    def __init__(self, api_manager: APIManager, db_manager: DatabaseManager, 
                 user_prompt: str, models: Dict[str, str], conversation_id: int,
                 start_stage: Stage = Stage.STARTED, resume_data: Optional[Dict] = None):
        super().__init__()
        self.api_manager = api_manager
        self.db_manager = db_manager
        self.user_prompt = user_prompt
        self.models = models
        self.conversation_id = conversation_id
        self.start_stage = start_stage
        self.resume_data = resume_data or {}
        
        self.conversation_manager: Optional[ConversationManager] = None
        self._pause_requested = False
        self._is_running = True

    def run(self):
        try:
            self.conversation_manager = ConversationManager(self.api_manager, self.db_manager)
            
            def progress_cb(step, msg):
                self.progress.emit(step, msg)
            
            def chunk_cb(model_key, chunk):
                self.chunk_received.emit(model_key, chunk)

            results = asyncio.run(
                self.conversation_manager.run_full_conversation(
                    self.user_prompt, 
                    self.models, 
                    self.conversation_id,
                    progress_callback=progress_cb,
                    chunk_callback=chunk_cb,
                    start_stage=self.start_stage,
                    resume_state=self.resume_data
                )
            )
            
            if self.conversation_manager._pause_requested:
                self.paused.emit()
            elif results and self._is_running:
                self.finished.emit(results)
            elif not self._is_running:
                self.error.emit("İşlem durduruldu.")
                
        except Exception as e:
            self.error.emit(f"Hata: {e}")

    def pause(self):
        """Mevcut adım bitince durdur"""
        self._pause_requested = True
        if self.conversation_manager:
            self.conversation_manager.request_pause()

    def stop(self):
        """Hemen durdur"""
        self._is_running = False
        if self.conversation_manager:
            self.conversation_manager.cancel()

class PromptViewerWindow(QDialog):
    """Prompt kayıtlarını görüntüleyen pencere."""
    def __init__(self, db_manager: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setWindowTitle("Gönderilen Prompt Kayıtları")
        
        # Görev 1: Window Flags & Boyutlandırma
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint)
        self.setMinimumSize(900, 600)
        self.resize(1100, 750)
        
        self.layout = QVBoxLayout(self)
        
        # Görev 2: Konuşma Seçici Arayüzü
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("<b>Konuşma Seçin:</b>"))
        
        self.conversation_combo = QComboBox()
        self.conversation_combo.setMinimumWidth(400)
        top_layout.addWidget(self.conversation_combo)
        
        self.refresh_button = QPushButton("Kayıtları Yenile")
        self.refresh_button.clicked.connect(self.loadConversations)
        top_layout.addWidget(self.refresh_button)
        
        self.layout.addLayout(top_layout)

        # Görev 3: Formatlı Markdown Görüntüleyici (QTextBrowser)
        self.text_browser = QTextBrowser()
        # Monospace font varsayılanı ekle
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(10)
        self.text_browser.setFont(font)
        self.layout.addWidget(self.text_browser)
        
        # Sinyal Bağlantıları
        self.conversation_combo.currentIndexChanged.connect(self.loadPrompts)
        
        # Başlangıçta veriyi yükle
        self.loadConversations()

    def loadConversations(self):
        """Veritabanından tüm konuşmaları combobox'a yükler."""
        self.conversation_combo.blockSignals(True)
        self.conversation_combo.clear()
        
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, responder1_model, responder2_model, critic_model, judge_model 
                FROM conversations 
                ORDER BY id DESC
            """)
            rows = cursor.fetchall()
        except Exception as e:
            logger.error(f"Konuşmalar listesi alınamadı: {e}")
            rows = []
            
        if not rows:
            self.conversation_combo.addItem("Henüz kaydedilmiş bir konuşma bulunmuyor.", None)
            self.conversation_combo.blockSignals(False)
            self.loadPrompts()
            return
            
        for conv_id, ts, r1, r2, crit, judge in rows:
            time_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
            models_list = [m for m in [r1, r2, crit, judge] if m]
            models_str = ", ".join(models_list)
            item_text = f"Konuşma #{conv_id} | {time_str} | Modeller: [{models_str}]"
            self.conversation_combo.addItem(item_text, conv_id)
            
        # Varsayılan olarak aktif veya en son konuşmayı seç
        parent = self.parent()
        target_conv_id = parent.current_conversation_id if parent else None
        found_index = 0
        if target_conv_id:
            for i in range(self.conversation_combo.count()):
                if self.conversation_combo.itemData(i) == target_conv_id:
                    found_index = i
                    break
        self.conversation_combo.setCurrentIndex(found_index)
        self.conversation_combo.blockSignals(False)
        self.loadPrompts()

    def loadPrompts(self):
        """Seçilen konuşmaya ait prompt ve yanıtları yükler ve formatlar."""
        try:
            idx = self.conversation_combo.currentIndex()
            if idx == -1:
                self.text_browser.setHtml("<h3>Henüz kaydedilmiş bir konuşma bulunmuyor.</h3>")
                return
                
            conv_id = self.conversation_combo.itemData(idx)
            if not conv_id:
                self.text_browser.setHtml("<h3>Geçersiz konuşma seçimi.</h3>")
                return

            history = self.db_manager.get_conversation_history(conv_id)
            messages = []
            
            import json
            for ts, model, stage, m_type, content, r_id in history:
                time_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                if m_type == 'prompt':
                    parsed = False
                    if content.strip().startswith('['):
                        try:
                            chatml = json.loads(content)
                            if isinstance(chatml, list):
                                for msg in chatml:
                                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                        messages.append({
                                            'time': time_str,
                                            'model': model,
                                            'role': msg['role'],
                                            'stage': stage,
                                            'content': msg['content']
                                        })
                                parsed = True
                        except Exception:
                            pass
                    if not parsed:
                        messages.append({
                            'time': time_str,
                            'model': model,
                            'role': 'user',
                            'stage': stage,
                            'content': content
                        })
                else:  # response
                    messages.append({
                        'time': time_str,
                        'model': model,
                        'role': 'assistant',
                        'stage': stage,
                        'content': content
                    })

            self.text_browser.clear()
            cursor = self.text_browser.textCursor()
            
            for msg in messages:
                # 1. Başlık Çubuğu (Hafif renkli arka planlı HTML kutusu)
                role_upper = msg['role'].upper()
                role_emoji = "⚙️" if "SYSTEM" in role_upper else ("👤" if "USER" in role_upper else "🤖")
                
                header_html = f"""
                <div style="background-color: #2b303c; color: #ffffff; padding: 8px 12px; margin-top: 15px; margin-bottom: 8px; border-radius: 4px; border-left: 5px solid #3b82f6; font-family: sans-serif; font-size: 12px;">
                    {role_emoji} <b>Model:</b> {msg['model']} &nbsp;|&nbsp; 
                    <b>Rol:</b> {role_upper} &nbsp;|&nbsp; 
                    <b>Zaman:</b> {msg['time']} &nbsp;|&nbsp; 
                    <b>Aşama:</b> {msg['stage']}
                </div>
                """
                cursor.insertHtml(header_html)
                
                # 2. İçerik (Markdown -> HTML)
                content_html = markdown.markdown(msg['content'])
                cursor.insertHtml(f'<div style="margin-left: 10px; margin-right: 10px; margin-top: 5px; margin-bottom: 20px; color: #e5e7eb; line-height: 1.6;">{content_html}</div>')
                
                # 3. Ayırıcı
                cursor.insertHtml('<hr style="border: 0; height: 1px; background: #374151; margin-bottom: 20px;"/>')
            
            # Auto-scroll
            self.text_browser.moveCursor(self.text_browser.textCursor().End)
            self.text_browser.ensureCursorVisible()
            
            self.setWindowTitle(f"Gönderilen Prompt Kayıtları - Konuşma #{conv_id} (Toplam Mesaj: {len(messages)})")
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Yenileme sırasında hata: {e}")

    def showEvent(self, event):
        self.loadConversations()
        super().showEvent(event)

class TelemetryViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 Yerel Telemetry İzleyici - Son 500 Kayıt")
        self.setMinimumSize(1100, 750)
        self.resize(1200, 850)
        
        # Standard window flags to allow maximize, minimize, close and remove the help button
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        self.initUI()
        self.refresh_data()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Top Label
        title_label = QLabel("📊 Yerel Telemetry İzleyici - Son 500 Kayıt (Agresif İzleme)")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # Splitter to divide tree and details
        self.splitter = QSplitter(Qt.Vertical)
        
        # Tree Widget
        self.tree = QTreeWidget()
        self.tree.setColumnCount(8)
        self.tree.setHeaderLabels([
            "Trace ID", "Span/Model", "Aşama", "TTFT (s)", "TPS", "Süre", "Durum", "Zaman"
        ])
        
        # Resize behavior
        header = self.tree.header()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch) # Stretch model column
        
        self.splitter.addWidget(self.tree)
        
        # Details Panel
        self.details_panel = QTextBrowser()
        self.details_panel.setPlaceholderText("Detaylarını görmek için listeden bir span seçin...")
        self.details_panel.setStyleSheet("background-color: #1e1e2e; color: #f8f8f2; font-family: Consolas, monospace;")
        self.splitter.addWidget(self.details_panel)
        
        # Set splitter sizes (e.g. 60% tree, 40% details)
        self.splitter.setSizes([450, 300])
        
        layout.addWidget(self.splitter)
        
        # Connect Selection Signal
        self.tree.itemSelectionChanged.connect(self.show_span_details)
        
        # Bottom Buttons
        btn_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("🔄 Yenile")
        self.refresh_btn.clicked.connect(self.refresh_data)
        btn_layout.addWidget(self.refresh_btn)
        
        self.clear_btn = QPushButton("🗑️ Temizle")
        self.clear_btn.clicked.connect(self.clear_data)
        btn_layout.addWidget(self.clear_btn)
        
        self.close_btn = QPushButton("Kapat")
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
        
    def refresh_data(self):
        from core.telemetry import TelemetryStorage
        import collections
        
        self.tree.clear()
        self.details_panel.clear()
        
        storage = TelemetryStorage.get_instance()
        spans = storage.get_all_spans()
        
        if not spans:
            return
            
        span_map = {s["span_id"]: s for s in spans}
        children_map = collections.defaultdict(list)
        for s in spans:
            if s["parent_id"]:
                children_map[s["parent_id"]].append(s)
                
        # Find root spans
        root_spans = []
        for s in spans:
            if not s["parent_id"] or s["parent_id"] not in span_map:
                root_spans.append(s)
                
        # Sort root spans chronologically by start_time
        root_spans.sort(key=lambda s: s["start_time"])
        
        def add_item(parent_item, span_dict):
            # Create QTreeWidgetItem
            item = QTreeWidgetItem(parent_item)
            item.setData(0, Qt.UserRole, span_dict["span_id"])
                
            # Formatting trace_id
            trace_id = span_dict["trace_id"]
            trace_id_str = str(trace_id) if trace_id < 1000000 else f"{trace_id:032x}"
            
            # Formatting name/model
            span_name = span_dict["name"]
            llm_model = span_dict.get("llm_model")
            span_model_str = llm_model if llm_model else span_name
            if span_name == "run_full_conversation":
                span_model_str = "Full Conversation"
            elif span_name.startswith("run_"):
                span_model_str = span_name.replace("run_", "").capitalize()
            elif span_name.startswith("tool_call:"):
                span_model_str = span_name.replace("tool_call:", "🔨 ")
                
            # Stage
            attributes = span_dict.get("attributes", {})
            stage_str = ""
            if span_name == "run_full_conversation":
                stage_str = f"Start: {attributes.get('stage.start', '')}"
            elif "stage.name" in attributes:
                stage_str = attributes.get("stage.name")
            elif "agent.role" in attributes:
                stage_str = attributes.get("agent.role")
            elif span_name == "ollama_api_call":
                stage_str = "API Call"
            elif span_name.startswith("tool_call:"):
                stage_str = "Tool/DB Call"
                
            # Performance metrics
            ttft_val = span_dict.get("ttft")
            tps_val = span_dict.get("tps")
            ttft_str = f"{ttft_val:.2f}" if ttft_val is not None else ""
            tps_str = f"{tps_val:.2f}" if tps_val is not None else ""
            
            # Duration
            duration_ms = span_dict.get("duration_ms", 0)
            duration_str = f"{duration_ms:.0f} ms"
            
            # Status
            status_str = span_dict.get("status", "OK")
            
            # Time
            start_time = span_dict.get("start_time", 0)
            time_str = datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S") if start_time else ""
            
            item.setText(0, trace_id_str)
            item.setText(1, span_model_str)
            item.setText(2, stage_str)
            item.setText(3, ttft_str)
            item.setText(4, tps_str)
            item.setText(5, duration_str)
            item.setText(6, status_str)
            item.setText(7, time_str)
            
            # Color code
            color = QColor("#81c784") if status_str == "OK" else QColor("#ff5252")
            brush = QBrush(color)
            for i in range(8):
                item.setForeground(i, brush)
                
            # Add child items
            children = children_map.get(span_dict["span_id"], [])
            children.sort(key=lambda s: s["start_time"])
            for child in children:
                add_item(item, child)
                
        for root in root_spans:
            add_item(self.tree, root)
            
        self.tree.expandAll()

    def show_span_details(self):
        selected_items = self.tree.selectedItems()
        if not selected_items:
            self.details_panel.clear()
            return
        item = selected_items[0]
        span_id = item.data(0, Qt.UserRole)
        
        from core.telemetry import TelemetryStorage
        spans = TelemetryStorage.get_instance().get_all_spans()
        span_dict = next((s for s in spans if s["span_id"] == span_id), None)
        if not span_dict:
            self.details_panel.clear()
            return
            
        # Build beautiful HTML report of the span details
        html = f"<div style='font-family: Arial, sans-serif; line-height: 1.5; padding: 10px;'>"
        html += f"<h2 style='color: #a6e22e; margin-bottom: 5px; border-bottom: 1px solid #3e3e5e; padding-bottom: 5px;'>🔍 Span Detayları: {span_dict['name']}</h2>"
        
        # Meta info table
        html += "<table style='width: 100%; border-collapse: collapse; margin-bottom: 15px;'>"
        html += f"<tr><td style='width: 150px; font-weight: bold; color: #66d9ef;'>Trace ID:</td><td>{span_dict['trace_id']}</td></tr>"
        html += f"<tr><td style='font-weight: bold; color: #66d9ef;'>Span ID:</td><td>{span_dict['span_id']}</td></tr>"
        if span_dict['parent_id']:
            html += f"<tr><td style='font-weight: bold; color: #66d9ef;'>Parent ID:</td><td>{span_dict['parent_id']}</td></tr>"
        html += f"<tr><td style='font-weight: bold; color: #66d9ef;'>Başlangıç:</td><td>{datetime.datetime.fromtimestamp(span_dict['start_time']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}</td></tr>"
        html += f"<tr><td style='font-weight: bold; color: #66d9ef;'>Toplam Süre:</td><td><span style='color: #fd971f; font-weight: bold;'>{span_dict['duration_ms']:.2f} ms</span></td></tr>"
        
        status_color = "#a6e22e" if span_dict['status'] == "OK" else "#f92672"
        html += f"<tr><td style='font-weight: bold; color: #66d9ef;'>Durum:</td><td><span style='color: {status_color}; font-weight: bold;'>{span_dict['status']}</span></td></tr>"
        html += "</table>"
        
        # Performance/Tokens
        attrs = span_dict.get("attributes", {})
        if span_dict.get("llm_model") or "token.total" in attrs:
            html += "<h3 style='color: #f92672; border-bottom: 1px solid #3e3e5e; padding-bottom: 2px;'>📊 LLM & Token İstatistikleri</h3>"
            html += "<ul>"
            if span_dict.get("llm_model"):
                html += f"<li><b>Kullanılan Model:</b> {span_dict['llm_model']}</li>"
            if "llm.quantization_level" in attrs:
                html += f"<li><b>Quantization Seviyesi:</b> {attrs['llm.quantization_level']}</li>"
            if "llm.vram_percent" in attrs:
                html += f"<li><b>VRAM Tahsis Yüzdesi:</b> {attrs['llm.vram_percent']}%</li>"
            if "token.prompt" in attrs:
                html += f"<li><b>Prompt Tokens:</b> {attrs['token.prompt']}</li>"
            if "token.completion" in attrs:
                html += f"<li><b>Completion Tokens:</b> {attrs['token.completion']}</li>"
            if "token.total" in attrs:
                html += f"<li><b>Toplam Token:</b> {attrs['token.total']}</li>"
            if span_dict.get("ttft") is not None:
                html += f"<li><b>Time to First Token (TTFT):</b> {span_dict['ttft']:.2f} s</li>"
            if span_dict.get("tps") is not None:
                html += f"<li><b>Tokens Per Second (TPS):</b> {span_dict['tps']:.2f} token/s</li>"
            html += "</ul>"
            
        # Inference Durations & Python Overhead
        if "duration.total_dur_ms" in attrs or "duration.python_overhead_ms" in attrs:
            html += "<h3 style='color: #66d9ef; border-bottom: 1px solid #3e3e5e; padding-bottom: 2px;'>⚡ Çıkarım (Inference) ve Donanım Süreleri</h3>"
            html += "<ul>"
            if "duration.total_dur_ms" in attrs:
                html += f"<li><b>Model Net Çıkarım Süresi:</b> {attrs['duration.total_dur_ms']:.2f} ms</li>"
            if "duration.load_dur_ms" in attrs:
                html += f"<li><b>Model Yükleme Süresi:</b> {attrs['duration.load_dur_ms']:.2f} ms</li>"
            if "duration.prompt_dur_ms" in attrs:
                html += f"<li><b>Prompt Değerlendirme Süresi:</b> {attrs['duration.prompt_dur_ms']:.2f} ms</li>"
            if "duration.eval_dur_ms" in attrs:
                html += f"<li><b>Yanıt Üretim Süresi:</b> {attrs['duration.eval_dur_ms']:.2f} ms</li>"
            if "duration.python_overhead_ms" in attrs:
                html += f"<li><b>Python Overhead (Sistem Gecikmesi):</b> <span style='color: #ae81ff;'>{attrs['duration.python_overhead_ms']:.2f} ms</span></li>"
            html += "</ul>"
            
        # Hardware info
        if "hw.gpu" in attrs:
            html += "<h3 style='color: #fd971f; border-bottom: 1px solid #3e3e5e; padding-bottom: 2px;'>💻 Yerel Donanım ve Profilleme</h3>"
            html += f"<ul>"
            html += f"<li><b>GPU Donanımı:</b> {attrs.get('hw.gpu', 'Unknown')}</li>"
            html += f"<li><b>Donanım Backend API:</b> {attrs.get('hw.backends', 'CPU')}</li>"
            html += "</ul>"

        # Attributes list
        if attrs:
            html += "<h3 style='color: #ae81ff; border-bottom: 1px solid #3e3e5e; padding-bottom: 2px;'>⚙️ Span Öznitelikleri (Attributes)</h3>"
            html += "<table style='width: 100%; border-collapse: collapse; border: 1px solid #3e3e5e; color: #f8f8f2;'>"
            for k, v in sorted(attrs.items()):
                # Prompts show in dedicated code blocks below
                if k in ["llm.system_prompt", "llm.user_prompt", "llm.raw_response"]:
                    continue
                html += f"<tr style='border-bottom: 1px solid #2e2e4e;'><td style='padding: 5px; font-weight: bold; width: 220px; color: #e6db74;'>{k}</td><td style='padding: 5px; font-family: monospace; word-break: break-all;'>{v}</td></tr>"
            html += "</table><br/>"

        # Prompts and response blocks
        if "llm.system_prompt" in attrs and attrs["llm.system_prompt"]:
            html += f"<h4 style='color: #e6db74; margin-bottom: 5px;'>📝 System Prompt:</h4>"
            html += f"<pre style='background: #272822; color: #f8f8f2; padding: 10px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; font-family: Consolas, monospace; border: 1px solid #3e3e5e;'>{attrs['llm.system_prompt']}</pre>"
        if "llm.user_prompt" in attrs and attrs["llm.user_prompt"]:
            html += f"<h4 style='color: #e6db74; margin-bottom: 5px;'>👤 User Prompt:</h4>"
            html += f"<pre style='background: #272822; color: #f8f8f2; padding: 10px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; font-family: Consolas, monospace; border: 1px solid #3e3e5e;'>{attrs['llm.user_prompt']}</pre>"
        if "llm.raw_response" in attrs and attrs["llm.raw_response"]:
            html += f"<h4 style='color: #e6db74; margin-bottom: 5px;'>🤖 LLM Yanıtı:</h4>"
            html += f"<pre style='background: #272822; color: #a6e22e; padding: 10px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; font-family: Consolas, monospace; border: 1px solid #3e3e5e;'>{attrs['llm.raw_response']}</pre>"

        # Errors & Stacktrace
        error_keys = [k for k in attrs.keys() if k.startswith("error.local.")]
        if error_keys:
            html += "<h3 style='color: #f92672; border-bottom: 1px solid #ef4444; padding-bottom: 2px;'>❌ Hata Esnasındaki Lokal Değişken Durumları</h3>"
            html += "<table style='width: 100%; border-collapse: collapse; border: 1px solid #ef4444; color: #f8f8f2;'>"
            for k in sorted(error_keys):
                var_name = k.replace("error.local.", "")
                html += f"<tr style='border-bottom: 1px solid #ef4444;'><td style='padding: 5px; font-weight: bold; color: #fca5a5; width: 150px;'>{var_name}</td><td style='padding: 5px; font-family: monospace; word-break: break-all;'>{attrs[k]}</td></tr>"
            html += "</table><br/>"
            
        html += "</div>"
        self.details_panel.setHtml(html)

    def clear_data(self):
        reply = QMessageBox.question(self, "Onay", 
            "Tüm telemetry kayıtları silinecek. Emin misiniz?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            from core.telemetry import TelemetryStorage
            TelemetryStorage.get_instance().clear_spans()
            self.refresh_data()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.db_manager = DatabaseManager()
        self.api_manager = APIManager()
        self.model_list = []
        self.worker = None
        self.current_conversation_id = None
        self.is_resuming = False
        
        # YENİ: Performans için chunk sayaçları
        self.chunk_counters = {
            'responder1': 0, 'responder2': 0, 'critic': 0,
            'defense1': 0, 'defense2': 0, 'final_word': 0, 'judge': 0
        }

        self.setWindowTitle("LLM Challenger Pro (Ollama)")
        self.setGeometry(100, 100, 1400, 900)
        
        self.prompt_viewer = PromptViewerWindow(self.db_manager, self)
        
        self.initUI()
        self.redirectOutput()
        self.loadModels()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_splitter = QSplitter(Qt.Vertical)
        main_layout = QHBoxLayout(main_widget)
        main_layout.addWidget(main_splitter)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_splitter = QSplitter(Qt.Horizontal)
        top_layout.addWidget(top_splitter)

        # Sol Panel
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(10)
        control_layout.setAlignment(Qt.AlignTop)

        control_layout.addWidget(QLabel("<b>Model Rollerini Seçin:</b>"))
        
        self.selectors = {}
        roles = {
            "responder1": "Yanıt Verici 1:",
            "responder2": "Yanıt Verici 2:",
            "critic": "Eleştirmen:",
            "judge": "Yüksek Hakim:"
        }
        
        for key, label_text in roles.items():
            role_layout = QHBoxLayout()
            role_label = QLabel(label_text)
            role_label.setFixedWidth(100)
            role_layout.addWidget(role_label)
            combo = QComboBox()
            self.selectors[key] = combo
            role_layout.addWidget(combo)
            control_layout.addLayout(role_layout)

        # YENİ: Ardışıl Mod seçeneği
        self.sequential_checkbox = QCheckBox("Ardışıl (Sequential) Mod")
        self.sequential_checkbox.setToolTip("Büyük modellerde VRAM aşımını önlemek için modelleri sırayla çalıştırır.")
        control_layout.addWidget(self.sequential_checkbox)

        # YENİ: Butonlar (Mola Ver ve Devam Et)
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Başlat")
        self.start_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.start_button.clicked.connect(self.startConversation)
        button_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("Mola Ver")
        self.pause_button.setFont(QFont("Arial", 11))
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pauseConversation)
        self.pause_button.setToolTip("Mevcut LLM çağrıları tamamlanınca durdur")
        button_layout.addWidget(self.pause_button)
        
        self.resume_button = QPushButton("Devam Et")
        self.resume_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.resume_button.setEnabled(False)
        self.resume_button.clicked.connect(self.resumeConversation)
        self.resume_button.setStyleSheet("background-color: #4CAF50; color: white;")
        button_layout.addWidget(self.resume_button)
        
        control_layout.addLayout(button_layout)
        
        # Diğer butonlar
        self.refresh_button = QPushButton("Model Listesini Yenile")
        self.refresh_button.clicked.connect(self.loadModels)
        control_layout.addWidget(self.refresh_button)
        
        self.prompt_viewer_button = QPushButton("Gönderilen Promptları Göster")
        self.prompt_viewer_button.clicked.connect(self.showPromptViewer)
        control_layout.addWidget(self.prompt_viewer_button)
        
        self.telemetry_button = QPushButton("📊 Telemetry")
        self.telemetry_button.clicked.connect(self.showTelemetryViewer)
        control_layout.addWidget(self.telemetry_button)
        
        control_layout.addSpacing(20)

        # İlerleme
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Hazır")
        self.progress_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.progress_label)
        
        control_layout.addSpacing(20)

        # Prompt girişi
        control_layout.addWidget(QLabel("<b>Prompt Girin:</b>"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Tartışma konusunu buraya yazın...")
        self.prompt_input.setMinimumHeight(150)
        control_layout.addWidget(self.prompt_input)

        top_splitter.addWidget(control_widget)

        # Sağ Panel (Tablar)
        self.results_tabs = QTabWidget()
        self.tab_widgets = {}
        self.current_texts = {}  # Biriken metinleri tut
        
        tab_names = [
            ("Yanıt 1", "responder1"),
            ("Yanıt 2", "responder2"), 
            ("Eleştiri", "critic"),
            ("Savunma 1", "defense1"),
            ("Savunma 2", "defense2"),
            ("Eleştirmen Son Söz", "final_word"),
            ("YÜKSEK HAKİM KARARI", "judge")
        ]
        
        for name, key in tab_names:
            text_browser = QTextEdit()
            text_browser.setReadOnly(True)
            self.results_tabs.addTab(text_browser, name)
            self.tab_widgets[key] = text_browser
            self.current_texts[key] = ""  # Başlangıçta boş
            
        top_splitter.addWidget(self.results_tabs)
        top_splitter.setSizes([400, 1000])
        
        main_splitter.addWidget(top_widget)

        # Alt Log Paneli
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.addWidget(QLabel("<b>Sistem Logları:</b>"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_output)
        
        main_splitter.addWidget(log_widget)
        main_splitter.setSizes([700, 200])

    def redirectOutput(self):
        self.stream_redirector = StreamRedirector()
        self.stream_redirector.textWritten.connect(self.updateLog)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.stream_redirector
        sys.stderr = self.stream_redirector

    def updateLog(self, text):
        self.log_output.moveCursor(self.log_output.textCursor().End)
        self.log_output.insertPlainText(text)
        self.original_stdout.write(text)

    def updateProgress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def appendChunkToTab(self, model_key: str, chunk: str):
        """
        Gelen chunk'ı ilgili tab'a Markdown formatında gösterir.
        Performans için her 3 chunk'ta bir veya noktalama işaretlerinde günceller.
        """
        tab_map = {
            'responder1': 'responder1',
            'responder2': 'responder2', 
            'critic': 'critic',
            'defense1': 'defense1',
            'defense2': 'defense2',
            'final_word': 'final_word',
            'judge': 'judge'
        }
        
        if model_key in tab_map:
            key = tab_map[model_key]
            self.current_texts[key] += chunk
            self.chunk_counters[key] += 1
            
            # Her 3 chunk'ta bir veya cümle sonlarında güncelle (performans optimizasyonu)
            counter = self.chunk_counters[key]
            last_char = chunk[-1] if chunk else ''
            
            if (counter % 3 == 0) or (last_char in {'.', '!', '?', '\n', ':', ';', '-'}):
                try:
                    # Qt5.14+ native Markdown desteği
                    self.tab_widgets[key].setMarkdown(self.current_texts[key])
                except AttributeError:
                    # Eski Qt sürümleri için fallback (daha yavaş)
                    self.tab_widgets[key].setHtml(markdown.markdown(self.current_texts[key]))
                
                # Auto-scroll (en alta kaydır)
                scrollbar = self.tab_widgets[key].verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())

    def loadModels(self):
        print("Modeller alınıyor...")
        # UI'ı yükleme durumuna getir (bloke et ve progress bar'ı loading moduna al)
        self.refresh_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # Belirsiz (indeterminate) loading modu
        self.progress_label.setText("Modeller yükleniyor...")
        for key, combo in self.selectors.items():
            combo.setEnabled(False)

        # Arka plan yükleme thread'ini başlat
        self.model_loader = ModelLoaderWorker(self.api_manager)
        self.model_loader.modelsLoaded.connect(self.onModelsLoaded)
        self.model_loader.errorOccurred.connect(self.onModelsLoadError)
        self.model_loader.start()

    def onModelsLoaded(self, models):
        self.model_list = models
        
        # UI elemanlarını tekrar etkinleştir
        self.refresh_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Hazır")
        for key, combo in self.selectors.items():
            combo.setEnabled(True)

        if not self.model_list:
            QMessageBox.warning(self, "Hata", "Ollama çalışmıyor veya model yok.")
            self.resume_button.setEnabled(False)
            return

        for key, combo in self.selectors.items():
            current = combo.currentText()
            combo.clear()
            combo.addItems(self.model_list)
            if current in self.model_list:
                combo.setCurrentText(current)

        # Modeller başarıyla yüklendikten sonra eksik konuşma var mı kontrol et
        self.checkForIncompleteConversation()

    def onModelsLoadError(self, error_msg):
        # Hata durumunda UI'ı geri yükle
        self.refresh_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Yükleme Hatası")
        for key, combo in self.selectors.items():
            combo.setEnabled(True)

        QMessageBox.critical(self, "Hata", f"Modeller yüklenirken hata oluştu: {error_msg}")
        self.checkForIncompleteConversation()

    def validateInputs(self) -> bool:
        models = {}
        for key, combo in self.selectors.items():
            if not combo.currentText():
                QMessageBox.warning(self, "Eksik", f"'{key}' için model seçin.")
                return False
            models[key] = combo.currentText()
        
        if not self.prompt_input.toPlainText().strip():
            QMessageBox.warning(self, "Eksik", "Prompt girin.")
            return False
            
        if models['responder1'] == models['responder2']:
            QMessageBox.warning(self, "Hata", "Yanıt vericiler farklı olmalıdır.")
            return False
            
        return True

    def clearTabs(self):
        """Tüm tabları temizle ve sayaçları sıfırla"""
        for key in self.tab_widgets:
            self.tab_widgets[key].clear()
            self.current_texts[key] = ""
            self.chunk_counters[key] = 0  # Sayaçları da sıfırla

    def startConversation(self):
        if not self.validateInputs():
            return
            
        models = {k: c.currentText() for k, c in self.selectors.items()}
        prompt = self.prompt_input.toPlainText().strip()
        
        self.clearTabs()
        self.current_conversation_id = self.db_manager.create_conversation(prompt, models)
        logger.info(f"Arayüz: Yeni tartışma başlatılıyor. ID: {self.current_conversation_id} | Model Konfigürasyonu: {models}")
        
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Ardışıl mod durumunu API Manager'a bildir
        self.api_manager.sequential_mode = self.sequential_checkbox.isChecked()
        
        self.worker = Worker(
            self.api_manager, self.db_manager, prompt, models, 
            self.current_conversation_id
        )
        self.worker.finished.connect(self.onConversationFinished)
        self.worker.error.connect(self.onConversationError)
        self.worker.progress.connect(self.updateProgress)
        self.worker.chunk_received.connect(self.appendChunkToTab)
        self.worker.paused.connect(self.onConversationPaused)
        self.worker.start()

    def pauseConversation(self):
        """Mola ver - mevcut adım bitince durur"""
        if self.worker and self.worker.isRunning():
            self.pause_button.setEnabled(False)
            self.pause_button.setText("Duraklatılıyor...")
            logger.info(f"Arayüz: Konuşma #{self.current_conversation_id} için duraklatma talebi gönderildi (mevcut adım bitince duracak).")
            self.worker.pause()

    def onConversationPaused(self):
        """Duraklatıldığında çağrılır"""
        logger.info(f"Arayüz: Konuşmaya mola verildi (Konuşma #{self.current_conversation_id}).")
        self.pause_button.setText("Mola Ver")
        self.pause_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.resume_button.setEnabled(True)
        self.progress_label.setText("Duraklatıldı")
        self.worker = None

    def checkForIncompleteConversation(self):
        """Başlangıçta tamamlanmamış konuşma var mı kontrol et"""
        incomplete = self.db_manager.get_incomplete_conversation()
        if incomplete:
            self.current_conversation_id, data = incomplete
            self.resume_button.setEnabled(True)
            self.resume_button.setToolTip(f"Konuşma #{self.current_conversation_id} kaldığı yerden devam eder")
            logger.info(f"Arayüz: Tamamlanmamış konuşma bulundu. Geri yüklenebilir ID: {self.current_conversation_id}")

    def resumeConversation(self):
        """Kaldığı yerden devam et"""
        incomplete = self.db_manager.get_incomplete_conversation()
        if not incomplete:
            QMessageBox.information(self, "Bilgi", "Devam edilecek konuşma yok.")
            return
            
        conv_id, data = incomplete
        self.current_conversation_id = conv_id
        
        # Model seçimlerini yükle
        models = data['models']
        for key, combo in self.selectors.items():
            if key in models and models[key] in self.model_list:
                combo.setCurrentText(models[key])
        
        # Prompt'u yükle
        self.prompt_input.setPlainText(data['prompt'])
        
        # Stage ve mevcut verileri al
        stage = Stage(data['stage'])
        resume_data = data['data']
        
        logger.info(f"Arayüz: Konuşma #{self.current_conversation_id} kaldığı yerden (Aşama: {stage.name}) devam ettiriliyor.")
        
        # UI'ı hazırla
        self.clearTabs()
        # Önceki metinleri geri yükle (varsa)
        for key, text in resume_data.items():
            if key in self.current_texts:
                self.current_texts[key] = text
                self.tab_widgets[key].setMarkdown(text)  # Markdown olarak göster
        
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.progress_bar.setValue(stage.value * 20)
        
        # Ardışıl mod durumunu API Manager'a bildir
        self.api_manager.sequential_mode = self.sequential_checkbox.isChecked()
        
        # Worker'ı resume modunda başlat
        self.worker = Worker(
            self.api_manager, self.db_manager, 
            data['prompt'], models, conv_id,
            start_stage=stage, resume_data=resume_data
        )
        self.worker.finished.connect(self.onConversationFinished)
        self.worker.error.connect(self.onConversationError)
        self.worker.progress.connect(self.updateProgress)
        self.worker.chunk_received.connect(self.appendChunkToTab)
        self.worker.paused.connect(self.onConversationPaused)
        self.worker.start()
        
        logger.info(f"Arayüz: Konuşma #{conv_id} başlatıldı.")

    def onConversationFinished(self, results: Dict):
        # Markdown olarak final render
        try:
            self.tab_widgets['responder1'].setMarkdown(results['responder1']['response'])
            self.tab_widgets['responder2'].setMarkdown(results['responder2']['response'])
            self.tab_widgets['critic'].setMarkdown(results['critic']['critique'])
            self.tab_widgets['defense1'].setMarkdown(results['responder1']['defense'])
            self.tab_widgets['defense2'].setMarkdown(results['responder2']['defense'])
            self.tab_widgets['final_word'].setMarkdown(results['critic']['final_word'])
            self.tab_widgets['judge'].setMarkdown(results['judge']['verdict'])
            
            self.results_tabs.setCurrentWidget(self.tab_widgets['judge'])
            logger.info(f"Arayüz: Konuşma #{self.current_conversation_id} başarıyla bitti ve tüm aşamalar render edildi.")
            QMessageBox.information(self, "Tamamlandı", "Tartışma tamamlandı!")
        except Exception as e:
            logger.error(f"Arayüz: Render hatası: {e}")
            
        self.resetUI()

    def onConversationError(self, error_message: str):
        logger.error(f"Arayüz: Konuşma #{self.current_conversation_id} hata ile sonlandı: {error_message}")
        QMessageBox.critical(self, "Hata", error_message)
        self.resetUI()

    def resetUI(self):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Mola Ver")
        self.resume_button.setEnabled(True)
        self.worker = None
        self.checkForIncompleteConversation()

    def showPromptViewer(self):
        conv_id = self.current_conversation_id or self.db_manager.get_latest_conversation_id()
        logger.info(f"Arayüz: Prompt kayıtları görüntüleme penceresi açıldı (Konuşma #{conv_id})")
        self.prompt_viewer.show()
        self.prompt_viewer.activateWindow()

    def showTelemetryViewer(self):
        logger.info("Arayüz: Yerel Telemetry İzleyici penceresi açıldı")
        dialog = TelemetryViewerDialog(self)
        dialog.exec_()

    def closeEvent(self, event):
        logger.info("Arayüz: Uygulama kapatılıyor...")
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, "Onay", 
                "Aktif konuşma var. Mevcut adım bitince durdurulsun mu?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            
            if reply == QMessageBox.Yes:
                logger.info("Arayüz: Kapatma öncesinde aktif konuşma duraklatılıyor...")
                self.worker.pause()
                self.worker.wait(5000)
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
                
        self.db_manager.close()
        sys.stdout = self.original_stdout
        event.accept()

if __name__ == "__main__":
    from core.logger_setup import setup_logger
    logger_setup = setup_logger()
    logger_setup.info("GUI Uygulaması başlatılıyor...")
    
    app = QApplication(sys.argv)
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    except ImportError:
        app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())