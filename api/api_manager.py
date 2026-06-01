# api/api_manager.py

import ollama
import os
import asyncio
import logging
import time
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Dict, Optional, Any, AsyncGenerator

load_dotenv()

logger = logging.getLogger("api_manager")

class APIManager:
    def __init__(self, host: Optional[str] = None):
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if not self.host:
            raise ValueError("OLLAMA_HOST adresi bulunamadı!")
        
        # timeout değeri (varsayılan 300.0 saniye büyük modellerin yüklenmesi için uygundur)
        try:
            timeout_val = float(os.getenv("OLLAMA_TIMEOUT", "300.0"))
        except ValueError:
            timeout_val = 300.0
        self.client = ollama.AsyncClient(host=self.host, timeout=timeout_val)
        self.sync_client = ollama.Client(host=self.host, timeout=timeout_val)
        
        # keep_alive değeri (varsayılan "0" yani hemen bellekten sil)
        keep_alive_env = os.getenv("OLLAMA_KEEP_ALIVE", "0")
        try:
            if keep_alive_env.isdigit():
                self.keep_alive = int(keep_alive_env)
            else:
                self.keep_alive = float(keep_alive_env)
        except ValueError:
            self.keep_alive = keep_alive_env
        
        # Concurrency sınırı (Semaphore)
        try:
            max_calls = int(os.getenv("MAX_CONCURRENT_CALLS", "2"))
        except ValueError:
            max_calls = 2
        self.semaphore = asyncio.Semaphore(max_calls)
        self.sequential_semaphore = asyncio.Semaphore(1)
        self.sequential_mode = False
        logger.info(f"APIManager başlatıldı: {self.host} (Max Concurrency: {max_calls})")
        
        # OpenTelemetry Tracing Başlat
        try:
            from core.telemetry import init_tracing
            init_tracing()
        except Exception as te:
            logger.warning(f"Telemetry başlatılırken hata oluştu: {te}")

    def get_model_list(self) -> List[str]:
        try:
            models_data = self.sync_client.list()
            model_names = []
            for model_item in models_data.get('models', []):
                name = model_item.get('name') or model_item.get('model')
                if name:
                    model_names.append(name)
            return model_names
        except Exception as e:
            logger.error(f"Model listesi alınamadı: {e}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ollama.ResponseError)),
        reraise=True
    )
    async def call_llm(self, model_name: str, messages: List[Dict[str, str]], 
                      temperature: float = 0.5, stream: bool = False) -> str:
        """
        Belirtilen Ollama modeline asenkron çağrı yapar.
        stream=True ise generator döner (streaming için).
        """
        sem = self.sequential_semaphore if self.sequential_mode else self.semaphore
        async with sem:
            try:
                logger.info(f"API Çağrısı: [{model_name}]'e istek gönderiliyor (stream={stream})...")
                
                if stream:
                    # Streaming mod: Generator döndür (bu fonksiyon async generator olmalı)
                    raise ValueError("Streaming için call_llm_stream kullanın")
                
                response = await self.client.chat(
                    model=model_name,
                    messages=messages,
                    options={"temperature": temperature},
                    keep_alive=self.keep_alive
                )
                content = response['message']['content']
                logger.info(f"API Yanıt: [{model_name}] tamamlandı ({len(content)} karakter)")
                return content
                
            except Exception as e:
                error_msg = f"API Hatası ({model_name}): {e}"
                logger.error(error_msg)
                raise

    async def call_llm_stream(self, model_name: str, messages: List[Dict[str, str]], 
                             temperature: float = 0.5, conv_id: Optional[int] = None,
                             db_manager: Optional[Any] = None) -> AsyncGenerator[str, None]:
        """
        Streaming LLM çağrısı. Chunk chunk yanıt döndürür.
        """
        sem = self.sequential_semaphore if self.sequential_mode else self.semaphore
        async with sem:
            if conv_id is None:
                conv_id = getattr(self, "current_conv_id", None)
            if db_manager is None:
                db_manager = getattr(self, "db_manager", None)

            # OpenTelemetry Tracing Span Başlat
            from core.telemetry import tracer
            import opentelemetry.trace as otel_trace

            prompt_len = 0
            if messages:
                if isinstance(messages, list):
                    prompt_len = sum(len(m.get('content', '')) for m in messages)
                else:
                    prompt_len = len(str(messages))

            span_ctx = None
            span = None
            try:
                span_ctx = tracer.start_as_current_span("ollama_api_call")
                if span_ctx:
                    span = span_ctx.__enter__()
                    if span:
                        span.set_attribute("llm.model", model_name)
                        span.set_attribute("prompt.length", prompt_len)
                        span.set_attribute("stream.enabled", True)
            except Exception as te:
                logger.warning(f"Telemetry error starting span: {te}")
                span_ctx = None

            start_time = time.perf_counter()
            ttft = 0.0
            first_chunk_received = False
            token_count = 0

            try:
                logger.info(f"Stream Başladı: [{model_name}]")
                stream = await self.client.chat(
                    model=model_name,
                    messages=messages,
                    options={"temperature": temperature},
                    stream=True,
                    keep_alive=self.keep_alive
                )
                
                async for chunk in stream:
                    content = chunk['message']['content']
                    if content:
                        if not first_chunk_received:
                            ttft = time.perf_counter() - start_time
                            first_chunk_received = True
                        token_count += max(1, len(content.strip().split()))
                        yield content
                        
            except Exception as e:
                logger.error(f"HATA Stream ({model_name}): {e}")
                if span:
                    try:
                        span.record_exception(e)
                        span.set_status(otel_trace.StatusCode.ERROR, str(e))
                    except Exception as te:
                        logger.warning(f"Telemetry error recording exception: {te}")
                raise
            finally:
                total_time = time.perf_counter() - start_time
                tps = token_count / max(total_time, 0.01)
                logger.info(f"📊 Perf | Conv: {conv_id} | Model: {model_name} | TTFT: {ttft:.2f}s | ~TPS: {tps:.1f} | ~Tokens: {token_count}")
                if db_manager and hasattr(db_manager, "record_performance") and conv_id is not None:
                    try:
                        db_manager.record_performance(conv_id, model_name, ttft, tps, token_count)
                    except Exception as db_err:
                        logger.error(f"Performans veritabanına yazılamadı: {db_err}")

                if span_ctx:
                    try:
                        if span:
                            span.set_attribute("response.duration_ms", int(total_time * 1000))
                        span_ctx.__exit__(None, None, None)
                    except Exception as te:
                        logger.warning(f"Telemetry error exiting span: {te}")