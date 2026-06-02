# core/telemetry.py

import os
import logging
import collections
import threading
import datetime
import time
import re
import inspect
import functools
import traceback
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanContext, TraceFlags, NonRecordingSpan, set_span_in_context
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, SimpleSpanProcessor

logger = logging.getLogger("telemetry")

# Global tracer, varsayılan olarak No-Op
tracer = otel_trace.get_tracer("noop")

class TelemetryStorage:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.spans = collections.deque(maxlen=500)
        self.perf_cache = {}  # (conv_id, model_name) -> list of performance metrics
        self.lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_perf(self, conv_id, model, ttft, tps, tokens):
        with self.lock:
            key = (conv_id, model)
            if key not in self.perf_cache:
                self.perf_cache[key] = []
            self.perf_cache[key].append({
                "ttft": ttft,
                "tps": tps,
                "token_count": tokens
            })

    def pop_perf(self, conv_id, model):
        with self.lock:
            key = (conv_id, model)
            if key in self.perf_cache and self.perf_cache[key]:
                return self.perf_cache[key].pop(0)
            return None

    def add_span(self, span_dict):
        with self.lock:
            self.spans.append(span_dict)

    def get_all_spans(self):
        with self.lock:
            return list(self.spans)

    def clear_spans(self):
        with self.lock:
            self.spans.clear()
            self.perf_cache.clear()

def get_system_ram_info():
    """Windows üzerinde sistem RAM bilgilerini (Total/Free GB) sorgular."""
    try:
        import subprocess
        output = subprocess.check_output(
            "powershell -Command \"Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory | Format-Table -HideTableHeaders\"",
            shell=True,
            text=True
        )
        nums = re.findall(r'\d+', output)
        if len(nums) >= 2:
            total_gb = int(nums[0]) / (1024 * 1024)
            free_gb = int(nums[1]) / (1024 * 1024)
            return round(total_gb, 2), round(free_gb, 2)
    except Exception:
        pass
    return 16.0, 8.0

def detect_hardware_backend():
    """İşlemcinin / Ekran kartının (Intel Arc vb.) yeteneklerini ve API backendlerini analiz eder."""
    import shutil
    backends = []
    
    # 1. Vulkan Tespiti
    if shutil.which("vulkaninfo") or os.path.exists("C:\\Windows\\System32\\vulkan-1.dll"):
        backends.append("Vulkan")
        
    # 2. DirectML / DirectX 12 Tespiti
    if os.path.exists("C:\\Windows\\System32\\DirectML.dll") or os.path.exists("C:\\Windows\\System32\\d3d12.dll"):
        backends.append("DirectML/Direct3D12")
        
    # 3. CUDA Tespiti
    if shutil.which("nvidia-smi") or os.path.exists("C:\\Windows\\System32\\nvcuda.dll"):
        backends.append("CUDA/Nvidia")
        
    # 4. OpenVINO Tespiti
    try:
        import openvino
        backends.append("OpenVINO")
    except ImportError:
        pass
        
    gpu_info = "Intel(R) Arc(TM) Graphics"  # Sistemde aktif GPU
    
    return {
        "hardware_backends": ", ".join(backends) if backends else "CPU",
        "gpu_info": gpu_info
    }

def get_model_telemetry_info(model_name: str):
    """Ollama API üzerinden aktif çalışan modelin quantization ve VRAM yüklenme istatistiklerini sorgular."""
    try:
        import ollama
        client = ollama.Client()
        running = client.ps()
        for m in running.models:
            if m.name == model_name or m.model == model_name:
                vram_percent = (m.size_vram / m.size) * 100 if m.size else 0
                quant = m.details.quantization_level if m.details else None
                return {
                    "quantization_level": quant,
                    "vram_size_bytes": m.size_vram,
                    "model_size_bytes": m.size,
                    "vram_percent": round(vram_percent, 2)
                }
        # Model aktif yüklenmemişse show() ile şemasına bak
        info = client.show(model_name)
        quant = info.details.quantization_level if hasattr(info, 'details') and info.details else None
        return {
            "quantization_level": quant,
            "vram_size_bytes": 0,
            "model_size_bytes": 0,
            "vram_percent": 0
        }
    except Exception:
        return {}

class ToolTracer:
    """SQLite sorguları, dosya I/O ve MCP çağrıları için decorator ve context manager."""
    def __init__(self, tool_name: str, schema: dict = None):
        self.tool_name = tool_name
        self.schema = schema or {}
        self.span_ctx = None
        self.span = None

    def __enter__(self):
        global tracer
        self.span_ctx = tracer.start_as_current_span(f"tool_call:{self.tool_name}")
        if self.span_ctx:
            self.span = self.span_ctx.__enter__()
            if self.span:
                self.span.set_attribute("tool.name", self.tool_name)
                for k, v in self.schema.items():
                    # Büyük parametreleri kırp
                    val_str = str(v)
                    if len(val_str) > 1000:
                        val_str = val_str[:1000] + "... [TRUNCATED]"
                    self.span.set_attribute(f"tool.arg.{k}", val_str)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span_ctx:
            if exc_type and self.span:
                self.span.record_exception(exc_val)
                self.span.set_status(otel_trace.StatusCode.ERROR, str(exc_val))
                try:
                    tb = exc_tb
                    while tb.tb_next:
                        tb = tb.tb_next
                    frame = tb.tb_frame
                    for name, val in frame.f_locals.items():
                        self.span.set_attribute(f"error.local.{name}", str(val)[:500])
                except Exception:
                    pass
            self.span_ctx.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                sc = dict(self.schema)
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    for k, v in bound.arguments.items():
                        if k != "self":
                            sc[k] = v
                except Exception:
                    pass
                async with ToolTracer(self.tool_name or func.__name__, sc):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                sc = dict(self.schema)
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    for k, v in bound.arguments.items():
                        if k != "self":
                            sc[k] = v
                except Exception:
                    pass
                with ToolTracer(self.tool_name or func.__name__, sc):
                    return func(*args, **kwargs)
            return sync_wrapper

class LLMSpanWrapper:
    """LLM API çağrısını sarmalayıp tüm zengin parametre, donanım, token ve hata bilgilerini toplayan sınıf."""
    def __init__(self, model_name: str, messages: list, temperature: float, stream: bool):
        self.model_name = model_name
        self.messages = messages
        self.temperature = temperature
        self.stream = stream
        self.span_ctx = None
        self.span = None
        self.start_time = None

    def __enter__(self):
        global tracer
        total_ram, free_ram = get_system_ram_info()
        hw_info = detect_hardware_backend()
        
        self.span_ctx = tracer.start_as_current_span("ollama_api_call")
        if self.span_ctx:
            self.span = self.span_ctx.__enter__()
            if self.span:
                self.start_time = time.perf_counter()
                
                # Zenginleştirilmiş Girdi Öznitelikleri (Bağlam)
                system_prompt = ""
                user_prompt = ""
                if isinstance(self.messages, list):
                    system_prompts = [m.get("content", "") for m in self.messages if isinstance(m, dict) and m.get("role") == "system"]
                    user_prompts = [m.get("content", "") for m in self.messages if isinstance(m, dict) and m.get("role") == "user"]
                    system_prompt = "\n".join(system_prompts)
                    user_prompt = "\n".join(user_prompts)
                
                self.span.set_attribute("llm.model", self.model_name)
                self.span.set_attribute("llm.system_prompt", system_prompt)
                self.span.set_attribute("llm.user_prompt", user_prompt)
                self.span.set_attribute("llm.temperature", self.temperature)
                self.span.set_attribute("llm.stream", self.stream)
                
                # Donanım ve Profilleme Etiketleri
                self.span.set_attribute("hw.gpu", hw_info["gpu_info"])
                self.span.set_attribute("hw.backends", hw_info["hardware_backends"])
                
                model_info = get_model_telemetry_info(self.model_name)
                if model_info:
                    self.span.set_attribute("llm.quantization_level", model_info.get("quantization_level") or "unknown")
                    self.span.set_attribute("llm.vram_percent", model_info.get("vram_percent", 0))
                
                # Donanım Kaynak Olayları (Events)
                self.span.add_event("memory_allocation_limits", {
                    "total_ram_gb": total_ram,
                    "available_ram_gb": free_ram,
                    "vram_allocated_bytes": model_info.get("vram_size_bytes", 0) if model_info else 0
                })
        return self

    def record_final_response(self, raw_response: str, response_obj):
        if self.span:
            self.span.set_attribute("llm.raw_response", raw_response)
            
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            total_dur = 0.0
            load_dur = 0.0
            prompt_dur = 0.0
            eval_dur = 0.0
            
            if response_obj:
                if isinstance(response_obj, dict):
                    prompt_tokens = response_obj.get("prompt_eval_count", 0) or 0
                    completion_tokens = response_obj.get("eval_count", 0) or 0
                    total_tokens = prompt_tokens + completion_tokens
                    
                    total_dur = (response_obj.get("total_duration", 0) or 0) / 1_000_000
                    load_dur = (response_obj.get("load_duration", 0) or 0) / 1_000_000
                    prompt_dur = (response_obj.get("prompt_eval_duration", 0) or 0) / 1_000_000
                    eval_dur = (response_obj.get("eval_duration", 0) or 0) / 1_000_000
                else:
                    prompt_tokens = getattr(response_obj, "prompt_eval_count", 0) or 0
                    completion_tokens = getattr(response_obj, "eval_count", 0) or 0
                    total_tokens = prompt_tokens + completion_tokens
                    
                    total_dur = (getattr(response_obj, "total_duration", 0) or 0) / 1_000_000
                    load_dur = (getattr(response_obj, "load_duration", 0) or 0) / 1_000_000
                    prompt_dur = (getattr(response_obj, "prompt_eval_duration", 0) or 0) / 1_000_000
                    eval_dur = (getattr(response_obj, "eval_duration", 0) or 0) / 1_000_000
                
                # Token Metrikleri
                self.span.set_attribute("token.prompt", prompt_tokens)
                self.span.set_attribute("token.completion", completion_tokens)
                self.span.set_attribute("token.total", total_tokens)
                
                # Çıkarım Zaman Detayları
                self.span.set_attribute("duration.total_dur_ms", total_dur)
                self.span.set_attribute("duration.load_dur_ms", load_dur)
                self.span.set_attribute("duration.prompt_dur_ms", prompt_dur)
                self.span.set_attribute("duration.eval_dur_ms", eval_dur)
                
                # Python Overhead Hesaplaması
                if self.start_time:
                    total_elapsed = (time.perf_counter() - self.start_time) * 1000
                    overhead = max(0.0, total_elapsed - total_dur)
                    self.span.set_attribute("duration.python_overhead_ms", overhead)

    def record_error(self, exception):
        if self.span:
            self.span.record_exception(exception)
            self.span.set_status(otel_trace.StatusCode.ERROR, str(exception))
            # Hata anındaki lokal değişken analizi (Local Variable Tracking)
            try:
                tb = exception.__traceback__
                while tb.tb_next:
                    tb = tb.tb_next
                frame = tb.tb_frame
                for name, val in frame.f_locals.items():
                    self.span.set_attribute(f"error.local.{name}", str(val)[:500])
            except Exception:
                pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span_ctx:
            if exc_type:
                self.record_error(exc_val)
            self.span_ctx.__exit__(exc_type, exc_val, exc_tb)

class TelemetryManager:
    """Projedeki tüm telemetri operasyonlarını yöneten ana gözlemlenebilirlik sınıfı."""
    @classmethod
    def trace_tool_execution(cls, tool_name: str, schema: dict = None):
        return ToolTracer(tool_name, schema)

    @classmethod
    def start_llm_span(cls, model_name: str, messages: list, temperature: float, stream: bool):
        return LLMSpanWrapper(model_name, messages, temperature, stream)

class LocalSpanExporter(SpanExporter):
    def __init__(self):
        super().__init__()

    def export(self, spans) -> SpanExportResult:
        storage = TelemetryStorage.get_instance()
        for span in spans:
            try:
                trace_id = span.context.trace_id
                span_id = span.context.span_id
                parent_id = span.parent.span_id if span.parent else None
                name = span.name
                
                start_time = span.start_time / 1_000_000_000 if span.start_time else 0
                end_time = span.end_time / 1_000_000_000 if span.end_time else 0
                duration_ms = (span.end_time - span.start_time) / 1_000_000 if span.start_time and span.end_time else 0
                
                attributes = dict(span.attributes) if span.attributes else {}
                
                status_code = span.status.status_code if hasattr(span.status, "status_code") else None
                if status_code == otel_trace.StatusCode.ERROR:
                    status_str = "ERROR"
                elif status_code == otel_trace.StatusCode.OK:
                    status_str = "OK"
                else:
                    status_str = "OK"
                
                if attributes.get("output.status") == "error":
                    status_str = "ERROR"
                    
                llm_model = attributes.get("llm.model")
                conversation_id = attributes.get("conversation.id", trace_id)
                
                # Kesin token ve perf metriklerini attributes'dan veya perf_cache'den al
                ttft = attributes.get("ttft")
                tps = attributes.get("tps")
                token_count = attributes.get("token.total")
                
                if llm_model:
                    perf = storage.pop_perf(conversation_id, llm_model)
                    if perf:
                        if ttft is None:
                            ttft = perf["ttft"]
                        if tps is None:
                            tps = perf["tps"]
                        if token_count is None:
                            token_count = perf["token_count"]
                
                span_dict = {
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "parent_id": parent_id,
                    "name": name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_ms": duration_ms,
                    "attributes": attributes,
                    "status": status_str,
                    "llm_model": llm_model,
                    "conversation_id": conversation_id,
                    "ttft": ttft,
                    "tps": tps,
                    "token_count": token_count
                }
                storage.add_span(span_dict)
            except Exception as e:
                logger.warning(f"Error in LocalSpanExporter.export for span: {e}")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

def init_tracing(service_name="llm_ensemble_arena", endpoint=None):
    """OpenTelemetry Tracing altyapısını yerel LocalSpanExporter ile başlatır."""
    global tracer
    enable_tracing = os.getenv("ENABLE_TRACING", "true").lower() == "true"
    
    if not enable_tracing:
        logger.info("OpenTelemetry Tracing pasif durumda (ENABLE_TRACING=false). No-Op kullanılıyor.")
        tracer = otel_trace.get_tracer("noop")
        return tracer

    try:
        otel_service_name = os.getenv("OTEL_SERVICE_NAME", service_name)
        resource = Resource(attributes={"service.name": otel_service_name})
        provider = TracerProvider(resource=resource)
        
        exporter = LocalSpanExporter()
        processor = SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)
        
        otel_trace.set_tracer_provider(provider)
        tracer = otel_trace.get_tracer(otel_service_name)
        logger.info(f"OpenTelemetry Tracing başarıyla başlatıldı (Yerel Exporter). Servis: {otel_service_name}")
    except Exception as e:
        logger.warning(f"OpenTelemetry başlatılırken hata oluştu (No-Op'a dönülüyor): {e}")
        tracer = otel_trace.get_tracer("noop")
        
    return tracer

def record_span(span_name: str, attributes: dict):
    """Basit ve tek seferlik span kaydı gerçekleştiren yardımcı metot."""
    try:
        with tracer.start_as_current_span(span_name) as span:
            for k, v in attributes.items():
                span.set_attribute(k, v)
    except Exception as e:
        logger.warning(f"Telemetry record_span hatası: {e}")

def get_conversation_context(conversation_id: int):
    """conversation_id değerini 128-bit (16-byte) Trace ID ile eşleştirir."""
    try:
        if not conversation_id or conversation_id <= 0:
            return None
        
        trace_id_val = conversation_id
        import random
        span_id_val = random.getrandbits(64)
        
        parent_context = SpanContext(
            trace_id=trace_id_val,
            span_id=span_id_val,
            is_remote=False,
            trace_flags=TraceFlags(0x01)
        )
        parent_span = NonRecordingSpan(parent_context)
        return set_span_in_context(parent_span)
    except Exception as e:
        logger.warning(f"Telemetry get_conversation_context hatası: {e}")
        return None

def patch_database_manager():
    """SQLite veritabanı işlemlerini dinamik olarak izlemek için monkey-patch uygular."""
    try:
        from db.database_manager import DatabaseManager
        if not getattr(DatabaseManager, "_is_telemetry_patched", False):
            # 1. Performance patch
            original_record = DatabaseManager.record_performance
            def patched_record_performance(self, conv_id, model, ttft, tps, tokens):
                try:
                    storage = TelemetryStorage.get_instance()
                    storage.record_perf(conv_id, model, ttft, tps, tokens)
                except Exception as e:
                    logger.warning(f"Error in patched record_performance: {e}")
                return original_record(self, conv_id, model, ttft, tps, tokens)
            DatabaseManager.record_performance = patched_record_performance
            
            # 2. SQLite Write operations tracing
            original_queue_write = DatabaseManager._queue_write
            def patched_queue_write(self, query, params=None):
                with TelemetryManager.trace_tool_execution("sqlite_write", {"query": query, "params": params}):
                    return original_queue_write(self, query, params)
            DatabaseManager._queue_write = patched_queue_write
            
            DatabaseManager._is_telemetry_patched = True
            logger.info("DatabaseManager has been successfully patched for aggressive telemetry.")
    except Exception as e:
        logger.error(f"DatabaseManager monkey-patch telemetry hatası: {e}")

def patch_ollama_client():
    """Ollama AsyncClient ve Client chat metotlarını zenginleştirilmiş tracing için sarmalar."""
    try:
        import ollama
        if not getattr(ollama, "_is_telemetry_patched", False):
            # 1. AsyncClient.chat Patch
            original_async_chat = ollama.AsyncClient.chat
            async def patched_async_chat(self_client, *args, **kwargs):
                model_name = kwargs.get("model") or (args[0] if len(args) > 0 else "unknown")
                messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
                options = kwargs.get("options") or {}
                temperature = options.get("temperature", 0.5)
                stream = kwargs.get("stream", False)
                
                with TelemetryManager.start_llm_span(model_name, messages, temperature, stream) as span_wrapper:
                    try:
                        if stream:
                            gen = await original_async_chat(self_client, *args, **kwargs)
                            async def wrap_generator(g):
                                full_response_parts = []
                                last_chunk = None
                                async for chunk in g:
                                    content = ""
                                    if hasattr(chunk, 'message') and chunk.message:
                                        content = chunk.message.content or ""
                                    elif isinstance(chunk, dict):
                                        content = chunk.get('message', {}).get('content', '')
                                    full_response_parts.append(content)
                                    last_chunk = chunk
                                    yield chunk
                                
                                full_response = "".join(full_response_parts)
                                span_wrapper.record_final_response(full_response, last_chunk)
                            return wrap_generator(gen)
                        else:
                            response = await original_async_chat(self_client, *args, **kwargs)
                            raw_txt = ""
                            if isinstance(response, dict):
                                raw_txt = response.get('message', {}).get('content', '')
                            elif hasattr(response, 'message') and response.message:
                                raw_txt = response.message.content or ""
                            span_wrapper.record_final_response(raw_txt, response)
                            return response
                    except Exception as e:
                        span_wrapper.record_error(e)
                        raise
            ollama.AsyncClient.chat = patched_async_chat

            # 2. Client.chat Patch
            original_sync_chat = ollama.Client.chat
            def patched_sync_chat(self_client, *args, **kwargs):
                model_name = kwargs.get("model") or (args[0] if len(args) > 0 else "unknown")
                messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
                options = kwargs.get("options") or {}
                temperature = options.get("temperature", 0.5)
                stream = kwargs.get("stream", False)
                
                with TelemetryManager.start_llm_span(model_name, messages, temperature, stream) as span_wrapper:
                    try:
                        if stream:
                            gen = original_sync_chat(self_client, *args, **kwargs)
                            def wrap_generator(g):
                                full_response_parts = []
                                last_chunk = None
                                for chunk in g:
                                    content = ""
                                    if hasattr(chunk, 'message') and chunk.message:
                                        content = chunk.message.content or ""
                                    elif isinstance(chunk, dict):
                                        content = chunk.get('message', {}).get('content', '')
                                    full_response_parts.append(content)
                                    last_chunk = chunk
                                    yield chunk
                                
                                full_response = "".join(full_response_parts)
                                span_wrapper.record_final_response(full_response, last_chunk)
                            return wrap_generator(gen)
                        else:
                            response = original_sync_chat(self_client, *args, **kwargs)
                            raw_txt = ""
                            if isinstance(response, dict):
                                raw_txt = response.get('message', {}).get('content', '')
                            elif hasattr(response, 'message') and response.message:
                                raw_txt = response.message.content or ""
                            span_wrapper.record_final_response(raw_txt, response)
                            return response
                    except Exception as e:
                        span_wrapper.record_error(e)
                        raise
            ollama.Client.chat = patched_sync_chat
            
            ollama._is_telemetry_patched = True
            logger.info("Ollama Clients have been successfully patched for aggressive telemetry.")
    except Exception as e:
        logger.error(f"Ollama client monkey-patch telemetry hatası: {e}")

# Telemetri patch işlemlerini otomatik tetikle
patch_database_manager()
patch_ollama_client()
