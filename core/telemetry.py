# core/telemetry.py

import os
import logging
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanContext, TraceFlags, NonRecordingSpan, set_span_in_context

logger = logging.getLogger("telemetry")

# Global tracer, varsayılan olarak No-Op
tracer = otel_trace.get_tracer("noop")

def init_tracing(service_name="llm_ensemble_arena", endpoint="http://localhost:4318/v1/traces"):
    """
    OpenTelemetry Tracing altyapısını başlatır.
    ENABLE_TRACING=true değilse veya bir hata oluşursa No-Op tracer döner.
    """
    global tracer
    enable_tracing = os.getenv("ENABLE_TRACING", "false").lower() == "true"
    
    if not enable_tracing:
        logger.info("OpenTelemetry Tracing pasif durumda (ENABLE_TRACING=false). No-Op kullanılıyor.")
        tracer = otel_trace.get_tracer("noop")
        return tracer

    try:
        otel_service_name = os.getenv("OTEL_SERVICE_NAME", service_name)
        resource = Resource(attributes={"service.name": otel_service_name})
        provider = TracerProvider(resource=resource)
        
        exporter_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
        exporter = OTLPSpanExporter(endpoint=exporter_endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        
        otel_trace.set_tracer_provider(provider)
        tracer = otel_trace.get_tracer(otel_service_name)
        logger.info(f"OpenTelemetry Tracing başarıyla başlatıldı. Servis: {otel_service_name}, Endpoint: {exporter_endpoint}")
    except Exception as e:
        logger.warning(f"OpenTelemetry başlatılırken hata oluştu (No-Op'a dönülüyor): {e}")
        tracer = otel_trace.get_tracer("noop")
        
    return tracer

def record_span(span_name: str, attributes: dict):
    """
    Basit ve tek seferlik span kaydı gerçekleştiren yardımcı metot.
    """
    try:
        with tracer.start_as_current_span(span_name) as span:
            for k, v in attributes.items():
                span.set_attribute(k, v)
    except Exception as e:
        logger.warning(f"Telemetry record_span hatası: {e}")

def get_conversation_context(conversation_id: int):
    """
    conversation_id değerini 128-bit (16-byte) Trace ID ile eşleştirir
    ve parent-child ilişkisi kurabilmek için uygun Context döner.
    """
    try:
        if not conversation_id or conversation_id <= 0:
            return None
        
        # 128-bit trace ID gereklidir. conversation_id'yi bu boyuta eşleyelim.
        # Örneğin conversation_id=11 ise trace_id=11 olur.
        trace_id_val = conversation_id
        # generate_span_id() opentelemetry.trace yerine opentelemetry.sdk.trace'den veya random/uuid formatından türetilmelidir.
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
