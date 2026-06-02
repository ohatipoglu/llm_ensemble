import pytest
import time
import asyncio
from core.telemetry import (
    TelemetryStorage, LocalSpanExporter, init_tracing, tracer,
    TelemetryManager, get_system_ram_info, detect_hardware_backend
)
from opentelemetry import trace as otel_trace

def test_telemetry_storage_singleton():
    storage1 = TelemetryStorage.get_instance()
    storage2 = TelemetryStorage.get_instance()
    assert storage1 is storage2

def test_telemetry_storage_limit():
    storage = TelemetryStorage.get_instance()
    storage.clear_spans()
    for i in range(600):
        storage.add_span({
            "span_id": i,
            "trace_id": 1,
            "parent_id": None,
            "name": f"span_{i}",
            "start_time": time.time(),
            "end_time": time.time(),
            "duration_ms": 1.0,
            "status": "OK"
        })
    
    spans = storage.get_all_spans()
    assert len(spans) == 500
    assert spans[0]["span_id"] == 100
    assert spans[-1]["span_id"] == 599

def test_local_span_export_and_hierarchy():
    init_tracing(service_name="test_service")
    storage = TelemetryStorage.get_instance()
    storage.clear_spans()
    
    with tracer.start_as_current_span("parent_span") as parent:
        parent.set_attribute("conversation.id", 42)
        with tracer.start_as_current_span("child_span") as child:
            child.set_attribute("llm.model", "test-model")
            
    spans = storage.get_all_spans()
    assert len(spans) >= 2
    
    child_exported = next((s for s in spans if s["name"] == "child_span"), None)
    parent_exported = next((s for s in spans if s["name"] == "parent_span"), None)
    
    assert child_exported is not None
    assert parent_exported is not None
    assert child_exported["parent_id"] == parent_exported["span_id"]
    assert child_exported["attributes"]["llm.model"] == "test-model"
    assert parent_exported["attributes"]["conversation.id"] == 42

def test_tool_tracer_decorator_sync():
    init_tracing(service_name="test_service")
    storage = TelemetryStorage.get_instance()
    storage.clear_spans()

    @TelemetryManager.trace_tool_execution("sync_dummy_tool", {"extra_param": "test_val"})
    def my_sync_function(a, b):
        return a + b

    res = my_sync_function(10, 20)
    assert res == 30

    spans = storage.get_all_spans()
    tool_span = next((s for s in spans if s["name"] == "tool_call:sync_dummy_tool"), None)
    assert tool_span is not None
    assert tool_span["attributes"]["tool.name"] == "sync_dummy_tool"
    assert tool_span["attributes"]["tool.arg.a"] == "10"
    assert tool_span["attributes"]["tool.arg.b"] == "20"
    assert tool_span["attributes"]["tool.arg.extra_param"] == "test_val"

@pytest.mark.asyncio
async def test_tool_tracer_decorator_async():
    init_tracing(service_name="test_service")
    storage = TelemetryStorage.get_instance()
    storage.clear_spans()

    @TelemetryManager.trace_tool_execution("async_dummy_tool")
    async def my_async_function(x):
        await asyncio.sleep(0.01)
        return x * 2

    res = await my_async_function(5)
    assert res == 10

    spans = storage.get_all_spans()
    tool_span = next((s for s in spans if s["name"] == "tool_call:async_dummy_tool"), None)
    assert tool_span is not None
    assert tool_span["attributes"]["tool.name"] == "async_dummy_tool"
    assert tool_span["attributes"]["tool.arg.x"] == "5"

def test_tool_tracer_context_manager():
    init_tracing(service_name="test_service")
    storage = TelemetryStorage.get_instance()
    storage.clear_spans()

    with TelemetryManager.trace_tool_execution("db_write", {"table": "users"}):
        pass

    spans = storage.get_all_spans()
    tool_span = next((s for s in spans if s["name"] == "tool_call:db_write"), None)
    assert tool_span is not None
    assert tool_span["attributes"]["tool.arg.table"] == "users"

def test_exception_local_variable_capturing():
    init_tracing(service_name="test_service")
    storage = TelemetryStorage.get_instance()
    storage.clear_spans()

    @TelemetryManager.trace_tool_execution("buggy_tool")
    def buggy_func(divisor):
        secret_var = "my_secret_token"
        return 100 / divisor

    with pytest.raises(ZeroDivisionError):
        buggy_func(0)

    spans = storage.get_all_spans()
    tool_span = next((s for s in spans if s["name"] == "tool_call:buggy_tool"), None)
    assert tool_span is not None
    assert tool_span["status"] == "ERROR"
    # Local variable check
    assert tool_span["attributes"]["error.local.secret_var"] == "my_secret_token"
    assert tool_span["attributes"]["error.local.divisor"] == "0"

def test_system_and_hardware_helpers():
    ram_total, ram_free = get_system_ram_info()
    assert ram_total > 0
    assert ram_free >= 0
    
    hw = detect_hardware_backend()
    assert "gpu_info" in hw
    assert "hardware_backends" in hw
