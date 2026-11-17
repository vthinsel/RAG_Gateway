import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from anyio import CancelScope
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.concurrency import run_in_threadpool

from rag_gateway.config import Settings
from rag_gateway.operations import OperationsStore

logger = logging.getLogger("uvicorn.error")
_runtime: ContextVar["Telemetry | None"] = ContextVar("telemetry_runtime", default=None)
_tenant: ContextVar[str] = ContextVar("usage_tenant", default="")
_request_id: ContextVar[str] = ContextVar("telemetry_request_id", default="")
_propagator = TraceContextTextMapPropagator()


class Telemetry:
    def __init__(
        self, settings: Settings, operations: OperationsStore,
        provider: TracerProvider | None = None,
    ) -> None:
        self.settings = settings
        self.operations = operations
        self.provider = provider
        if self.provider is None and settings.otel_traces_enabled:
            self.provider = TracerProvider(
                resource=Resource.create({"service.name": settings.otel_service_name}),
                sampler=ParentBased(TraceIdRatioBased(settings.otel_traces_sample_ratio)),
            )
            self.provider.add_span_processor(BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_traces_endpoint)
            ))
        self.tracer = (
            self.provider.get_tracer("rag_gateway") if self.provider else trace.NoOpTracer()
        )

    def close(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()


@contextmanager
def request_telemetry(
    runtime: Telemetry | None, headers: dict[str, str],
) -> Generator[None, None, None]:
    token = _runtime.set(runtime)
    request_token = _request_id.set("")
    tenant_token = _tenant.set("")
    try:
        if runtime is None:
            yield
        else:
            with runtime.tracer.start_as_current_span(
                "HTTP request", context=_propagator.extract(headers),
                kind=trace.SpanKind.SERVER, record_exception=False, set_status_on_exception=False,
            ):
                try:
                    yield
                except BaseException as error:
                    mark_span_error(error)
                    raise
    finally:
        _tenant.reset(tenant_token)
        _request_id.reset(request_token)
        _runtime.reset(token)


def correlate_request(request_id: str) -> str:
    _request_id.set(request_id)
    trace.get_current_span().set_attribute("rag.request_id", request_id)
    return current_trace_id()


def current_trace_id() -> str:
    context = trace.get_current_span().get_span_context()
    return f"{context.trace_id:032x}" if context.is_valid else ""


@contextmanager
def usage_tenant(tenant_id: str) -> Generator[None, None, None]:
    token = _tenant.set(tenant_id)
    try:
        yield
    finally:
        _tenant.reset(token)


@contextmanager
def telemetry_span(stage: str) -> Generator[trace.Span, None, None]:
    runtime = _runtime.get()
    tracer = runtime.tracer if runtime else trace.NoOpTracer()
    with tracer.start_as_current_span(
        stage, attributes={"rag.request_id": _request_id.get()},
        record_exception=False, set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except BaseException as error:
            mark_span_error(error)
            raise


def mark_span_error(error: BaseException) -> None:
    span = trace.get_current_span()
    span.set_attribute("error.type", type(error).__name__)
    span.set_status(trace.StatusCode.ERROR)


def finish_request_span(method: str, route: str, status: int, failed: bool) -> None:
    span = trace.get_current_span()
    span.update_name(f"{method} {route}")
    span.set_attributes({
        "http.request.method": method, "http.route": route, "http.response.status_code": status,
    })
    if failed:
        span.set_status(trace.StatusCode.ERROR)


def trace_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if _runtime.get() is not None:
        _propagator.inject(headers)
    return headers


def annotate_model_span(provider: str, operation: str, model: str) -> None:
    trace.get_current_span().set_attributes({
        "gen_ai.provider.name": provider, "gen_ai.operation.name": operation,
        "gen_ai.request.model": model,
    })


def _token_count(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return value if type(value) is int and value >= 0 else None


def record_model_usage(
    provider: str, operation: str, model: str, usage: Any,
    outcome: str = "success", tenant_id: str | None = None,
) -> None:
    runtime = _runtime.get()
    if runtime is None:
        return
    tokens = usage if isinstance(usage, dict) else {}
    if any(
        key in tokens and tokens[key] is not None and _token_count(tokens, key) is None
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        logger.warning(
            "Invalid upstream token counts ignored: request_id=%s provider=%s operation=%s",
            _request_id.get(), provider, operation,
        )
    input_tokens = _token_count(tokens, "prompt_tokens")
    output_tokens = _token_count(tokens, "completion_tokens")
    total_tokens = _token_count(tokens, "total_tokens")
    if operation == "embedding":
        input_tokens = total_tokens
        output_tokens = 0 if total_tokens is not None else None
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    pricing = runtime.settings.model_pricing.get(f"{provider}:{operation}:{model}")
    rates = pricing.model_dump(exclude_none=True) if pricing else {}
    cost: float | None = None
    if rates:
        cost = rates.get("per_request_usd", 0.0)
        for field, count in (
            ("input_per_million_usd", input_tokens),
            ("output_per_million_usd", output_tokens),
        ):
            rate = rates.get(field)
            if rate is not None:
                if count is None:
                    cost = None
                    break
                cost += count * rate / 1_000_000
            elif count not in (0, None) and "per_request_usd" not in rates:
                cost = None
                break
        if "per_request_usd" not in rates and (
            input_tokens is None or output_tokens is None
        ):
            cost = None
    event = {
        "request_id": _request_id.get(), "trace_id": current_trace_id(),
        "tenant_id": tenant_id if tenant_id is not None else _tenant.get(),
        "provider": provider, "operation": operation, "model": model, "outcome": outcome,
        "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens,
        "estimated_cost_usd": cost, "pricing": rates,
    }
    try:
        runtime.operations.save_model_usage(event)
    except sqlite3.Error:
        logger.error(
            "Model usage persistence failed: request_id=%s provider=%s operation=%s",
            _request_id.get(), provider, operation,
        )
        raise


async def persist_model_usage(
    provider: str, operation: str, model: str, usage: Any,
    outcome: str = "success", tenant_id: str | None = None,
) -> None:
    if _runtime.get() is None:
        return
    with CancelScope(shield=True):
        await run_in_threadpool(
            record_model_usage, provider, operation, model, usage, outcome, tenant_id,
        )


def usage_prometheus(rows: list[dict[str, Any]]) -> str:
    from rag_gateway.observability import _label

    lines: list[str] = []
    for metric, field, help_text in (
        ("calls", "calls", "Recorded model API calls."),
        ("failed_calls", "failed_calls", "Failed or incomplete model API calls."),
        ("missing_usage_calls", "missing_usage_calls", "Calls without reported total token usage."),
        ("unpriced_calls", "unpriced_calls", "Calls without a complete cost estimate."),
        ("estimated_cost_usd", "estimated_cost_usd", "Sum of known estimated costs in USD."),
    ):
        name = f"rag_gateway_model_{metric}_total"
        lines.extend([f"# HELP {name} {help_text}", f"# TYPE {name} counter"])
        for row in rows:
            if row[field] is not None:
                labels = ",".join(
                    f'{key}="{_label(row[key])}"'
                    for key in ("tenant_id", "provider", "operation", "model")
                )
                lines.append(f"{name}{{{labels}}} {row[field]}")
    name = "rag_gateway_model_tokens_total"
    lines.extend([f"# HELP {name} Reported model token usage.", f"# TYPE {name} counter"])
    for row in rows:
        labels = ",".join(
            f'{key}="{_label(row[key])}"'
            for key in ("tenant_id", "provider", "operation", "model")
        )
        for kind in ("input", "output", "total"):
            if row[f"{kind}_tokens"] is not None:
                lines.append(f'{name}{{{labels},token_type="{kind}"}} {row[f"{kind}_tokens"]}')
    return "\n".join(lines) + "\n"
