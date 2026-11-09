import json
import logging
import re
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from traceback import extract_tb
from typing import Any, Literal, TypedDict
from uuid import uuid4

import httpx
from pymongo.errors import PyMongoError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("uvicorn.error")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_request_scope: ContextVar[Scope | None] = ContextVar("request_scope", default=None)


class RequestEvent(TypedDict):
    request_id: str
    timestamp: str
    method: str
    route: str
    status: int | Literal["in progress"]
    duration_ms: float
    outcome: Literal["success", "error", "in progress"]
    details: dict[str, Any]


class RouteMetrics(TypedDict):
    method: str
    route: str
    requests: int
    errors: int
    average_duration_ms: float


class MonitoringSnapshot(TypedDict):
    requests_total: int
    requests_failed: int
    requests_in_flight: int
    recent_average_duration_ms: float
    recent_requests: list[RequestEvent]
    routes: list[RouteMetrics]
    retained_request_limit: int


class RequestMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Counter[tuple[str, str, int]] = Counter()
        self._durations: dict[tuple[str, str], float] = defaultdict(float)
        self._duration_counts: Counter[tuple[str, str]] = Counter()
        self._application_failures: Counter[tuple[str, str]] = Counter()
        self._active: dict[str, tuple[float, str, Scope]] = {}
        self._recent: deque[RequestEvent] = deque(maxlen=500)

    def begin(self, request_id: str, method: str, scope: Scope) -> None:
        if scope.get("path") == "/v1/admin/monitoring":
            return
        with self._lock:
            self._active[request_id] = (time.perf_counter(), method, scope)

    def observe(
        self,
        method: str,
        route: str,
        status: int,
        duration: float,
        request_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        timestamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._lock:
            if request_id:
                self._active.pop(request_id, None)
            if route in {"/metrics", "/v1/admin/monitoring"}:
                return
            self._requests[(method, route, status)] += 1
            self._durations[(method, route)] += duration
            self._duration_counts[(method, route)] += 1
            diagnostics = deepcopy(details or {})
            failed = status >= 400 or bool(diagnostics.get("error"))
            if failed and status < 400:
                self._application_failures[(method, route)] += 1
            self._recent.append(
                {
                    "request_id": request_id,
                    "timestamp": timestamp,
                    "method": method,
                    "route": route,
                    "status": status,
                    "duration_ms": round(duration * 1000, 2),
                    "outcome": "error" if failed else "success",
                    "details": diagnostics,
                }
            )

    def snapshot(self) -> MonitoringSnapshot:
        now = time.perf_counter()
        with self._lock:
            requests = list(self._requests.items())
            durations = dict(self._durations)
            duration_counts = dict(self._duration_counts)
            active = list(self._active.items())
            recent = list(reversed(self._recent))
            application_failures = dict(self._application_failures)

        route_totals: dict[tuple[str, str], dict[str, int]] = {}
        for (method, route, status), count in requests:
            key = (method, route)
            totals = route_totals.setdefault(key, {"requests": 0, "errors": 0})
            totals["requests"] += count
            if status >= 400:
                totals["errors"] += count
        routes: list[RouteMetrics] = [
            {
                "method": method,
                "route": route,
                "requests": totals["requests"],
                "errors": totals["errors"] + application_failures.get((method, route), 0),
                "average_duration_ms": round(
                    durations[(method, route)] / duration_counts[(method, route)] * 1000,
                    2,
                ),
            }
            for (method, route), totals in sorted(route_totals.items())
        ]
        in_flight: list[RequestEvent] = []
        for request_id, (started, method, scope) in active:
            route = getattr(scope.get("route"), "path", "in progress")
            in_flight.append(
                {
                    "request_id": request_id,
                    "timestamp": datetime.fromtimestamp(
                        time.time() - (now - started), UTC
                    ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                    "method": method,
                    "route": route,
                    "status": "in progress",
                    "duration_ms": round((now - started) * 1000, 2),
                    "outcome": "in progress",
                    "details": _snapshot_details(scope, now),
                }
            )
        total_requests = sum(count for _, count in requests)
        total_errors = sum(
            count for (_, _, status), count in requests if status >= 400
        ) + sum(application_failures.values())
        recent_average = (
            round(sum(float(item["duration_ms"]) for item in recent) / len(recent), 2)
            if recent
            else 0
        )
        return {
            "requests_total": total_requests,
            "requests_failed": total_errors,
            "requests_in_flight": len(in_flight),
            "recent_average_duration_ms": recent_average,
            "recent_requests": [*in_flight, *recent][:500],
            "routes": routes,
            "retained_request_limit": 500,
        }

    def prometheus(self) -> str:
        with self._lock:
            requests = sorted(self._requests.items())
            durations = sorted(self._durations.items())
            duration_counts = sorted(self._duration_counts.items())
        lines = [
            "# HELP rag_gateway_http_requests_total Total HTTP requests handled.",
            "# TYPE rag_gateway_http_requests_total counter",
        ]
        lines.extend(
            f'rag_gateway_http_requests_total{{method="{method}",route="{_label(route)}",'
            f'status="{status}"}} {count}'
            for (method, route, status), count in requests
        )
        lines.extend(
            [
                (
                    "# HELP rag_gateway_http_request_duration_seconds_count "
                    "Number of HTTP requests observed for the duration summary."
                ),
                "# TYPE rag_gateway_http_request_duration_seconds_count counter",
            ]
        )
        lines.extend(
            f'rag_gateway_http_request_duration_seconds_count{{method="{method}",'
            f'route="{_label(route)}"}} {count}'
            for (method, route), count in duration_counts
        )
        lines.extend(
            [
                (
                    "# HELP rag_gateway_http_request_duration_seconds_sum "
                    "Cumulative HTTP request duration in seconds."
                ),
                "# TYPE rag_gateway_http_request_duration_seconds_sum counter",
            ]
        )
        lines.extend(
            f'rag_gateway_http_request_duration_seconds_sum{{method="{method}",'
            f'route="{_label(route)}"}} {duration:.6f}'
            for (method, route), duration in durations
        )
        return "\n".join(lines) + "\n"


def _snapshot_details(scope: Scope, now: float) -> dict[str, Any]:
    state = scope.get("state", {})
    with state.get("timing_lock", nullcontext()):
        details = deepcopy(state.get("diagnostics", {}))
        stack = state.get("timing_stack", [])
        if stack:
            details["active_stage"] = {
                "name": stack[-1]["name"],
                "elapsed_ms": round(max(0, now - stack[-1]["started"]) * 1000, 2),
            }
    return details


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


request_metrics = RequestMetrics()


def record_request_error(scope: Scope, error: Exception, status: int) -> None:
    cause = error
    chain = [type(error).__name__]
    while cause.__cause__ is not None and len(chain) < 5:
        cause = cause.__cause__
        chain.append(type(cause).__name__)
    try:
        message = HTTPStatus(status).phrase
    except ValueError:
        message = "Request failed"
    failure: dict[str, Any] = {
        "type": type(cause).__name__, "exception_chain": chain,
        "message": message, "status": status,
        "traceback": [
            {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
            for frame in extract_tb(cause.__traceback__)[-8:]
        ],
    }
    if isinstance(cause, httpx.TimeoutException):
        failure["category"] = "upstream_timeout"
    elif isinstance(cause, httpx.HTTPStatusError):
        failure["category"] = "upstream_http_error"
        failure["upstream_status"] = cause.response.status_code
    elif isinstance(cause, httpx.RequestError):
        failure["category"] = "upstream_connection_error"
    elif isinstance(cause, PyMongoError):
        failure["category"] = "mongodb_error"
    else:
        failure["category"] = "request_error"
    if isinstance(cause, httpx.HTTPError):
        try:
            failure["upstream_host"] = cause.request.url.host
        except RuntimeError:
            pass
    diagnostics = scope.setdefault("state", {}).setdefault("diagnostics", {})
    diagnostics["error"] = failure


@contextmanager
def processing_stage(stage: str) -> Iterator[None]:
    with _measure_stage(_request_scope.get(), stage):
        yield


@contextmanager
def _measure_stage(scope: Scope | None, stage: str) -> Iterator[None]:
    state = scope.setdefault("state", {}) if scope is not None else {}
    lock = state.setdefault("timing_lock", threading.RLock())
    with lock:
        diagnostics = state.setdefault("diagnostics", {})
        diagnostics["stage"] = stage
        started = time.perf_counter()
        stack = state.setdefault("timing_stack", [])
        frame = {"name": stage, "started": started}
        stack.append(frame)
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        with lock:
            timings = diagnostics.setdefault("stage_durations_ms", {})
            timings[stage] = round(timings.get(stage, 0) + (time.perf_counter() - started) * 1000, 2)
            counts = diagnostics.setdefault("stage_counts", {})
            counts[stage] = counts.get(stage, 0) + 1
            stack.remove(frame)
            if not failed and stack:
                diagnostics["stage"] = stack[-1]["name"]


@asynccontextmanager
async def request_stage(scope: Scope | None, stage: str) -> AsyncIterator[None]:
    with _measure_stage(scope, stage):
        yield


class ObservabilityMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        supplied_id = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied_id if _REQUEST_ID.fullmatch(supplied_id) else uuid4().hex
        started = time.perf_counter()
        status = 500
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["diagnostics"] = {"stage": "routing"}
        scope_token = _request_scope.set(scope)
        response_bytes = 0
        request_metrics.begin(request_id, scope["method"], scope)

        async def send_with_request_id(message: Message) -> None:
            nonlocal status, response_bytes
            if message["type"] == "http.response.start":
                status = message["status"]
                state["diagnostics"]["response_headers_ms"] = round(
                    (time.perf_counter() - started) * 1000, 2
                )
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = response_headers
            elif message["type"] == "http.response.body":
                response_bytes += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception as error:
            record_request_error(scope, error, 500)
            raise
        finally:
            duration = time.perf_counter() - started
            state["diagnostics"]["response_bytes"] = response_bytes
            if status < 400 and not state["diagnostics"].get("error"):
                state["diagnostics"]["stage"] = "completed"
            route = getattr(scope.get("route"), "path", "unmatched")
            request_metrics.observe(
                scope["method"], route, status, duration, request_id=request_id,
                details=state["diagnostics"],
            )
            if route not in {"/metrics", "/v1/admin/monitoring"}:
                failed = status >= 400 or bool(state["diagnostics"].get("error"))
                error_status = state["diagnostics"].get("error", {}).get("status", 0)
                level = logging.ERROR if max(status, error_status) >= 500 else (
                    logging.WARNING if failed else logging.INFO
                )
                logger.log(
                    level,
                    json.dumps(
                        {
                            "event": "http_request",
                            "request_id": request_id,
                            "method": scope["method"],
                            "route": route,
                            "status": status,
                            "duration_ms": round(duration * 1000, 2),
                            "outcome": "error" if failed else "success",
                            "details": state["diagnostics"],
                        },
                        separators=(",", ":"),
                    )
                )
            _request_scope.reset(scope_token)
