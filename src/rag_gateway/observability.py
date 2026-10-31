import json
import logging
import re
import threading
import time
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime
from typing import Literal, TypedDict
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("uvicorn.error")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RequestEvent(TypedDict):
    request_id: str
    timestamp: str
    method: str
    route: str
    status: int | Literal["in progress"]
    duration_ms: float


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
            self._recent.append(
                {
                    "request_id": request_id,
                    "timestamp": timestamp,
                    "method": method,
                    "route": route,
                    "status": status,
                    "duration_ms": round(duration * 1000, 2),
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
                "errors": totals["errors"],
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
                }
            )
        total_requests = sum(count for _, count in requests)
        total_errors = sum(
            count for (_, _, status), count in requests if status >= 400
        )
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


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


request_metrics = RequestMetrics()


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
        request_metrics.begin(request_id, scope["method"], scope)

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration = time.perf_counter() - started
            route = getattr(scope.get("route"), "path", "unmatched")
            request_metrics.observe(
                scope["method"], route, status, duration, request_id=request_id
            )
            if route not in {"/metrics", "/v1/admin/monitoring"}:
                logger.info(
                    json.dumps(
                        {
                            "event": "http_request",
                            "request_id": request_id,
                            "method": scope["method"],
                            "route": route,
                            "status": status,
                            "duration_ms": round(duration * 1000, 2),
                        },
                        separators=(",", ":"),
                    )
                )
