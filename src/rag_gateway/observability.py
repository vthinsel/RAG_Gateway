import json
import logging
import re
import threading
import time
from collections import Counter, defaultdict
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("uvicorn.error")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RequestMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Counter[tuple[str, str, int]] = Counter()
        self._durations: dict[tuple[str, str], float] = defaultdict(float)
        self._duration_counts: Counter[tuple[str, str]] = Counter()

    def observe(self, method: str, route: str, status: int, duration: float) -> None:
        with self._lock:
            self._requests[(method, route, status)] += 1
            self._durations[(method, route)] += duration
            self._duration_counts[(method, route)] += 1

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
            if route != "/metrics":
                request_metrics.observe(scope["method"], route, status, duration)
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
