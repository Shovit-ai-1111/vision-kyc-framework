import asyncio
from datetime import datetime, timezone
import json
import logging
import time
from uuid import uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
        }
        for key in (
            "request_id",
            "method",
            "route",
            "status",
            "duration_ms",
            "error_code",
            "error_type",
            "mode",
            "decision",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


def configure_logging(level: str) -> None:
    logger = logging.getLogger("kyc")
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(level.upper())
    logger.propagate = False


class RequestMiddleware:
    """Bound body bytes before JSON/multipart parsing, including chunked bodies."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app, self.max_bytes = app, max_bytes
        self.slots = asyncio.Semaphore(8)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        start = time.monotonic()
        status = 500

        async def wrapped_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = list(message.get("headers", []))
                headers += [
                    (b"x-request-id", request_id.encode()),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                ]
                message["headers"] = headers
            await send(message)

        async def reject(code: int, name: str, message: str) -> None:
            await JSONResponse(
                status_code=code,
                content={
                    "error": {
                        "code": name,
                        "message": message,
                        "request_id": request_id,
                    }
                },
            )(scope, receive, wrapped_send)

        try:
            # Keep local inference and in-memory request buffers bounded.
            if self.slots.locked():
                await reject(503, "busy", "Server is busy. Please retry shortly")
                return
            async with self.slots:
                if scope["method"] in {"POST", "PUT", "PATCH"}:
                    length = dict(scope.get("headers", [])).get(b"content-length")
                    if length is not None:
                        try:
                            too_large = int(length) > self.max_bytes or int(length) < 0
                        except ValueError:
                            await reject(
                                400, "invalid_length", "Invalid content length"
                            )
                            return
                        if too_large:
                            await reject(
                                413,
                                "request_size",
                                "Request body exceeds the configured limit",
                            )
                            return
                    chunks, count = [], 0
                    try:
                        async with asyncio.timeout(30):
                            while True:
                                message = await receive()
                                if message["type"] == "http.disconnect":
                                    return
                                chunk = message.get("body", b"")
                                count += len(chunk)
                                if count > self.max_bytes:
                                    await reject(
                                        413,
                                        "request_size",
                                        "Request body exceeds the configured limit",
                                    )
                                    return
                                chunks.append(chunk)
                                if not message.get("more_body", False):
                                    break
                    except TimeoutError:
                        await reject(408, "request_timeout", "Upload timed out")
                        return
                    body = b"".join(chunks)
                    delivered = False

                    async def replay() -> Message:
                        nonlocal delivered
                        if not delivered:
                            delivered = True
                            return {
                                "type": "http.request",
                                "body": body,
                                "more_body": False,
                            }
                        return await receive()

                    await self.app(scope, replay, wrapped_send)
                else:
                    await self.app(scope, receive, wrapped_send)
        finally:
            route = getattr(scope.get("route"), "path", "/unmatched")
            logging.getLogger("kyc").info(
                "http_request",
                extra={
                    "event": "http_request",
                    "request_id": request_id,
                    "method": scope["method"],
                    "route": route,
                    "status": status,
                    "duration_ms": round((time.monotonic() - start) * 1000, 2),
                },
            )
