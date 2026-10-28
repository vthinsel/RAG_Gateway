import uvicorn

from rag_gateway.config import get_settings


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "rag_gateway.main:app",
        host=settings.gateway_bind_address,
        port=settings.gateway_port,
    )
