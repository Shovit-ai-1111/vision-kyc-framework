import asyncio
from contextlib import asynccontextmanager, suppress
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.core.config import Settings
from app.core.errors import KycError
from app.core.middleware import RequestMiddleware, configure_logging
from app.services.registry import Registry
from app.services.pipeline import Pipeline


def create_app(settings=None, registry=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        configure_logging(settings.log_level)
        key_file = settings.data_dir / "reviewer-key"
        if not settings.reviewer_key and key_file.is_file():
            settings.reviewer_key = key_file.read_text().strip()
        app.state.settings = settings
        app.state.registry = registry if registry is not None else Registry(settings)
        app.state.pipeline = Pipeline(settings, app.state.registry)

        async def cleanup():
            while True:
                await asyncio.sleep(60)

                # Cleanup shares the same lock as requests to avoid races.
                def run():
                    with app.state.pipeline.lock:
                        app.state.pipeline.cleanup()

                await asyncio.to_thread(run)

        task = asyncio.create_task(cleanup())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        app.state.registry.close()

    app = FastAPI(
        title="VisionKYC — Evidence Pipeline", version="3.0.0", lifespan=lifespan
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Reviewer-Key"],
    )
    app.add_middleware(RequestMiddleware, max_bytes=settings.max_request_bytes)

    @app.exception_handler(KycError)
    async def kyc_error(request: Request, error: KycError):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.get("/health/live")
    def live():
        return {"alive": True}

    @app.get("/health/ready")
    def ready(request: Request):
        status = request.app.state.registry.status()
        return JSONResponse(status_code=200 if status["ready"] else 503, content=status)

    app.include_router(router)
    return app


app = create_app()
