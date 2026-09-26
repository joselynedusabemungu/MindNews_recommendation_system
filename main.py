from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastapi import FastAPI
from config.database import dispose_database, initialize_database
from config.settings import resolve_model_path, settings
from routers import health, inference
from services.engine import RecommendationEngine

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine = RecommendationEngine()
    await initialize_database()
    await engine.load(resolve_model_path(settings.model_path))
    app.state.engine = engine
    try:
        yield
    finally:
        await dispose_database()
        

app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(inference.router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": settings.app_name, "status": "online"}
