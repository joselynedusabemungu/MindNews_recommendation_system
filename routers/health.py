from typing import Any
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])

@router.get("/live", summary="Liveness probe")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/ready", response_model=None, summary="Readiness probe")
async def readiness(request: Request) -> Any:
    engine = request.app.state.engine
    if not engine.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "model_not_loaded"},
        )
    return {"status": "ready", "model": engine.metadata}
