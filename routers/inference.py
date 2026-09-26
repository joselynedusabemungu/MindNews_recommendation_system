from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from config.database import get_session
from repositories.telemetry_repo import TelemetryRepository
from schemas.recommendations import RecommendationRequest, RecommendationResponse
from services.recommendation import RecommendationService

router = APIRouter(tags=["recommendations"])

def get_recommendation_service(
    request: Request, session: AsyncSession = Depends(get_session)
) -> RecommendationService:
    return RecommendationService(request.app.state.engine, TelemetryRepository(session))


@router.post("/recommend", response_model=RecommendationResponse, summary="Recommend news")
async def recommend(
    payload: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    return await service.recommend(payload)
