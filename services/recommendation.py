from time import perf_counter
from repositories.telemetry_repo import TelemetryRepository
from schemas.recommendations import RecommendationRequest, RecommendationResponse
from services.engine import RecommendationEngine

class RecommendationService:
    def __init__(self, engine: RecommendationEngine, telemetry: TelemetryRepository) -> None:
        self._engine = engine
        self._telemetry = telemetry

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        started_at = perf_counter()
        recommendation_items = await self._engine.predict(request.click_history, request.top_k)
        recommended_ids = [item["news_id"] for item in recommendation_items]
        latency_ms = (perf_counter() - started_at) * 1000
        await self._telemetry.record_inference(request.user_id, recommended_ids, latency_ms)
        return RecommendationResponse(
            user_id=request.user_id,
            recommended_news_ids=recommended_ids,
            recommendations=recommendation_items,
            latency_ms=round(latency_ms, 3),
        )

__all__ = ["RecommendationService"]
