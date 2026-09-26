import json
from collections.abc import Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from models.telemetry import InferenceTelemetry

class TelemetryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_inference(
        self, user_id: str, predicted_news_ids: Sequence[str], latency_ms: float
    ) -> InferenceTelemetry:
        record = InferenceTelemetry(
            user_id=user_id,
            predicted_news_ids=json.dumps(list(predicted_news_ids)),
            latency_ms=latency_ms,
        )
        self._session.add(record)
        try:
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
        await self._session.refresh(record)
        return record

__all__ = ["TelemetryRepository"]
