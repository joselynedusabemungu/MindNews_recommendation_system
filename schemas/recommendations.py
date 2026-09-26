from pydantic import BaseModel, ConfigDict, Field, field_validator

class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    news_id: str
    title: str
    category: str
    similarity_score: float = Field(ge=0, le=1)
    reason: str


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=128)
    click_history: list[str] = Field(default_factory=list, max_length=5000)
    top_k: int = Field(default=10, ge=1, le=1000)

    @field_validator("user_id", mode="before")
    @classmethod
    def normalize_user_id(cls, value: object) -> str:
        return str(value).strip()

    @field_validator("click_history", mode="before")
    @classmethod
    def normalize_history(cls, value: object) -> list[str]:
        if value is None:
            return []
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str
    recommended_news_ids: list[str]
    recommendations: list[RecommendationItem]
    latency_ms: float = Field(ge=0)

__all__ = ["RecommendationItem", "RecommendationRequest", "RecommendationResponse"]
