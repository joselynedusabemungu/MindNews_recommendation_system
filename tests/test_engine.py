import pickle
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from schemas.recommendations import RecommendationRequest
from services.engine import RecommendationEngine

@pytest.fixture
def artifact_path(tmp_path):
    news = np.array(["N1", "N2", "N3"])
    vectorizer = TfidfVectorizer()
    matrix = csr_matrix(vectorizer.fit_transform(["space launch news", "space mission update", "cooking recipe"]), dtype=np.float32)
    path = tmp_path / "model.pkl"
    with path.open("wb") as output:
        pickle.dump({
            "vectorizer": vectorizer,
            "item_matrix": matrix,
            "news": __import__("pandas").DataFrame({
                "news_id": news,
                "title": ["Space launch news", "Space mission update", "Cooking recipe"],
                "category": ["science", "science", "food"],
            }),
            "news_id_to_index": {item: index for index, item in enumerate(news)},
            "training_metadata": {"dataset": "test"},
        }, output)
    return path


@pytest.mark.asyncio
async def test_engine_loads_and_excludes_seen_items(artifact_path):
    engine = RecommendationEngine()
    await engine.load(artifact_path)
    recommendations = await engine.predict(["N1"], top_k=2)
    assert engine.is_ready
    assert all(item["news_id"] != "N1" for item in recommendations)
    assert len(recommendations) == 2
    assert recommendations[0]["title"]
    assert recommendations[0]["reason"]


def test_request_normalizes_history_and_rejects_extra_fields():
    request = RecommendationRequest(user_id=" user-1 ", click_history=["N1", "N1", " N2 "])
    assert request.user_id == "user-1"
    assert request.click_history == ["N1", "N2"]
    with pytest.raises(ValueError):
        RecommendationRequest(user_id="u", unexpected=True)
