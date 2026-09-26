import asyncio
import pickle
from pathlib import Path
from typing import Any
import numpy as np
from scipy.sparse import csr_matrix

class ModelNotLoadedError(RuntimeError):
    pass

class RecommendationEngine:
    def __init__(self) -> None:
        self._artifact: dict[str, Any] | None = None
        self._model_path: Path | None = None

    @property
    def is_ready(self) -> bool:
        return self._artifact is not None

    @property
    def metadata(self) -> dict[str, Any]:
        if self._artifact is None:
            return {"ready": False}
        return {"ready": True, **self._artifact.get("training_metadata", {})}

    async def load(self, model_path: Path) -> None:
        resolved_path = model_path if model_path.is_absolute() else Path.cwd() / model_path
        artifact = await asyncio.to_thread(self._load_pickle, resolved_path)
        self._artifact = artifact
        self._model_path = resolved_path

    @staticmethod
    def _load_pickle(model_path: Path) -> dict[str, Any]:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
        try:
            with model_path.open("rb") as artifact_file:
                artifact = pickle.load(artifact_file)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The model artifact was trained with incompatible package versions. "
                "Delete data/artifacts/model.pkl and retrain the notebook using this "
                "same virtual environment."
            ) from exc
        required = {"item_matrix", "news_id_to_index", "news"}
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"Model artifact is missing required keys: {sorted(missing)}")
        return artifact

    async def predict(self, click_history: list[str], top_k: int) -> list[dict[str, Any]]:
        if self._artifact is None:
            raise ModelNotLoadedError("Recommendation model has not been loaded")
        return await asyncio.to_thread(self._predict_sync, click_history, top_k)

    def _predict_sync(self, click_history: list[str], top_k: int) -> list[dict[str, Any]]:
        assert self._artifact is not None
        matrix = csr_matrix(self._artifact["item_matrix"], dtype=np.float32)
        mapping: dict[str, int] = self._artifact["news_id_to_index"]
        news = self._artifact["news"]
        news_ids = news["news_id"].to_numpy()
        known_history = list(dict.fromkeys(item for item in click_history if item in mapping))
        if not known_history:
            recommendation_indices = range(min(top_k, len(news)))
            return [self._item(news, index, 0.0, "A cold-start recommendation from the catalog.") for index in recommendation_indices]

        indices = [mapping[item] for item in known_history]
        user_profile = csr_matrix(matrix[indices].sum(axis=0), dtype=np.float32)
        numerator = matrix.dot(user_profile.T).toarray().ravel()
        item_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
        profile_norm = float(np.sqrt(user_profile.multiply(user_profile).sum()))
        scores = numerator / (item_norms * profile_norm + 1e-12)
        scores[indices] = -np.inf
        recommendation_indices = np.argsort(-scores)[:top_k]
        clicked_titles = [str(news.iloc[index]["title"]) for index in indices]
        return [
            self._item(
                news,
                int(index),
                float(max(0.0, min(1.0, scores[index]))),
                self._reason(str(news.iloc[index]["title"]), clicked_titles),
            )
            for index in recommendation_indices
        ]

    @staticmethod
    def _item(news: Any, index: int, score: float, reason: str) -> dict[str, Any]:
        row = news.iloc[index]
        return {
            "news_id": str(row["news_id"]),
            "title": str(row.get("title", "Untitled article")),
            "category": str(row.get("category", "news")),
            "similarity_score": score,
            "reason": reason,
        }

    @staticmethod
    def _reason(title: str, clicked_titles: list[str]) -> str:
        clicked_words = {word.lower() for text in clicked_titles for word in text.split() if len(word) > 3}
        overlap = [word for word in title.split() if word.lower() in clicked_words]
        if overlap:
            return f"Similar language to your history, especially: {', '.join(dict.fromkeys(overlap[:3]))}."
        return "Ranks highly because its title vector is close to your reading profile."


__all__ = ["ModelNotLoadedError", "RecommendationEngine"]
