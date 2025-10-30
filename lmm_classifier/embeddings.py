"""Embedding utilities for SigLIP retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - exercised in integration environments
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - handled gracefully during testing
    np = None  # type: ignore

try:  # pragma: no cover
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    Image = Any  # type: ignore

try:  # pragma: no cover - exercised in integration environments
    from sklearn.neighbors import NearestNeighbors
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without sklearn
    NearestNeighbors = None  # type: ignore

try:  # pragma: no cover - exercised in integration environments
    from transformers import AutoModel, AutoProcessor
except ModuleNotFoundError:  # pragma: no cover - fallback when transformers unavailable
    AutoModel = AutoProcessor = None  # type: ignore

from .logging_config import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class EmbeddingRecord:
    """A single indexed embedding with metadata."""

    vector: Any
    label: str
    metadata: dict


class SiglipEmbeddingModel:
    """Thin wrapper around the SigLIP-so400m vision encoder."""

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384") -> None:
        if AutoModel is None or AutoProcessor is None:  # pragma: no cover
            raise RuntimeError("transformers is required to use SigLIP embeddings")
        self.model_name = model_name
        LOGGER.info("Loading SigLIP model: %s", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

    def __call__(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Generate normalized embeddings for a batch of images."""

        if np is None:  # pragma: no cover - requires numpy in production
            raise RuntimeError("numpy is required to compute embeddings")
        inputs = self.processor(images=images, return_tensors="pt")
        with np.errstate(all="ignore"):
            outputs = self.model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            raw = outputs.pooler_output
        else:
            raw = outputs.last_hidden_state[:, 0]
        embeddings = raw.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)
        return embeddings


class EmbeddingIndex:
    """In-memory retrieval index backed by scikit-learn."""

    def __init__(self, embeddings: Iterable[EmbeddingRecord], metric: str = "cosine") -> None:
        if np is None:  # pragma: no cover - requires numpy in production
            raise RuntimeError("numpy is required to build the retrieval index")
        if NearestNeighbors is None:  # pragma: no cover
            raise RuntimeError("scikit-learn is required to build the retrieval index")
        self.metric = metric
        self.records: List[EmbeddingRecord] = list(embeddings)
        if not self.records:
            raise ValueError("Embedding index requires at least one record")
        matrix = np.stack([record.vector for record in self.records])
        self._neighbors = NearestNeighbors(metric=metric)
        self._neighbors.fit(matrix)
        LOGGER.info("Initialized retrieval index with %d items", len(self.records))

    def save(self, path: Path | str) -> None:
        """Persist the index to disk."""

        if np is None:  # pragma: no cover
            raise RuntimeError("numpy is required to save the retrieval index")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "embeddings": np.stack([record.vector for record in self.records]),
            "labels": [record.label for record in self.records],
            "metadata": [record.metadata for record in self.records],
            "metric": self.metric,
        }
        np.savez_compressed(path, **data)
        LOGGER.info("Saved index to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "EmbeddingIndex":
        """Load an index from disk."""

        if np is None:  # pragma: no cover
            raise RuntimeError("numpy is required to load the retrieval index")
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            embeddings = data["embeddings"]
            labels = data["labels"].tolist()
            metadata = data["metadata"].tolist()
            metric = str(data["metric"])
        records = [
            EmbeddingRecord(vector=embeddings[i], label=labels[i], metadata=metadata[i])
            for i in range(len(labels))
        ]
        return cls(records, metric=metric)

    def query(self, vector: np.ndarray, top_k: int) -> List[Tuple[EmbeddingRecord, float]]:
        """Return the top-k nearest neighbors and their similarity scores."""

        distances, indices = self._neighbors.kneighbors(vector.reshape(1, -1), n_neighbors=top_k)
        results: List[Tuple[EmbeddingRecord, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            record = self.records[idx]
            score = 1 - dist if self.metric == "cosine" else -dist
            results.append((record, float(score)))
        return results


__all__ = ["SiglipEmbeddingModel", "EmbeddingIndex", "EmbeddingRecord"]
