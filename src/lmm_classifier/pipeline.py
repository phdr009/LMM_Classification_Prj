"""Hybrid retrieval + reranking pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

try:  # pragma: no cover
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = Any  # type: ignore

from .config import PipelineConfig
from .embeddings import EmbeddingIndex, SiglipEmbeddingModel
from .logging_config import get_logger
from .rerank import QwenVLReranker, RerankCandidate

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class ClassificationResult:
    """Result of a classification request."""

    label: str
    score: float
    candidates: List[RerankCandidate]
    used_reranker: bool


class HybridClassifier:
    """Train-free classification pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        embedding_model: Optional[SiglipEmbeddingModel] = None,
        index: Optional[EmbeddingIndex] = None,
        reranker: Optional[QwenVLReranker] = None,
    ) -> None:
        self.config = config
        self.embedding_model = embedding_model or SiglipEmbeddingModel()
        if index is None:
            if not config.retrieval.index_path:
                raise ValueError("Embedding index path must be provided in configuration")
            LOGGER.info("Loading embedding index from %s", config.retrieval.index_path)
            self.index = EmbeddingIndex.load(config.retrieval.index_path)
        else:
            self.index = index
        self.reranker = reranker or (
            QwenVLReranker() if config.rerank.enabled else None
        )

    def classify(self, image: Any) -> ClassificationResult:
        """Classify an image by retrieval and reranking."""

        embedding = self.embedding_model([image])[0]
        retrieval = self.index.query(embedding, self.config.retrieval.top_k)
        if not retrieval:
            raise RuntimeError("Embedding index returned no candidates")
        candidates = [
            RerankCandidate(label=record.label, score=score, metadata=record.metadata)
            for record, score in retrieval
        ]
        LOGGER.debug("Retrieved candidates: %s", candidates)
        top_candidate = candidates[0]
        if top_candidate.score < self.config.retrieval.open_set_threshold and self.config.allow_unknown:
            LOGGER.info("Prediction below open-set threshold; returning unknown")
            return ClassificationResult(
                label="unknown",
                score=top_candidate.score,
                candidates=candidates,
                used_reranker=False,
            )
        gap = (
            top_candidate.score - candidates[1].score
            if len(candidates) > 1
            else float("inf")
        )
        if (
            top_candidate.score >= self.config.retrieval.confidence_threshold
            and gap >= self.config.rerank.min_delta
        ):
            LOGGER.info("Confident prediction from retrieval: %s", top_candidate.label)
            return ClassificationResult(
                label=top_candidate.label,
                score=top_candidate.score,
                candidates=candidates,
                used_reranker=False,
            )
        if not self.reranker:
            LOGGER.info("Reranker disabled; returning top retrieval candidate")
            return ClassificationResult(
                label=top_candidate.label,
                score=top_candidate.score,
                candidates=candidates,
                used_reranker=False,
            )
        reranked = self.reranker.rerank(
            image=image,
            candidates=candidates,
            prompt_template=self.config.rerank.prompt_template,
        )
        LOGGER.info("Reranker selected: %s", reranked[0].label)
        return ClassificationResult(
            label=reranked[0].label,
            score=reranked[0].score,
            candidates=reranked,
            used_reranker=True,
        )


__all__ = ["HybridClassifier", "ClassificationResult"]
