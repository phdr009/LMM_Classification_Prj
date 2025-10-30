"""Configuration models for the hybrid classification pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:  # pragma: no cover
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass(slots=True)
class RetrievalConfig:
    """Configuration for the embedding retrieval stage."""

    index_path: Optional[Path] = None
    top_k: int = 10
    confidence_threshold: float = 0.6
    open_set_threshold: float = 0.35
    metric: str = "cosine"


@dataclass(slots=True)
class RerankConfig:
    """Configuration for the LMM reranker stage."""

    enabled: bool = True
    max_batch_size: int = 8
    min_delta: float = 0.05
    prompt_template: str = (
        "Given an input image description and candidate class labels,"
        " choose the best matching label."
        "\nImage: {image_description}\nCandidates: {candidates}"
    )


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration for the hybrid pipeline."""

    labels: List[str]
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    allow_unknown: bool = True

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PipelineConfig":
        """Load a pipeline configuration from a YAML file."""

        with open(path, "r", encoding="utf-8") as handle:
            if yaml is None:  # pragma: no cover - requires PyYAML in production
                raise RuntimeError("pyyaml is required to load configuration files")
            data = yaml.safe_load(handle)
        return cls(
            labels=data["labels"],
            retrieval=RetrievalConfig(**data.get("retrieval", {})),
            rerank=RerankConfig(**data.get("rerank", {})),
            allow_unknown=data.get("allow_unknown", True),
        )

    def to_yaml(self, path: Path | str) -> None:
        """Persist the configuration to a YAML file."""

        data = {
            "labels": self.labels,
            "retrieval": {
                "index_path": str(self.retrieval.index_path)
                if self.retrieval.index_path
                else None,
                "top_k": self.retrieval.top_k,
                "confidence_threshold": self.retrieval.confidence_threshold,
                "open_set_threshold": self.retrieval.open_set_threshold,
                "metric": self.retrieval.metric,
            },
            "rerank": {
                "enabled": self.rerank.enabled,
                "max_batch_size": self.rerank.max_batch_size,
                "min_delta": self.rerank.min_delta,
                "prompt_template": self.rerank.prompt_template,
            },
            "allow_unknown": self.allow_unknown,
        }
        with open(path, "w", encoding="utf-8") as handle:
            if yaml is None:  # pragma: no cover - requires PyYAML in production
                raise RuntimeError("pyyaml is required to persist configuration files")
            yaml.safe_dump(data, handle)


__all__ = ["PipelineConfig", "RetrievalConfig", "RerankConfig"]
