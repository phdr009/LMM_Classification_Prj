"""Unit tests for the hybrid pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from lmm_classifier.config import PipelineConfig, RetrievalConfig, RerankConfig
from lmm_classifier.pipeline import HybridClassifier


@dataclass
class DummyRecord:
    label: str
    metadata: dict


class DummyIndex:
    def __init__(self, responses: List[List[tuple[str, float]]]) -> None:
        self.responses = responses
        self.calls = 0

    def query(self, _vector: Any, top_k: int):
        del top_k
        result = self.responses[self.calls]
        self.calls += 1
        return [
            (DummyRecord(label=label, metadata={}), score) for label, score in result
        ]


class DummyEmbed:
    def __call__(self, images):
        del images
        return [[1.0, 0.0, 0.0]]


class DummyReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, image, candidates, prompt_template):
        del image, prompt_template
        self.calls += 1
        return list(reversed(candidates))


@dataclass
class DummyImage:
    size: tuple[int, int] = (2, 2)


def create_image() -> DummyImage:
    return DummyImage()


def test_confident_retrieval_skips_rerank(monkeypatch):
    config = PipelineConfig(
        labels=["a", "b"],
        retrieval=RetrievalConfig(confidence_threshold=0.5, open_set_threshold=0.2),
        rerank=RerankConfig(enabled=True, min_delta=0.05),
        allow_unknown=True,
    )
    index = DummyIndex([[("a", 0.9), ("b", 0.5)]])
    classifier = HybridClassifier(
        config=config,
        embedding_model=DummyEmbed(),
        index=index,
        reranker=DummyReranker(),
    )
    image = create_image()
    result = classifier.classify(image)
    assert result.label == "a"
    assert not result.used_reranker


def test_open_set_unknown(monkeypatch):
    config = PipelineConfig(
        labels=["a", "b"],
        retrieval=RetrievalConfig(confidence_threshold=0.5, open_set_threshold=0.3),
        rerank=RerankConfig(enabled=True),
        allow_unknown=True,
    )
    index = DummyIndex([[("a", 0.2), ("b", 0.1)]])
    classifier = HybridClassifier(
        config=config,
        embedding_model=DummyEmbed(),
        index=index,
        reranker=DummyReranker(),
    )
    image = create_image()
    result = classifier.classify(image)
    assert result.label == "unknown"
    assert not result.used_reranker


def test_uncertain_invokes_reranker(monkeypatch):
    config = PipelineConfig(
        labels=["a", "b"],
        retrieval=RetrievalConfig(confidence_threshold=0.8, open_set_threshold=0.2),
        rerank=RerankConfig(enabled=True, min_delta=0.05),
        allow_unknown=True,
    )
    index = DummyIndex([[("a", 0.5), ("b", 0.49)]])
    reranker = DummyReranker()
    classifier = HybridClassifier(
        config=config,
        embedding_model=DummyEmbed(),
        index=index,
        reranker=reranker,
    )
    image = create_image()
    result = classifier.classify(image)
    assert result.label == "b"
    assert result.used_reranker
    assert reranker.calls == 1


def test_rerank_disabled_returns_top_candidate(monkeypatch):
    config = PipelineConfig(
        labels=["a", "b"],
        retrieval=RetrievalConfig(confidence_threshold=0.8, open_set_threshold=0.2),
        rerank=RerankConfig(enabled=False),
        allow_unknown=True,
    )
    index = DummyIndex([[("a", 0.5), ("b", 0.49)]])
    classifier = HybridClassifier(
        config=config,
        embedding_model=DummyEmbed(),
        index=index,
        reranker=None,
    )
    image = create_image()
    result = classifier.classify(image)
    assert result.label == "a"
    assert not result.used_reranker


def test_small_gap_triggers_reranker():
    config = PipelineConfig(
        labels=["a", "b"],
        retrieval=RetrievalConfig(confidence_threshold=0.5, open_set_threshold=0.2),
        rerank=RerankConfig(enabled=True, min_delta=0.2),
        allow_unknown=True,
    )
    index = DummyIndex([[("a", 0.7), ("b", 0.69)]])
    reranker = DummyReranker()
    classifier = HybridClassifier(
        config=config,
        embedding_model=DummyEmbed(),
        index=index,
        reranker=reranker,
    )
    image = create_image()
    result = classifier.classify(image)
    assert result.label == "b"
    assert result.used_reranker


def test_pipeline_builds_qwen_reranker_from_config(monkeypatch):
    loaded_index = DummyIndex([[("a", 0.8), ("b", 0.7)]])

    class DummySiglip:
        def __call__(self, images):
            del images
            return [[1.0, 0.0, 0.0]]

    created_rerankers = []

    class DummyQwen:
        def __init__(self, model_name: str) -> None:
            created_rerankers.append(model_name)

        def rerank(self, image, candidates, prompt_template):
            del image, candidates, prompt_template
            return []

    monkeypatch.setattr("lmm_classifier.pipeline.SiglipEmbeddingModel", DummySiglip)
    monkeypatch.setattr(
        "lmm_classifier.pipeline.EmbeddingIndex.load", lambda _path: loaded_index
    )
    monkeypatch.setattr("lmm_classifier.pipeline.QwenVLReranker", DummyQwen)

    config = PipelineConfig(
        labels=["a", "b"],
        retrieval=RetrievalConfig(
            index_path="/tmp/index.npz",
            confidence_threshold=0.9,
            open_set_threshold=0.2,
        ),
        rerank=RerankConfig(enabled=True, model_name="Qwen/Qwen2-VL-7B-Instruct"),
        allow_unknown=False,
    )

    classifier = HybridClassifier(config=config)
    assert isinstance(classifier.reranker, DummyQwen)
    assert created_rerankers == ["Qwen/Qwen2-VL-7B-Instruct"]
