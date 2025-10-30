"""Tests for the embedding index builder."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from lmm_classifier.index_builder import build_index_from_directory


class DummyEmbeddingModel:
    """Deterministic embedding model used for tests."""

    def __init__(self) -> None:
        self.calls: List[int] = []

    def __call__(self, images):  # type: ignore[override]
        batch_size = len(images)
        self.calls.append(batch_size)
        # Produce simple embeddings derived from image size to keep values stable.
        vectors = []
        for image in images:
            width, height = image.size
            vectors.append([width, height, width + height])
        return np.asarray(vectors, dtype=np.float32)


class DummyIndex:
    def __init__(self, records, metric="cosine") -> None:  # type: ignore[no-untyped-def]
        self.records = list(records)
        self.metric = metric
        self.saved_path: Path | None = None

    def save(self, path):  # type: ignore[no-untyped-def]
        self.saved_path = Path(path)


def create_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (8 + color[0] % 3, 8 + color[1] % 3), color)
    image.save(path)


def test_build_index_from_directory(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    (root / "cat").mkdir(parents=True)
    (root / "dog").mkdir()
    create_image(root / "cat" / "a.jpg", (255, 0, 0))
    create_image(root / "cat" / "b.png", (0, 255, 0))
    create_image(root / "dog" / "c.jpg", (0, 0, 255))

    model = DummyEmbeddingModel()
    monkeypatch.setattr("lmm_classifier.index_builder.SiglipEmbeddingModel", lambda: model)
    monkeypatch.setattr("lmm_classifier.index_builder.EmbeddingIndex", DummyIndex)

    output_path = tmp_path / "index.npz"
    index = build_index_from_directory(
        image_root=root,
        batch_size=2,
        output_path=output_path,
    )

    assert isinstance(index, DummyIndex)
    # All three images should be embedded with the specified batch sizes.
    assert model.calls == [2, 1]
    assert len(index.records) == 3
    assert index.saved_path == output_path
    labels = sorted(record.label for record in index.records)
    assert labels == ["cat", "cat", "dog"]
    metadata_paths = sorted(record.metadata["path"] for record in index.records)
    assert metadata_paths == ["cat/a.jpg", "cat/b.png", "dog/c.jpg"]
