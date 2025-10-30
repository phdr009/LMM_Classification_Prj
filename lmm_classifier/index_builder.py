"""Utilities to build SigLIP embedding indices from large image corpora."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

try:  # pragma: no cover - exercised only when PIL is available
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - handled in limited environments
    Image = None  # type: ignore

from .embeddings import EmbeddingIndex, EmbeddingRecord, SiglipEmbeddingModel
from .logging_config import get_logger, setup_logging

LOGGER = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class CorpusImage:
    """Represents a single image and its associated label."""

    path: Path
    label: str


def iter_corpus(root: Path) -> Iterator[CorpusImage]:
    """Yield labelled images from a directory tree.

    The builder supports two corpus layouts:

    ``root/<label>/*.ext``
        Traditional directory-per-label organisation. Nested subdirectories are
        allowed and the folder name becomes the label.

    ``root/*.ext``
        Flat collections without label folders. The label is derived from the
        filename stem (e.g. ``sample.jpg`` -> ``sample``).

    Files with unsupported extensions are ignored.
    """

    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            label = entry.name
            for image_path in sorted(entry.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield CorpusImage(path=image_path, label=label)
        elif entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield CorpusImage(path=entry, label=entry.stem)


def batched(iterable: Iterable[CorpusImage], batch_size: int) -> Iterator[List[CorpusImage]]:
    """Group an iterable into lists of ``batch_size`` elements."""

    batch: List[CorpusImage] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_index_from_directory(
    image_root: Path | str,
    *,
    embedding_model: Optional[SiglipEmbeddingModel] = None,
    batch_size: int = 32,
    metadata_hook: Optional[Callable[[CorpusImage], dict]] = None,
    output_path: Path | str | None = None,
    metric: str = "cosine",
) -> EmbeddingIndex:
    """Generate a retrieval index for a large labelled corpus.

    Parameters
    ----------
    image_root:
        Root directory containing labelled images. Either organise images into
        per-label folders (``root/<label>/*.ext``) or place them directly under
        the root to derive labels from filenames.
    embedding_model:
        Optional pre-loaded SigLIP embedding model. When ``None`` the default
        ``SiglipEmbeddingModel`` is instantiated.
    batch_size:
        Number of images to embed per forward pass. Tune to balance throughput
        and GPU memory usage. The default works on CPU but can be increased for
        accelerated hardware.
    metadata_hook:
        Optional callable that receives a :class:`CorpusImage` and returns a
        metadata dictionary merged into the stored :class:`EmbeddingRecord`.
    output_path:
        When provided, the built index is persisted to this path via
        :meth:`EmbeddingIndex.save`.
    metric:
        Distance metric used by :class:`EmbeddingIndex` (defaults to cosine).
    """

    if Image is None:  # pragma: no cover - requires Pillow during runtime
        raise RuntimeError("Pillow is required to load images for embedding generation")

    root = Path(image_root)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Image root {root} does not exist or is not a directory")

    model = embedding_model or SiglipEmbeddingModel()

    records: List[EmbeddingRecord] = []
    total_images = 0

    for batch_num, batch in enumerate(batched(iter_corpus(root), batch_size=batch_size), start=1):
        LOGGER.info("Embedding batch %d with %d images", batch_num, len(batch))
        images: List[Image.Image] = []
        filtered_batch: List[CorpusImage] = []
        for corpus_image in batch:
            try:
                with Image.open(corpus_image.path) as img:
                    images.append(img.convert("RGB"))
                    filtered_batch.append(corpus_image)
            except OSError as exc:  # pragma: no cover - depends on filesystem state
                LOGGER.warning("Failed to load image %s: %s", corpus_image.path, exc)
        if not images:
            continue
        vectors = model(images)
        for corpus_image, vector in zip(filtered_batch, vectors):
            metadata = {
                "path": str(corpus_image.path.relative_to(root)),
                "label": corpus_image.label,
            }
            if metadata_hook:
                metadata.update(metadata_hook(corpus_image))
            records.append(
                EmbeddingRecord(
                    vector=vector,
                    label=corpus_image.label,
                    metadata=metadata,
                )
            )
        total_images += len(filtered_batch)

    if not records:
        raise RuntimeError("No embeddings were generated; check the corpus directory")

    LOGGER.info("Generated embeddings for %d images", total_images)
    index = EmbeddingIndex(records, metric=metric)
    if output_path is not None:
        index.save(output_path)
    return index


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a SigLIP embedding index from an image corpus")
    parser.add_argument(
        "image_root",
        type=Path,
        help="Root directory containing labelled images (folders per label or flat files)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination .npz file for the index")
    parser.add_argument("--batch-size", type=int, default=32, help="Images per SigLIP forward pass")
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "l1", "l2"],
        help="Distance metric passed to scikit-learn's NearestNeighbors",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point to build an embedding index."""

    setup_logging()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    build_index_from_directory(
        image_root=args.image_root,
        batch_size=args.batch_size,
        output_path=args.output,
        metric=args.metric,
    )
    LOGGER.info("Index saved to %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())
