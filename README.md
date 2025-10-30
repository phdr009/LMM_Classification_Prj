# LMM Classification Project

Hybrid train-free image pattern classification pipeline combining SigLIP retrieval and Qwen2-VL reranking.

## Features

- **Embedding retrieval**: SigLIP-so400m patch14 384 encoder produces normalized embeddings for efficient similarity search over large corpora (20k+ images).
- **Adaptive reranking**: When retrieval confidence is low, the pipeline queries Qwen2-VL-7B to disambiguate ambiguous candidates.
- **Open/closed set support**: Threshold policies allow forcing predictions into a fixed label set or emitting an `unknown` label when the match confidence is low.
- **Configuration driven**: YAML configuration controls thresholds, label lists, and reranking behaviour.
- **Logging**: Centralized logging configuration with optional JSON output.
- **Testing**: Pytest unit tests covering decision logic for retrieval, reranking, and open-set handling.

## Project Layout

```
├── config
│   └── default.yaml
├── lmm_classifier
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── index_builder.py
│   ├── logging_config.py
│   ├── pipeline.py
│   └── rerank.py
├── tests
│   ├── test_index_builder.py
│   └── test_pipeline.py
└── pyproject.toml
```

## Generating SigLIP embeddings for a large corpus

Use the provided CLI to embed a labelled corpus (20k+ images) into a reusable
retrieval index. The expected directory layout is `root/<label>/*.jpg` (nested
folders are allowed). The builder streams images in batches so it can operate on
large datasets without exhausting memory.

```bash
python -m lmm_classifier.index_builder /path/to/corpus --output /path/to/index.npz --batch-size 64
```

The resulting `.npz` file can be referenced from `config/default.yaml` via the
`retrieval.index_path` setting.

You can also call the helper directly if you prefer Python scripts:

```python
from pathlib import Path

from lmm_classifier import build_index_from_directory

build_index_from_directory(
    image_root=Path("/data/corpus"),
    batch_size=64,
    output_path=Path("/data/siglip_index.npz"),
)
```

## Configuration

The pipeline is configured via YAML (see `config/default.yaml`). Key options include:

- `labels`: Closed-set label list. Set `allow_unknown: true` to enable open-set fallbacks.
- `retrieval.index_path`: Path to the persisted SigLIP embedding index generated with the corpus builder CLI.
- `retrieval.confidence_threshold` / `retrieval.open_set_threshold`: Confidence gates controlling when to trust retrieval results or emit `unknown`.
- `rerank.enabled`: Toggle Qwen2-VL reranking when retrieval is uncertain.
- `rerank.model_name`: Hugging Face model identifier for the LMM (defaults to `Qwen/Qwen2-VL-7B-Instruct`).
- `rerank.prompt_template`: Natural-language prompt that guides the LMM selection.

## Usage

```python
from pathlib import Path
from PIL import Image

from lmm_classifier import PipelineConfig, HybridClassifier

config = PipelineConfig.from_yaml(Path("config/default.yaml"))
classifier = HybridClassifier(config)
image = Image.open("/path/to/image.jpg")
result = classifier.classify(image)
print(result.label, result.score, result.used_reranker)
```

## Running Tests

```bash
pytest
```
