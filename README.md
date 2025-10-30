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
├── src
│   └── lmm_classifier
│       ├── __init__.py
│       ├── config.py
│       ├── embeddings.py
│       ├── logging_config.py
│       ├── pipeline.py
│       └── rerank.py
├── tests
│   └── test_pipeline.py
└── pyproject.toml
```

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
