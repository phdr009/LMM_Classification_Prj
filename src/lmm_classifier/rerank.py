"""Reranking stage using Qwen2-VL-7B."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

try:  # pragma: no cover - exercised in integration environments
    from transformers import AutoModelForCausalLM, AutoProcessor
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test environments
    AutoModelForCausalLM = AutoProcessor = None  # type: ignore

from .logging_config import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class RerankCandidate:
    """Candidate returned by the retrieval stage."""

    label: str
    score: float
    metadata: dict


class QwenVLReranker:
    """Use the Qwen2-VL LMM to disambiguate uncertain predictions."""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct") -> None:
        if AutoModelForCausalLM is None or AutoProcessor is None:  # pragma: no cover
            raise RuntimeError("transformers is required to use the reranker")
        self.model_name = model_name
        LOGGER.info("Loading Qwen2-VL model: %s", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

    def rerank(
        self,
        image: Any,
        candidates: Sequence[RerankCandidate],
        prompt_template: str,
    ) -> List[RerankCandidate]:
        """Return candidates ordered by LMM preference."""

        formatted_candidates = ", ".join(candidate.label for candidate in candidates)
        prompt = prompt_template.format(
            image_description="Input image provided as tensor.",
            candidates=formatted_candidates,
        )
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        outputs = self.model.generate(**inputs, max_new_tokens=16)
        text_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        LOGGER.debug("Qwen reranker output: %s", text_output)
        ordered = sorted(
            candidates,
            key=lambda candidate: text_output.find(candidate.label)
            if candidate.label in text_output
            else float("inf"),
        )
        return ordered


__all__ = ["QwenVLReranker", "RerankCandidate"]
