"""Engine wrapper: pplx-decider-flash-v1.2 on Apple silicon (MPS).

What is theirs: the checkpoint, the joint request protocol (v3), the readout
head, the temperature. What is ours: the device choice, the dtype choice on
non-CUDA hardware, the question set, and the aggregation.

One thing to know: their loader sets dtype = bfloat16 on cuda and float32
everywhere else. On MPS we prefer bfloat16 (the weights ship in BF16 and the
Mac has the memory), with a float32 fallback if the kernel path misbehaves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

TEMPERATURE_NOTE = 1.5539092221462472

# Three phrasings of the same proposition. If the model is worth trusting, they
# land close together; the distance between them is reported to the user.
PHRASINGS = (
    'The answer to the question "{q}" is yes.',
    'If someone asks "{q}", the correct answer is yes.',
    'Taking the situation into account, "{q}" should be answered yes.',
)

EVIDENCE_LEVELS = (
    "no evidence, only the question itself",
    "a vague hint, nothing checkable",
    "some relevant specifics",
    "concrete numbers, constraints and details",
)


@dataclass
class Read:
    p: float
    spread: float
    evidence: float
    readings: list[float]
    usage: dict
    seconds: float


class CoinEngine:
    def __init__(self, checkpoint: str, device: str = "mps", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoProcessor  # noqa: F401  (import check before the heavy load)
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(checkpoint) / "source" / "src"))
        from pplx_decider import Decider

        self.device = device
        self.torch = torch
        self._dtype_patch = dtype
        t0 = time.time()
        self.decider = self._load(Decider, checkpoint, device, dtype)
        self.load_seconds = round(time.time() - t0, 1)
        self.model_name = "pplx-decider-flash-v1.2"

    def _load(self, Decider, checkpoint, device, dtype):
        """Load, then force the dtype we want on non-CUDA hardware."""
        torch = self.torch
        target = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        patch = patch_qwen3_5_dtype(torch, target) if device != "cuda" else None
        try:
            d = Decider.from_pretrained(checkpoint, device=device, fast=False)
        finally:
            if patch:
                patch()
        if device != "cuda":
            # their loader picked float32 for non-CUDA; move to the requested dtype
            d.backbone.to(target)
            d.readout.to(target)
        return d

    def read(self, question: str, context: str = "") -> Read:
        question = question.strip()
        context = context.strip()
        if context:
            state = (
                "Situation, as described by the person asking:\n"
                f"{context}\n\n"
                f"Question they are asking: {question}"
            )
        else:
            state = f"Question: {question}"

        requests = []
        keys = []
        for i, template in enumerate(PHRASINGS):
            key = f"v{i}"
            keys.append(key)
            requests.append(
                {
                    "state": state,
                    "questions": {
                        key: {
                            "type": "noul",
                            "instructions": template.format(q=question),
                            "criteria": {"true": "yes", "false": "no"},
                        }
                    },
                }
            )
        requests.append(
            {
                "state": state,
                "questions": {
                    "evidence": {
                        "type": "score",
                        "instructions": "How much concrete evidence does the text contain about this question?",
                        "criteria": list(EVIDENCE_LEVELS),
                    }
                },
            }
        )

        t0 = time.time()
        responses = self.decider.decide(requests)
        seconds = time.time() - t0

        readings = [float(r["answers"][k]["noul"]) for r, k in zip(responses[:3], keys)]
        ev = responses[3]["answers"]["evidence"]
        evidence = float(ev["score"]) / (len(EVIDENCE_LEVELS) - 1)
        usage = {
            "input_tokens": sum(int(r["usage"]["input_tokens"]) for r in responses),
            "output_tokens": 0,
            "sequences": sum(int(r["usage"]["sequences"]) for r in responses),
        }
        return Read(
            p=sum(readings) / len(readings),
            spread=max(readings) - min(readings),
            evidence=max(0.0, min(1.0, evidence)),
            readings=readings,
            usage=usage,
            seconds=round(seconds, 2),
        )


def patch_qwen3_5_dtype(torch, target):
    """Force Qwen3_5Model.from_pretrained to a dtype, without touching their source."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    original = m.Qwen3_5Model.from_pretrained.__func__

    def wrapper(cls, *args, **kwargs):
        kwargs["dtype"] = target
        return original(cls, *args, **kwargs)

    m.Qwen3_5Model.from_pretrained = classmethod(wrapper)

    def undo():
        m.Qwen3_5Model.from_pretrained = classmethod(original)

    return undo
