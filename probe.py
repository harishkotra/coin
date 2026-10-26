"""Smoke test: does the checkpoint answer on MPS, how fast, and does it move with evidence."""
import json
import sys
import time

from app.engine import CoinEngine

CKPT = sys.argv[1] if len(sys.argv) > 1 else "./ckpt"
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "mps"
DTYPE = sys.argv[3] if len(sys.argv) > 3 else "bfloat16"

CASES = [
    (
        "Is the sky blue?",
        "",
    ),
    (
        "Should I take the offer?",
        "Offer is 40% more base than today. The commute goes from 20 minutes to 95 minutes each way, "
        "four days a week. I have a two year old. The team I would join has lost three engineers in "
        "five months and the hiring manager is the third manager in that role in two years. "
        "Savings runway today is 11 months.",
    ),
    (
        "Should I take the offer?",
        "The pay is a bit better and the office is a bit further. It seems like a good company.",
    ),
    (
        "Will this feature ship by Friday?",
        "Three of the five tickets are still open. One of them is the auth migration, which the "
        "engineer owning it has never touched before and which touches every route. No QA slot is "
        "booked. Today is Wednesday.",
    ),
]

t0 = time.time()
eng = CoinEngine(CKPT, device=DEVICE, dtype=DTYPE)
print(f"loaded in {time.time() - t0:.1f}s on {DEVICE} {DTYPE}", flush=True)

for q, ctx in CASES:
    r = eng.read(q, ctx)
    print(
        json.dumps(
            {
                "q": q,
                "ctx_chars": len(ctx),
                "p": round(r.p, 4),
                "spread": round(r.spread, 4),
                "evidence": round(r.evidence, 3),
                "readings": [round(x, 3) for x in r.readings],
                "seconds": r.seconds,
                "in_tokens": r.usage["input_tokens"],
            }
        ),
        flush=True,
    )
