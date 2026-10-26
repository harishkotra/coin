"""Coin — the oracle that admits when it doesn't know.

Single process: FastAPI serves the JSON API and the static front-end, and holds
one warm pplx-decider-flash-v1.2 on the GPU.

Privacy: the question is stored (that is the wall). The situation you paste is
NOT stored, only used for the read.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
CKPT = os.environ.get("COIN_CKPT", str(ROOT / "ckpt"))
DB = Path(os.environ.get("COIN_DB", str(ROOT / "data" / "coin.db")))
DEVICE = os.environ.get("COIN_DEVICE", "mps")
DTYPE = os.environ.get("COIN_DTYPE", "bfloat16")

state: dict = {"engine": None, "error": None, "ready_at": None, "busy": False}
lock = threading.Lock()


def db() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, timeout=10)
    con.execute(
        """CREATE TABLE IF NOT EXISTS reads (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              question TEXT NOT NULL,
              p REAL NOT NULL,
              spread REAL NOT NULL,
              evidence REAL NOT NULL,
              created_at REAL NOT NULL
           )"""
    )
    con.commit()
    return con


def load_engine() -> None:
    try:
        from app.engine import CoinEngine

        eng = CoinEngine(CKPT, device=DEVICE, dtype=DTYPE)
        state["engine"] = eng
        state["ready_at"] = time.time()
        print(f"[coin] model ready on {DEVICE} in {eng.load_seconds}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[coin] model failed: {state['error']}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db().close()
    threading.Thread(target=load_engine, daemon=True).start()
    yield


app = FastAPI(title="Coin", lifespan=lifespan, docs_url=None, redoc_url=None)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=220)
    context: str = Field(default="", max_length=6000)


@app.get("/api/health")
def health():
    eng = state["engine"]
    return {
        "model_loaded": eng is not None,
        "device": DEVICE if eng else None,
        "dtype": DTYPE,
        "error": state["error"],
        "load_seconds": getattr(eng, "load_seconds", None),
    }


@app.post("/api/decide")
def decide(ask: Ask):
    eng = state["engine"]
    if eng is None:
        if state["error"]:
            raise HTTPException(503, f"the model did not load: {state['error']}")
        raise HTTPException(503, "the model is still loading, try again in a few seconds")

    if not lock.acquire(timeout=120):
        raise HTTPException(429, "one read at a time on this machine, try again")
    try:
        read = eng.read(ask.question, ask.context)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"the read failed: {type(exc).__name__}") from exc
    finally:
        lock.release()

    con = db()
    try:
        con.execute(
            "INSERT INTO reads (question, p, spread, evidence, created_at) VALUES (?,?,?,?,?)",
            (ask.question.strip(), read.p, read.spread, read.evidence, time.time()),
        )
        con.commit()
    finally:
        con.close()

    return {
        "question": ask.question.strip(),
        "p": read.p,
        "spread": read.spread,
        "evidence": read.evidence,
        "readings": len(read.readings),
        "model": eng.model_name,
        "params": "4B",
        "temperature": 1.5539092221462472,
        "device": DEVICE,
        "usage": read.usage,
        "seconds": read.seconds,
    }


@app.get("/api/wall")
def wall(sort: str = "closest", limit: int = 12):
    limit = max(1, min(50, limit))
    con = db()
    try:
        if sort == "recent":
            rows = con.execute(
                "SELECT question, p FROM reads GROUP BY question ORDER BY MAX(created_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = con.execute(
                """SELECT question, p FROM reads
                   WHERE id IN (SELECT MAX(id) FROM reads GROUP BY question)
                   ORDER BY ABS(p - 0.5) ASC, created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
    finally:
        con.close()
    return [{"question": q, "p": p} for q, p in rows]


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
app.mount("/", StaticFiles(directory=WEB), name="static")
