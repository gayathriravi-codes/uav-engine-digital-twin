"""
api/server.py -- FastAPI wrapper for the AeroTwin frontend.

Run from the repo root:
    pip install fastapi "uvicorn[standard]"
    uvicorn api.server:app --reload --port 8000

Endpoints
  WS   /stream?fault=overheat&severity=1.0&seed=7&tick_seconds=1.0&session=demo
  GET  /flight?fault=&severity=&seed=      -> list of ticks (for scrub / replay)
  POST /whatif {fault, severity, seed?}    -> list of ticks (counterfactual flight)
  POST /confirm {session, t, decision}     -> operator decision, audit-logged
  GET  /health                             -> faults available, live-hook status
"""
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.adapter import FAULTS, HOOKS, iter_flight, new_state

app = FastAPI(title="AeroTwin API")
app.add_middleware(  # local dev only; tighten before any public deployment
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

SESSIONS = {}
LOG_PATH = os.path.join("logs", "api_confirmations.jsonl")


class WhatIf(BaseModel):
    fault: str
    severity: float = 1.0
    seed: Optional[int] = None


class Confirm(BaseModel):
    session: str = "default"
    t: int
    decision: str  # "confirm" | "reject"


def _check_fault(fault):
    if fault not in FAULTS:
        raise HTTPException(400, f"unknown fault '{fault}', choose from {FAULTS}")


@app.get("/health")
def health():
    return {"faults": FAULTS,
            "hooks_wired": {k: v is not None for k, v in HOOKS.items()}}


@app.get("/flight")
def flight(fault: str = "none", severity: float = 1.0, seed: Optional[int] = None):
    _check_fault(fault)
    return list(iter_flight(fault, severity, seed))


@app.post("/whatif")
def whatif(req: WhatIf):
    # Regenerates the flight through the same fault injectors, as whatif_engine v2
    # does. Swap in whatif_engine directly once its signature is known.
    _check_fault(req.fault)
    return list(iter_flight(req.fault, req.severity, req.seed))


@app.post("/confirm")
def confirm(req: Confirm):
    if req.decision not in ("confirm", "reject"):
        raise HTTPException(400, "decision must be 'confirm' or 'reject'")
    state = SESSIONS.get(req.session)
    if state is None:
        raise HTTPException(404, f"no active session '{req.session}'")
    status = "confirmed" if req.decision == "confirm" else "rejected"
    state["confirmations"][req.t] = status
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": req.session, "t": req.t, "decision": status,
        }) + "\n")
    return {"ok": True, "t": req.t, "status": status}


@app.websocket("/stream")
async def stream(ws: WebSocket, fault: str = "none", severity: float = 1.0,
                 seed: Optional[int] = None, tick_seconds: float = 1.0,
                 session: str = "default"):
    await ws.accept()
    if fault not in FAULTS:
        await ws.send_json({"type": "error", "detail": f"unknown fault '{fault}'"})
        await ws.close()
        return
    state = new_state()
    SESSIONS[session] = state
    gen = iter_flight(fault, severity, seed, state)
    try:
        while True:
            # inference is CPU-bound; keep the event loop free
            tick = await run_in_threadpool(next, gen, None)
            if tick is None:
                await ws.send_json({"type": "end"})
                break
            await ws.send_json(tick)
            await asyncio.sleep(tick_seconds)
    except WebSocketDisconnect:
        pass
    finally:
        SESSIONS.pop(session, None)