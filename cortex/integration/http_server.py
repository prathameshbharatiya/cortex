"""
Cortex HTTP Server
==================
REST API transport for the CortexServer.

Exposes the full Cortex pipeline over HTTP/JSON.
Any language, any framework, any robot stack can call this.

Endpoints
---------
  POST /certify              → certify an action
  POST /record               → record an execution outcome
  POST /context              → get a validated CTX
  POST /memory               → store a memory record
  GET  /health               → server health and adapter status

Run
---
    from cortex.integration.http_server import run_http_server
    run_http_server(server, host="0.0.0.0", port=8765)

Or as a module:
    python -m cortex.integration.http_server
"""

from __future__ import annotations

import time
import uuid
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel as PydanticBase
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from cortex.integration.protocol import (
    CertifyRequest, RecordOutcomeRequest,
    GetContextRequest, StoreMemoryRequest, HealthRequest,
)
from cortex.integration.server import CortexServer


# ── Pydantic models for FastAPI ───────────────────────────────────────────────

if _FASTAPI_AVAILABLE:

    class CertifyBody(PydanticBase):
        action_type:      str   = "move_ee"
        source:           str   = "unknown"
        intent:           str   = ""
        target_position:  list  = [0.0, 0.0, 0.0]
        target_quaternion: list = [1.0, 0.0, 0.0, 0.0]
        max_speed_ms:     float | None = None
        max_force_n:      float | None = None
        timeout_ms:       float = 5000.0
        task:             str   = ""
        ee_position:      list  = [0.0, 0.0, 0.0]
        ee_quaternion:    list  = [1.0, 0.0, 0.0, 0.0]
        joint_positions:  list  = []
        joint_torques:    list  = []
        gripper_open:     bool  = True
        humans_nearby:    bool  = False
        emergency_stop:   bool  = False
        in_singularity:   bool  = False
        confidence_floor: float = 0.60
        validity_window_ms: float = 500.0
        skip_physics:     bool  = False

    class RecordBody(PydanticBase):
        trace_id:         str
        ctx_id:           str
        action_id:        str
        task:             str
        outcome_class:    str
        description:      str   = ""
        predicted_confidence: float = 0.5
        memory_record_ids: list = []
        ee_position:      list  = []

    class ContextBody(PydanticBase):
        task:             str
        ee_position:      list  = [0.0, 0.0, 0.0]
        ee_quaternion:    list  = [1.0, 0.0, 0.0, 0.0]
        joint_positions:  list  = []
        joint_torques:    list  = []
        gripper_open:     bool  = True
        humans_nearby:    bool  = False
        emergency_stop:   bool  = False
        top_k:            int   = 10
        skip_physics:     bool  = False

    class MemoryBody(PydanticBase):
        source:       str
        memory_type:  str   = "episodic"
        content_text: str
        content:      dict  = {}
        confidence:   float = 0.9


def create_app(cortex_server: CortexServer) -> "FastAPI":
    """Create the FastAPI application bound to a CortexServer instance."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError("fastapi is required: pip install fastapi uvicorn")

    app = FastAPI(
        title="Cortex",
        description="Context Certification Infrastructure for Physical AI",
        version="0.1.0",
    )

    def _rid() -> str:
        return str(uuid.uuid4())[:8]

    from fastapi import Depends
    from cortex.security.api_keys import require_permission

    @app.post("/certify")
    async def certify(body: CertifyBody, _=Depends(require_permission("certify"))) -> dict:
        req  = CertifyRequest(request_id=_rid(), **body.model_dump())
        resp = cortex_server.certify(req)
        return resp.__dict__

    @app.post("/record")
    async def record(body: RecordBody, _=Depends(require_permission("record"))) -> dict:
        req  = RecordOutcomeRequest(request_id=_rid(), **body.model_dump())
        resp = cortex_server.record_outcome(req)
        return resp.__dict__

    @app.post("/context")
    async def context(body: ContextBody, _=Depends(require_permission("context"))) -> dict:
        req  = GetContextRequest(request_id=_rid(), **body.model_dump())
        resp = cortex_server.get_context(req)
        return resp.__dict__

    @app.post("/memory")
    async def memory(body: MemoryBody, _=Depends(require_permission("memory"))) -> dict:
        req  = StoreMemoryRequest(request_id=_rid(), **body.model_dump())
        resp = cortex_server.store_memory(req)
        return resp.__dict__

    @app.get("/health")
    async def health() -> dict:
        # Health is unauthenticated — Kubernetes liveness probes need this
        resp = cortex_server.health(HealthRequest(request_id=_rid()))
        return resp.__dict__

    @app.get("/")
    async def root() -> dict:
        return {
            "name":    "Cortex",
            "version": "0.1.0",
            "tagline": "We do not trust AI decisions. We certify them.",
            "endpoints": ["/certify", "/record", "/context", "/memory", "/health"],
        }

    return app


def run_http_server(
    server: CortexServer,
    host:   str = "0.0.0.0",
    port:   int = 8765,
) -> None:
    """Run the Cortex HTTP server. Blocks until stopped."""
    try:
        import uvicorn
        app = create_app(server)
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError:
        raise ImportError("uvicorn is required: pip install uvicorn")


def main() -> None:
    """
    Entry point for the `cortex-serve-http` CLI command.

    Loads configuration from the full priority chain, then starts the HTTP server.

    Quick start:
        cortex-serve-http
        cortex-serve-http --port 8765
        CORTEX_HTTP_PORT=9000 cortex-serve-http
    """
    import argparse
    import logging
    import os

    parser = argparse.ArgumentParser(
        prog="cortex-serve-http",
        description="Start the Cortex HTTP REST server.",
    )
    parser.add_argument("--host",   default=None, help="Bind host (overrides config)")
    parser.add_argument("--port",   default=None, type=int, help="Bind port (overrides config)")
    parser.add_argument("--config", default=None, help="Path to cortex.yaml")
    args = parser.parse_args()

    if args.config:
        os.environ["CORTEX_CONFIG_FILE"] = args.config

    from cortex.config import get_settings
    from cortex.config.factory import build_server

    cfg = get_settings()
    if args.host: cfg.http.host = args.host
    if args.port: cfg.http.port = args.port

    logging.basicConfig(
        level=getattr(logging, cfg.server.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    log = logging.getLogger("cortex.http")
    log.info(cfg.display())

    core_server = build_server(cfg)
    log.info("Cortex HTTP server starting on %s:%d", cfg.http.host, cfg.http.port)
    run_http_server(core_server, host=cfg.http.host, port=cfg.http.port)


if __name__ == "__main__":
    main()
