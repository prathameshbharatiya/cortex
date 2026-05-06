"""
cortex-health — check whether Cortex and its adapters are reachable.

Usage:
    cortex-health
    cortex-health --server grpc://localhost:50051
    cortex-health --server http://localhost:8765
    cortex-health --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _check_local() -> dict:
    """Instantiate the full local stack and verify everything initialises."""
    results: dict = {"mode": "local", "components": {}}

    # Core gate
    try:
        from cortex.gate.certification import CertificationGate
        g = CertificationGate()
        results["components"]["certification_gate"] = {"ok": True, "note": "initialised"}
    except Exception as exc:
        results["components"]["certification_gate"] = {"ok": False, "error": str(exc)}

    # Memory gateway
    try:
        from cortex.memory.gateway import MemoryGateway
        from cortex.memory.adapters import InMemoryAdapter
        gw = MemoryGateway()
        gw.register(InMemoryAdapter(), priority=1)
        results["components"]["memory_gateway"] = {"ok": True, "adapters": 1}
    except Exception as exc:
        results["components"]["memory_gateway"] = {"ok": False, "error": str(exc)}

    # Redis (optional)
    try:
        import redis as _redis
        r = _redis.from_url("redis://localhost:6379", socket_connect_timeout=1)
        r.ping()
        results["components"]["redis"] = {"ok": True, "url": "redis://localhost:6379"}
    except ImportError:
        results["components"]["redis"] = {"ok": None, "note": "redis extra not installed"}
    except Exception as exc:
        results["components"]["redis"] = {"ok": False, "error": str(exc)}

    return results


def _check_grpc(address: str) -> dict:
    try:
        import grpc
        from cortex.integration.grpc import cortex_pb2 as pb
        from cortex.integration.grpc import cortex_pb2_grpc as pb_grpc
        channel = grpc.insecure_channel(address.replace("grpc://", ""))
        stub = pb_grpc.CortexServiceStub(channel)
        req = pb.HealthRequest()
        resp = stub.Health(req, timeout=3)
        return {"mode": "grpc", "address": address, "ok": True, "status": resp.status}
    except Exception as exc:
        return {"mode": "grpc", "address": address, "ok": False, "error": str(exc)}


def _check_http(address: str) -> dict:
    try:
        import urllib.request
        url = address.rstrip("/") + "/health"
        with urllib.request.urlopen(url, timeout=3) as r:
            data = json.loads(r.read())
        return {"mode": "http", "address": address, "ok": True, **data}
    except Exception as exc:
        return {"mode": "http", "address": address, "ok": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cortex-health",
        description="Check Cortex and adapter connectivity.",
    )
    parser.add_argument(
        "--server", "-s",
        default=None,
        help="Remote server address (grpc://host:port or http://host:port). "
             "Omit to check local stack.",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    t0 = time.monotonic()

    if args.server is None:
        result = _check_local()
    elif args.server.startswith("grpc://"):
        result = _check_grpc(args.server)
    elif args.server.startswith("http://") or args.server.startswith("https://"):
        result = _check_http(args.server)
    else:
        print(f"Unknown server scheme: {args.server}", file=sys.stderr)
        sys.exit(2)

    result["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    # Exit 1 if any component is not ok
    all_ok = _all_ok(result)
    sys.exit(0 if all_ok else 1)


def _all_ok(result: dict) -> bool:
    if "ok" in result and result["ok"] is False:
        return False
    for v in result.get("components", {}).values():
        if isinstance(v, dict) and v.get("ok") is False:
            return False
    return True


def _print_human(result: dict) -> None:
    GREEN = "\033[92m"
    RED   = "\033[91m"
    GRAY  = "\033[90m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    mode = result.get("mode", "unknown")
    print(f"\n{BOLD}Cortex health check{RESET}  [{mode}]  {result['latency_ms']}ms\n")

    if "components" in result:
        for name, info in result["components"].items():
            ok = info.get("ok")
            if ok is True:
                icon = f"{GREEN}✓{RESET}"
            elif ok is False:
                icon = f"{RED}✗{RESET}"
            else:
                icon = f"{GRAY}–{RESET}"
            note = info.get("error") or info.get("note") or ""
            print(f"  {icon}  {name:<28} {GRAY}{note}{RESET}")
    elif "ok" in result:
        ok = result["ok"]
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        addr = result.get("address", "")
        err  = result.get("error", "")
        print(f"  {icon}  {addr}  {GRAY}{err}{RESET}")

    print()


if __name__ == "__main__":
    main()
