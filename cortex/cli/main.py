"""
cortex — unified CLI
=====================
Entry point for the `cortex` command, dispatching sub-commands:

  cortex keygen          Generate an Ed25519 signing key pair
  cortex validate-ctx    Validate a .ctx file signature and schema
  cortex status          Show gate + memory health status
  cortex simulate        Run the simulation harness
  cortex serve           Start the gRPC certification server

Usage
-----
    cortex --help
    cortex keygen --out keys/
    cortex validate-ctx path/to/file.ctx
    cortex status
    cortex simulate
    cortex simulate --tag safety
"""

from __future__ import annotations

import argparse
import sys


# ── Sub-command handlers ──────────────────────────────────────────────────────

def cmd_keygen(args: argparse.Namespace) -> int:
    """Generate an Ed25519 key pair for .ctx signing."""
    import os
    from cortex.protocol.signing import CTXSigner

    signer   = CTXSigner()
    out_dir  = args.out or "."
    os.makedirs(out_dir, exist_ok=True)

    sk_path  = os.path.join(out_dir, "cortex_signing.key")
    vk_path  = os.path.join(out_dir, "cortex_verify.key")

    signer.save_to_file(sk_path)
    with open(vk_path, "w") as f:
        f.write(signer.verify_key_hex + "\n")

    print(f"Signing key  → {sk_path}")
    print(f"Verify key   → {vk_path}")
    print(f"Signing key (hex): {signer.signing_key_hex}")
    print(f"Verify  key (hex): {signer.verify_key_hex}")
    print("\nKeep the signing key secret. Distribute only the verify key.")
    return 0


def cmd_validate_ctx(args: argparse.Namespace) -> int:
    """Validate a .ctx file (schema + optional signature check)."""
    from cortex.protocol.validator import validate_ctx_file

    result = validate_ctx_file(args.path)
    for msg in result.errors:
        print(f"  ERROR: {msg}")
    for msg in result.warnings:
        print(f"  WARN:  {msg}")
    if result.valid:
        print("\n✓ Context file is valid.")
        return 0
    else:
        print("\n✗ Validation failed.")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show CertificationGate and memory adapter health."""
    import cortex
    from cortex.gate.certification import CertificationGate
    from cortex.memory.gateway import MemoryGateway

    print("Cortex Status")
    print("─" * 40)

    # Gate instantiation
    try:
        gate = CertificationGate()
        print(f"  Gate:          OK  (sentinel={type(gate.sentinel).__name__})")
    except Exception as exc:
        print(f"  Gate:          ERROR — {exc}")

    # Memory gateway
    try:
        gw = MemoryGateway()
        print(f"  MemoryGateway: OK  ({gw.adapter_count} adapters)")
    except Exception as exc:
        print(f"  MemoryGateway: ERROR — {exc}")

    # Version
    try:
        import cortex._version as _v
        version = getattr(_v, "__version__", "unknown")
    except ImportError:
        version = "unknown"
    print(f"  Version:       {version}")
    print("─" * 40)
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run the simulation harness against standard or tagged scenarios."""
    from cortex.sim.harness import SimHarness, SimReporter

    harness = SimHarness()
    if args.tag:
        harness.add_standard_scenarios()
        results = harness.run_tagged(args.tag)
        if not results:
            print(f"No scenarios found for tag '{args.tag}'.")
            return 1
    else:
        harness.add_standard_scenarios()
        results = harness.run_all()

    reporter = SimReporter(results)
    reporter.print_summary()
    return 0 if reporter.all_passed else 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the gRPC certification server."""
    try:
        from cortex.integration.grpc.server import main as grpc_main
        grpc_main()
        return 0
    except ImportError as exc:
        print(f"gRPC server not available: {exc}")
        print("Install with: pip install 'cortex-gate[grpc]'")
        return 1


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortex",
        description="Cortex Certification Infrastructure CLI",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print Cortex version and exit.",
    )

    subs = parser.add_subparsers(dest="command", metavar="<command>")

    # keygen
    p_keygen = subs.add_parser("keygen", help="Generate Ed25519 signing key pair")
    p_keygen.add_argument(
        "--out", "-o", default=".",
        help="Output directory for key files (default: current directory)",
    )

    # validate-ctx
    p_val = subs.add_parser("validate-ctx", help="Validate a .ctx wire-format file")
    p_val.add_argument("path", help="Path to the .ctx file to validate")

    # status
    subs.add_parser("status", help="Show gate and memory health")

    # simulate
    p_sim = subs.add_parser("simulate", help="Run simulation harness")
    p_sim.add_argument(
        "--tag", "-t", default="",
        help="Only run scenarios with this tag (e.g. 'safety', 'smoke')",
    )
    p_sim.add_argument(
        "--scenario", "-s", default="",
        help="Run a single scenario by name",
    )

    # serve
    p_serve = subs.add_parser("serve", help="Start the gRPC certification server")
    p_serve.add_argument("--port", type=int, default=50051)

    return parser


def main() -> None:
    parser  = _build_parser()
    args    = parser.parse_args()

    if args.version:
        try:
            from cortex._version import __version__
            print(f"cortex {__version__}")
        except ImportError:
            print("cortex (version unknown)")
        sys.exit(0)

    handlers = {
        "keygen":       cmd_keygen,
        "validate-ctx": cmd_validate_ctx,
        "status":       cmd_status,
        "simulate":     cmd_simulate,
        "serve":        cmd_serve,
    }

    if args.command not in handlers:
        parser.print_help()
        sys.exit(0)

    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
