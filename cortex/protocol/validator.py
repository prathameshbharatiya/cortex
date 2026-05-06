"""
CTX Validator
=============
Validates a .ctx file: schema integrity, version compatibility,
and optional cryptographic signature verification.

CLI usage
---------
    python -m cortex.protocol.validator myfile.ctx
    python -m cortex.protocol.validator myfile.ctx --verify-key <hex>
    python -m cortex.protocol.validator myfile.ctx --json-dump

Programmatic usage
------------------
    from cortex.protocol.validator import validate_ctx_file

    result = validate_ctx_file("path/to/file.ctx")
    if result.valid:
        ctx = result.ctx
    else:
        print(result.errors)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cortex.models.context import Context


@dataclass
class ValidationResult:
    valid:      bool
    version:    str = ""
    ctx:        "Context | None" = None
    errors:     list[str] = field(default_factory=list)
    warnings:   list[str] = field(default_factory=list)
    signature_verified: bool = False


def validate_ctx_file(
    path: str,
    verify_key_hex: str | None = None,
) -> ValidationResult:
    """
    Validate a .ctx file programmatically.

    Parameters
    ----------
    path            : path to the .ctx file
    verify_key_hex  : hex-encoded Ed25519 verify key (optional)

    Returns
    -------
    ValidationResult with .valid bool, .ctx (if valid), and .errors list
    """
    result = ValidationResult(valid=False)

    # ── Step 1: Load binary ───────────────────────────────────────────────────
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        result.errors.append(f"Cannot read file: {exc}")
        return result

    if len(raw) == 0:
        result.errors.append("File is empty")
        return result

    # ── Step 2: Parse MessagePack envelope ────────────────────────────────────
    try:
        import msgpack
        envelope = msgpack.unpackb(raw, raw=False)
    except Exception as exc:
        result.errors.append(f"Not a valid .ctx binary (MessagePack parse failed): {exc}")
        return result

    if not isinstance(envelope, dict):
        result.errors.append("Root object must be a MessagePack dict")
        return result

    # ── Step 3: Version check ─────────────────────────────────────────────────
    version = envelope.get("v", "")
    if not version:
        result.errors.append("Missing 'v' (version) field")
        return result

    result.version = version

    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        result.errors.append(f"Invalid version format: {version!r}")
        return result

    from cortex.protocol.ctx_format import CTX_VERSION
    expected_major = int(CTX_VERSION.split(".")[0])
    if major != expected_major:
        result.errors.append(
            f"Incompatible version {version!r}: "
            f"this runtime supports major={expected_major}"
        )
        return result

    # ── Step 4: Deserialise Context ───────────────────────────────────────────
    try:
        from cortex.protocol.ctx_format import CTXSerializer
        serializer = CTXSerializer()
        ctx = serializer.from_bytes(raw)
        result.ctx = ctx
    except Exception as exc:
        result.errors.append(f"Context deserialisation failed: {exc}")
        return result

    # ── Step 5: Signature verification (optional) ─────────────────────────────
    if verify_key_hex:
        try:
            from cortex.protocol.signing import CTXSigner, CTXSignatureError
            # Signature must be passed alongside the .ctx file in a sidecar
            sig_path = path + ".sig"
            try:
                with open(sig_path, "rb") as fh:
                    sig_bytes = fh.read()
                CTXSigner.verify_with_key(raw, sig_bytes, bytes.fromhex(verify_key_hex))
                result.signature_verified = True
            except FileNotFoundError:
                result.warnings.append(
                    f"No signature sidecar found at {sig_path!r} — skipping verification"
                )
            except CTXSignatureError as exc:
                result.errors.append(f"Signature verification failed: {exc}")
                return result
        except CTXSignatureError as exc:
            if exc.install_required:
                result.warnings.append("PyNaCl not installed — skipping signature verification")
            else:
                result.errors.append(str(exc))
                return result

    result.valid = True
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="cortex-validate",
        description="Validate a .ctx file: schema, version, and optional signature",
    )
    parser.add_argument("ctx_file", help="Path to the .ctx file")
    parser.add_argument(
        "--verify-key",
        metavar="HEX",
        help="Hex-encoded Ed25519 verify key for signature verification",
    )
    parser.add_argument(
        "--json-dump",
        action="store_true",
        help="Print the deserialised Context as JSON",
    )
    args = parser.parse_args(argv)

    result = validate_ctx_file(args.ctx_file, verify_key_hex=args.verify_key)

    if result.valid:
        print(f"✓ VALID  [{result.version}]  {args.ctx_file}")
        if result.signature_verified:
            print("  ✓ Signature verified")
        for w in result.warnings:
            print(f"  ⚠ {w}")
        if args.json_dump and result.ctx:
            from cortex.protocol.ctx_format import CTXSerializer
            print(CTXSerializer().to_json(result.ctx))
        return 0
    else:
        print(f"✗ INVALID  {args.ctx_file}")
        for err in result.errors:
            print(f"  ✗ {err}")
        for w in result.warnings:
            print(f"  ⚠ {w}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
