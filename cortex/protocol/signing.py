"""
CTX Cryptographic Signing
==========================
Every .ctx packet can be signed with Ed25519 before transmission.
The signature covers the complete serialised packet payload.
Any modification after signing is detectable.

Algorithm:    Ed25519 (libsodium via PyNaCl)
Key size:     32 bytes signing key, 32 bytes verify key
Signature:    64 bytes
Verify time:  ~0.3ms on modern hardware

Dependency:   PyNaCl (pip install PyNaCl)
              Falls back to CTXSignatureError(install_required=True) if absent.

Key management
--------------
    # Generate a fresh key pair
    signer = CTXSigner()
    print(signer.signing_key_hex)   # store this securely
    print(signer.verify_key_hex)    # distribute to verifiers

    # Reconstruct from stored key
    signer = CTXSigner.from_hex(signing_key_hex)

    # Sign a payload
    sig = signer.sign(payload_bytes)

    # Verify (using only the public verify key)
    ok = CTXSigner.verify_with_key(payload_bytes, sig, verify_key_bytes)

CLI
---
    cortex keygen --output ./cortex.key
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any


class CTXSignatureError(Exception):
    """Raised on signature verification failure or key errors."""

    def __init__(self, message: str, install_required: bool = False) -> None:
        super().__init__(message)
        self.install_required = install_required


class CTXSigner:
    """
    Ed25519 signer for .ctx packets.

    Requires PyNaCl:  pip install PyNaCl
    """

    def __init__(self, signing_key_bytes: bytes | None = None) -> None:
        try:
            from nacl.signing import SigningKey
        except ImportError as exc:
            raise CTXSignatureError(
                "PyNaCl is not installed. Install with: pip install PyNaCl",
                install_required=True,
            ) from exc

        if signing_key_bytes is not None:
            self._key = SigningKey(signing_key_bytes)
        else:
            self._key = SigningKey.generate()

    # ── Class methods ─────────────────────────────────────────────────────────

    @classmethod
    def from_hex(cls, hex_key: str) -> "CTXSigner":
        """Reconstruct signer from hex-encoded signing key."""
        return cls(bytes.fromhex(hex_key))

    @classmethod
    def from_file(cls, path: str) -> "CTXSigner":
        """Load signing key from a .key file (JSON with 'signing_key_hex')."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if "signing_key_hex" not in data:
            raise CTXSignatureError(f"Key file {path!r} missing 'signing_key_hex' field")
        return cls.from_hex(data["signing_key_hex"])

    @staticmethod
    def verify_with_key(payload: bytes, signature: bytes, verify_key_bytes: bytes) -> bool:
        """
        Verify signature against payload using a raw verify key.
        Returns True on success.  Raises CTXSignatureError on failure.
        """
        try:
            from nacl.signing import VerifyKey
            from nacl.exceptions import BadSignatureError
        except ImportError as exc:
            raise CTXSignatureError(
                "PyNaCl is not installed. Install with: pip install PyNaCl",
                install_required=True,
            ) from exc

        try:
            VerifyKey(verify_key_bytes).verify(payload, signature)
            return True
        except BadSignatureError as exc:
            raise CTXSignatureError("CTX signature verification failed — packet may be tampered") from exc

    # ── Instance methods ──────────────────────────────────────────────────────

    def sign(self, payload: bytes) -> bytes:
        """Sign payload.  Returns 64-byte Ed25519 signature."""
        return bytes(self._key.sign(payload).signature)

    def verify(self, payload: bytes, signature: bytes) -> bool:
        """Verify using this signer's own verify key."""
        return CTXSigner.verify_with_key(payload, signature, self.verify_key_bytes)

    def save_to_file(self, path: str) -> None:
        """Save key pair to JSON file.  Keep signing_key_hex secret."""
        key_data = {
            "signing_key_hex": self.signing_key_hex,
            "verify_key_hex":  self.verify_key_hex,
            "algorithm":       "Ed25519",
            "format":          "cortex-key-v1",
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(key_data, fh, indent=2)
        # Restrict permissions on Unix-like systems
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def signing_key_bytes(self) -> bytes:
        return bytes(self._key)

    @property
    def verify_key_bytes(self) -> bytes:
        return bytes(self._key.verify_key)

    @property
    def signing_key_hex(self) -> str:
        return self.signing_key_bytes.hex()

    @property
    def verify_key_hex(self) -> str:
        return self.verify_key_bytes.hex()

    @property
    def verify_key_b64(self) -> str:
        return base64.b64encode(self.verify_key_bytes).decode()


# ── Convenience: sign+pack, verify+unpack ─────────────────────────────────────

def sign_ctx_bytes(payload: bytes, signer: CTXSigner) -> dict[str, bytes]:
    """
    Sign a .ctx payload and return a signed envelope dict:
      {
        "payload":    <bytes>  — the original .ctx binary
        "signature":  <bytes>  — 64-byte Ed25519 signature
        "verify_key": <bytes>  — 32-byte public verify key
      }
    Recipients can verify without the signing key.
    """
    return {
        "payload":    payload,
        "signature":  signer.sign(payload),
        "verify_key": signer.verify_key_bytes,
    }


def verify_signed_envelope(envelope: dict[str, bytes]) -> bytes:
    """
    Verify a signed envelope from sign_ctx_bytes().
    Returns the original payload on success.
    Raises CTXSignatureError on failure.
    """
    CTXSigner.verify_with_key(
        envelope["payload"],
        envelope["signature"],
        envelope["verify_key"],
    )
    return envelope["payload"]
