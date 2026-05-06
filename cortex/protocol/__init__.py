"""
cortex.protocol
===============
The .ctx wire format — the universal interface between AI intention and
physical execution.

A .ctx packet is:
  - Versioned (forward/backward compatible)
  - Binary (MessagePack for compactness and speed)
  - Cryptographically signed (Ed25519 — 0.3ms verify)
  - Language-agnostic (any runtime that speaks MessagePack can parse it)

Public API
----------
    from cortex.protocol import CTXSerializer, CTXSigner, CTXSignatureError
    from cortex.protocol.validator import validate_ctx_file

    serializer = CTXSerializer()
    raw   = serializer.to_bytes(ctx)
    ctx2  = serializer.from_bytes(raw)

    signer    = CTXSigner()
    sig       = signer.sign(raw)
    verified  = signer.verify(raw, sig, signer.verify_key_bytes)
"""

from cortex.protocol.ctx_format import CTXSerializer, CTX_VERSION
from cortex.protocol.signing    import CTXSigner, CTXSignatureError
from cortex.protocol.schema     import generate_json_schema

__all__ = [
    "CTXSerializer",
    "CTX_VERSION",
    "CTXSigner",
    "CTXSignatureError",
    "generate_json_schema",
]
