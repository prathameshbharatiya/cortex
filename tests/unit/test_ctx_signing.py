"""
tests/unit/test_ctx_signing.py
================================
Tests for CTX Ed25519 signing: key generation, sign, verify, tamper detection.
"""

from __future__ import annotations

import time

import pytest

from cortex.protocol.signing import (
    CTXSigner,
    CTXSignatureError,
    sign_ctx_bytes,
    verify_signed_envelope,
)


class TestCTXSigner:

    def test_generates_key_pair(self) -> None:
        s = CTXSigner()
        assert len(s.signing_key_bytes) == 32
        assert len(s.verify_key_bytes)  == 32

    def test_hex_properties(self) -> None:
        s = CTXSigner()
        assert len(s.signing_key_hex) == 64
        assert len(s.verify_key_hex)  == 64

    def test_sign_returns_64_bytes(self) -> None:
        s   = CTXSigner()
        sig = s.sign(b"hello cortex")
        assert isinstance(sig, bytes)
        assert len(sig) == 64

    def test_verify_self(self) -> None:
        s       = CTXSigner()
        payload = b"certify this action"
        sig     = s.sign(payload)
        assert s.verify(payload, sig) is True

    def test_verify_with_key(self) -> None:
        s       = CTXSigner()
        payload = b"pick red block at 0.3 0.0 0.5"
        sig     = s.sign(payload)
        result  = CTXSigner.verify_with_key(payload, sig, s.verify_key_bytes)
        assert result is True

    def test_tamper_detection(self) -> None:
        """Modifying payload after signing must be detected."""
        s         = CTXSigner()
        payload   = b"original payload"
        sig       = s.sign(payload)
        tampered  = b"tampered payload"
        with pytest.raises(CTXSignatureError):
            CTXSigner.verify_with_key(tampered, sig, s.verify_key_bytes)

    def test_wrong_key_detection(self) -> None:
        """Verifying with a different key must fail."""
        signer1 = CTXSigner()
        signer2 = CTXSigner()
        payload = b"some ctx data"
        sig     = signer1.sign(payload)
        with pytest.raises(CTXSignatureError):
            CTXSigner.verify_with_key(payload, sig, signer2.verify_key_bytes)

    def test_from_hex_reconstruction(self) -> None:
        """Reconstruct signer from hex key and verify it produces the same key."""
        s1  = CTXSigner()
        s2  = CTXSigner.from_hex(s1.signing_key_hex)
        assert s2.signing_key_hex  == s1.signing_key_hex
        assert s2.verify_key_hex   == s1.verify_key_hex

    def test_from_hex_signs_same(self) -> None:
        s1      = CTXSigner()
        s2      = CTXSigner.from_hex(s1.signing_key_hex)
        payload = b"deterministic test"
        sig1    = s1.sign(payload)
        sig2    = s2.sign(payload)
        assert sig1 == sig2

    def test_save_load_key_file(self, tmp_path) -> None:
        s1   = CTXSigner()
        path = str(tmp_path / "test.key")
        s1.save_to_file(path)
        s2 = CTXSigner.from_file(path)
        assert s2.signing_key_hex == s1.signing_key_hex

    def test_different_signers_produce_different_keys(self) -> None:
        s1 = CTXSigner()
        s2 = CTXSigner()
        assert s1.signing_key_bytes != s2.signing_key_bytes


class TestSignedEnvelope:

    def test_sign_and_verify_envelope(self) -> None:
        s       = CTXSigner()
        payload = b"full ctx binary"
        env     = sign_ctx_bytes(payload, s)
        result  = verify_signed_envelope(env)
        assert result == payload

    def test_tampered_envelope_raises(self) -> None:
        s       = CTXSigner()
        payload = b"untouched payload"
        env     = sign_ctx_bytes(payload, s)
        bad_env = {**env, "payload": b"tampered!"}
        with pytest.raises(CTXSignatureError):
            verify_signed_envelope(bad_env)
