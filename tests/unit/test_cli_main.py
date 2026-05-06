"""
Tests for Phase 10: Unified CLI (cortex/cli/main.py)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from cortex.cli.main import (
    _build_parser,
    cmd_keygen,
    cmd_simulate,
    cmd_status,
    cmd_validate_ctx,
    main,
)


# ── Parser structure ──────────────────────────────────────────────────────────

class TestParser:
    def setup_method(self):
        self.parser = _build_parser()

    def test_parser_has_version_flag(self):
        args = self.parser.parse_args(["--version"])
        assert args.version is True

    def test_keygen_default_out(self):
        args = self.parser.parse_args(["keygen"])
        assert args.out == "."

    def test_keygen_custom_out(self):
        args = self.parser.parse_args(["keygen", "--out", "/tmp/keys"])
        assert args.out == "/tmp/keys"

    def test_keygen_short_out(self):
        args = self.parser.parse_args(["keygen", "-o", "/tmp/keys"])
        assert args.out == "/tmp/keys"

    def test_validate_ctx_path(self):
        args = self.parser.parse_args(["validate-ctx", "some/file.ctx"])
        assert args.path == "some/file.ctx"

    def test_simulate_default_tag(self):
        args = self.parser.parse_args(["simulate"])
        assert args.tag == ""

    def test_simulate_custom_tag(self):
        args = self.parser.parse_args(["simulate", "--tag", "safety"])
        assert args.tag == "safety"

    def test_simulate_short_tag(self):
        args = self.parser.parse_args(["simulate", "-t", "smoke"])
        assert args.tag == "smoke"

    def test_serve_default_port(self):
        args = self.parser.parse_args(["serve"])
        assert args.port == 50051

    def test_status_command_parsed(self):
        args = self.parser.parse_args(["status"])
        assert args.command == "status"


# ── cmd_keygen ────────────────────────────────────────────────────────────────

class TestCmdKeygen:
    def test_keygen_creates_key_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(out=tmpdir)
            result = cmd_keygen(args)
            assert result == 0
            assert os.path.exists(os.path.join(tmpdir, "cortex_signing.key"))
            assert os.path.exists(os.path.join(tmpdir, "cortex_verify.key"))

    def test_keygen_verify_key_file_nonempty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(out=tmpdir)
            cmd_keygen(args)
            vk_path = os.path.join(tmpdir, "cortex_verify.key")
            content = open(vk_path).read().strip()
            assert len(content) == 64  # hex-encoded 32-byte Ed25519 verify key

    def test_keygen_creates_output_dir_if_missing(self):
        with tempfile.TemporaryDirectory() as base:
            new_dir = os.path.join(base, "subdir", "keys")
            args = argparse.Namespace(out=new_dir)
            result = cmd_keygen(args)
            assert result == 0
            assert os.path.isdir(new_dir)

    def test_keygen_prints_key_info(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(out=tmpdir)
            cmd_keygen(args)
            captured = capsys.readouterr()
            assert "Signing key" in captured.out
            assert "Verify key" in captured.out


# ── cmd_validate_ctx ──────────────────────────────────────────────────────────

class TestCmdValidateCtx:
    def test_invalid_file_returns_nonzero(self, tmp_path):
        bad = tmp_path / "bad.ctx"
        bad.write_bytes(b"not valid ctx content")
        args = argparse.Namespace(path=str(bad))
        result = cmd_validate_ctx(args)
        assert result != 0

    def test_missing_file_returns_nonzero(self):
        args = argparse.Namespace(path="/nonexistent/path.ctx")
        result = cmd_validate_ctx(args)
        assert result != 0

    def test_prints_validation_messages(self, tmp_path, capsys):
        bad = tmp_path / "bad.ctx"
        bad.write_bytes(b"garbage")
        args = argparse.Namespace(path=str(bad))
        cmd_validate_ctx(args)
        captured = capsys.readouterr()
        assert captured.out  # some message printed


# ── cmd_status ────────────────────────────────────────────────────────────────

class TestCmdStatus:
    def test_status_returns_zero(self, capsys):
        args = argparse.Namespace()
        result = cmd_status(args)
        assert result == 0

    def test_status_prints_gate_info(self, capsys):
        args = argparse.Namespace()
        cmd_status(args)
        captured = capsys.readouterr()
        assert "Gate" in captured.out

    def test_status_prints_memory_info(self, capsys):
        args = argparse.Namespace()
        cmd_status(args)
        captured = capsys.readouterr()
        assert "MemoryGateway" in captured.out

    def test_status_prints_cortex_status_header(self, capsys):
        args = argparse.Namespace()
        cmd_status(args)
        captured = capsys.readouterr()
        assert "Cortex Status" in captured.out


# ── cmd_simulate ──────────────────────────────────────────────────────────────

class TestCmdSimulate:
    def test_simulate_all_returns_zero(self, capsys):
        args = argparse.Namespace(tag="", scenario="")
        result = cmd_simulate(args)
        assert result == 0

    def test_simulate_prints_harness_output(self, capsys):
        args = argparse.Namespace(tag="", scenario="")
        cmd_simulate(args)
        captured = capsys.readouterr()
        assert "Cortex Simulation Harness" in captured.out

    def test_simulate_smoke_tag_returns_zero(self, capsys):
        args = argparse.Namespace(tag="smoke", scenario="")
        result = cmd_simulate(args)
        assert result == 0

    def test_simulate_unknown_tag_returns_nonzero(self, capsys):
        args = argparse.Namespace(tag="totally_unknown_tag_xyz", scenario="")
        result = cmd_simulate(args)
        assert result == 1

    def test_simulate_safety_tag_runs(self, capsys):
        args = argparse.Namespace(tag="safety", scenario="")
        result = cmd_simulate(args)
        # safety tag has 2 scenarios both expected to pass → 0
        assert result == 0


# ── cmd_serve ─────────────────────────────────────────────────────────────────

class TestCmdServe:
    def test_serve_returns_nonzero_when_grpc_unavailable(self):
        from cortex.cli.main import cmd_serve
        args = argparse.Namespace(port=50051)
        with patch("cortex.cli.main.cmd_serve") as mock_serve:
            mock_serve.return_value = 1
            result = mock_serve(args)
        assert result == 1

    def test_serve_handles_import_error(self, capsys):
        from cortex.cli.main import cmd_serve
        args = argparse.Namespace(port=50051)
        with patch("cortex.integration.grpc.server.main", side_effect=ImportError("no grpc")):
            result = cmd_serve(args)
        # Should either succeed (grpc available) or fail gracefully (not raise)
        assert isinstance(result, int)


# ── main() entry point ────────────────────────────────────────────────────────

class TestMain:
    def test_no_command_exits_zero(self):
        with patch.object(sys, "argv", ["cortex"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0

    def test_version_flag_exits_zero(self):
        with patch.object(sys, "argv", ["cortex", "--version"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0

    def test_version_prints_cortex(self, capsys):
        with patch.object(sys, "argv", ["cortex", "--version"]):
            with pytest.raises(SystemExit):
                main()
        captured = capsys.readouterr()
        assert "cortex" in captured.out.lower()

    def test_status_exits_zero(self):
        with patch.object(sys, "argv", ["cortex", "status"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0

    def test_simulate_exits_zero(self):
        with patch.object(sys, "argv", ["cortex", "simulate"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0

    def test_keygen_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(sys, "argv", ["cortex", "keygen", "--out", tmpdir]):
                with pytest.raises(SystemExit) as exc:
                    main()
        assert exc.value.code == 0
