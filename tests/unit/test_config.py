"""
tests/unit/test_config.py
==========================
Configuration system — loading, validation, env vars, file loading, secrets masking.
"""

from __future__ import annotations

import os
import tempfile
import pytest

from cortex.config.settings import (
    CortexSettings, GateSettings, RedisSettings, Mem0Settings,
)
from cortex.config import get_settings, reset_settings


class TestDefaults:

    def test_default_loads_without_error(self):
        cfg = CortexSettings()
        assert cfg is not None

    def test_gate_defaults(self):
        cfg = CortexSettings()
        assert cfg.gate.human_override_threshold == 0.35
        assert cfg.gate.replan_threshold == 0.50

    def test_redis_disabled_by_default(self):
        cfg = CortexSettings()
        assert not cfg.redis.enabled

    def test_metrics_disabled_by_default(self):
        cfg = CortexSettings()
        assert not cfg.metrics.enabled

    def test_tracing_disabled_by_default(self):
        cfg = CortexSettings()
        assert cfg.tracing.exporter == "none"

    def test_lifecycle_enabled_by_default(self):
        cfg = CortexSettings()
        assert cfg.lifecycle.enabled


class TestYamlLoading:

    def test_from_yaml_basic(self):
        cfg = CortexSettings.from_yaml("""
server:
  platform_id: arm_01
  deployment_id: prod
""")
        assert cfg.server.platform_id == "arm_01"
        assert cfg.server.deployment_id == "prod"

    def test_from_yaml_gate_thresholds(self):
        cfg = CortexSettings.from_yaml("""
gate:
  human_override_threshold: 0.45
  replan_threshold: 0.60
  sentinel_weight: 0.50
  physicore_weight: 0.35
  memory_weight: 0.15
""")
        assert cfg.gate.human_override_threshold == 0.45
        assert cfg.gate.replan_threshold == 0.60

    def test_from_yaml_redis(self):
        cfg = CortexSettings.from_yaml("""
redis:
  enabled: true
  url: redis://prod:6379
  ttl_seconds: 7200
""")
        assert cfg.redis.enabled
        assert cfg.redis.url == "redis://prod:6379"
        assert cfg.redis.ttl_seconds == 7200

    def test_from_yaml_observability(self):
        cfg = CortexSettings.from_yaml("""
metrics:
  enabled: true
  port: 9091
tracing:
  exporter: console
  sample_rate: 0.5
""")
        assert cfg.metrics.enabled
        assert cfg.metrics.port == 9091
        assert cfg.tracing.exporter == "console"
        assert cfg.tracing.sample_rate == 0.5

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("server:\n  platform_id: from_file\n")
            fname = f.name
        try:
            cfg = CortexSettings.from_file(fname)
            assert cfg.server.platform_id == "from_file"
        finally:
            os.unlink(fname)

    def test_from_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            CortexSettings.from_file("/nonexistent/path/cortex.yaml")

    def test_empty_yaml_uses_defaults(self):
        cfg = CortexSettings.from_yaml("")
        assert cfg.gate.human_override_threshold == 0.35


class TestValidation:

    def test_replan_below_human_override_raises(self):
        with pytest.raises(Exception):
            CortexSettings.from_yaml("""
gate:
  human_override_threshold: 0.60
  replan_threshold: 0.50
  sentinel_weight: 0.50
  physicore_weight: 0.35
  memory_weight: 0.15
""")

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(Exception):
            CortexSettings.from_yaml("""
gate:
  human_override_threshold: 0.35
  replan_threshold: 0.50
  sentinel_weight: 0.70
  physicore_weight: 0.35
  memory_weight: 0.15
""")

    def test_weights_summing_to_one_passes(self):
        cfg = CortexSettings.from_yaml("""
gate:
  human_override_threshold: 0.35
  replan_threshold: 0.50
  sentinel_weight: 0.50
  physicore_weight: 0.35
  memory_weight: 0.15
""")
        total = cfg.gate.sentinel_weight + cfg.gate.physicore_weight + cfg.gate.memory_weight
        assert abs(total - 1.0) < 0.001

    def test_redis_bad_url_raises(self):
        with pytest.raises(Exception):
            RedisSettings(url="not-a-redis-url")

    def test_redis_valid_url_passes(self):
        s = RedisSettings(url="redis://localhost:6379")
        assert s.url.startswith("redis://")

    def test_mem0_enabled_without_key_raises(self):
        with pytest.raises(Exception):
            Mem0Settings(enabled=True, api_key=None)

    def test_mem0_enabled_with_key_passes(self):
        s = Mem0Settings(enabled=True, api_key="test-key")
        assert s.api_key == "test-key"

    def test_grpc_port_out_of_range_raises(self):
        from cortex.config.settings import GrpcSettings
        with pytest.raises(Exception):
            GrpcSettings(port=99999)


class TestSecretsMasking:

    def test_password_masked_in_display(self):
        cfg = CortexSettings.from_yaml("""
redis:
  enabled: true
  url: redis://localhost:6379
  password: super_secret_password
""")
        display = cfg.display()
        assert "super_secret_password" not in display
        assert "***" in display

    def test_weaviate_key_masked(self):
        cfg = CortexSettings.from_yaml("""
weaviate:
  enabled: false
  api_key: wcs-abc123
""")
        display = cfg.display()
        assert "wcs-abc123" not in display
        assert "***" in display

    def test_mem0_key_masked(self):
        cfg = CortexSettings.from_yaml("""
mem0:
  enabled: true
  api_key: mem0-xyz-secret
""")
        display = cfg.display()
        assert "mem0-xyz-secret" not in display

    def test_url_not_masked(self):
        """URLs are not secret — they must appear in display()."""
        cfg = CortexSettings.from_yaml("""
redis:
  enabled: true
  url: redis://prod-host:6379
""")
        display = cfg.display()
        assert "redis://prod-host:6379" in display


class TestEnvVarLoading:

    def test_env_platform_id(self, monkeypatch):
        monkeypatch.setenv("CORTEX_SERVER_PLATFORM_ID", "env_arm_01")
        reset_settings()
        cfg = CortexSettings()
        assert cfg.server.platform_id == "env_arm_01"
        reset_settings()

    def test_env_redis_enabled(self, monkeypatch):
        monkeypatch.setenv("CORTEX_REDIS_ENABLED", "true")
        monkeypatch.setenv("CORTEX_REDIS_URL", "redis://env-host:6379")
        reset_settings()
        cfg = CortexSettings()
        assert cfg.redis.enabled
        assert cfg.redis.url == "redis://env-host:6379"
        reset_settings()

    def test_env_gate_threshold(self, monkeypatch):
        monkeypatch.setenv("CORTEX_GATE_HUMAN_OVERRIDE_THRESHOLD", "0.42")
        monkeypatch.setenv("CORTEX_GATE_REPLAN_THRESHOLD", "0.55")
        reset_settings()
        cfg = CortexSettings()
        assert abs(cfg.gate.human_override_threshold - 0.42) < 0.001
        reset_settings()


class TestSingleton:

    def test_get_settings_returns_same_instance(self):
        reset_settings()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_clears_singleton(self):
        reset_settings()
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        # After reset, a new instance is created
        assert s1 is not s2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
