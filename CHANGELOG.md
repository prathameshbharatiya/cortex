# Changelog

All notable changes to **cortex-gate** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Prometheus metrics endpoint (`/metrics`)
- OpenTelemetry trace export
- `cortex.yaml` configuration file support
- Kubernetes Helm chart
- Full test suite (unit + integration)

---

## [0.1.0] — 2025-04-28

### Added

**Core Certification Engine**
- `CertificationGate` — the mandatory boundary between AI intent and physical action
- Five execution states: `EXECUTE`, `EXECUTE_WITH_CONSTRAINTS`, `REPLAN_REQUIRED`, `SAFE_HALT`, `HUMAN_OVERRIDE_REQUIRED`
- Three-validator hierarchy: Sentinel (safety veto) → PhysiCore (physics) → Memory (consistency)
- `ConfidenceContract` — bounded, structured confidence with uncertainty sources
- `DecisionTrace` — full auditability of every certification decision

**Memory Layer**
- `MemoryGateway` — pluggable, multi-adapter memory routing with automatic failover
- Adapters: `InMemoryAdapter`, `JSONFileAdapter`, `RedisAdapter`, `WeaviateAdapter`, `Mem0Adapter`
- Temporal validity enforcement — expired records never reach the certification gate

**Context Engine**
- `ContextEngine` — assembles validated CTX packets from task, robot state, and memory
- `SensorCrossChecker` — validates sensor consistency before context assembly
- `ConstraintLoader` — loads and applies safety constraints per deployment
- `PhysicsHorizonBuilder` — integrates PhysiCore 12-step predictive lookahead

**Relevance Engine**
- `RelevanceEngine` — multi-factor memory ranking pipeline
- Four scorers: `TaskRelevanceScorer`, `StateRelevanceScorer`, `TemporalValidityScorer`, `OutcomeRelevanceScorer`

**Experience & Feedback Loop**
- `ExperienceTracker` — records execution outcomes, computes surprise scores
- `LifecycleManager` — manages memory record aging, pruning, and maintenance

**Integration Layer**
- gRPC server — low-latency robot fleet transport
- HTTP REST server — language-agnostic transport (FastAPI)
- Python SDK (`CortexSDK`) — four-line integration API
- ROS 2 node (`CortexNode`) — native ROS 2 integration

**PhysiCore Engine**
- Hardware bridge for physical robot platforms
- Sentinel fault observer, Lyapunov stability checker
- MAVLink bridge for firmware integration

**Packaging (this release)**
- `pyproject.toml` — PEP 621 compliant packaging
- Optional dependency groups: `grpc`, `redis`, `weaviate`, `mem0`, `http`, `physicore`, `bridge`, `dev`
- `py.typed` marker — PEP 561 typed package
- `MANIFEST.in` — source distribution manifest
- `CHANGELOG.md`, `SECURITY.md`
- GitHub Actions workflows: CI (`test.yml`), release (`release.yml`)

---

## Format

### Entry categories
- **Added** — new features
- **Changed** — changes to existing behaviour
- **Deprecated** — features that will be removed in a future release
- **Removed** — features removed in this release
- **Fixed** — bug fixes
- **Security** — security fixes (also see `SECURITY.md`)

[Unreleased]: https://github.com/cortex-ai/cortex/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cortex-ai/cortex/releases/tag/v0.1.0
