# This file is the single source of truth for Cortex versioning.
# It is read by pyproject.toml (via hatch-vcs or statically), setup.py,
# and cortex/__init__.py. Never hardcode the version anywhere else.
#
# Version follows Semantic Versioning 2.0.0 (semver.org):
#   MAJOR — breaking API change
#   MINOR — backwards-compatible new capability
#   PATCH — backwards-compatible bug fix
#
# Pre-release suffixes:
#   0.1.0a1  — alpha (internal, incomplete)
#   0.1.0b1  — beta (feature-complete, may have bugs)
#   0.1.0rc1 — release candidate (production-ready candidate)
#   0.1.0    — stable release

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

# Build metadata (set by CI; never commit non-empty values)
__build__ = ""
__commit__ = ""
