# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Active  |

Security fixes are backported to the current stable minor release only.

## Reporting a Vulnerability

**Do not file a public GitHub issue for security vulnerabilities.**

Cortex is infrastructure for physical AI systems. A vulnerability here can affect real robots in the real world. We take security reports seriously and will respond within 48 hours.

### How to report

Email: **security@cortex.ai**

Include:
- A description of the vulnerability
- Steps to reproduce
- The affected version(s)
- Your assessment of impact (CVSS score if possible)
- Whether you would like to be credited in the fix

### What to expect

1. **48 hours** — acknowledgement of your report
2. **7 days** — our initial assessment and severity determination
3. **30 days** — patch release for confirmed high/critical vulnerabilities
4. **90 days** — coordinated disclosure (you may publish after this window or after the fix ships, whichever comes first)

We will not pursue legal action against researchers who follow responsible disclosure.

## Security Design Notes

Cortex is designed as a **trust boundary**, not a trust consumer. Key design decisions:

- The Certification Gate is the only path to `robot.execute()`. There is no bypass.
- Sentinel has absolute veto power — it cannot be overridden by confidence scores.
- All certification decisions produce a `DecisionTrace` for audit.
- The gRPC and HTTP servers do not authenticate callers by default in v0.1.x — this is a known gap being addressed in Part 6 (Security hardening). **Do not expose Cortex servers to untrusted networks without adding authentication.**

## Known Security Limitations (v0.1.x)

- No authentication on gRPC or HTTP endpoints
- No mTLS between Cortex and memory adapters
- No rate limiting on certification requests
- API keys and connection strings are read from environment variables with no rotation mechanism

These are tracked and will be addressed in the security hardening milestone.
