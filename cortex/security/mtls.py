"""
cortex.security.mtls
====================
Mutual TLS (mTLS) helpers for the Cortex gRPC server.

mTLS enforces that:
  - The server presents a certificate the client trusts (standard TLS).
  - The client presents a certificate the server trusts (mutual).
  - Only robots with a certificate signed by the Cortex CA can connect.

This provides a hardware-level trust boundary: a robot cannot call certify()
unless it holds a valid certificate issued by your CA.

Certificate layout
------------------
    certs/
        ca/
            ca.crt          — CA certificate (shared across fleet)
            ca.key          — CA private key  (keep offline / in HSM)
        server/
            server.crt      — Server certificate (signed by CA)
            server.key      — Server private key
        robots/
            arm_east_01.crt — Per-robot certificate (signed by CA)
            arm_east_01.key — Per-robot private key

Generating certificates (development)
--------------------------------------
    cortex-gen-certs --output ./certs --robots arm_east_01,arm_west_02
    # Generates CA, server cert, and one cert per robot.

Environment variables
---------------------
CORTEX_TLS_CERT_FILE    — path to server certificate (PEM)
CORTEX_TLS_KEY_FILE     — path to server private key  (PEM)
CORTEX_TLS_CA_FILE      — path to CA certificate for client verification (mTLS)
CORTEX_TLS_ENABLED      — set to "1" to enable TLS (default: "0")
CORTEX_TLS_MTLS         — set to "1" to require client certificates (default: "0")

Usage
-----
    from cortex.security.mtls import load_server_credentials, load_client_credentials

    # Server side (gRPC)
    creds = load_server_credentials()
    server.add_secure_port("[::]:50051", creds)

    # Client side (SDK)
    creds = load_client_credentials()
    channel = grpc.secure_channel("cortex:50051", creds)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


# ── Server credentials ────────────────────────────────────────────────────────

def load_server_credentials(
    cert_file: str | None = None,
    key_file:  str | None = None,
    ca_file:   str | None = None,
    require_client_cert: bool | None = None,
) -> "grpc.ServerCredentials":
    """
    Build gRPC ServerCredentials from certificate files.

    Parameters are read from environment variables if not provided:
        CORTEX_TLS_CERT_FILE, CORTEX_TLS_KEY_FILE, CORTEX_TLS_CA_FILE
        CORTEX_TLS_MTLS (require_client_cert)

    Returns grpc.ServerCredentials ready to pass to server.add_secure_port().
    Raises FileNotFoundError if any certificate file is missing.
    """
    try:
        import grpc
    except ImportError as e:
        raise ImportError("pip install 'cortex-gate[grpc]' to use mTLS") from e

    cert = _read_cert(
        cert_file or os.environ.get("CORTEX_TLS_CERT_FILE", ""),
        "server certificate",
    )
    key = _read_cert(
        key_file or os.environ.get("CORTEX_TLS_KEY_FILE", ""),
        "server private key",
    )

    ca_path = ca_file or os.environ.get("CORTEX_TLS_CA_FILE", "")
    ca = _read_cert(ca_path, "CA certificate") if ca_path else None

    mtls = require_client_cert
    if mtls is None:
        mtls = os.environ.get("CORTEX_TLS_MTLS", "0") == "1"

    if ca and mtls:
        log.info("mTLS enabled — client certificates required")
        return grpc.ssl_server_credentials(
            [(key, cert)],
            root_certificates=ca,
            require_client_auth=True,
        )
    elif ca:
        log.info("TLS enabled — client certificates optional")
        return grpc.ssl_server_credentials(
            [(key, cert)],
            root_certificates=ca,
            require_client_auth=False,
        )
    else:
        log.info("TLS enabled — server-only (no client certificate verification)")
        return grpc.ssl_server_credentials([(key, cert)])


def load_client_credentials(
    ca_file:   str | None = None,
    cert_file: str | None = None,
    key_file:  str | None = None,
) -> "grpc.ChannelCredentials":
    """
    Build gRPC ChannelCredentials for a client (SDK / robot).

    For mTLS: provide cert_file and key_file (robot-specific certificate).
    For server-only TLS: provide only ca_file.
    For public CA: omit all parameters (uses system root certificates).

    Returns grpc.ChannelCredentials ready to pass to grpc.secure_channel().
    """
    try:
        import grpc
    except ImportError as e:
        raise ImportError("pip install 'cortex-gate[grpc]' to use mTLS") from e

    ca_path   = ca_file   or os.environ.get("CORTEX_TLS_CA_FILE",   "")
    cert_path = cert_file or os.environ.get("CORTEX_TLS_CERT_FILE", "")
    key_path  = key_file  or os.environ.get("CORTEX_TLS_KEY_FILE",  "")

    ca   = _read_cert(ca_path,   "CA certificate")   if ca_path   else None
    cert = _read_cert(cert_path, "client certificate") if cert_path else None
    key  = _read_cert(key_path,  "client private key") if key_path  else None

    if cert and key:
        log.info("mTLS client credentials loaded")
        return grpc.ssl_channel_credentials(
            root_certificates=ca,
            private_key=key,
            certificate_chain=cert,
        )
    else:
        log.info("TLS client credentials (no client certificate)")
        return grpc.ssl_channel_credentials(root_certificates=ca)


# ── Certificate generation (development only) ─────────────────────────────────

def generate_dev_certs(
    output_dir: str = "./certs",
    robots: list[str] | None = None,
    days: int = 365,
) -> dict[str, str]:
    """
    Generate a self-signed CA, server certificate, and per-robot certificates.
    FOR DEVELOPMENT AND TESTING ONLY.

    Requires the `cryptography` package: pip install cryptography

    Returns a dict of {name: path} for all generated files.
    """
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
    except ImportError as e:
        raise ImportError(
            "pip install cryptography to generate development certificates"
        ) from e

    out = Path(output_dir)
    (out / "ca").mkdir(parents=True, exist_ok=True)
    (out / "server").mkdir(parents=True, exist_ok=True)
    (out / "robots").mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}

    # ── CA ────────────────────────────────────────────────────────────────────
    ca_key = _gen_rsa_key()
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Cortex Dev CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Cortex"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=days * 3))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    files["ca_key"]  = _write_key(ca_key,  out / "ca" / "ca.key")
    files["ca_cert"] = _write_cert(ca_cert, out / "ca" / "ca.crt")
    log.info("Generated CA certificate: %s", files["ca_cert"])

    # ── Server ────────────────────────────────────────────────────────────────
    srv_key  = _gen_rsa_key()
    srv_cert = _sign_cert(srv_key, ca_cert, ca_key, "cortex-server", days,
                          san_dns=["localhost", "cortex", "cortex-grpc"])
    files["server_key"]  = _write_key(srv_key,  out / "server" / "server.key")
    files["server_cert"] = _write_cert(srv_cert, out / "server" / "server.crt")
    log.info("Generated server certificate: %s", files["server_cert"])

    # ── Robots ────────────────────────────────────────────────────────────────
    for robot_id in (robots or []):
        rkey   = _gen_rsa_key()
        rcert  = _sign_cert(rkey, ca_cert, ca_key, robot_id, days)
        safe   = robot_id.replace("/", "_")
        files[f"robot_{safe}_key"]  = _write_key(rkey,  out / "robots" / f"{safe}.key")
        files[f"robot_{safe}_cert"] = _write_cert(rcert, out / "robots" / f"{safe}.crt")
        log.info("Generated robot certificate: %s (%s)", robot_id, files[f"robot_{safe}_cert"])

    print(f"\nDev certificates written to {out}/")
    print(f"  CA:     {out}/ca/ca.crt")
    print(f"  Server: {out}/server/server.crt + server.key")
    if robots:
        print(f"  Robots: {out}/robots/<robot_id>.crt + .key")
    print(f"\nTo use:")
    print(f"  export CORTEX_TLS_ENABLED=1")
    print(f"  export CORTEX_TLS_CERT_FILE={out}/server/server.crt")
    print(f"  export CORTEX_TLS_KEY_FILE={out}/server/server.key")
    print(f"  export CORTEX_TLS_CA_FILE={out}/ca/ca.crt")
    print(f"  export CORTEX_TLS_MTLS=1  # require robot client certs\n")
    return files


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_cert(path: str, label: str) -> bytes:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cortex {label} not found: {p}")
    return p.read_bytes()


def _gen_rsa_key():
    from cryptography.hazmat.primitives.asymmetric import rsa
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _sign_cert(subject_key, ca_cert, ca_key, cn: str, days: int,
               san_dns: list[str] | None = None):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    import datetime

    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(subject_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
    )
    if san_dns:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(d) for d in san_dns]),
            critical=False,
        )
    return builder.sign(ca_key, hashes.SHA256())


def _write_key(key, path: Path) -> str:
    from cryptography.hazmat.primitives import serialization
    path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    path.chmod(0o600)
    return str(path)


def _write_cert(cert, path: Path) -> str:
    from cryptography.hazmat.primitives import serialization
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main_gen_certs() -> None:
    """cortex-gen-certs — generate development TLS certificates."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="cortex-gen-certs",
        description="Generate development TLS/mTLS certificates for Cortex.",
    )
    parser.add_argument("--output", default="./certs",     help="Output directory (default: ./certs)")
    parser.add_argument("--robots", default="",            help="Comma-separated robot IDs")
    parser.add_argument("--days",   default=365, type=int, help="Certificate validity in days")
    args = parser.parse_args()

    robots = [r.strip() for r in args.robots.split(",") if r.strip()] if args.robots else []
    generate_dev_certs(output_dir=args.output, robots=robots, days=args.days)
