"""
cortex.security
===============
Authentication and transport security for the Cortex stack.

    from cortex.security import require_permission, ApiKeyInterceptor
    from cortex.security import load_server_credentials, generate_api_key

API key auth (HTTP):
    @app.post("/certify")
    async def certify(..., _=Depends(require_permission("certify"))):
        ...

API key auth (gRPC):
    server = grpc.server(..., interceptors=[ApiKeyInterceptor()])

mTLS (gRPC):
    creds = load_server_credentials()
    server.add_secure_port("[::]:50051", creds)

Key generation:
    cortex-keygen
    cortex-gen-certs --robots arm_east_01,arm_west_02
"""

from cortex.security.api_keys import (
    ApiKeyRecord,
    ApiKeyStore,
    ApiKeyInterceptor,
    generate_api_key,
    hash_key,
    get_store,
    reset_store,
    require_permission,
    ALL_PERMISSIONS,
    PREFIX_LIVE,
    PREFIX_DEV,
)
from cortex.security.mtls import (
    load_server_credentials,
    load_client_credentials,
    generate_dev_certs,
)

__all__ = [
    # API keys
    "ApiKeyRecord",
    "ApiKeyStore",
    "ApiKeyInterceptor",
    "generate_api_key",
    "hash_key",
    "get_store",
    "reset_store",
    "require_permission",
    "ALL_PERMISSIONS",
    "PREFIX_LIVE",
    "PREFIX_DEV",
    # mTLS
    "load_server_credentials",
    "load_client_credentials",
    "generate_dev_certs",
]
