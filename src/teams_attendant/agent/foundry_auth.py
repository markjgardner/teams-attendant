"""Azure Foundry authentication helpers.

Supports both API-key and identity-based (``DefaultAzureCredential``)
authentication.  When ``api_key`` is provided it is used directly as a
Bearer token; otherwise a short-lived token is obtained via the
``azure-identity`` library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from teams_attendant.config import AzureFoundryConfig

log = structlog.get_logger()

_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"

# Cached token provider (created once per process)
_token_provider = None


def _get_identity_token() -> str:
    """Obtain a bearer token via ``DefaultAzureCredential``."""
    global _token_provider  # noqa: PLW0603

    if _token_provider is None:
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as exc:
            raise ImportError(
                "azure-identity is required for identity-based authentication. "
                "Install it with: pip install azure-identity"
            ) from exc

        credential = DefaultAzureCredential()
        _token_provider = get_bearer_token_provider(
            credential, _COGNITIVE_SERVICES_SCOPE,
        )
        log.info("foundry_auth.identity_provider_created")

    return _token_provider()


def build_auth_headers(config: AzureFoundryConfig) -> dict[str, str]:
    """Build authentication headers for an Azure Foundry request.

    Uses ``api_key`` when available, otherwise falls back to
    ``DefaultAzureCredential`` identity-based auth.
    """
    if config.api_key:
        return {
            "api-key": config.api_key,
            "Authorization": f"Bearer {config.api_key}",
        }

    token = _get_identity_token()
    return {
        "Authorization": f"Bearer {token}",
    }
