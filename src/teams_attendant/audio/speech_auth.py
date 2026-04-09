"""Azure Speech authentication helpers.

Supports both API-key and identity-based (``DefaultAzureCredential``)
authentication.  When ``key`` is provided, a ``SpeechConfig`` is created
with a subscription key; otherwise a short-lived token is obtained via
the ``azure-identity`` library and used as an authorization token.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from teams_attendant.config import AzureSpeechConfig

log = structlog.get_logger()

_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


def build_speech_config(config: AzureSpeechConfig) -> Any:
    """Build an Azure ``SpeechConfig`` using API key or identity auth.

    When ``config.key`` is set it is used as a subscription key.
    Otherwise, a bearer token is obtained via ``DefaultAzureCredential``.

    Returns an ``azure.cognitiveservices.speech.SpeechConfig`` instance.
    """
    import azure.cognitiveservices.speech as speechsdk

    if config.key:
        log.debug("speech_auth.using_key")
        return speechsdk.SpeechConfig(
            subscription=config.key,
            region=config.region,
        )

    log.info("speech_auth.using_identity")
    token = _get_identity_token()
    return speechsdk.SpeechConfig(
        auth_token=token,
        region=config.region,
    )


def _get_identity_token() -> str:
    """Obtain a bearer token via ``DefaultAzureCredential``."""
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise ImportError(
            "azure-identity is required for identity-based authentication. "
            "Install it with: pip install azure-identity"
        ) from exc

    credential = DefaultAzureCredential()
    token = credential.get_token(_COGNITIVE_SERVICES_SCOPE)
    return token.token
