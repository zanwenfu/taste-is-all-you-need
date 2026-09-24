"""Explicit Azure routing supplied by the trusted controller.

No dotenv discovery, personal-key fallback, or mutable process-global client.
Deployment names are routing labels; dated served-model IDs identify prices
and experimental provenance. Only verified Global Standard rates are admitted.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from taste.providers.base import ProtocolFailure

AZURE_PLANNER_MODEL = "gpt-6-astra-2026-09-03"
AZURE_WORKER_MODEL = "gpt-6-sol-2026-09-22"
AZURE_MONITOR_MODEL = AZURE_WORKER_MODEL
_MODELS = frozenset({AZURE_PLANNER_MODEL, AZURE_WORKER_MODEL})


@dataclass(frozen=True)
class AzureDeployment:
    model: str
    deployment: str
    deployment_type: str = "GlobalStandard"

    def __post_init__(self) -> None:
        if self.model not in _MODELS:
            raise ProtocolFailure("Azure model has no verified deployment/pricing profile")
        if not isinstance(self.deployment, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.deployment):
            raise ProtocolFailure("invalid Azure deployment name")
        if self.deployment_type != "GlobalStandard":
            raise ProtocolFailure("only Azure GlobalStandard pricing is verified")


@dataclass(frozen=True)
class AzureOpenAIConfig:
    base_url: str
    api_key: str = field(repr=False)
    deployments: tuple[AzureDeployment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key.strip() or self.api_key != self.api_key.strip():
            raise ProtocolFailure("AZURE_OPENAI_API_KEY must be explicitly supplied")
        if not isinstance(self.base_url, str):
            raise ProtocolFailure("invalid Azure OpenAI v1 endpoint")
        parts = urlsplit(self.base_url)
        # Reject credentials, ports, escapes, redirects/proxies and arbitrary
        # compatible hosts. This path is specifically an Azure credit route.
        if (
            parts.scheme != "https" or parts.query or parts.fragment
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]*\.(?:openai\.azure\.com|services\.ai\.azure\.com)", parts.netloc)
            or parts.path not in {"/openai/v1", "/openai/v1/"}
            or self.base_url != self.base_url.strip()
        ):
            raise ProtocolFailure("Azure requires a resource HTTPS /openai/v1 endpoint")
        if type(self.deployments) is not tuple or not self.deployments or any(type(d) is not AzureDeployment for d in self.deployments):
            raise ProtocolFailure("Azure deployments must be a nonempty immutable tuple")
        if len({d.model for d in self.deployments}) != len(self.deployments) or len({d.deployment for d in self.deployments}) != len(self.deployments):
            raise ProtocolFailure("Azure deployment/model bindings must be unique")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/") + "/")

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str], *, deployments: tuple[AzureDeployment, ...],
    ) -> AzureOpenAIConfig:
        """Read only the two named Azure values from a trusted host mapping."""
        return cls(
            base_url=environment.get("AZURE_OPENAI_BASE_URL", ""),
            api_key=environment.get("AZURE_OPENAI_API_KEY", ""),
            deployments=deployments,
        )

    def deployment_for(self, model: str) -> AzureDeployment:
        for binding in self.deployments:
            if binding.model == model:
                return binding
        raise ProtocolFailure("model is not admitted by this Azure-only configuration")
