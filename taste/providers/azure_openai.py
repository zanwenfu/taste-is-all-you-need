"""Explicit routing supplied by the trusted controller: Azure OpenAI, or Amazon Bedrock.

No dotenv discovery, personal-key fallback, or mutable process-global client.
Deployment names are routing labels; dated served-model IDs identify prices
and experimental provenance. Only verified Global Standard rates are admitted.

**The Bedrock route.** The same configuration names an Amazon Bedrock runtime
endpoint (``https://bedrock-runtime.<region>.amazonaws.com/``) instead of an
Azure resource, with each model's global cross-region inference profile as its
deployment and a Bedrock API key as the key. It goes through the same
admission, journal, budget fence and settlement as the Azure route; only the
adapter differs (``taste.providers._bedrock``). The names say Azure because
that route came first: the endpoint and key still cross process boundaries as
AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_KEY, whichever route they are.
Only global profiles are admitted: their list prices are the table's (checked
2026-10-16 on Bedrock's own offers), while a geographic profile (``us.``) is
billed 10% more and is not priced here.
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
AZURE_LUNA_MODEL = "gpt-5.6-luna-2026-07-09"
AZURE_LUNA6_MODEL = "gpt-6-luna-2026-09-22"
# Dated served models with a verified Azure Global Standard price. A role may
# be assigned any of them, or of Bedrock's below; nothing else can be routed.
AZURE_MODELS = frozenset({AZURE_PLANNER_MODEL, AZURE_WORKER_MODEL, AZURE_LUNA_MODEL, AZURE_LUNA6_MODEL})
# The served models Bedrock offers with a verified global price, and the one
# inference profile each is reached by. Bedrock names no dated version for the
# OpenAI models; its model card for GPT-6 Luna starts on 2026-09-22, the
# Azure deployment's version.
BEDROCK_PROFILES = {
    AZURE_LUNA6_MODEL: "global.openai.gpt-6-luna",
    AZURE_WORKER_MODEL: "global.openai.gpt-6-sol",
    "claude-sonnet-5-5": "global.anthropic.claude-sonnet-5-5",
    "claude-opus-5-5": "global.anthropic.claude-opus-5-5",
    "claude-haiku-5-5": "global.anthropic.claude-haiku-5-5",
}
BEDROCK_MODELS = frozenset(BEDROCK_PROFILES)
# Every model some explicit route can serve. Which route serves it is checked
# where the route is configured.
ROUTED_MODELS = AZURE_MODELS | BEDROCK_MODELS
_MODELS = ROUTED_MODELS
_BEDROCK_HOST = re.compile(r"bedrock-runtime\.[a-z]{2}(?:-[a-z]+)+-[0-9]\.amazonaws\.com")


def route_of(endpoint: str) -> str:
    """Which route an endpoint is: ``bedrock`` for a Bedrock runtime, else ``azure_openai``."""
    return "bedrock" if _BEDROCK_HOST.fullmatch(urlsplit(str(endpoint)).netloc or "") else "azure_openai"


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
        # compatible hosts. This path is specifically a credit route: an Azure
        # resource's v1 API, or Bedrock's runtime in one region.
        bedrock = route_of(self.base_url) == "bedrock"
        if (
            parts.scheme != "https" or parts.query or parts.fragment
            or not (bedrock or re.fullmatch(
                r"[a-z0-9][a-z0-9-]*\.(?:openai\.azure\.com|services\.ai\.azure\.com)", parts.netloc))
            or parts.path not in ({"", "/"} if bedrock else {"/openai/v1", "/openai/v1/"})
            or self.base_url != self.base_url.strip()
        ):
            raise ProtocolFailure("Azure requires a resource HTTPS /openai/v1 endpoint, "
                                  "Bedrock its regional runtime endpoint")
        if type(self.deployments) is not tuple or not self.deployments or any(type(d) is not AzureDeployment for d in self.deployments):
            raise ProtocolFailure("Azure deployments must be a nonempty immutable tuple")
        if len({d.model for d in self.deployments}) != len(self.deployments) or len({d.deployment for d in self.deployments}) != len(self.deployments):
            raise ProtocolFailure("Azure deployment/model bindings must be unique")
        for binding in self.deployments:
            # One route, one way to each model: Azure serves only its own
            # models, and Bedrock each model through its global profile.
            if bedrock and BEDROCK_PROFILES.get(binding.model) != binding.deployment:
                raise ProtocolFailure("Bedrock serves a model only through its verified global profile")
            if not bedrock and binding.model not in AZURE_MODELS:
                raise ProtocolFailure("model is not served by an Azure deployment")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/") + "/")

    @property
    def route(self) -> str:
        """``azure_openai`` or ``bedrock``: which adapter this route is called through."""
        return route_of(self.base_url)

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
