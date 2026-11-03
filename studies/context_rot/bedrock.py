"""A small Amazon Bedrock Converse client for the judges of the context-rot reproductions.

Judges ask one question and read one word, so this client does only that: one user
message, an optional system message, a reasoning effort, a cap on output tokens. It
reads only the reply's text block; a reasoning block, when Bedrock returns one, is
counted in the output tokens and never read as the verdict.

The token comes from ``AWS_BEARER_TOKEN_BEDROCK`` (load it from the Taste ``.env``
with ``set -a; source <(grep '^AWS_BEARER_TOKEN_BEDROCK=' .env); set +a``) and is
never printed, logged or kept on an error.

A call is sent again only when it provably cost nothing: a 429 (the service declined
it) or a connection that was never made. Any other failure is raised.

Temperature is never sent. The OpenAI reasoning models on Bedrock refuse it beside a
reasoning effort (taste/providers/_bedrock.py), so a judge that the paper ran at
temperature 0 runs here at the model's default, with the effort set instead. That
difference is recorded beside every verdict.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from urllib.parse import quote

import httpx

BASE_URL = "https://bedrock-runtime.us-east-1.amazonaws.com/"
RESENDS = 5

# Prices in USD per million tokens, for estimates only (input, output, where read).
# GPT-5.4: OpenAI's list price, assumed equal on Bedrock (Bedrock's page did not list it,
# Nov 3 2026). gpt-oss-120b: Bedrock's page lists only Asia Pacific (Sydney).
PRICES = {
    "global.openai.gpt-5.4": (2.50, 15.00, "OpenAI list price, assumed for Bedrock"),
    "openai.gpt-oss-120b-1:0": (0.1545, 0.618, "Bedrock, Sydney on-demand"),
}


class BedrockError(RuntimeError):
    """A refusal or fault from Bedrock: status and message only, never the request."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Bedrock {status}: {message[:300]}")
        self.status = status


@dataclass(frozen=True)
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    request_id: str
    seconds: float

    def cost(self, model: str) -> float | None:
        price = PRICES.get(model)
        if price is None:
            return None
        return (self.input_tokens * price[0] + self.output_tokens * price[1]) / 1_000_000


class Converse:
    """One Bedrock model, asked one question at a time. Safe to share across threads."""

    def __init__(self, model: str, *, effort: str | None = "low", max_tokens: int = 2048,
                 token: str | None = None, base_url: str = BASE_URL, wait_seconds: float = 2.0,
                 timeout_seconds: float = 120.0, transport: httpx.BaseTransport | None = None) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.wait_seconds = wait_seconds
        self._token = token if token is not None else os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
        if not self._token:
            raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is not set")
        self._url = f"{base_url}model/{quote(model, safe='')}/converse"
        self._client = httpx.Client(transport=transport, timeout=httpx.Timeout(timeout_seconds, connect=30.0))

    def __repr__(self) -> str:
        return f"Converse(model={self.model!r}, effort={self.effort!r}, max_tokens={self.max_tokens})"

    def settings(self) -> dict:
        return {"model": self.model, "effort": self.effort, "max_tokens": self.max_tokens,
                "temperature": "model default (not sent)"}

    def ask(self, user: str, system: str | None = None) -> Reply:
        body: dict = {"messages": [{"role": "user", "content": [{"text": user}]}],
                      "inferenceConfig": {"maxTokens": self.max_tokens}}
        if system:
            body["system"] = [{"text": system}]
        if self.effort:
            body["additionalModelRequestFields"] = {"reasoning": {"effort": self.effort}}
        headers = {"authorization": f"Bearer {self._token}", "content-type": "application/json",
                   "accept": "application/json"}
        content = json.dumps(body, ensure_ascii=False).encode()
        for resend in range(RESENDS + 1):
            started = time.monotonic()
            try:
                response = self._client.post(self._url, content=content, headers=headers)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                if resend == RESENDS:
                    raise
            else:
                if response.status_code == 200:
                    return _reply(response, time.monotonic() - started)
                if response.status_code != 429 or resend == RESENDS:
                    raise BedrockError(response.status_code, _message(response))
            time.sleep(self.wait_seconds * (2 ** resend))
        raise AssertionError("unreachable")  # pragma: no cover


def _message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("message", ""))
    except ValueError:
        return response.text


def _reply(response: httpx.Response, seconds: float) -> Reply:
    payload = response.json()
    blocks = payload.get("output", {}).get("message", {}).get("content", [])
    text = "".join(block["text"] for block in blocks if isinstance(block, dict) and "text" in block)
    usage = payload.get("usage", {})
    return Reply(text=text, input_tokens=int(usage.get("inputTokens", 0)),
                 output_tokens=int(usage.get("outputTokens", 0)), stop_reason=str(payload.get("stopReason", "")),
                 request_id=response.headers.get("x-amzn-requestid", ""), seconds=round(seconds, 3))
