

def test_anthropic_client_carries_an_explicit_timeout() -> None:
    """Defect 36: the SDK's default-timeout heuristic refuses non-streaming
    requests whose max_tokens could exceed ten minutes; with the 32K ceiling
    that was every request. The client must be built with its own timeout."""
    import inspect

    from taste.providers import _anthropic

    assert _anthropic.ANTHROPIC_TIMEOUT_S >= 600
    assert "timeout=ANTHROPIC_TIMEOUT_S" in inspect.getsource(_anthropic)


def test_opus_requests_omit_temperature_and_say_so() -> None:
    """The central planner pins an opus model and asked for temperature=0.0.

    Measured against the live API: opus answers any request carrying
    ``temperature`` with HTTP 400, "`temperature` is deprecated for this
    model" -- non-retryable. So every planner call this system ever attempted
    failed at the provider, and no test caught it because every planner test
    injects a fake LLM.

    Dropping the parameter is the only way to call the model at all, but a
    silent drop would leave the manifest claiming a setting that never
    applied. See ``Completion.effective_sampling``.
    """
    from types import SimpleNamespace

    from taste.providers._anthropic import AnthropicProvider
    from taste.providers.base import CompletionRequest, SamplingConfig

    sent: dict[str, object] = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            model=kwargs["model"],
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cache_creation=None,
            ),
        )

    provider = AnthropicProvider(api_key="not-used")
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))

    def request(model: str) -> CompletionRequest:
        return CompletionRequest(
            model=model,
            system=[{"type": "text", "text": "s"}],
            messages=[{"role": "user", "content": "u"}],
            tools=None,
            max_tokens=16,
            sampling=SamplingConfig(temperature=0.0),
            role="planner",
        )

    opus = provider.complete(request("claude-opus-4-7"))
    assert "temperature" not in sent, "opus rejects the parameter outright"
    assert opus.effective_sampling == {"temperature": None, "dropped": ["temperature"]}

    sent.clear()
    sonnet = provider.complete(request("claude-sonnet-4-6"))
    assert sent["temperature"] == 0.0, "models that accept it must still get it"
    assert sonnet.effective_sampling == {"temperature": 0.0}
