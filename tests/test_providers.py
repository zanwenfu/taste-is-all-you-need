

def test_anthropic_client_carries_an_explicit_timeout() -> None:
    """Defect 36: the SDK's default-timeout heuristic refuses non-streaming
    requests whose max_tokens could exceed ten minutes; with the 32K ceiling
    that was every request. The client must be built with its own timeout."""
    import inspect

    from taste.providers import _anthropic

    assert _anthropic.ANTHROPIC_TIMEOUT_S >= 600
    assert "timeout=ANTHROPIC_TIMEOUT_S" in inspect.getsource(_anthropic)
