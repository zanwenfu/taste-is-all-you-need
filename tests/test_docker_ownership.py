"""A container name or reusable key cannot substitute for ownership/configuration."""

import pytest

from taste.execution import DockerProvider
from tests.test_execution import _respawning_client


def test_separate_providers_in_one_process_never_remove_each_others_containers():
    client = _respawning_client()
    one = DockerProvider(client=client)
    two = DockerProvider(client=client)
    first = one.open(key="same", image="synthetic")
    second = two.open(key="same", image="synthetic")
    assert first.container.name != second.container.name
    assert not first.container.removed and not second.container.removed
    one.close_all()
    assert first.container.removed and not second.container.removed


@pytest.mark.parametrize("keys", [("a/b", "a_b"), ("a:b", "a_b"), ("x" * 240 + "a", "x" * 240 + "b")])
def test_normalized_and_truncated_names_cannot_destroy_another_key(keys):
    client = _respawning_client()
    provider = DockerProvider(client=client)
    first = provider.open(key=keys[0], image="synthetic")
    second = provider.open(key=keys[1], image="synthetic")
    assert first.container.name != second.container.name
    assert not first.container.removed and not second.container.removed


@pytest.mark.parametrize("changed", ["image", "network_mode", "platform", "env_prefix"])
def test_cached_launch_configuration_must_match_exactly(changed):
    client = _respawning_client()
    provider = DockerProvider(client=client)
    first = provider.open(key="same", image="synthetic", network_mode="none")
    image, network = "synthetic", "none"
    if changed == "image":
        image = "different"
    elif changed == "network_mode":
        network = "bridge"
    elif changed == "platform":
        provider.platform = "linux/arm64"
    else:
        provider.env_prefix = "different environment setup"
    with pytest.raises(RuntimeError, match="configuration"):
        provider.open(key="same", image=image, network_mode=network)
    assert not first.container.removed and len(client.containers.spawned) == 1


def test_identical_request_reuses_its_owned_container_and_exposes_actual_network_mode():
    provider = DockerProvider(client=_respawning_client())
    first = provider.open(key="same", image="synthetic", network_mode="bridge")
    assert provider.open(key="same", image="synthetic", network_mode="bridge") is first
    assert first.network_mode == "bridge" and first.platform == "linux/amd64"


def test_close_releases_configuration_identity_before_reopening():
    provider = DockerProvider(client=_respawning_client())
    first = provider.open(key="same", image="synthetic", network_mode="none")
    first.close()
    second = provider.open(key="same", image="other", network_mode="bridge")
    assert second is not first and second.image == "other" and second.network_mode == "bridge"
