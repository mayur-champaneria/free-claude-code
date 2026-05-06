"""Tests for the LQQQ provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from providers.base import ProviderConfig
from providers.lqqq import LQQQ_DEFAULT_BASE, LqqqProvider
from providers.lqqq.request import build_request_body

# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def test_lqqq_default_base_url() -> None:
    assert LQQQ_DEFAULT_BASE == "https://api.lqqq.cc/v1"


def test_lqqq_provider_uses_default_base_when_none() -> None:
    config = ProviderConfig(api_key="test-key")
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = LqqqProvider(config)
    assert provider._base_url == LQQQ_DEFAULT_BASE


def test_lqqq_provider_uses_custom_base_url() -> None:
    config = ProviderConfig(api_key="test-key", base_url="https://custom.lqqq.cc/v1")
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = LqqqProvider(config)
    assert provider._base_url == "https://custom.lqqq.cc/v1"


def test_lqqq_provider_name_is_lqqq() -> None:
    config = ProviderConfig(api_key="test-key")
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = LqqqProvider(config)
    assert provider._provider_name == "LQQQ"


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lqqq_lists_openai_compatible_model_ids() -> None:
    config = ProviderConfig(api_key="test-key")
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = LqqqProvider(config)

    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(id="claude-sonnet-4-20250514"),
                SimpleNamespace(id="gpt-4o"),
            ]
        ),
    ):
        ids = await provider.list_model_ids()
        assert ids == frozenset({"claude-sonnet-4-20250514", "gpt-4o"})


# ---------------------------------------------------------------------------
# Request body building
# ---------------------------------------------------------------------------


def test_build_request_body_basic() -> None:
    from tests.provider_request_mocks import make_openai_compat_stream_request

    request = make_openai_compat_stream_request(model="claude-sonnet-4-20250514")
    body = build_request_body(request, thinking_enabled=False)
    assert body["model"] == "claude-sonnet-4-20250514"


def test_build_request_body_preserves_model_name() -> None:
    from tests.provider_request_mocks import make_openai_compat_stream_request

    request = make_openai_compat_stream_request(model="gpt-4o")
    body = build_request_body(request, thinking_enabled=False)
    assert body["model"] == "gpt-4o"


def test_build_request_body_with_thinking_enabled() -> None:
    from tests.provider_request_mocks import make_openai_compat_stream_request

    request = make_openai_compat_stream_request(model="claude-sonnet-4-20250514")
    body = build_request_body(request, thinking_enabled=True)
    assert body["model"] == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_lqqq_in_provider_catalog() -> None:
    from config.provider_catalog import PROVIDER_CATALOG, SUPPORTED_PROVIDER_IDS

    assert "lqqq" in PROVIDER_CATALOG
    assert "lqqq" in SUPPORTED_PROVIDER_IDS


def test_lqqq_in_provider_factories() -> None:
    from providers.registry import PROVIDER_FACTORIES

    assert "lqqq" in PROVIDER_FACTORIES


def test_lqqq_catalog_descriptor_fields() -> None:
    from config.provider_catalog import PROVIDER_CATALOG

    desc = PROVIDER_CATALOG["lqqq"]
    assert desc.provider_id == "lqqq"
    assert desc.transport_type == "openai_chat"
    assert desc.credential_env == "LQQQ_API_KEY"
    assert desc.credential_attr == "lqqq_api_key"
    assert desc.default_base_url == "https://api.lqqq.cc/v1"
    assert desc.proxy_attr == "lqqq_proxy"
    assert "chat" in desc.capabilities
    assert "streaming" in desc.capabilities
    assert "tools" in desc.capabilities
    assert "thinking" in desc.capabilities


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------


def test_settings_has_lqqq_fields() -> None:
    from config.settings import Settings

    s = Settings.model_construct(
        model="lqqq/claude-sonnet-4-20250514",
        lqqq_api_key="test-lqqq-key",
        lqqq_proxy="http://proxy:8080",
    )
    assert s.lqqq_api_key == "test-lqqq-key"
    assert s.lqqq_proxy == "http://proxy:8080"


def test_settings_validates_lqqq_model_format() -> None:
    from config.settings import Settings

    s = Settings.model_construct(model="lqqq/some-model")
    assert s.provider_type == "lqqq"
    assert s.model_name == "some-model"
