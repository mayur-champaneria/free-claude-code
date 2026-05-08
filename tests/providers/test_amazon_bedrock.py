from __future__ import annotations

import json
import struct
import zlib
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from api.models.anthropic import MessagesRequest
from config.bedrock import BedrockSettings
from providers.amazon_bedrock import AmazonBedrockProvider
from providers.amazon_bedrock.client import (
    _extract_foundation_model_ids,
    _extract_inference_profile_ids,
)
from providers.base import ProviderConfig
from providers.model_listing import ProviderModelInfo


class _AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def _provider() -> AmazonBedrockProvider:
    return AmazonBedrockProvider(
        ProviderConfig(api_key="aws-default-chain"),
        bedrock_settings=BedrockSettings(
            region="us-east-1",
            access_key_id="AKIATEST",
            secret_access_key="secret",
        ),
    )


def _request(**overrides) -> MessagesRequest:
    data = {
        "model": "deepseek.v3.2",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "metadata": {"user_id": "ignored"},
        "system": [{"type": "text", "text": "Be brief."}],
    }
    data.update(overrides)
    return MessagesRequest.model_validate(data)


def _eventstream_frame(headers: dict[str, str], payload: dict) -> bytes:
    header_bytes = b"".join(
        _eventstream_string_header(name, value) for name, value in headers.items()
    )
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
    total_length = 12 + len(header_bytes) + len(payload_bytes) + 4
    prelude_prefix = struct.pack(">II", total_length, len(header_bytes))
    prelude_crc = struct.pack(">I", zlib.crc32(prelude_prefix) & 0xFFFFFFFF)
    message_without_crc = prelude_prefix + prelude_crc + header_bytes + payload_bytes
    message_crc = struct.pack(">I", zlib.crc32(message_without_crc) & 0xFFFFFFFF)
    return message_without_crc + message_crc


def _eventstream_string_header(name: str, value: str) -> bytes:
    name_bytes = name.encode()
    value_bytes = value.encode()
    return (
        struct.pack(">B", len(name_bytes))
        + name_bytes
        + struct.pack(">BH", 7, len(value_bytes))
        + value_bytes
    )


def test_build_request_body_uses_bedrock_converse_shape():
    provider = _provider()

    body = provider._build_request_body(_request())

    assert body["model"] == "deepseek.v3.2"
    assert body["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    assert body["inferenceConfig"] == {"maxTokens": 64}
    assert body["system"] == [{"text": "Be brief."}]
    assert "stream" not in body
    assert "metadata" not in body
    assert "requestMetadata" not in body


@pytest.mark.asyncio
async def test_send_stream_request_posts_signed_bedrock_runtime_path() -> None:
    provider = _provider()
    with (
        patch.object(
            provider,
            "_signed_headers",
            return_value={"Authorization": "AWS4-HMAC-SHA256 signed"},
        ) as signed_headers,
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=httpx.Response(200),
        ) as send,
    ):
        await provider._send_stream_request(
            provider._build_request_body(_request(model="qwen.qwen3-coder-test-v1:0"))
        )

    assert send.await_args is not None
    request = send.await_args.args[0]
    assert str(request.url).endswith(
        "/model/qwen.qwen3-coder-test-v1%3A0/converse-stream"
    )
    assert b'"inferenceConfig":{"maxTokens":64}' in request.content
    assert b'"model"' not in request.content
    signed_headers.assert_called_once()


@pytest.mark.asyncio
async def test_stream_response_decodes_bedrock_eventstream_to_anthropic_sse() -> None:
    provider = _provider()
    frame = _eventstream_frame(
        {":message-type": "event", ":event-type": "contentBlockDelta"},
        {"contentBlockIndex": 0, "delta": {"text": "hello"}},
    )
    response = httpx.Response(
        200,
        stream=_AsyncBytes([frame[:10], frame[10:]]),
        request=httpx.Request("POST", "https://bedrock-runtime.test"),
    )

    with patch.object(
        provider,
        "_send_stream_request",
        new_callable=AsyncMock,
        return_value=response,
    ):
        chunks = [chunk async for chunk in provider.stream_response(_request())]

    text = "".join(chunks)
    assert "event: content_block_start\n" in text
    assert "event: content_block_delta\n" in text
    assert '"text": "hello"' in text


def test_extracts_streaming_foundation_and_inference_profile_ids() -> None:
    foundation = _extract_foundation_model_ids(
        {
            "modelSummaries": [
                {
                    "modelId": "deepseek.v3.2",
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                },
                {"modelId": "qwen.qwen3-coder-30b-a3b-v1:0"},
                {
                    "modelId": "text.no-stream",
                    "responseStreamingSupported": False,
                },
            ]
        }
    )
    profiles = _extract_inference_profile_ids(
        {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "eu.deepseek.v3.2",
                    "status": "ACTIVE",
                },
                {"inferenceProfileId": "inactive.model", "status": "INACTIVE"},
            ]
        }
    )

    assert foundation == frozenset({"deepseek.v3.2", "qwen.qwen3-coder-30b-a3b-v1:0"})
    assert profiles == frozenset({"eu.deepseek.v3.2"})


@pytest.mark.asyncio
async def test_list_model_infos_combines_foundation_models_and_profiles() -> None:
    provider = _provider()
    with patch.object(
        provider,
        "_get_control_json",
        new_callable=AsyncMock,
        side_effect=[
            {
                "modelSummaries": [
                    {
                        "modelId": "deepseek.v3.2",
                        "outputModalities": ["TEXT"],
                        "responseStreamingSupported": True,
                    }
                ]
            },
            {
                "inferenceProfileSummaries": [
                    {
                        "inferenceProfileId": "eu.qwen.qwen3-coder-30b-a3b-v1:0",
                        "status": "ACTIVE",
                    }
                ]
            },
        ],
    ):
        infos = await provider.list_model_infos()

    assert infos == frozenset(
        {
            ProviderModelInfo(
                "deepseek.v3.2",
                supports_thinking=None,
            ),
            ProviderModelInfo(
                "eu.qwen.qwen3-coder-30b-a3b-v1:0",
                supports_thinking=None,
            ),
        }
    )
