"""Amazon Bedrock provider implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote, urlencode

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from botocore.eventstream import EventStreamBuffer
from botocore.session import get_session

from config.bedrock import BedrockSettings
from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from providers.exceptions import (
    APIError,
    AuthenticationError,
    ModelListResponseError,
    RateLimitError,
    ServiceUnavailableError,
)
from providers.model_listing import ProviderModelInfo

from .request import build_request_body

_PROVIDER_NAME = "BEDROCK"
_SERVICE_BEDROCK = "bedrock"
_SERVICE_BEDROCK_RUNTIME = "bedrock"


class AmazonBedrockProvider(AnthropicMessagesTransport):
    """Amazon Bedrock Runtime provider using the model-neutral Converse API."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        bedrock_settings: BedrockSettings,
    ):
        self._bedrock_settings = bedrock_settings
        self._region = bedrock_settings.resolved_region()
        runtime_base_url = (
            config.base_url or f"https://bedrock-runtime.{self._region}.amazonaws.com"
        )
        super().__init__(
            config,
            provider_name=_PROVIDER_NAME,
            default_base_url=runtime_base_url,
        )
        self._control_base_url = (
            bedrock_settings.control_base_url
            or f"https://bedrock.{self._region}.amazonaws.com"
        ).rstrip("/")
        self._aws_session = get_session()
        if bedrock_settings.profile.strip():
            self._aws_session.set_config_variable(
                "profile", bedrock_settings.profile.strip()
            )
        self._aws_session.set_config_variable("region", self._region)

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict[str, Any]:
        """Internal helper for tests and direct request dispatch."""
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Invoke Bedrock ConverseStream with AWS event-stream response framing."""
        model_id = body.get("model")
        if not isinstance(model_id, str) or not model_id.strip():
            msg = "Bedrock request is missing the target model id."
            raise APIError(msg, status_code=400)

        encoded_model_id = quote(model_id, safe="")
        url = f"{self._base_url}/model/{encoded_model_id}/converse-stream"
        payload_body = dict(body)
        payload_body.pop("model", None)
        payload = json.dumps(payload_body, separators=(",", ":")).encode("utf-8")
        headers = self._signed_headers(
            method="POST",
            url=url,
            body=payload,
            service=_SERVICE_BEDROCK_RUNTIME,
            headers={
                "Accept": "application/vnd.amazon.eventstream",
                "Content-Type": "application/json",
            },
        )
        request = self._client.build_request(
            "POST", url, content=payload, headers=headers
        )
        return await self._client.send(request, stream=True)

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return streaming text model and inference-profile ids from Bedrock."""
        model_payload = await self._get_control_json("/foundation-models")
        profile_payload = await self._get_control_json(
            "/inference-profiles", {"maxResults": "1000"}
        )
        model_ids = set(_extract_foundation_model_ids(model_payload))
        model_ids.update(_extract_inference_profile_ids(profile_payload))
        return frozenset(
            ProviderModelInfo(model_id=model_id, supports_thinking=None)
            for model_id in model_ids
        )

    async def _get_control_json(
        self, path: str, params: dict[str, str] | None = None
    ) -> Any:
        url = self._control_url(path, params)
        headers = self._signed_headers(
            method="GET",
            url=url,
            body=b"",
            service=_SERVICE_BEDROCK,
            headers={"Accept": "application/json"},
        )
        response = await self._client.get(url, headers=headers)
        try:
            response.raise_for_status()
            return response.json()
        except ValueError as exc:
            msg = "BEDROCK model-list response is malformed: invalid JSON"
            raise ModelListResponseError(msg) from exc
        finally:
            await response.aclose()

    def _control_url(self, path: str, params: dict[str, str] | None = None) -> str:
        query = f"?{urlencode(params)}" if params else ""
        return f"{self._control_base_url}{path}{query}"

    def _signed_headers(
        self,
        *,
        method: str,
        url: str,
        body: bytes,
        service: str,
        headers: dict[str, str],
    ) -> dict[str, str]:
        request = AWSRequest(method=method, url=url, data=body, headers=headers)
        SigV4Auth(self._credentials(), service, self._region).add_auth(request)
        return dict(request.headers.items())

    def _credentials(self) -> Any:
        settings = self._bedrock_settings
        if settings.access_key_id.strip() and settings.secret_access_key.strip():
            return Credentials(
                settings.access_key_id.strip(),
                settings.secret_access_key.strip(),
                settings.session_token.strip() or None,
            ).get_frozen_credentials()

        credentials = self._aws_session.get_credentials()
        if credentials is None:
            raise AuthenticationError(
                "AWS credentials were not found. Configure AWS_ACCESS_KEY_ID/"
                "AWS_SECRET_ACCESS_KEY, AWS_PROFILE, or the default AWS credential "
                "chain for Amazon Bedrock."
            )
        return credentials.get_frozen_credentials()

    async def _iter_stream_chunks(
        self,
        response: httpx.Response,
        *,
        state: Any,
        thinking_enabled: bool,
    ) -> AsyncIterator[str]:
        """Decode AWS event-stream frames and yield Anthropic SSE chunks."""
        buffer = EventStreamBuffer()
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            buffer.add_data(chunk)
            for message in buffer:
                event = _message_to_sse_event(message)
                if event is None:
                    continue
                output_event = self._transform_stream_event(
                    event,
                    state,
                    thinking_enabled=thinking_enabled,
                )
                if output_event is None:
                    continue
                for line in output_event.splitlines(keepends=True):
                    yield line


def _message_to_sse_event(message: Any) -> str | None:
    headers = message.headers
    message_type = headers.get(":message-type")
    event_type = headers.get(":event-type")
    if message_type in {"error", "exception"}:
        raise _bedrock_stream_error(event_type, message.payload)

    payload = _json_payload(message.payload)
    if isinstance(event_type, str) and event_type:
        payload = {event_type: payload}
    return _converse_event_to_sse(payload)


def _converse_event_to_sse(payload: dict[str, Any]) -> str | None:
    if "messageStart" in payload:
        role = payload.get("messageStart", {}).get("role", "assistant")
        return _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "bedrock",
                    "type": "message",
                    "role": role if role in {"assistant", "user"} else "assistant",
                    "content": [],
                    "model": "",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    content_start = payload.get("contentBlockStart")
    if isinstance(content_start, dict):
        return _content_block_start_to_sse(content_start)

    content_delta = payload.get("contentBlockDelta")
    if isinstance(content_delta, dict):
        return _content_block_delta_to_sse(content_delta)

    content_stop = payload.get("contentBlockStop")
    if isinstance(content_stop, dict):
        index = content_stop.get("contentBlockIndex")
        if isinstance(index, int):
            return _sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            )

    message_stop = payload.get("messageStop")
    if isinstance(message_stop, dict):
        stop_reason = _stop_reason(message_stop.get("stopReason"))
        prefix = _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        )
        return prefix + _sse("message_stop", {"type": "message_stop"})

    if any(
        key in payload
        for key in (
            "internalServerException",
            "modelStreamErrorException",
            "serviceUnavailableException",
            "throttlingException",
            "validationException",
        )
    ):
        name, body = next(iter(payload.items()))
        raw_message = body.get("message") if isinstance(body, dict) else None
        message = raw_message if isinstance(raw_message, str) else name
        raise _bedrock_stream_error(name, json.dumps({"message": message}).encode())

    return None


def _content_block_start_to_sse(content_start: dict[str, Any]) -> str | None:
    index = content_start.get("contentBlockIndex")
    if not isinstance(index, int):
        return None
    start = content_start.get("start")
    if not isinstance(start, dict):
        return None
    tool_use = start.get("toolUse")
    if not isinstance(tool_use, dict):
        return None
    tool_use_id = tool_use.get("toolUseId")
    name = tool_use.get("name")
    return _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": tool_use_id if isinstance(tool_use_id, str) else "",
                "name": name if isinstance(name, str) else "",
                "input": {},
            },
        },
    )


def _content_block_delta_to_sse(content_delta: dict[str, Any]) -> str | None:
    index = content_delta.get("contentBlockIndex")
    if not isinstance(index, int):
        return None
    delta = content_delta.get("delta")
    if not isinstance(delta, dict):
        return None
    text = delta.get("text")
    if isinstance(text, str):
        return _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            },
        )
    tool_use = delta.get("toolUse")
    if isinstance(tool_use, dict):
        partial = tool_use.get("input")
        return _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": partial if isinstance(partial, str) else "",
                },
            },
        )
    reasoning = delta.get("reasoningContent")
    if isinstance(reasoning, dict):
        reasoning_text = reasoning.get("text")
        signature = reasoning.get("signature")
        payload_delta: dict[str, Any] = {
            "type": "thinking_delta",
            "thinking": reasoning_text if isinstance(reasoning_text, str) else "",
        }
        if isinstance(signature, str):
            payload_delta = {"type": "signature_delta", "signature": signature}
        return _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": payload_delta,
            },
        )
    return None


def _sse(event_name: str, payload: dict[str, Any]) -> str:
    return (
        f"event: {event_name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
    )


def _stop_reason(value: Any) -> str:
    if value == "tool_use":
        return "tool_use"
    if value == "max_tokens":
        return "max_tokens"
    if value in {"stop_sequence", "end_turn"}:
        return "end_turn"
    return "end_turn"


def _json_payload(payload: bytes) -> dict[str, Any]:
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise APIError("Bedrock stream chunk was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise APIError("Bedrock stream chunk JSON was not an object.")
    return data


def _bedrock_stream_error(event_type: Any, payload: bytes) -> Exception:
    body = _json_payload(payload) if payload else {}
    message = body.get("message") or body.get("Message") or "Bedrock stream error"
    if not isinstance(message, str):
        message = "Bedrock stream error"
    status = _stream_error_status(event_type)
    if status == 429:
        return RateLimitError(message, raw_error=body)
    if status == 503:
        return ServiceUnavailableError(message, raw_error=body)
    return APIError(message, status_code=status, raw_error=body)


def _stream_error_status(event_type: Any) -> int:
    if event_type in {"throttlingException", "modelNotReadyException"}:
        return 429
    if event_type == "modelTimeoutException":
        return 408
    if event_type == "validationException":
        return 400
    if event_type == "serviceUnavailableException":
        return 503
    if event_type == "modelStreamErrorException":
        return 424
    return 500


def _extract_foundation_model_ids(payload: Any) -> frozenset[str]:
    summaries = payload.get("modelSummaries") if isinstance(payload, dict) else None
    if not isinstance(summaries, list):
        msg = "BEDROCK model-list response is malformed: expected modelSummaries array"
        raise ModelListResponseError(msg)

    model_ids: set[str] = set()
    for item in summaries:
        if not isinstance(item, dict):
            continue
        model_id = item.get("modelId")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        if item.get("responseStreamingSupported") is False:
            continue
        output_modalities = item.get("outputModalities")
        if isinstance(output_modalities, list) and "TEXT" not in output_modalities:
            continue
        model_ids.add(model_id)
    return frozenset(model_ids)


def _extract_inference_profile_ids(payload: Any) -> frozenset[str]:
    summaries = (
        payload.get("inferenceProfileSummaries") if isinstance(payload, dict) else None
    )
    if not isinstance(summaries, list):
        msg = (
            "BEDROCK inference-profile response is malformed: expected "
            "inferenceProfileSummaries array"
        )
        raise ModelListResponseError(msg)

    profile_ids: set[str] = set()
    for item in summaries:
        if not isinstance(item, dict):
            continue
        profile_id = item.get("inferenceProfileId")
        profile_arn = item.get("inferenceProfileArn")
        if item.get("status") not in (None, "ACTIVE"):
            continue
        if isinstance(profile_id, str) and profile_id.strip():
            profile_ids.add(profile_id)
        if isinstance(profile_arn, str) and profile_arn.strip():
            profile_ids.add(profile_arn)
    return frozenset(profile_ids)
