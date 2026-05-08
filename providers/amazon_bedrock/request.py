"""Amazon Bedrock Converse request builder."""

from __future__ import annotations

from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import (
    dump_raw_messages_request,
    sanitize_native_messages_thinking_policy,
)


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    blocks: list[dict[str, Any]] = []
    for block in content:
        converted = _content_block(block)
        if converted is not None:
            blocks.append(converted)
    return blocks or [{"text": ""}]


def _content_block(block: Any) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return {"text": str(block)}

    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text")
        return {"text": text if isinstance(text, str) else ""}
    if block_type == "tool_use":
        tool_id = block.get("id")
        name = block.get("name")
        tool_input = block.get("input")
        return {
            "toolUse": {
                "toolUseId": tool_id if isinstance(tool_id, str) else "",
                "name": name if isinstance(name, str) else "",
                "input": tool_input if isinstance(tool_input, dict) else {},
            }
        }
    if block_type == "tool_result":
        return _tool_result_block(block)
    if block_type == "thinking" and isinstance(block.get("signature"), str):
        return {
            "reasoningContent": {
                "reasoningText": {
                    "text": block.get("thinking", ""),
                    "signature": block["signature"],
                }
            }
        }
    if block_type == "redacted_thinking":
        data = block.get("data")
        return {
            "reasoningContent": {
                "redactedContent": data if isinstance(data, str) else ""
            }
        }
    return None


def _tool_result_block(block: dict[str, Any]) -> dict[str, Any]:
    tool_use_id = block.get("tool_use_id")
    content = block.get("content")
    result_content: list[dict[str, Any]] = []
    if isinstance(content, str):
        result_content.append({"text": content})
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text" and isinstance(item.get("text"), str):
                    result_content.append({"text": item["text"]})
                else:
                    result_content.append({"json": item})
            else:
                result_content.append({"text": str(item)})
    elif isinstance(content, dict):
        result_content.append({"json": content})
    else:
        result_content.append({"text": ""})

    return {
        "toolResult": {
            "toolUseId": tool_use_id if isinstance(tool_use_id, str) else "",
            "content": result_content,
        }
    }


def _messages(messages: Any, *, thinking_enabled: bool) -> list[dict[str, Any]]:
    sanitized = sanitize_native_messages_thinking_policy(
        messages, thinking_enabled=thinking_enabled
    )
    if not isinstance(sanitized, list):
        return []

    converted: list[dict[str, Any]] = []
    for message in sanitized:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        converted.append(
            {"role": role, "content": _content_blocks(message.get("content", ""))}
        )
    return converted


def _system_blocks(system: Any) -> list[dict[str, str]] | None:
    if system is None:
        return None
    if isinstance(system, str):
        return [{"text": system}]
    if not isinstance(system, list):
        return None

    blocks: list[dict[str, str]] = []
    for block in system:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            blocks.append({"text": block["text"]})
    return blocks or None


def _inference_config(request: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if isinstance(request.get("max_tokens"), int):
        config["maxTokens"] = request["max_tokens"]
    else:
        config["maxTokens"] = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    if isinstance(request.get("stop_sequences"), list):
        config["stopSequences"] = request["stop_sequences"]
    if isinstance(request.get("temperature"), int | float):
        config["temperature"] = request["temperature"]
    if isinstance(request.get("top_p"), int | float):
        config["topP"] = request["top_p"]
    return config


def _tool_config(request: dict[str, Any]) -> dict[str, Any] | None:
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        return None

    converted_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        tool_spec: dict[str, Any] = {"name": name}
        description = tool.get("description")
        if isinstance(description, str) and description:
            tool_spec["description"] = description
        input_schema = tool.get("input_schema")
        if isinstance(input_schema, dict):
            tool_spec["inputSchema"] = {"json": input_schema}
        converted_tools.append({"toolSpec": tool_spec})

    if not converted_tools:
        return None

    config: dict[str, Any] = {"tools": converted_tools}
    tool_choice = _tool_choice(request.get("tool_choice"))
    if tool_choice is not None:
        config["toolChoice"] = tool_choice
    return config


def _tool_choice(tool_choice: Any) -> dict[str, Any] | None:
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return {"auto": {}}
    if choice_type == "any":
        return {"any": {}}
    if choice_type == "tool":
        name = tool_choice.get("name")
        if isinstance(name, str) and name:
            return {"tool": {"name": name}}
    return None


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict[str, Any]:
    """Build a model-neutral Bedrock ConverseStream request body."""
    logger.debug(
        "BEDROCK_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    request = dump_raw_messages_request(request_data)
    body: dict[str, Any] = {
        "model": request.get("model"),
        "messages": _messages(
            request.get("messages"), thinking_enabled=thinking_enabled
        ),
        "inferenceConfig": _inference_config(request),
    }

    system = _system_blocks(request.get("system"))
    if system is not None:
        body["system"] = system
    tool_config = _tool_config(request)
    if tool_config is not None:
        body["toolConfig"] = tool_config
    extra_body = request.get("extra_body")
    if isinstance(extra_body, dict):
        additional = extra_body.get("additionalModelRequestFields")
        if isinstance(additional, dict):
            body["additionalModelRequestFields"] = additional
        response_paths = extra_body.get("additionalModelResponseFieldPaths")
        if isinstance(response_paths, list):
            body["additionalModelResponseFieldPaths"] = response_paths

    logger.debug(
        "BEDROCK_REQUEST: conversion done msgs={} tools={}",
        len(body.get("messages", [])),
        len(body.get("toolConfig", {}).get("tools", [])),
    )
    return body
