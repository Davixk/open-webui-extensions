"""
title: Auto Anthropic
author: @nokodo
description: clean, plug and play Claude manifold pipeline with support for all the latest features from Anthropic
version: 0.2.0-beta1
required_open_webui_version: ">= 0.5.0"
license: see extension documentation file `auto_claude.md` (License section) for the licensing terms.
repository_url: https://nokodo.net/github/open-webui-extensions
funding_url: https://ko-fi.com/nokodo
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from collections import defaultdict
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    DefaultDict,
    Iterable,
    Literal,
    Optional,
    cast,
)

from anthropic import Anthropic
from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
    RedactedThinkingBlock,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
)
from pydantic import BaseModel, Field

LogLevel = Literal["debug", "info", "warning", "error"]


async def emit_status(
    description: str,
    emitter: Any,
    status: Literal["in_progress", "complete", "error"] = "complete",
    extra_data: Optional[dict] = None,
):
    if not emitter:
        raise ValueError("emitter is required to emit status updates")

    await emitter(
        {
            "type": "status",
            "data": {
                "description": description,
                "status": status,
                "done": status in ("complete", "error"),
                "error": status == "error",
                **(extra_data or {}),
            },
        }
    )


REASONING_EFFORT_BUDGET_TOKEN_MAP = {
    "none": None,
    "low": 4_000,
    "medium": 16_000,
    "high": 32_000,
    "max": 48_000,
}

MAX_COMBINED_TOKENS = 128_000

MODEL_SPECS = {
    "claude-sonnet-4-5-20250929": {
        "max_output_tokens": 64_000,
        "supports_thinking": True,
    },
    "claude-sonnet-4-5": {"max_output_tokens": 64_000, "supports_thinking": True},
    "claude-haiku-4-5-20251001": {
        "max_output_tokens": 64_000,
        "supports_thinking": True,
    },
    "claude-haiku-4-5": {"max_output_tokens": 64_000, "supports_thinking": True},
    "claude-opus-4-1-20250805": {
        "max_output_tokens": 32_000,
        "supports_thinking": True,
    },
    "claude-opus-4-1": {"max_output_tokens": 32_000, "supports_thinking": True},
    "claude-sonnet-4-20250514": {
        "max_output_tokens": 64_000,
        "supports_thinking": True,
    },
    "claude-sonnet-4-0": {"max_output_tokens": 64_000, "supports_thinking": True},
    "claude-sonnet-4-latest": {
        "max_output_tokens": 64_000,
        "supports_thinking": True,
    },
    "claude-opus-4-20250514": {"max_output_tokens": 32_000, "supports_thinking": True},
    "claude-opus-4-0": {"max_output_tokens": 32_000, "supports_thinking": True},
    "claude-opus-4-latest": {"max_output_tokens": 32_000, "supports_thinking": True},
    "claude-3-7-sonnet-20250219": {
        "max_output_tokens": 64_000,
        "supports_thinking": True,
    },
    "claude-3-7-sonnet-latest": {
        "max_output_tokens": 64_000,
        "supports_thinking": True,
    },
}

CLAUDE_MODELS = list(MODEL_SPECS.keys())


class Pipe:
    """Claude pipeline that talks directly to the Anthropic Messages API."""

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default=os.getenv("ANTHROPIC_API_KEY", ""),
            description="Anthropic API key",
        )
        ttft_as_thinking: bool = Field(
            default=False,
            description="show 'thinking...' status while waiting for first token (entertaining loading indicator)",
        )
        debug_mode: bool = Field(
            default=False,
            description="enable debug logging",
        )

    def __init__(self) -> None:
        self.type = "manifold"
        self.id = "auto_claude"
        self.valves = self.Valves()
        self.log("Anthropic-native Claude pipeline initialized", level="debug")
        self._thinking_start_time: Optional[float] = None

    async def on_startup(self):
        self.log(f"on_startup:{__name__}")

    async def on_shutdown(self):
        self.log(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        self.log("Valves updated")

    def log(self, message: Any, level: LogLevel = "info"):
        if level == "debug" and not self.valves.debug_mode:
            return
        if level not in {"debug", "info", "warning", "error"}:
            level = "info"
        logger = logging.getLogger()
        getattr(logger, level, logger.info)(message)

    def _anthropic_client(self) -> Anthropic:
        if not self.valves.ANTHROPIC_API_KEY:
            raise RuntimeError("missing ANTHROPIC_API_KEY in valves")
        return Anthropic(api_key=self.valves.ANTHROPIC_API_KEY)

    def get_anthropic_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for model_name in CLAUDE_MODELS:
            specs = MODEL_SPECS[model_name]
            models.append(
                {
                    "id": f"anthropic-native/{model_name}",
                    "name": model_name,
                    "context_length": 200_000,
                    "supports_vision": True,
                    "supports_thinking": specs["supports_thinking"],
                    "max_output_tokens": specs["max_output_tokens"],
                }
            )
        self.log(f"Available native models: {models}", level="debug")
        return models

    def pipes(self) -> list[dict[str, Any]]:
        self.log("native pipes called", level="debug")
        return self.get_anthropic_models()

    async def thinking_status(
        self,
        status: Literal["started", "completed"],
        emitter: Callable[[Any], Awaitable[None]],
    ):
        current_time = time.time()
        if status == "started":
            await emit_status(
                description="thinking",
                emitter=emitter,
                status="in_progress",
            )
            self._thinking_start_time = current_time
        else:
            if self._thinking_start_time is None:
                raise RuntimeError("thinking_start_time not set")
            thinking_duration = current_time - self._thinking_start_time
            await emit_status(
                description=f"thought for {thinking_duration:.1f}s",
                emitter=emitter,
                status="complete",
            )
            self._thinking_start_time = None

    async def execute_tool(
        self,
        tool_use: dict[str, Any],
        tools: dict[str, Any],
    ) -> dict[str, Any] | str:
        tool_name = tool_use.get("name") or tool_use.get("function", {}).get("name")
        if not tool_name:
            return {
                "tool_call": tool_use,
                "result": None,
                "error": "Missing tool name in tool_use payload",
            }
        try:
            tool = tools.get(tool_name) if tools else None
            if not tool:
                raise ValueError(f"tool '{tool_name}' not found")
            arguments = tool_use.get("input")
            if arguments is None and tool_use.get("function"):
                arg_json = tool_use["function"].get("arguments") or "{}"
                try:
                    arguments = json.loads(arg_json)
                except json.JSONDecodeError:
                    arguments = {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"value": arguments}
            if arguments is None:
                arguments = {}
            result = await tool["callable"](**arguments)
            return json.dumps(result)
        except json.JSONDecodeError:
            return {
                "tool_call": tool_use,
                "result": None,
                "error": f"Failed to parse arguments for tool '{tool_name}'",
            }
        except Exception as exc:  # noqa: BLE001
            self.log(f"Error executing tool '{tool_name}': {exc}", "error")
            return {
                "tool_call": tool_use,
                "result": None,
                "error": f"Error executing tool '{tool_name}': {exc}",
            }

    async def query_anthropic_sdk(
        self,
        messages: list[dict[str, Any]],
        event_emitter: Callable[[Any], Awaitable[None]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        stream: Optional[bool] = None,
        tool_choice: Optional[Literal["none", "auto", "required", "any"]] = None,
        thinking_config: Optional[dict[str, Any]] = None,
        host_tools: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator | str:
        if model is None:
            model = "claude-sonnet-4-latest"
        if max_tokens is None:
            max_tokens = 16_000
        if temperature is None:
            temperature = 1
        if stream is None:
            stream = True

        client = self._anthropic_client()

        if stream:
            return self._anthropic_stream_handler(
                client=client,
                model=model,
                host_messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop,
                tools=tools,
                tool_choice=tool_choice,
                thinking_config=thinking_config,
                host_tools=host_tools,
                event_emitter=event_emitter,
            )

        request_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._build_anthropic_messages(messages),
        }

        system_prompt = self._build_system_prompt(messages)
        if system_prompt:
            request_kwargs["system"] = system_prompt
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if stop:
            request_kwargs["stop_sequences"] = stop
        if thinking_config is not None:
            request_kwargs["thinking"] = thinking_config

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            request_kwargs["tools"] = cast(Any, converted_tools)

        converted_tool_choice = self._convert_tool_choice(tool_choice)
        if converted_tool_choice:
            request_kwargs["tool_choice"] = cast(Any, converted_tool_choice)

        response = client.messages.create(**request_kwargs)  # type: ignore[arg-type]

        if response.content:
            return "".join(
                getattr(block, "text", "")
                for block in response.content
                if getattr(block, "type", None) == "text"
            )
        return ""

    def _build_system_prompt(self, messages: list[dict[str, Any]]) -> Optional[str]:
        parts: list[str] = []
        for message in messages:
            if message.get("role") == "system":
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
        return "\n\n".join(filter(None, parts)) or None

    def _coerce_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (int, float, bool)):
            return str(content)
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, list):
            return "\n".join(self._coerce_text(item) for item in content)
        return str(content)

    def _convert_openai_tool_calls(
        self, tool_calls: Optional[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        if not tool_calls:
            return converted
        for call in tool_calls:
            function = call.get("function", {})
            block: dict[str, Any] = {
                "type": "tool_use",
                "id": call.get("id"),
                "name": function.get("name"),
            }
            arguments = function.get("arguments")
            if arguments:
                try:
                    block["input"] = json.loads(arguments)
                except json.JSONDecodeError:
                    block["input"] = arguments
            converted.append(block)
        return converted

    def _convert_assistant_content(
        self,
        content: Any,
        tool_calls: Optional[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "image"}:
                    blocks.append(item)
                elif isinstance(item, str):
                    blocks.append({"type": "text", "text": item})
        elif content:
            blocks.append({"type": "text", "text": self._coerce_text(content)})
        blocks.extend(self._convert_openai_tool_calls(tool_calls))
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        return blocks

    def _convert_user_content(self, content: Any) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "image"}:
                    blocks.append(item)
                elif isinstance(item, str):
                    blocks.append({"type": "text", "text": item})
        elif content is not None:
            blocks.append({"type": "text", "text": self._coerce_text(content)})
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        return blocks

    def _build_anthropic_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        anthropic_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "assistant":
                structured = message.get("_anthropic_content")
                if structured:
                    content_blocks = copy.deepcopy(structured)
                else:
                    content_blocks = self._convert_assistant_content(
                        message.get("content"),
                        message.get("tool_calls"),
                    )
                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": content_blocks,
                    }
                )
            elif role == "user":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": self._convert_user_content(message.get("content")),
                    }
                )
            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": self._coerce_text(message.get("content")),
                            }
                        ],
                    }
                )
        return anthropic_messages

    def _convert_tools(
        self, tools: Optional[list[dict[str, Any]]]
    ) -> Optional[list[dict[str, Any]]]:
        if not tools:
            return None
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function", {})
            converted.append(
                {
                    "type": "custom",
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "input_schema": function.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                }
            )
        return converted or None

    def _convert_tool_choice(
        self, tool_choice: Optional[str]
    ) -> Optional[dict[str, Any]]:
        if tool_choice in {None, "auto"}:
            return {"type": "auto"} if tool_choice else None
        if tool_choice == "none":
            return {"type": "none"}
        if tool_choice in {"required", "any"}:
            return {"type": "any"}
        return None

    async def _anthropic_stream_handler(
        self,
        client: Anthropic,
        model: str,
        host_messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        stop_sequences: Optional[list[str]],
        tools: Optional[list[dict[str, Any]]],
        tool_choice: Optional[str],
        thinking_config: Optional[dict[str, Any]],
        host_tools: Optional[dict[str, Any]],
        event_emitter: Callable[[Any], Awaitable[None]],
    ) -> AsyncIterator[str]:
        first_output = True
        first_iteration_with_text = True
        first_iteration_after_tool_call = False

        prepped_tools = self._convert_tools(tools)
        prepped_tool_choice = self._convert_tool_choice(tool_choice)

        while True:
            system_prompt = self._build_system_prompt(host_messages)
            anthropic_messages = self._build_anthropic_messages(host_messages)

            stream_params = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": anthropic_messages,
                "temperature": temperature,
            }
            if thinking_config is not None:
                stream_params["thinking"] = thinking_config
            if system_prompt:
                stream_params["system"] = system_prompt
            if top_p is not None:
                stream_params["top_p"] = top_p
            if stop_sequences:
                stream_params["stop_sequences"] = stop_sequences
            if prepped_tools:
                stream_params["tools"] = prepped_tools
            if prepped_tool_choice:
                stream_params["tool_choice"] = prepped_tool_choice

            assistant_blocks: list[dict[str, Any]] = []
            block_buffer: DefaultDict[int, dict[str, Any]] = defaultdict(dict)
            collected_tool_uses: list[dict[str, Any]] = []
            stop_reason: Optional[str] = None

            self.log(f"Stream params: {list(stream_params.keys())}", "debug")
            with client.messages.stream(**stream_params) as stream:
                for event in stream:
                    if isinstance(event, RawContentBlockStartEvent):
                        block_index = event.index
                        block = event.content_block
                        block_type = block.type
                        prepared: dict[str, Any] = {"type": block_type}

                        if isinstance(block, ThinkingBlock):
                            prepared["thinking"] = ""
                            prepared["signature"] = block.signature
                        elif isinstance(block, RedactedThinkingBlock):
                            prepared["thinking"] = ""
                        elif isinstance(block, TextBlock):
                            initial_text = block.text or ""
                            prepared["text"] = initial_text
                            if initial_text:
                                if (
                                    first_iteration_after_tool_call
                                    and not first_iteration_with_text
                                ):
                                    first_iteration_after_tool_call = False
                                    yield "\n\n---\n\n"
                                if first_output:
                                    first_output = False
                                    if self.valves.ttft_as_thinking:
                                        await self.thinking_status(
                                            "completed", emitter=event_emitter
                                        )
                                if first_iteration_with_text:
                                    first_iteration_with_text = False
                                yield initial_text
                        elif isinstance(block, ToolUseBlock):
                            prepared["id"] = block.id
                            prepared["name"] = block.name
                            prepared["input"] = block.input

                        block_buffer[block_index] = prepared
                        assistant_blocks.append(prepared)

                        if isinstance(block, ToolUseBlock):
                            collected_tool_uses.append(prepared)
                    elif isinstance(event, RawContentBlockDeltaEvent):
                        block_index = event.index
                        delta = event.delta
                        block_state = block_buffer[block_index]

                        if isinstance(delta, ThinkingDelta):
                            block_state["thinking"] = (
                                block_state.get("thinking", "") + delta.thinking
                            )
                        elif isinstance(delta, SignatureDelta):
                            block_state["signature"] = (
                                block_state.get("signature", "") + delta.signature
                            )
                        elif isinstance(delta, TextDelta):
                            if (
                                first_iteration_after_tool_call
                                and not first_iteration_with_text
                            ):
                                first_iteration_after_tool_call = False
                                yield "\n\n---\n\n"
                            if first_output:
                                first_output = False
                                if self.valves.ttft_as_thinking:
                                    await self.thinking_status(
                                        "completed", emitter=event_emitter
                                    )
                            if first_iteration_with_text:
                                first_iteration_with_text = False
                            block_state["text"] = (
                                block_state.get("text", "") + delta.text
                            )
                            yield delta.text
                        elif isinstance(delta, InputJSONDelta):
                            # Initialize as empty string if not exists, then append
                            current = block_state.get("input", "")
                            if isinstance(current, dict):
                                current = ""
                            block_state["input"] = current + delta.partial_json
                    elif isinstance(event, RawMessageDeltaEvent):
                        if event.delta.stop_reason:
                            stop_reason = stop_reason or event.delta.stop_reason
                    elif isinstance(event, RawMessageStopEvent):
                        # Message stop event doesn't have stop_reason directly
                        pass
                    elif event.type == "error":
                        raise RuntimeError(f"Anthropic stream error: {event}")

            self.log(f"Stream completed: stop_reason={stop_reason}", "debug")
            if collected_tool_uses:
                self.log(f"Collected {len(collected_tool_uses)} tool uses", "debug")
            for block in assistant_blocks:
                if block.get("type") == "tool_use" and isinstance(
                    block.get("input"), str
                ):
                    try:
                        block["input"] = json.loads(block["input"])
                    except json.JSONDecodeError:
                        pass

            if stop_reason == "tool_use" and collected_tool_uses:
                self.log(f"Executing {len(collected_tool_uses)} tools", "debug")
                if first_output and self.valves.ttft_as_thinking:
                    await self.thinking_status("completed", emitter=event_emitter)
                    first_output = False

                assistant_text = "".join(
                    block.get("text", "")
                    for block in assistant_blocks
                    if block.get("type") == "text"
                )
                host_messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_text,
                        "_anthropic_content": copy.deepcopy(assistant_blocks),
                        "tool_calls": self._build_openai_style_tool_calls(
                            collected_tool_uses
                        ),
                    }
                )

                if not host_tools:
                    raise RuntimeError("host tools are required for tool execution")

                for tool_use in collected_tool_uses:
                    tool_result = await self.execute_tool(tool_use, host_tools)
                    tool_use_id = tool_use.get("id")
                    result_payload = self._coerce_text(tool_result)
                    host_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": result_payload,
                        }
                    )

                first_iteration_after_tool_call = True
                first_iteration_with_text = True
                continue

            break

    def _build_openai_style_tool_calls(
        self, tool_uses: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for tool_use in tool_uses:
            arguments = tool_use.get("input") or {}
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            tool_calls.append(
                {
                    "id": tool_use.get("id"),
                    "type": "function",
                    "function": {
                        "name": tool_use.get("name"),
                        "arguments": arguments,
                    },
                }
            )
        return tool_calls

    def _object_to_dict(self, obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump(mode="python", exclude_none=True)
            except TypeError:
                return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return {}

    def setup_params(
        self, body: dict[str, Any]
    ) -> tuple[str, int, Optional[dict[str, Any]]]:
        model_full = body.get("model", "anthropic-native/claude-sonnet-4-5-20250929")
        model = model_full.split("/", 1)[1] if "/" in model_full else model_full
        reasoning_effort = body.get("reasoning_effort", "none")
        budget_tokens = REASONING_EFFORT_BUDGET_TOKEN_MAP.get(reasoning_effort)
        if (
            not budget_tokens
            and reasoning_effort is not None
            and reasoning_effort not in REASONING_EFFORT_BUDGET_TOKEN_MAP
        ):
            try:
                budget_tokens = int(reasoning_effort)
            except ValueError:
                self.log(
                    f"Failed to convert reasoning effort to int: {reasoning_effort}",
                    "warning",
                )
                budget_tokens = None
        max_tokens = body.get("max_tokens", 64_000)
        thinking_config = None
        if budget_tokens:
            combined_tokens = budget_tokens + max_tokens
            if combined_tokens > MAX_COMBINED_TOKENS:
                self.log(
                    "Combined thinking and output tokens exceed Anthropic limit",
                    "error",
                )
                raise ValueError(
                    "invalid request. please contact your system administrator."
                )
            thinking_config = {"type": "enabled", "budget_tokens": budget_tokens}
        return model, max_tokens, thinking_config

    async def auto_claude(
        self,
        body: dict[str, Any],
        event_emitter: Callable[[Any], Awaitable[None]],
        host_tools: Optional[dict[str, Any]] = None,
    ):
        model, max_tokens, thinking_config = self.setup_params(body)
        return await self.query_anthropic_sdk(
            model=model,
            event_emitter=event_emitter,
            messages=body.get("messages", []),
            max_tokens=max_tokens,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            stop=body.get("stop"),
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            stream=body.get("stream"),
            thinking_config=thinking_config,
            host_tools=host_tools,
        )

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __tools__: Optional[dict[str, Any]] = None,
        __task__: Optional[str] = None,
    ) -> str | AsyncIterator[str]:
        self.log(f"native pipe called with body: {body}", level="debug")
        if not __event_emitter__:
            raise RuntimeError("event emitter is required")
        if self.valves.ttft_as_thinking:
            await self.thinking_status("started", emitter=__event_emitter__)
        if __task__ == "function_calling":
            return ""
        return await self.auto_claude(
            body=body,
            event_emitter=__event_emitter__,
            host_tools=__tools__,
        )
