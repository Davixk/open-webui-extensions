"""
title: Auto Anthropic
author: @nokodo
description: clean, plug and play Claude manifold pipeline with support for all the latest features from Anthropic
version: 0.1.0-alpha7
required_open_webui_version: ">= 0.5.0"
license: see extension documentation file `auto_claude.md` (License section) for the licensing terms.
repository_url: https://nokodo.net/github/open-webui-extensions
funding_url: https://ko-fi.com/nokodo
"""

import json
import logging
import os
import time
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Optional,
    Union,
)

import requests
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel, Field

LogLevel = Literal["debug", "info", "warning", "error"]


async def emit_status(
    description: str,
    emitter: Callable[[Any], Awaitable[None]],
    status: Literal["in_progress", "complete", "error"] = "complete",
    done: Optional[bool] = None,
):
    """Emit a status event with sensible defaults.

    Defaults:
    - status defaults to "complete"
    - done defaults to True unless status == "in_progress" (then False)
    """
    if not emitter:
        raise ValueError("emitter is required")
    if done is None:
        done = status != "in_progress"
    await emitter(
        {
            "type": "status",
            "data": {
                "description": description,
                "status": status,
                "done": done,
            },
        }
    )


# Reasoning effort to token budget mapping
REASONING_EFFORT_BUDGET_TOKEN_MAP = {
    "none": None,
    "low": 4_000,
    "medium": 16_000,
    "high": 32_000,
    "max": 48_000,
}

# Maximum combined token limit for Claude 4
MAX_COMBINED_TOKENS = 128_000

CLAUDE_MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-latest",
    "claude-opus-4-latest",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-haiku-4-5-20251001",
]

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_PDF_SIZE = 32 * 1024 * 1024  # 32 MB
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 5


async def query_anthropic_sdk(
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    system_message: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    stop: Optional[list[str]],
    tools: Optional[list[dict[str, Any]]],
    tool_choice: Optional[dict[str, Any]],
    stream: bool,
    thinking_config: Optional[dict[str, Any]],
) -> Union[str, AsyncIterator[dict[str, Any]]]:
    """Query Anthropic SDK with native support for both streaming and non-streaming.

    This is a standalone function that uses the official Anthropic Python SDK.
    Tool calls work properly with both streaming and non-streaming modes.

    Args:
        api_key: Anthropic API key
        model: Model name (e.g., "claude-sonnet-4-5-20250929")
        messages: List of message dicts with role and content
        system_message: Optional system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        stop: Stop sequences
        tools: List of tool definitions in OpenAI format (will be converted)
        tool_choice: Tool choice configuration
        stream: Whether to stream responses
        thinking_config: Extended thinking configuration

    Returns:
        If stream=False: Complete response as string, or JSON string with tool_calls
        If stream=True: AsyncIterator yielding dicts with {"type": "content"|"tool_calls", ...}
    """
    client = Anthropic(api_key=api_key)

    # Convert tools from OpenAI format to Anthropic format if provided
    anthropic_tools = None
    if tools:
        anthropic_tools = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append(
                    {
                        "name": func.get("name"),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )
            else:
                # Already in Anthropic format
                anthropic_tools.append(tool)

    # Build extra kwargs for thinking
    extra_kwargs = {}
    if thinking_config:
        extra_kwargs["thinking"] = thinking_config

    if stream:
        # Return async iterator for streaming
        return _stream_anthropic_sdk(
            client=client,
            model=model,
            messages=messages,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            tools=anthropic_tools,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )
    else:
        # Non-streaming: call API and return result
        try:
            # Build kwargs dynamically to avoid None values
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,  # type: ignore
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if system_message:
                kwargs["system"] = system_message
            if top_p is not None:
                kwargs["top_p"] = top_p
            if stop:
                kwargs["stop_sequences"] = stop
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools  # type: ignore
            if tool_choice:
                kwargs["tool_choice"] = tool_choice  # type: ignore

            kwargs.update(extra_kwargs)

            response = client.messages.create(**kwargs)

            # Check for tool use
            tool_use_blocks = [
                block for block in response.content if block.type == "tool_use"
            ]

            if tool_use_blocks:
                # Return tool calls as JSON
                tool_calls = [
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                    for block in tool_use_blocks
                ]
                return json.dumps({"tool_calls": tool_calls})

            # Return text content
            text_blocks = [block for block in response.content if block.type == "text"]
            return text_blocks[0].text if text_blocks else ""

        except Exception as e:
            return f"Error: {e}"


async def _stream_anthropic_sdk(
    client: Anthropic,
    model: str,
    messages: list[dict[str, Any]],
    system_message: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    stop: Optional[list[str]],
    tools: Optional[list[dict[str, Any]]],
    tool_choice: Optional[dict[str, Any]],
    extra_kwargs: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Stream response from Anthropic SDK and yield content chunks and tool calls."""
    try:
        # Build kwargs dynamically
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,  # type: ignore
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_message:
            kwargs["system"] = system_message
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop:
            kwargs["stop_sequences"] = stop
        if tools:
            kwargs["tools"] = tools  # type: ignore
        if tool_choice:
            kwargs["tool_choice"] = tool_choice  # type: ignore

        kwargs.update(extra_kwargs)

        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                # Yield text content as it streams
                if event.type == "content_block_delta":
                    delta = event.delta
                    # Check if delta has text attribute (TextDelta)
                    if hasattr(delta, "text") and delta.text:  # type: ignore
                        yield {"type": "content", "content": delta.text}  # type: ignore

                # Collect tool use when complete
                elif event.type == "content_block_stop":
                    content_block = event.content_block  # type: ignore
                    if hasattr(content_block, "type") and content_block.type == "tool_use":  # type: ignore
                        tool_call = {
                            "id": content_block.id,  # type: ignore
                            "type": "function",
                            "function": {
                                "name": content_block.name,  # type: ignore
                                "arguments": json.dumps(content_block.input),  # type: ignore
                            },
                        }
                        yield {"type": "tool_calls", "tool_calls": [tool_call]}

    except Exception as e:
        yield {"type": "error", "error": str(e)}


class Pipe:
    """Claude manifold pipeline with extended thinking and multimodal support."""

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default=os.getenv("ANTHROPIC_API_KEY", ""),
            description="Anthropic API key",
        )
        debug_mode: bool = Field(
            default=False,
            description="enable debug logging",
        )

    def __init__(self) -> None:
        self.type = "manifold"
        self.id = "auto_claude"
        self.valves = self.Valves()
        self.url = "https://api.anthropic.com/v1/messages"
        self.log("Claude pipeline initialized")

    async def on_startup(self):
        """Called when the pipeline starts up."""
        self.log(f"on_startup:{__name__}")

    async def on_shutdown(self):
        """Called when the pipeline shuts down."""
        self.log(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        """Called when valves are updated."""
        self.log("Valves updated")

    def log(self, message: str, level: LogLevel = "info"):
        if level == "debug" and not self.valves.debug_mode:
            return
        if level not in {"debug", "info", "warning", "error"}:
            level = "info"

        logger = logging.getLogger()
        getattr(logger, level, logger.info)(message)

    def get_anthropic_models(self) -> list[dict[str, Any]]:
        """Return available Claude models with thinking variants."""
        models: list[dict[str, Any]] = []

        for model_name in CLAUDE_MODELS:
            # Standard model
            models.append(
                {
                    "id": f"anthropic/{model_name}",
                    "name": model_name,
                    "context_length": 200_000,
                    "supports_vision": True,
                    "supports_thinking": False,
                    "max_output_tokens": 64_000,
                }
            )

            # Extended thinking variant
            models.append(
                {
                    "id": f"anthropic/{model_name}-thinking",
                    "name": f"{model_name} (thinking)",
                    "context_length": 200_000,
                    "supports_vision": True,
                    "supports_thinking": True,
                    "max_output_tokens": 64_000,
                }
            )

        self.log(f"Available models: {models}", level="debug")

        return models

    def pipes(self) -> list[dict[str, Any]]:
        """Return model list for host"""
        self.log("pipes called", level="debug")
        return self.get_anthropic_models()

    async def thinking_status(
        self,
        status: Literal["started", "completed"],
        emitter: Callable[[Any], Awaitable[None]],
    ):
        """Emit thinking status events."""

        current_time = time.time()

        if status == "started":
            await emit_status(
                description="thinking...",
                emitter=emitter,
                status="in_progress",
            )
            self.thinking_start_time = current_time
        else:
            if not hasattr(self, "thinking_start_time"):
                raise RuntimeError("thinking_start_time not set")
            thinking_duration = current_time - self.thinking_start_time
            await emit_status(
                description=f"thought for {thinking_duration:.1f}s",
                emitter=emitter,
                status="complete",
            )

    def _process_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process messages for Anthropic format."""
        processed: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            parts: list[dict[str, Any]] = []

            if isinstance(content, list):
                for part in content:
                    try:
                        part_type = part.get("type")
                        if part_type == "text":
                            parts.append({"type": "text", "text": part.get("text", "")})
                        elif part_type == "image_url":
                            parts.append(self._process_image(part))
                        elif part_type == "pdf_url":
                            pdf_part = self._process_pdf(part)
                            if pdf_part:
                                parts.append(pdf_part)
                        elif part_type == "tool_result":
                            # Handle tool results from function calls
                            parts.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": part.get("tool_use_id"),
                                    "content": part.get("content", ""),
                                }
                            )
                        elif part_type == "tool_use":
                            # Handle tool use blocks (assistant's tool calls)
                            parts.append(
                                {
                                    "type": "tool_use",
                                    "id": part.get("id"),
                                    "name": part.get("name"),
                                    "input": part.get("input", {}),
                                }
                            )
                    except Exception as e:
                        self.log(f"Content part skipped: {e}", "warning")
            else:
                parts.append({"type": "text", "text": str(content or "")})

            processed.append({"role": role, "content": parts})

        return processed

    def _process_image(self, image_data: dict[str, Any]) -> dict[str, Any]:
        """Process image data for Anthropic format."""
        url = image_data.get("image_url", {}).get("url")
        if not url:
            raise ValueError("Missing image URL")

        if url.startswith("data:image"):
            header, b64data = url.split(",", 1)
            mime = header.split(":", 1)[1].split(";", 1)[0]

            if mime not in SUPPORTED_IMAGE_TYPES:
                raise ValueError(f"Unsupported image type: {mime}")

            size = len(b64data) * 3 // 4
            if size > MAX_IMAGE_SIZE:
                raise ValueError("Image exceeds size limit")

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": b64data,
                },
            }
        else:
            # URL validation
            resp = requests.head(url, allow_redirects=True, timeout=5)
            content_length = int(resp.headers.get("content-length", 0))
            if content_length > MAX_IMAGE_SIZE:
                raise ValueError("Image at URL exceeds size limit")

            return {
                "type": "image",
                "source": {"type": "url", "url": url},
            }

    def _process_pdf(self, pdf_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process PDF data for Anthropic format."""
        url = pdf_data.get("pdf_url", {}).get("url")
        if not url:
            return None

        if url.startswith("data:application/pdf"):
            _, b64data = url.split(",", 1)
            size = len(b64data) * 3 // 4
            if size > MAX_PDF_SIZE:
                raise ValueError("PDF exceeds size limit")

            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": b64data,
                },
            }
        else:
            resp = requests.head(url, allow_redirects=True, timeout=5)
            content_length = int(resp.headers.get("content-length", 0))
            if content_length > MAX_PDF_SIZE:
                raise ValueError("PDF at URL exceeds size limit")

            return {
                "type": "document",
                "source": {"type": "url", "url": url},
            }

    async def anthropic_sdk_wrapper(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system_message: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        stop: Optional[list[str]],
        tools: Optional[list[dict[str, Any]]],
        tool_choice: Optional[dict[str, Any]],
        stream: bool,
        thinking_config: Optional[dict[str, Any]],
    ) -> Union[str, AsyncIterator[dict[str, Any]]]:
        """Wrapper method that injects API key and calls the standalone Anthropic SDK function.

        This is a thin wrapper around query_anthropic_sdk() that provides the API key from valves.
        """
        return await query_anthropic_sdk(
            api_key=self.valves.ANTHROPIC_API_KEY,
            model=model,
            messages=messages,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            thinking_config=thinking_config,
        )

    async def execute_tool(
        self,
        tool_call: dict[str, Any],
        tools: dict[str, Any],
    ) -> Union[dict[str, Any], str]:
        """Execute a single tool call and return the result."""
        tool_name = tool_call["function"]["name"]

        try:
            tool = tools.get(tool_name)
            if not tool:
                raise ValueError(f"Tool '{tool_name}' not found")

            arguments_str = tool_call["function"]["arguments"]
            if arguments_str:
                parsed_args = json.loads(arguments_str)
            else:
                parsed_args = {}

            # Execute the tool
            result = await tool["callable"](**parsed_args)

            return json.dumps(result)
        except json.JSONDecodeError:
            return {
                "tool_call": tool_call,
                "result": None,
                "error": f"Failed to parse arguments for tool '{tool_name}'",
            }
        except Exception as e:
            self.log(f"Error executing tool '{tool_name}': {e}", "error")
            return {
                "tool_call": tool_call,
                "result": None,
                "error": f"Error executing tool '{tool_name}': {str(e)}",
            }

    async def query_openai_sdk(
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
        tool_choice: Optional[Literal["none", "auto", "required"]] = None,
        thinking_config: Optional[dict[str, Any]] = None,
        host_tools: Optional[dict[str, Any]] = None,
    ) -> Union[AsyncIterator, str]:
        """Query OpenAI SDK with automatic tool calling loop."""

        if model is None:
            model = "claude-sonnet-4-latest"
        if max_tokens is None:
            max_tokens = 16_000
        if temperature is None:
            temperature = 1
        if stop is None:
            stop = []
        if tools is None:
            tools = []
        if tool_choice is None:
            tool_choice = "auto"
        if stream is None:
            stream = True

        client = OpenAI(
            api_key=self.valves.ANTHROPIC_API_KEY,
            base_url="https://api.anthropic.com/v1/",
        )

        extra_body = {}
        if thinking_config:
            extra_body["thinking"] = thinking_config

        if stream:
            return self._openai_sdk_stream_handler(
                client=client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                extra_body=extra_body if extra_body else None,
                host_tools=host_tools,
                event_emitter=event_emitter,
            )
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or None,
                tools=tools,  # type: ignore
                tool_choice=tool_choice,
                stream=False,
                extra_body=extra_body if extra_body else None,
            )

            await self.thinking_status("completed", emitter=event_emitter)
            return response.choices[0].message.content

    async def _openai_sdk_stream_handler(
        self,
        client: OpenAI,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        stop: Optional[list[str]],
        tools: Optional[list[dict[str, Any]]],
        extra_body: Optional[dict[str, Any]],
        host_tools: Optional[dict[str, Any]],
        event_emitter: Callable[[Any], Awaitable[None]],
        tool_choice: Literal["none", "auto", "required"],
    ) -> AsyncIterator:
        """Stream responses with automatic tool calling loop."""

        first_iteration = True
        first_iteration_after_tool_call = False

        while True:
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or None,
                tools=tools,  # type: ignore
                tool_choice=tool_choice,
                stream=True,
                extra_body=extra_body,
            )

            collected_content = ""
            collected_tool_calls = []
            finish_reason = None

            for chunk in response:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                if delta.content:
                    collected_content += delta.content
                    if first_iteration_after_tool_call:
                        first_iteration_after_tool_call = False
                        yield "\n\n---\n\n"
                    if first_iteration:
                        first_iteration = False
                        await self.thinking_status("completed", emitter=event_emitter)
                    yield delta.content

                # Collect tool calls as they stream
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        while len(collected_tool_calls) <= tc.index:
                            collected_tool_calls.append(
                                {
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": None, "arguments": ""},
                                }
                            )

                        if tc.id:
                            collected_tool_calls[tc.index]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                collected_tool_calls[tc.index]["function"][
                                    "name"
                                ] = tc.function.name
                            if tc.function.arguments:
                                collected_tool_calls[tc.index]["function"][
                                    "arguments"
                                ] += tc.function.arguments

            if finish_reason == "tool_calls":
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if collected_content:
                    assistant_msg["content"] = collected_content
                if collected_tool_calls:
                    assistant_msg["tool_calls"] = collected_tool_calls
                messages.append(assistant_msg)

                # Execute tools
                if not host_tools:
                    raise RuntimeError("Host tools are required for tool execution")

                for tool_call in collected_tool_calls:
                    tool_result = await self.execute_tool(tool_call, host_tools)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": (tool_result),
                        }
                    )
                first_iteration_after_tool_call = True

            else:
                break

    def setup_params(
        self, body: dict[str, Any]
    ) -> tuple[str, int, Optional[dict[str, Any]]]:
        """Setup and validate model, max_tokens, and thinking config.

        Returns:
            tuple: (model_name, max_tokens, thinking_config)
        """
        model_full = body.get("model", "anthropic/claude-sonnet-4-5-20250929")

        # Clean model name
        model = model_full.split("/", 1)[1] if "/" in model_full else model_full
        thinking_requested = model.endswith("-thinking")
        if thinking_requested:
            model = model.replace("-thinking", "")

        # Handle reasoning effort for Claude 4 models
        reasoning_effort = body.get("reasoning_effort", "none")
        budget_tokens = REASONING_EFFORT_BUDGET_TOKEN_MAP.get(reasoning_effort)

        # Allow users to input an integer value representing budget tokens
        if (
            not budget_tokens
            and reasoning_effort is not None
            and reasoning_effort not in REASONING_EFFORT_BUDGET_TOKEN_MAP.keys()
        ):
            try:
                budget_tokens = int(reasoning_effort)
            except ValueError:
                self.log(
                    f"Failed to convert reasoning effort to int: {reasoning_effort}",
                    "warning",
                )
                budget_tokens = None

        # Calculate max_tokens for thinking if needed
        max_tokens = body.get("max_tokens", 64_000)
        thinking_config = None

        if thinking_requested or budget_tokens:
            if budget_tokens:
                # Check combined token limit
                combined_tokens = budget_tokens + max_tokens

                if combined_tokens > MAX_COMBINED_TOKENS:
                    error_msg = f"Error: Combined tokens (budget_tokens {budget_tokens} + max_tokens {max_tokens} = {combined_tokens}) exceeds the maximum limit of {MAX_COMBINED_TOKENS}"
                    self.log(error_msg, "error")
                    raise ValueError(error_msg)

                max_tokens = combined_tokens
                thinking_config = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
            else:
                # Default thinking for -thinking models
                thinking_config = {
                    "type": "enabled",
                    "budget_tokens": 32_000,
                }

        return model, max_tokens, thinking_config

    async def auto_claude(
        self,
        body: dict[str, Any],
        event_emitter: Callable[[Any], Awaitable[None]],
        host_tools: Optional[dict[str, Any]] = None,
    ):
        """Process a single message and yield response chunks."""

        # Setup model, max_tokens, and thinking config
        model, max_tokens, thinking_config = self.setup_params(body)

        # Process messages for Anthropic format
        processed_messages = self._process_messages(body.get("messages", []))

        # Here we will also decide whether to use the native Anthropic SDK or the OpenAI SDK for processing - FUTURE TASK

        return await self.query_openai_sdk(
            model=model,
            event_emitter=event_emitter,
            messages=processed_messages,
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
    ) -> Union[str, AsyncIterator[str]]:
        """Main pipeline entry point."""

        self.log(f"pipe called with body: {body}", level="debug")

        if not __event_emitter__:
            raise RuntimeError("Event emitter is required")
        await self.thinking_status("started", emitter=__event_emitter__)

        # If host is trying to handle tools, opt out and do it ourselves
        if __task__ == "function_calling":
            return ""

        if not self.valves.ANTHROPIC_API_KEY:
            raise RuntimeError("Missing ANTHROPIC_API_KEY in valves")

        return await self.auto_claude(
            body=body,
            event_emitter=__event_emitter__,
            host_tools=__tools__,
        )
