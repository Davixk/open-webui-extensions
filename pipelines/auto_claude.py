"""
title: Auto Anthropic
author: @nokodo
description: clean, plug and play Claude manifold pipeline with support for all the latest features from Anthropic
version: 0.1.0-alpha5
required_open_webui_version: ">= 0.5.0"
license: see extension documentation file `auto_claude.md` (License section) for the licensing terms.
repository_url: https://nokodo.net/github/open-webui-extensions
funding_url: https://ko-fi.com/nokodo
"""

import asyncio
import json
import logging
import os
import random
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Literal,
    Optional,
    Union,
)

import requests
from anthropic import Anthropic
from open_webui.utils.misc import pop_system_message
from openai import OpenAI
from pydantic import BaseModel, Field

LogLevel = Literal["debug", "info", "warning", "error"]

LOG_FORMAT = "[Auto Claude][{level}] {message}"
LOGGER_NAME = "open_webui.extensions.auto_claude"


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


# ============================================================================
# Pipeline Class
# ============================================================================


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

        logger = logging.getLogger(LOGGER_NAME)
        getattr(logger, level, logger.info)(message)

        print(LOG_FORMAT.format(level=level, message=message))

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
        """Return model list for Open WebUI."""
        self.log("pipes called", level="debug")
        return self.get_anthropic_models()

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

    def _build_headers(self) -> dict[str, str]:
        """Build request headers - minimal set for Claude 4."""
        return {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
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
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]],
    ) -> dict[str, Any]:
        """Execute a single tool call and return the result."""
        tool_name = tool_call["function"]["name"]

        try:
            tool = tools.get(tool_name)
            if not tool:
                raise ValueError(f"Tool '{tool_name}' not found")

            arguments_str = tool_call["function"]["arguments"]
            if arguments_str:
                parsed_args = json.loads(arguments_str)
                if __event_emitter__:
                    await emit_status(
                        f"Executing tool '{tool_name}' with arguments: {parsed_args}",
                        __event_emitter__,
                        "in_progress",
                    )
            else:
                parsed_args = {}

            # Execute the tool
            result = await tool["callable"](**parsed_args)

            return {
                "tool_call": tool_call,
                "result": json.dumps(result),
                "error": None,
            }
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
        """Query OpenAI SDK (pointing to Anthropic API) with support for both streaming and non-streaming"""
        client = OpenAI(
            api_key=self.valves.ANTHROPIC_API_KEY,
            base_url="https://api.anthropic.com/v1/",
        )

        # Process messages
        processed_messages = self._process_messages(messages)

        # Add system message
        if system_message:
            processed_messages = [
                {"role": "system", "content": system_message}
            ] + processed_messages

        # Build extra_body
        extra_body = {}
        if thinking_config:
            extra_body["thinking"] = thinking_config

        if stream:
            # Return async iterator for streaming
            return self._stream_openai_sdk(
                client=client,
                model=model,
                messages=processed_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                extra_body=extra_body if extra_body else None,
            )
        else:
            # Non-streaming: return complete response
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=processed_messages,  # type: ignore
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop or None,
                    tools=tools,  # type: ignore
                    tool_choice=tool_choice,  # type: ignore
                    stream=False,
                    extra_body=extra_body if extra_body else None,
                )

                # Handle tool calls
                if (
                    hasattr(response.choices[0].message, "tool_calls")
                    and response.choices[0].message.tool_calls
                ):
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in response.choices[0].message.tool_calls
                    ]
                    return json.dumps({"tool_calls": tool_calls})

                # Return message content
                return response.choices[0].message.content or ""

            except Exception as e:
                self.log(f"OpenAI SDK request failed: {e}", "error")
                return f"Error: {e}"

    async def _stream_openai_sdk(
        self,
        client: OpenAI,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        stop: Optional[list[str]],
        tools: Optional[list[dict[str, Any]]],
        tool_choice: Optional[dict[str, Any]],
        extra_body: Optional[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream response from OpenAI SDK and yield content chunks and tool calls."""
        stream = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or None,
            tools=tools,  # type: ignore
            tool_choice=tool_choice,  # type: ignore
            stream=True,
            extra_body=extra_body,
        )

        tool_calls_buffer: dict[int, dict[str, Any]] = {}

        for chunk in stream:
            self.log(f"Received chunk: {chunk}", "debug")

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Yield content
            if delta.content:
                yield {"type": "content", "content": delta.content}

            # Collect tool calls
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    idx = tool_call.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tool_call.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    if tool_call.id:
                        tool_calls_buffer[idx]["id"] = tool_call.id
                    if tool_call.function:
                        if tool_call.function.name:
                            tool_calls_buffer[idx]["function"][
                                "name"
                            ] = tool_call.function.name
                        if tool_call.function.arguments:
                            tool_calls_buffer[idx]["function"][
                                "arguments"
                            ] += tool_call.function.arguments

        # Yield tool calls at end if any
        if tool_calls_buffer:
            tool_calls_list = [
                tool_calls_buffer[i] for i in sorted(tool_calls_buffer.keys())
            ]
            yield {"type": "tool_calls", "tool_calls": tool_calls_list}

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __tools__: Optional[dict[str, Any]] = None,
        __task__: Optional[str] = None,
    ) -> Union[str, Generator[str, None, None], AsyncIterator[str]]:
        """Main pipeline entry point."""

        self.log(f"pipe called with body: {body}", level="debug")

        # If Open WebUI is trying to handle tools, opt out and do it ourselves
        if __task__ == "function_calling":
            yield ""
            return

        if not self.valves.ANTHROPIC_API_KEY:
            error_msg = "Error: Anthropic API key not configured"
            if __event_emitter__:
                await emit_status(error_msg, __event_emitter__, "error")
            yield error_msg
            return

        # Extract system message and parse model
        system_message, messages = pop_system_message(body.get("messages", []))

        # Convert system_message to string if it's a dict
        system_str = None
        if system_message:
            if isinstance(system_message, dict):
                system_str = system_message.get("content", str(system_message))
            else:
                system_str = str(system_message)

        model_full = body.get("model", "anthropic/claude-sonnet-4-5-20250929")

        # Clean model name
        model = model_full.split("/", 1)[1] if "/" in model_full else model_full
        thinking_requested = model.endswith("-thinking")
        if thinking_requested:
            model = model.replace("-thinking", "")

        # Handle reasoning effort for Claude 4 models (all our models support this)
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
                    if __event_emitter__:
                        await emit_status(error_msg, __event_emitter__, "error")
                    yield error_msg
                    return

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

        # Determine if we're streaming
        is_streaming = body.get("stream", False)

        # Tool calling loop - like the inspiration module
        while True:
            if __event_emitter__:
                await emit_status(
                    "Generating response...", __event_emitter__, "in_progress"
                )

            # Call the model using query_openai_sdk
            result = await self.query_openai_sdk(
                model=model,
                messages=body["messages"],
                system_message=system_str,
                max_tokens=max_tokens,
                temperature=(
                    body.get("temperature", 0.7) if not thinking_config else 1.0
                ),
                top_p=body.get("top_p") if not thinking_config else None,
                stop=body.get("stop", []),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                stream=is_streaming,
                thinking_config=thinking_config,
            )

            # Handle streaming response
            if is_streaming:
                tool_calls_made: list[dict[str, Any]] = []
                message_content = ""

                # Process streaming chunks
                async for chunk in result:  # type: ignore
                    if chunk.get("type") == "content":
                        message_content += chunk["content"]
                        yield chunk["content"]
                    elif chunk.get("type") == "tool_calls":
                        tool_calls_made = chunk["tool_calls"]
            else:
                # Non-streaming: result is already a complete string or JSON
                result_str = str(result)

                # Check if it's a tool call response
                try:
                    result_json = json.loads(result_str)
                    if "tool_calls" in result_json:
                        tool_calls_made = result_json["tool_calls"]
                        message_content = ""
                    else:
                        tool_calls_made = []
                        message_content = result_str
                        yield result_str
                except json.JSONDecodeError:
                    # Not JSON, just regular text
                    tool_calls_made = []
                    message_content = result_str
                    yield result_str

            # Add assistant message to body
            if message_content:
                body["messages"].append(
                    {
                        "role": "assistant",
                        "content": message_content,
                    }
                )

            # If no tool calls, we're done
            if not tool_calls_made:
                if __event_emitter__:
                    await emit_status("Done", __event_emitter__, "complete")
                break

            # If tools were requested but not provided, error
            if not __tools__:
                error_msg = "Tool calls requested but no tools provided"
                self.log(error_msg, "error")
                yield f"\n\nError: {error_msg}"
                break

            # Add tool calls to messages
            body["messages"].append(
                {
                    "role": "assistant",
                    "tool_calls": tool_calls_made,
                }
            )

            # Execute tools
            if __event_emitter__:
                await emit_status(
                    "Executing tools...", __event_emitter__, "in_progress"
                )

            tool_results = []
            for tool_call in tool_calls_made:
                result = await self.execute_tool(
                    tool_call, __tools__, __event_emitter__
                )
                tool_results.append(result)

                # Yield tool execution results for display
                if result["error"]:
                    yield f'\n\n<details><summary>Error executing {tool_call["function"]["name"]}</summary>\n{result["error"]}\n</details>\n\n'
                else:
                    yield f'\n\n<details><summary>Executed {tool_call["function"]["name"]}</summary>\nResult: {result["result"]}\n</details>\n\n'

            # Add tool results to messages
            for result in tool_results:
                body["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call"]["id"],
                        "content": (
                            result["error"] if result["error"] else result["result"]
                        ),
                    }
                )

            if __event_emitter__:
                await emit_status(
                    "Tool execution complete", __event_emitter__, "complete"
                )

        return

    async def _pipe_with_requests(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system_message: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        stop: list[str],
        tools: Optional[list[dict[str, Any]]],
        tool_choice: Optional[dict[str, Any]],
        stream: bool,
        thinking_config: Optional[dict[str, Any]],
    ) -> Union[str, Generator[str, None, None]]:
        """Handle request using requests library (legacy implementation)."""
        headers = self._build_headers()

        # Process messages for Anthropic format
        processed_messages = self._process_messages(messages)

        # Build Anthropic API payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": processed_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        # Add system message if present
        if system_message:
            payload["system"] = system_message

        # Add optional parameters
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if stop:
            payload["stop_sequences"] = stop

        # Convert tools from OpenAI format to Anthropic format if needed
        if tools:
            converted_tools = []
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "function":
                    func = tool.get("function", {})
                    converted_tools.append(
                        {
                            "name": func.get("name"),
                            "description": func.get("description", ""),
                            "input_schema": func.get("parameters", {}),
                        }
                    )
                else:
                    converted_tools.append(tool)
            payload["tools"] = converted_tools

            if tool_choice:
                payload["tool_choice"] = tool_choice

        # Add thinking configuration if present
        if thinking_config:
            payload["thinking"] = thinking_config

        # Handle streaming vs non-streaming
        if stream:
            return self._stream_response(self.url, headers, payload)
        else:
            try:
                data, _ = await self._send_request(self.url, headers, payload)
                if not data:
                    return "Error: Empty response"

                content_blocks = data.get("content", [])

                # Handle tool calls
                tool_use_blocks = [
                    b for b in content_blocks if b.get("type") == "tool_use"
                ]
                if tool_use_blocks:
                    tool_calls = [
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"]),
                            },
                        }
                        for block in tool_use_blocks
                    ]
                    return json.dumps({"tool_calls": tool_calls})

                # Handle thinking blocks
                thinking_blocks = [
                    b for b in content_blocks if b.get("type") == "thinking"
                ]
                thinking_text = ""
                if thinking_blocks and thinking_config:
                    thinking_text = (
                        f"<think>{thinking_blocks[0].get('thinking', '')}</think>\n\n"
                    )

                # Extract regular text
                text_blocks = [b for b in content_blocks if b.get("type") == "text"]
                response_text = text_blocks[0].get("text", "") if text_blocks else ""

                return thinking_text + response_text

            except Exception as e:
                self.log(f"Request failed: {e}", "error")
                return f"Error: {e}"

    def _stream_response(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> Generator[str, None, None]:
        """Handle streaming response with thinking token support."""
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=(3.05, 300)
            ) as resp:
                if resp.status_code != 200:
                    yield f"Error: HTTP {resp.status_code}: {resp.text}"
                    return

                is_thinking = False
                tool_use_buffer = None

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue

                    line = raw_line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue

                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")

                    if event_type == "content_block_start":
                        block = event.get("content_block", {})
                        block_type = block.get("type")
                        if block_type == "thinking":
                            is_thinking = True
                            yield "<think>"
                        elif block_type == "text":
                            is_thinking = False
                            yield block.get("text", "")
                        elif block_type == "tool_use":
                            # Initialize tool use tracking
                            tool_use_buffer = {
                                "id": block.get("id"),
                                "name": block.get("name"),
                                "input": "",
                            }

                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        delta_type = delta.get("type")
                        if is_thinking and delta_type == "thinking_delta":
                            # Stream thinking tokens
                            yield delta.get("thinking", "")
                        elif is_thinking and delta_type == "signature_delta":
                            # End of thinking, start of response
                            yield "\n</think>\n\n"
                            is_thinking = False
                        elif not is_thinking and delta_type == "text_delta":
                            # Stream regular text tokens
                            yield delta.get("text", "")
                        elif delta_type == "input_json_delta":
                            # Accumulate tool input
                            if tool_use_buffer:
                                tool_use_buffer["input"] += delta.get(
                                    "partial_json", ""
                                )

                    elif event_type == "content_block_stop":
                        if is_thinking:
                            yield "</think>"
                            is_thinking = False
                        elif tool_use_buffer:
                            # Emit complete tool call
                            try:
                                tool_call = {
                                    "id": tool_use_buffer["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tool_use_buffer["name"],
                                        "arguments": tool_use_buffer["input"],
                                    },
                                }
                                yield json.dumps({"tool_calls": [tool_call]})
                                tool_use_buffer = None
                            except Exception as e:
                                self.log(f"Error processing tool call: {e}", "error")

                    elif event_type == "message_stop":
                        break

        except Exception as e:
            yield f"Error: {e}"

    async def _send_request(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """Send request with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = await asyncio.to_thread(
                    requests.post,
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(3.05, 300),
                )

                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage", {})

                    # Cache metrics (for potential future use)
                    metrics = {
                        "cache_creation_input_tokens": usage.get(
                            "cache_creation_input_tokens", 0
                        ),
                        "cache_read_input_tokens": usage.get(
                            "cache_read_input_tokens", 0
                        ),
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                    }

                    return data, metrics

                elif resp.status_code in RETRY_STATUS_CODES:
                    delay = (2**attempt) + random.uniform(0, 1)
                    self.log(
                        f"Retrying after {resp.status_code}, attempt {attempt + 1}/{MAX_RETRIES}, delay {delay:.1f}s",
                        "warning",
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-retryable error
                    try:
                        return resp.json(), None
                    except Exception:
                        return {"error": {"message": resp.text}}, None

            except Exception as e:
                delay = (2**attempt) + random.uniform(0, 1)
                self.log(
                    f"Request exception: {e}, attempt {attempt + 1}/{MAX_RETRIES}, delay {delay:.1f}s",
                    "warning",
                )
                await asyncio.sleep(delay)

        self.log("Max retries exceeded", "error")
        return None, None
