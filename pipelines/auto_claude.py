"""
title: Auto Anthropic
author: @nokodo
description: clean, plug and play Claude manifold pipeline with support for all the latest features from Anthropic
version: 0.4.2
required_open_webui_version: ">= 0.5.0"
license: see extension documentation file `auto_claude.md` (License section) for the licensing terms.
repository_url: https://nokodo.net/github/open-webui-extensions
funding_url: https://ko-fi.com/nokodo
"""

from __future__ import annotations

import base64
import copy
import io
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
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

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
    "claude-opus-4-5": {"max_output_tokens": 64_000, "supports_thinking": True},
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
CLAUDE_MAX_IMAGE_SIZE_MB = 4.0
CLAUDE_IMAGE_FORMAT = "WEBP"

LogLevel = Literal["debug", "info", "warning", "error"]
ImageFormat = Literal["WEBP", "JPEG", "PNG"]


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


def b64_url_to_image(b64_url: str) -> Image.Image:
    """
    Converts a base64 data URL to a PIL Image.

    Args:
        b64_url: Base64 data URL (e.g., "data:image/png;base64,iVBORw0KG...")

    Returns:
        PIL Image object with EXIF orientation applied
    """
    # Strip data URL prefix if present
    if "base64," in b64_url:
        b64_data = b64_url.split("base64,", 1)[1]
    else:
        b64_data = b64_url

    image_bytes = base64.b64decode(b64_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Apply EXIF orientation to prevent rotation issues with phone photos
    return ImageOps.exif_transpose(image) or image


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
        allow_assistant_images: bool = Field(
            default=False,
            description="allow image blocks in assistant messages (not yet supported by Anthropic API)",
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
                    "info": {"params": {"function_calling": "native"}},
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

    def ensure_image_under_size(
        self,
        image: Image.Image | str,
        max_size_mb: float = CLAUDE_MAX_IMAGE_SIZE_MB,
        image_format: ImageFormat = CLAUDE_IMAGE_FORMAT,
        initial_quality: int = 90,
        min_quality: int = 60,
    ) -> tuple[str, str]:
        """
        Ensures an image is under the specified size when base64 encoded.
        If original data is provided and already under limit, returns it unchanged.
        Otherwise converts to WEBP format for optimal compression.

        Args:
            image: PIL Image object to compress
            original_b64: Original base64 data (if available)
            original_format: Original media type (if available)
            max_size_mb: Maximum size in megabytes for base64 encoded result
            image_format: Target format for conversion (default WEBP)
            initial_quality: Starting quality (1-100)
            min_quality: Minimum acceptable quality before resizing

        Returns:
            tuple of (base64_string, media_type)
        """
        max_bytes = int(max_size_mb * 1024 * 1024)

        if isinstance(image, str):
            # Input is base64 data URL.
            original_input = image  # Keep original for fallback
            # Extract media type and base64 data
            if image.startswith("data:") and ";base64," in image:
                header, b64_data = image.split(",", 1)
                declared_media_type = header.split(";")[0].split(":")[1]
            else:
                declared_media_type = "application/octet-stream"
                b64_data = image

            # Calculate actual base64 string size (bytes)
            # The base64 string length IS the size that will be sent to the API
            size_bytes = len(b64_data)

            # Always verify the actual image format matches declared media type
            # by decoding and checking the real format
            try:
                image_bytes = base64.b64decode(b64_data)
                pil_image = Image.open(io.BytesIO(image_bytes))
                actual_format = pil_image.format  # e.g., "JPEG", "PNG", "WEBP", "GIF"

                # Map PIL format to media type
                format_to_media_type = {
                    "JPEG": "image/jpeg",
                    "PNG": "image/png",
                    "WEBP": "image/webp",
                    "GIF": "image/gif",
                }
                if actual_format:
                    actual_media_type = format_to_media_type.get(
                        actual_format, f"image/{actual_format.lower()}"
                    )
                else:
                    actual_media_type = "application/octet-stream"

                self.log(
                    f"Image format check: declared={declared_media_type}, actual={actual_media_type} "
                    f"(PIL format={actual_format}), size={size_bytes / (1024 * 1024):.2f} MB",
                    "debug",
                )

                if size_bytes <= max_bytes:
                    # Use the ACTUAL media type, not the declared one
                    if declared_media_type != actual_media_type:
                        self.log(
                            f"Media type mismatch: header says {declared_media_type} but image is actually {actual_media_type}. "
                            f"Using actual media type.",
                            "warning",
                        )
                    self.log(
                        f"Image already under limit ({size_bytes / (1024 * 1024):.2f} MB), skipping conversion",
                        "info",
                    )
                    return b64_data, actual_media_type

                # Need to compress - convert PIL image with EXIF orientation
                image = ImageOps.exif_transpose(pil_image) or pil_image

            except Exception as e:
                self.log(f"Failed to verify image format: {e}", "warning")
                # Fall back to converting the image anyway
                image = b64_url_to_image(original_input)

        # Need to compress - start timing
        compression_start = time.time()
        img = image.copy()
        quality = initial_quality
        compression_iterations = 0

        while True:
            compression_iterations += 1
            buffer = io.BytesIO()
            # Use method=3 for WEBP (good balance of speed/compression)
            # method=6 is extremely slow, method=3 is ~4x faster with similar quality
            img.save(buffer, format=image_format, quality=quality, method=3)
            size = buffer.tell()

            # Check base64 size (base64 is ~1.37x larger than raw bytes)
            estimated_b64_size = (size * 4 + 2) // 3

            if estimated_b64_size <= max_bytes:
                compression_time = time.time() - compression_start
                self.log(
                    f"Image compressed to {estimated_b64_size / (1024 * 1024):.2f} MB "
                    f"with quality={quality}, format={image_format} "
                    f"in {compression_time:.2f}s ({compression_iterations} iterations)"
                )
                b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return b64_string, f"image/{image_format.lower()}"

            # Try reducing quality first
            if quality > min_quality:
                quality -= 5
                continue

            # If quality is too low, resize the image
            new_width = int(img.width * 0.9)
            new_height = int(img.height * 0.9)

            if new_width < 100 or new_height < 100:
                # Image is too small, just return what we have
                compression_time = time.time() - compression_start
                self.log(
                    f"Image compression reached minimum size after {compression_time:.2f}s "
                    f"({compression_iterations} iterations), final size: {estimated_b64_size / (1024 * 1024):.2f} MB"
                )
                b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return b64_string, f"image/{image_format.lower()}"

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            quality = initial_quality

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

    def _convert_image_url_to_anthropic(
        self, image_block: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Convert OpenAI-style image_url block to Anthropic's image format."""
        image_url_obj = image_block.get("image_url")
        if not image_url_obj:
            return None

        url = (
            image_url_obj.get("url")
            if isinstance(image_url_obj, dict)
            else image_url_obj
        )
        if not url or not isinstance(url, str):
            return None

        # Handle base64 data URLs (e.g., "data:image/jpeg;base64,...")
        if url.startswith("data:"):
            try:
                # Convert to PIL Image and ensure it's under size limit
                compressed_data, media_type = self.ensure_image_under_size(image=url)

                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": compressed_data,
                    },
                }
            except Exception as e:
                self.log(f"Failed to process image: {e}", "warning")
                return None

        # Handle regular URLs
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            },
        }

    def _process_content_item(
        self, item: Any, allow_images: bool = True
    ) -> Optional[dict[str, Any]]:
        """Process a single content item (text, image, or image_url) into Anthropic format."""
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "text":
                return item
            elif item_type == "image":
                return item if allow_images else None
            elif item_type == "image_url":
                if not allow_images:
                    return None
                converted = self._convert_image_url_to_anthropic(item)
                return converted if converted else None
        elif isinstance(item, str):
            return {"type": "text", "text": item}
        return None

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
        # TODO: Remove this when Anthropic supports images in assistant messages
        allow_images = self.valves.allow_assistant_images

        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                processed = self._process_content_item(item, allow_images=allow_images)
                if processed:
                    blocks.append(processed)
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
                processed = self._process_content_item(item, allow_images=True)
                if processed:
                    blocks.append(processed)
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
                                self.log(
                                    f"TextBlock with initial_text: {repr(initial_text[:50])} "
                                    f"(after_tool_call={first_iteration_after_tool_call})",
                                    "debug",
                                )
                                if (
                                    first_iteration_after_tool_call
                                    and not first_iteration_with_text
                                ):
                                    self.log(
                                        "Yielding separator before initial_text",
                                        "debug",
                                    )
                                    yield "\n\n---\n\n"
                                    first_iteration_after_tool_call = False
                                if first_output:
                                    first_output = False
                                    if self.valves.ttft_as_thinking:
                                        await self.thinking_status(
                                            "completed", emitter=event_emitter
                                        )
                                if first_iteration_with_text:
                                    first_iteration_with_text = False
                                self.log(
                                    f"Yielding initial_text: {repr(initial_text[:50])}",
                                    "debug",
                                )
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
                            self.log(
                                f"TextDelta: {repr(delta.text[:50])} "
                                f"(after_tool_call={first_iteration_after_tool_call})",
                                "debug",
                            )
                            if first_iteration_after_tool_call:
                                self.log("Yielding separator from TextDelta", "debug")
                                yield "\n\n---\n\n"
                                first_iteration_after_tool_call = False
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
                            self.log(
                                f"Yielding delta.text: {repr(delta.text[:50])}", "debug"
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
        __metadata__: Optional[dict[str, Any]] = None,
    ) -> str | AsyncIterator[str]:
        self.log(f"native pipe called with body: {body}", level="debug")
        if not __event_emitter__:
            raise RuntimeError("event emitter is required")
        if self.valves.ttft_as_thinking:
            await self.thinking_status("started", emitter=__event_emitter__)
        if __task__ == "function_calling":
            return ""

        # Merge tools from metadata if present
        if __metadata__ and "tools" in __metadata__:
            metadata_tools = __metadata__["tools"]
            if isinstance(metadata_tools, dict):
                merged_tools = {**metadata_tools}
                if __tools__:
                    merged_tools.update(__tools__)
                __tools__ = merged_tools

        return await self.auto_claude(
            body=body,
            event_emitter=__event_emitter__,
            host_tools=__tools__,
        )
