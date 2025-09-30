"""
title: Auto Anthropic
author: @nokodo
description: clean, plug and play Claude manifold pipeline with support for all the latest features from Anthropic
version: 0.1.0-alpha4
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
from open_webui.utils.misc import pop_system_message
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
                        if part.get("type") == "text":
                            parts.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            parts.append(self._process_image(part))
                        elif part.get("type") == "pdf_url":
                            pdf_part = self._process_pdf(part)
                            if pdf_part:
                                parts.append(pdf_part)
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

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Union[str, Generator[str, None, None], AsyncIterator[str]]:
        """Main pipeline entry point."""

        self.log(f"pipe called with body: {body}", level="debug")

        if not self.valves.ANTHROPIC_API_KEY:
            error_msg = "Error: Anthropic API key not configured"
            if __event_emitter__:
                await emit_status(error_msg, __event_emitter__, "error")
            return error_msg

        # Remove unnecessary keys
        for key in ["user", "chat_id", "title"]:
            body.pop(key, None)

        # Extract system message and parse model
        system_message, messages = pop_system_message(body.get("messages", []))
        model_full = body.get("model", "anthropic/claude-sonnet-4-5-20250929")

        # Clean model name
        model = model_full.split("/", 1)[1] if "/" in model_full else model_full
        thinking_requested = model.endswith("-thinking")
        if thinking_requested:
            model = model.replace("-thinking", "")

        processed_messages = self._process_messages(messages)

        # Build base payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": processed_messages,
            "max_tokens": body.get("max_tokens", 64_000),
            "temperature": body.get("temperature", 0.7),
            "stop_sequences": body.get("stop", []),
            "stream": body.get("stream", False),
        }

        # Add system message if present
        if system_message:
            payload["system"] = str(system_message)

        # Add optional parameters only if explicitly provided
        if "top_k" in body:
            payload["top_k"] = body["top_k"]

        # Handle top_p (mutually exclusive with temperature in some cases)
        if "top_p" in body:
            payload["top_p"] = body["top_p"]

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

        # Apply thinking configuration if requested or reasoning effort specified
        if thinking_requested or budget_tokens:
            if budget_tokens:
                # Check combined token limit
                max_tokens = payload.get("max_tokens", 64_000)
                combined_tokens = budget_tokens + max_tokens

                if combined_tokens > MAX_COMBINED_TOKENS:
                    error_msg = f"Error: Combined tokens (budget_tokens {budget_tokens} + max_tokens {max_tokens} = {combined_tokens}) exceeds the maximum limit of {MAX_COMBINED_TOKENS}"
                    self.log(error_msg, "error")
                    if __event_emitter__:
                        await emit_status(error_msg, __event_emitter__, "error")
                    return error_msg

                payload["max_tokens"] = combined_tokens
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
                # Thinking requires temperature 1.0 and doesn't support top_p, top_k
                payload["temperature"] = 1.0
                if "top_k" in payload:
                    del payload["top_k"]
                if "top_p" in payload:
                    del payload["top_p"]
            else:
                # Default thinking for -thinking models
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 32_000,
                }

        headers = self._build_headers()

        # Handle streaming vs non-streaming
        if body.get("stream", False):
            return self._stream_response(self.url, headers, payload)
        else:
            try:
                data, _ = await self._send_request(self.url, headers, payload)
                if not data:
                    return "Error: Empty response"

                # Extract text from response
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        return block.get("text", "")
                return ""

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
                    elif event_type == "content_block_stop":
                        if is_thinking:
                            yield "</think>"
                            is_thinking = False
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
