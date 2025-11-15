"""
title: Auto Image
author: @nokodo
description: native image generation and editing tools using built-in image engines
author_email: nokodo@nokodo.net
author_url: https://nokodo.net
funding_url: https://ko-fi.com/nokodo
repository_url: https://nokodo.net/github/open-webui-extensions
version: 0.2.0
required_open_webui_version: >= 0.6.0
requirements:
license: see extension documentation file `auto_image.md` (License section) for the licensing terms.
"""

import json
from typing import Any, Literal, Optional

from open_webui.main import Request, app
from open_webui.models.users import Users
from open_webui.routers.images import (
    EditImageForm,
    GenerateImageForm,
    image_edits,
    image_generations,
)
from pydantic import BaseModel, Field


async def emit_status(
    description: str,
    emitter: Any,
    status: Literal["in_progress", "complete", "error"] = "complete",
    done: Optional[bool] = None,
    error: Optional[bool] = None,
):
    """emit status updates to the UI."""
    if not emitter:
        raise ValueError("emitter is required to emit status updates")

    await emitter(
        {
            "type": "status",
            "data": {
                "description": description,
                "done": done if done is not None else status in ("complete", "error"),
                "error": error if error is not None else status == "error",
                "status": status,
            },
        }
    )


async def emit_files(images: list[dict], emitter: Any):
    """Emit generated image files to the UI."""
    if not emitter:
        raise ValueError("Emitter is required to emit files")

    await emitter(
        {
            "type": "files",
            "data": {
                "files": [
                    {
                        "type": "image",
                        "url": image["url"],
                    }
                    for image in images
                ]
            },
        }
    )


async def get_request() -> Request:
    """Get a mock Request object for internal API calls."""
    return Request(scope={"type": "http", "app": app})


class Tools:
    class Valves(BaseModel):
        IMAGE_SIZE: str = Field(
            default="1024x1024",
            description="default image size (e.g., 512x512, 1024x1024)",
        )
        IMAGE_STEPS: Optional[int] = Field(
            default=None,
            description="number of generation steps (if supported by the engine)",
        )
        DEFAULT_N: int = Field(
            default=1,
            description="default number of images to generate per prompt",
            ge=1,
            le=10,
        )
        CHECK_CHAT_FILES: bool = Field(
            default=False,
            description="whether to check for chat files (images, etc.) in the current conversation context",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_image",
                    "description": "Generate images from text descriptions using the configured image generation engine. Use this tool when the user explicitly requests image generation or visualization of a concept.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "A detailed description of the image to generate. Be specific and descriptive.",
                            },
                            "negative_prompt": {
                                "type": "string",
                                "description": "Optional description of what should NOT appear in the image (if supported by the engine).",
                            },
                            "n": {
                                "type": "integer",
                                "description": "Number of images to generate (1-10). Defaults to 1.",
                                "minimum": 1,
                                "maximum": 10,
                            },
                            "size": {
                                "type": "string",
                                "description": "Image size in format WIDTHxHEIGHT (e.g., 512x512, 1024x1024). Omit unless user requests a specific size.",
                            },
                        },
                        "required": ["prompt"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_image",
                    "description": "Edit or modify existing images using AI. Use this tool when the user wants to modify, alter, or change an existing image based on a text prompt. The tool automatically accesses images from the current conversation context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "A detailed description of the edits to make to the image. Be specific about what should change.",
                            },
                            "image_index": {
                                "type": "integer",
                                "description": "Index of the user-uploaded image to edit (0 for first, -1 for most recent). Defaults to -1 (most recent).",
                            },
                            "negative_prompt": {
                                "type": "string",
                                "description": "Optional description of what should NOT appear in the edited image (if supported by the engine).",
                            },
                            "n": {
                                "type": "integer",
                                "description": "Number of edited variations to generate (1-10). Defaults to 1.",
                                "minimum": 1,
                                "maximum": 10,
                            },
                            "size": {
                                "type": "string",
                                "description": "Output image size in format WIDTHxHEIGHT (e.g., 512x512, 1024x1024). Omit unless user requests a specific size.",
                            },
                        },
                        "required": ["prompt"],
                    },
                },
            },
        ]

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        __event_emitter__: Any = None,
        __user__: Optional[dict] = None,
    ) -> str:
        """generate images from a text prompt using the native image generation engine."""
        if __event_emitter__:
            await emit_status(
                "creating image",
                status="in_progress",
                done=False,
                emitter=__event_emitter__,
            )

        try:
            if __user__ is None:
                raise ValueError("user information is required")

            user = Users.get_user_by_id(__user__["id"])
            if user is None:
                raise ValueError("user not found")

            form_data = GenerateImageForm(
                prompt=prompt,
                n=n if n is not None else self.valves.DEFAULT_N,
                negative_prompt=negative_prompt,
                size=size if size is not None else self.valves.IMAGE_SIZE,
            )

            images = await image_generations(
                request=await get_request(),
                form_data=form_data,
                user=user,
            )

            if images is None:
                raise ValueError("image generation returned no results")

            if __event_emitter__:
                await emit_status(
                    f"image{'s' if len(images) > 1 else ''} created",
                    status="complete",
                    done=True,
                    emitter=__event_emitter__,
                )
                await emit_files(images, emitter=__event_emitter__)

            return json.dumps(
                {
                    "status": "image generation completed successfully!",
                    "image_count": len(images),
                    "images": [
                        {
                            "url": image["url"],
                            "prompt": prompt,
                        }
                        for image in images
                    ],
                }
            )

        except Exception as e:
            if __event_emitter__:
                await emit_status(
                    f"failed to generate image: {str(e)}",
                    status="error",
                    done=True,
                    error=True,
                    emitter=__event_emitter__,
                )
            return json.dumps(
                {
                    "status": "image generation failed",
                    "error": str(e),
                }
            )

    async def edit_image(
        self,
        prompt: str,
        image_index: int = -1,
        negative_prompt: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        __event_emitter__: Any = None,
        __user__: Optional[dict] = None,
        __files__: Optional[list] = None,
        __messages__: Optional[list] = None,
    ) -> str:
        """edit an existing image using the configured image editing engine."""
        if __event_emitter__:
            await emit_status(
                "editing image",
                status="in_progress",
                done=False,
                emitter=__event_emitter__,
            )

        try:
            if __user__ is None:
                raise ValueError("user information is required")

            user = Users.get_user_by_id(__user__["id"])
            if user is None:
                raise ValueError("user not found")

            # Extract images from messages
            image_files = []

            # Check __files__ first (chat-level files)
            if self.valves.CHECK_CHAT_FILES and __files__:
                image_files.extend([f for f in __files__ if f.get("type") == "image"])

            # Then check messages for image_url content
            if __messages__:
                for message in reversed(__messages__):  # Start from most recent
                    if message.get("role") == "user":
                        content = message.get("content")
                        # Handle list content (OpenAI format with images)
                        if isinstance(content, list):
                            for item in content:
                                if item.get("type") == "image_url":
                                    image_url = item.get("image_url", {}).get("url")
                                    if image_url:
                                        image_files.append(
                                            {"type": "image", "url": image_url}
                                        )

            if not image_files:
                raise ValueError(
                    "no image files found in the conversation. Please attach an image first."
                )

            # Validate image_index (supports negative indexing)
            try:
                image_url = image_files[image_index].get("url")
            except IndexError:
                raise ValueError(
                    f"image index {image_index} is out of range. Only {len(image_files)} image(s) available."
                )

            form_data = EditImageForm(
                image=image_url,
                prompt=prompt,
                n=n if n is not None else self.valves.DEFAULT_N,
                negative_prompt=negative_prompt,
                size=size if size is not None else self.valves.IMAGE_SIZE,
            )

            images = await image_edits(
                request=await get_request(),
                form_data=form_data,
                user=user,
            )

            if images is None:
                raise ValueError("image editing returned no results")

            if __event_emitter__:
                await emit_status(
                    f"image{'s' if len(images) > 1 else ''} edited",
                    status="complete",
                    done=True,
                    emitter=__event_emitter__,
                )
                await emit_files(images, emitter=__event_emitter__)

            return json.dumps(
                {
                    "status": "image editing completed successfully!",
                    "image_count": len(images),
                    "images": [
                        {
                            "url": image["url"],
                            "prompt": prompt,
                        }
                        for image in images
                    ],
                }
            )

        except Exception as e:
            if __event_emitter__:
                await emit_status(
                    f"failed to edit image: {str(e)}",
                    status="error",
                    done=True,
                    error=True,
                    emitter=__event_emitter__,
                )
            return json.dumps(
                {
                    "status": "image editing failed",
                    "error": str(e),
                }
            )
