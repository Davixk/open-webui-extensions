"""
title: Extended Thinking Toggle
author: @nokodo
description: Toggle extended thinking mode with configurable reasoning effort
version: 1.0.0-alpha1
required_open_webui_version: ">= 0.5.0"
license: see extension documentation file `extended_thinking.md` (License section) for the licensing terms.
"""

from typing import Any, Awaitable, Callable, Literal, Optional

from pydantic import BaseModel, Field


class Action:
    """Action to toggle extended thinking mode with configurable reasoning effort."""

    class Valves(BaseModel):
        reasoning_effort: Literal["minimal", "low", "medium", "high", "max"] = Field(
            default="medium",
            description="reasoning effort level: minimal, low, medium, high, max",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def action(
        self,
        body: dict[str, Any],
        __user__: Optional[dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[Any], Awaitable[Any]]] = None,
        __model__: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """toggle extended thinking mode."""

        reasoning_effort = self.valves.reasoning_effort

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "notification",
                    "data": {
                        "type": "success",
                        "content": f"reasoning effort set to: {reasoning_effort}",
                    },
                }
            )

        return {
            "content": f"💭 **extended thinking mode**\n\nreasoning effort: `{reasoning_effort}`\n\nthis will apply to your next message.",
        }
