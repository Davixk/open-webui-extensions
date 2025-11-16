"""
title: Memory Tools
author: @nokodo
description: native tools to interact with the built-in memory system
author_email: nokodo@nokodo.net
author_url: https://nokodo.net
funding_url: https://ko-fi.com/nokodo
repository_url: https://nokodo.net/github/open-webui-extensions
version: 1.0.0-alpha1
required_open_webui_version: >= 0.5.0
license: see extension documentation file `memory_tools.md` (License section) for the licensing terms.
"""

import json
import logging
from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import Request
from open_webui.main import app as webui_app
from open_webui.models.users import Users
from open_webui.retrieval.vector.main import SearchResult
from open_webui.routers.memories import QueryMemoryForm, query_memory
from pydantic import BaseModel, Field

LogLevel = Literal["debug", "info", "warning", "error"]


class Memory(BaseModel):
    """Single memory entry with metadata."""

    memory_id: str = Field(..., description="ID of the memory")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    content: str = Field(..., description="Content of the memory")
    similarity_score: Optional[float] = Field(
        None,
        description="Similarity score (0 to 1 - higher is more similar to query)",
    )


def log(message: Any, level: LogLevel = "info"):
    """Simple logger utility."""

    logger = logging.getLogger("memory_tools")
    getattr(logger, level, logger.info)(message)


def searchresults_to_memories(results: SearchResult) -> list[Memory]:
    """Convert SearchResult to list of Memory objects."""
    memories = []

    if not results.ids or not results.documents or not results.metadatas:
        return memories

    for batch_idx, (ids_batch, docs_batch, metas_batch) in enumerate(
        zip(results.ids, results.documents, results.metadatas)
    ):
        distances_batch = results.distances[batch_idx] if results.distances else None

        for doc_idx, (mem_id, content, meta) in enumerate(
            zip(ids_batch, docs_batch, metas_batch)
        ):
            if not meta:
                continue

            created_at = datetime.fromtimestamp(
                meta.get("created_at", meta.get("timestamp", 0))
            )
            updated_at = datetime.fromtimestamp(
                meta.get("updated_at", meta.get("created_at", meta.get("timestamp", 0)))
            )

            # Extract similarity score if available
            similarity_score = None
            if distances_batch is not None and doc_idx < len(distances_batch):
                similarity_score = round(distances_batch[doc_idx], 3)

            mem = Memory(
                memory_id=mem_id,
                created_at=created_at,
                updated_at=updated_at,
                content=content,
                similarity_score=similarity_score,
            )
            memories.append(mem)

    return memories


class Tools:
    class Valves(BaseModel):
        memory_max_k: int = Field(
            default=15,
            ge=1,
            le=50,
            description="maximum number of memories to retrieve",
        )
        minimum_similarity_threshold: float = Field(
            default=0.65,
            ge=0.0,
            le=1.0,
            description="minimum similarity score to include memories (0-1)",
        )
        show_status: bool = Field(
            default=True,
            description="show status messages during memory retrieval",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.emitter = None
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_memories",
                    "description": "Search and retrieve memories from your long term memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Vector search query",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    async def emit_status(
        self,
        description: str,
        status: str = "complete",
        done: Optional[bool] = None,
        error: Optional[bool] = None,
    ):
        """Emit status update if emitter is available and show_status is enabled."""
        if not self.emitter or not self.valves.show_status:
            return

        await self.emitter(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "status": status,
                    "done": (
                        done if done is not None else status in ("complete", "error")
                    ),
                    "error": error if error is not None else status == "error",
                },
            }
        )

    async def retrieve_memories(
        self,
        query: str,
        __event_emitter__: Any = None,
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Retrieve memories from the user's memory store based on a query.

        Args:
            query: Natural language query for memory retrieval
            limit: Maximum number of memories to retrieve
            __event_emitter__: Event emitter for status updates
            __user__: User information dictionary

        Returns:
            JSON string containing retrieved memories with metadata
        """
        self.emitter = __event_emitter__

        if __user__ is None:
            return json.dumps(
                {"error": "user information is required to retrieve memories"}
            )

        # Get user model
        user = Users.get_user_by_id(__user__["id"])
        if not user:
            return json.dumps({"error": "user not found"})

        await self.emit_status(
            description="recalling memories",
            status="in_progress",
        )

        try:
            # Query the memory store
            results = await query_memory(
                request=Request(scope={"type": "http", "app": webui_app}),
                form_data=QueryMemoryForm(content=query, k=self.valves.memory_max_k),
                user=user,
            )

            # Convert results to Memory objects
            memories = searchresults_to_memories(results) if results else []
            log(memories)

            # Filter by similarity threshold if configured
            memories = [
                mem
                for mem in memories
                if mem.similarity_score is not None
                and mem.similarity_score >= self.valves.minimum_similarity_threshold
            ]

            response = {
                "query": query,
                "relevant_memories": len(memories),
                "memories": [
                    memory.model_dump(
                        exclude={"memory_id", "similarity_score"}, mode="json"
                    )
                    for memory in memories
                ],
                "message": (
                    "found relevant memories matching your query"
                    if memories
                    else "no relevant memories found matching your query"
                ),
            }

            await self.emit_status(
                description=f"recalled {len(memories)} memories",
                status="complete",
            )

            return json.dumps(response)

        except Exception as e:
            error_msg = "failed to retrieve memories"
            await self.emit_status(
                description=error_msg,
                status="error",
            )
            return json.dumps({"message": error_msg, "error": str(e)})
