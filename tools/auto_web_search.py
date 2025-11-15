"""
title: Auto Web Search
author: @nokodo
description: make native automated web searches using the built-in web search engine
author_email: nokodo@nokodo.net
author_url: https://nokodo.net
funding_url: https://ko-fi.com/nokodo
repository_url: https://nokodo.net/github/open-webui-extensions
version: 1.0.0-alpha1
required_open_webui_version: >= 0.6.0
requirements: aiohttp
license: see extension documentation file `auto_web_search.md` (License section) for the licensing terms.
"""

import json
from typing import Any, Literal, Optional, cast
from urllib.parse import urlparse

import aiohttp
from open_webui.main import Request, app
from open_webui.models.users import UserModel, Users
from open_webui.retrieval.utils import get_content_from_url
from open_webui.routers.retrieval import SearchForm, process_web_search
from pydantic import BaseModel, Field


async def emit_status(
    description: str,
    emitter: Any,
    status: Literal[
        "in_progress", "complete", "error", "web_search", "web_search_queries_generated"
    ] = "complete",
    extra_data: Optional[dict] = None,
    done: Optional[bool] = None,
    error: Optional[bool] = None,
):
    if not emitter:
        raise ValueError("Emitter is required to emit status updates")
    if extra_data is None:
        extra_data = {}

    if status in ("in_progress", "complete", "error"):
        extra_data["status"] = status
    else:
        extra_data["action"] = status
    """ if status == "web_search":
        status_key["action"] = "web_search"
    else:
        status_key["status"] = status """

    await emitter(
        {
            "type": "status",
            "data": {
                "description": description,
                "done": done if done is not None else status in ("complete", "error"),
                "error": error if error is not None else status == "error",
                **(extra_data or {}),
            },
        }
    )


async def get_request() -> Request:
    return Request(scope={"type": "http", "app": app})


class Tools:
    class Valves(BaseModel):
        SEARCH_MODE: Literal["native", "perplexica"] = Field(
            default="native",
            description="Search mode (native or perplexica)",
        )
        PERPLEXICA_BASE_URL: str = Field(
            default="http://host.docker.internal:3001",
            description="Base URL for the Perplexica API",
        )
        PERPLEXICA_OPTIMIZATION_MODE: Literal["speed", "balanced"] = Field(
            default="balanced",
            description="Search optimization mode (speed or balanced)",
        )
        PERPLEXICA_CHAT_MODEL: str = Field(
            default="gpt-5-chat-latest", description="Default chat model"
        )
        PERPLEXICA_EMBEDDING_MODEL: str = Field(
            default="bge-m3:latest", description="Default embedding model"
        )
        OLLAMA_BASE_URL: str = Field(
            default="http://host.docker.internal:11434",
            description="Base URL for Ollama API",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for factual information, current events, or specific topics. Only use this tool when a search query is explicitly needed or when the user asks for information that requires looking up current or factual data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_queries": {
                                "type": "array",
                                "description": "An array of search queries.",
                                "items": {
                                    "type": "string",
                                    "title": "Search Query",
                                    "description": "A search query can be anything from a simple search term to a complex question.",
                                },
                                "minItems": 1,
                                "maxItems": 5,
                            }
                        },
                        "required": ["search_queries"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_url_content",
                    "description": "Browse and retrieve the full content from any URL including webpages, articles, YouTube videos, and other online resources. Use this whenever you need to access content from a specific link.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to browse and retrieve content from.",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

    async def web_search(
        self,
        search_queries: list[str],
        __event_emitter__: Any = None,
        __user__: Optional[dict] = None,
    ) -> str:
        """Search the web for a query."""
        if __user__ is None:
            raise ValueError("User information is required")

        search_mode = self.valves.SEARCH_MODE
        user = Users.get_user_by_id(__user__["id"])
        if user is None:
            raise ValueError("User not found")

        if search_mode == "perplexica":
            return await perplexica_web_search(
                search_queries,
                base_url=self.valves.PERPLEXICA_BASE_URL,
                optimization_mode=self.valves.PERPLEXICA_OPTIMIZATION_MODE,
                chat_model=self.valves.PERPLEXICA_CHAT_MODEL,
                embedding_model=self.valves.PERPLEXICA_EMBEDDING_MODEL,
                emitter=__event_emitter__,
                user=user,
            )
        elif search_mode == "native":
            return await native_web_search(
                search_queries, emitter=__event_emitter__, user=user
            )
        else:
            raise ValueError(f"Unknown search mode: {search_mode}")

    async def fetch_url_content(
        self,
        url: str,
        __event_emitter__: Any = None,
        __user__: Optional[dict] = None,
    ) -> str:
        """Fetch content from a URL."""
        if __user__ is None:
            raise ValueError("User information is required")

        user = Users.get_user_by_id(__user__["id"])
        if user is None:
            raise ValueError("User not found")

        return await fetch_url(url, emitter=__event_emitter__, user=user)


async def fetch_url(url: str, emitter: Any, user: UserModel) -> str:
    """Fetch content from a URL using the native web loader."""
    try:
        # Extract domain name from URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc or parsed_url.path.split("/")[0]

        await emit_status(
            f"browsing {domain}",
            status="in_progress",
            emitter=emitter,
            done=False,
        )

        request = await get_request()
        content, docs = get_content_from_url(request, url)

        for doc in docs:
            metadata = doc.metadata or {}
            await emitter(
                {
                    "type": "citation",
                    "data": {
                        "document": [doc.page_content],
                        "metadata": [metadata],
                        "source": {
                            "name": metadata.get("title")
                            or metadata.get("source")
                            or url
                        },
                    },
                }
            )

        await emit_status(
            f"read webpage from {domain}",
            status="complete",
            emitter=emitter,
            extra_data={"url": url},
        )

        return json.dumps(
            {
                "status": "success",
                "url": url,
                "content": content,
                "documents": [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in docs
                ],
            }
        )

    except Exception as e:
        await emit_status(
            "failed to read webpage",
            status="error",
            emitter=emitter,
            error=True,
        )
        return json.dumps(
            {
                "status": "error",
                "url": url,
                "error": str(e),
            }
        )


async def native_web_search(
    search_queries: list[str], emitter: Any, user: UserModel
) -> str:
    """Search using the native search engine."""
    try:
        await emit_status(
            "searching the web",
            extra_data={"queries": search_queries},
            status="web_search_queries_generated",
            done=False,
            emitter=emitter,
        )

        form = SearchForm.model_validate({"queries": search_queries})
        result = await process_web_search(
            request=await get_request(), form_data=form, user=user
        )

        items = cast(list[dict[str, Any]], result["docs"])
        item_count = cast(int, result["loaded_count"])

        search_results = cast(
            list[dict[str, str]],
            [
                {
                    "source": item["metadata"]["source"],
                    "content": item["content"],
                }
                for item in items
            ],
        )

        if emitter:
            for sr in search_results:
                await emitter(
                    {
                        "type": "citation",
                        "data": {
                            "document": [sr["content"]],
                            "metadata": [{"source": sr["source"]}],
                            "source": {"name": sr["source"]},
                        },
                    }
                )

        await emit_status(
            f"searched {item_count} website{'s' if item_count != 1 else ''}",
            status="web_search",
            done=True,
            extra_data={"urls": [sr["source"] for sr in search_results]},
            emitter=emitter,
        )

        return json.dumps(
            {
                "status": "web search completed successfully!",
                "result_count": item_count,
                "results": search_results,
            }
        )

    except Exception as e:
        await emit_status(
            "encountered an error while searching the web",
            status="web_search",
            done=True,
            error=True,
            emitter=emitter,
        )
        return json.dumps(
            {
                "status": "web search failed",
                "error": str(e),
            }
        )


async def perplexica_web_search(
    search_queries: list[str],
    base_url: str,
    optimization_mode: str,
    chat_model: str,
    embedding_model: str,
    emitter: Any,
    user: UserModel,
) -> Any:
    """Search using the Perplexica API."""
    # fallback for legacy code
    query = search_queries[0]

    await emit_status(f"Initiating search for: {query}", emitter=emitter)

    # Fixed: Use proper nested structure like the working Pipe
    payload = {
        "focusMode": "webSearch",
        "optimizationMode": optimization_mode,
        "query": query,
        "chatModel": {
            "provider": "ollama",
            "name": chat_model,
        },
        "embeddingModel": {
            "provider": "ollama",
            "name": embedding_model,
        },
        "history": [],  # Changed from None to empty list
    }

    # Fixed: Clean up request body like the working Pipe
    payload = {k: v for k, v in payload.items() if v is not None}
    payload = {k: v for k, v in payload.items() if v != "default"}

    try:
        await emit_status(
            "Sending request to Perplexica API", status="in_progress", emitter=emitter
        )

        # Fixed: Use aiohttp instead of requests for proper async handling
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url.rstrip('/')}/api/search",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                result = await response.json()

        # Emit main content as citation
        if emitter:
            await emitter(
                {
                    "type": "citation",
                    "data": {
                        "document": [result["message"]],
                        "metadata": [{"source": "Perplexica Search"}],
                        "source": {"name": "Perplexica"},
                    },
                }
            )

        # Emit each source as a citation
        if result.get("sources") and emitter:
            for source in result["sources"]:
                await emitter(
                    {
                        "type": "citation",
                        "data": {
                            "document": [source["pageContent"]],
                            "metadata": [{"source": source["metadata"]["url"]}],
                            "source": {"name": source["metadata"]["title"]},
                        },
                    }
                )

        await emit_status(
            "search completed successfully",
            status="complete",
            emitter=emitter,
        )

        # Format response with citations
        response_text = f"{result['message']}\n\nSources:\n"
        response_text += "- Perplexica Search\n"
        for source in result.get("sources", []):
            response_text += (
                f"- {source['metadata']['title']}: {source['metadata']['url']}\n"
            )
        return response_text

    except Exception as e:
        error_msg = f"error performing search: {str(e)}"
        await emit_status(error_msg, status="error", emitter=emitter)
        return error_msg
