"""
title: Auto Memory
author: @nokodo
description: automatically identify and store valuable information from chats as Memories.
author_email: nokodo@nokodo.net
author_url: https://nokodo.net
repository_url: https://nokodo.net/github/open-webui-extensions
version: 1.1.0-alpha1
required_open_webui_version: >= 0.5.0
funding_url: https://ko-fi.com/nokodo
license: see extension documentation file `auto_memory.md` (License section) for the licensing terms.
"""

import asyncio
import json
import logging
import re
import threading
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from urllib.parse import urlparse

from fastapi import HTTPException, Request
from open_webui.main import app as webui_app
from open_webui.models.users import UserModel, Users
from open_webui.retrieval.vector.main import SearchResult
from open_webui.routers.memories import (
    AddMemoryForm,
    MemoryUpdateModel,
    QueryMemoryForm,
    add_memory,
    delete_memory_by_id,
    query_memory,
    update_memory_by_id,
)
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, create_model

LogLevel = Literal["debug", "info", "warning", "error"]

STRINGIFIED_MESSAGE_TEMPLATE = "-{index}. {role}: ```{content}```"


UNIFIED_SYSTEM_PROMPT = """\
You are maintaining a collection of Memories - individual "journal entries" or facts about a user, each automatically timestamped upon creation or update.

You will be provided with:
1. Recent messages from a conversation (displayed with negative indices; -1 is the most recent overall message)
2. Any existing related memories that might potentially be relevant

Your job is to determine what actions to take on the memory collection based on the User's **latest** message (-2).

<key_instructions>
## Instructions
1. Focus ONLY on the **User's most recent message** (-2). Older messages provide context but should not generate new memories unless explicitly referenced in the latest message.
2. Each Memory should represent **a single fact or statement**. Never combine multiple facts into one Memory.
3. When the User's latest message contradicts existing memories, **update the existing memory** rather than creating a conflicting new one.
4. If memories are exact duplicates or direct conflicts about the same topic, **consolidate them by updating or deleting** as appropriate.
5. **Link related Memories** by including brief references when relevant to maintain semantic connections.
6. Capture anything valuable for **personalizing future interactions** with the User.
7. Always **honor memory requests**, whether direct from the User ("remember this", "forget that", "update X") or implicit through the Assistant's commitment ("I'll remember that", "I'll keep that in mind"). Treat these as strong signals to store, update, or delete the referenced information.
8. Each memory must be **self-contained and understandable without external context.** Avoid ambiguous references like "it", "that", or "there" - instead, include the specific subject being referenced. For example, prefer "User's new TV broke" over "It broke".
9. Be alert to **sarcasm, jokes, and non-literal language.** If the User's statement appears to be hyperbole, sarcasm, or non-literal rather than a factual claim, do not store it as a memory.
10. When determining which memory is "most recent" for conflict resolution, **refer to the `created_at` or `update_at` timestamps** from the existing memories.
</key_instructions>

<what_to_extract>
## What you WANT to extract
- Personal preferences, opinions, and feelings
- Long-term personal information (likely true for months/years)
- Future-oriented statements ("from now on", "going forward")
- Direct memory requests ("remember that", "note this", "forget that")
- Hobbies, interests, skills
- Important life details (job, education, relationships, location)
- Long term goals, plans, aspirations
- Recurring patterns or habits
- Strong likes/dislikes affecting future conversations
</what_to_extract>

<what_not_to_extract>
## What you do NOT want to extract
- User/assistant names (already in profile)
- User gender, age and birthdate (already in profile)
- ANY kind of short-term or ephemeral information that is unlikely to be relevant in future conversations
- Information the assistant confirms is already known
- Content from translation/rewrite/summarization/similar tasks ("Please help me write my essay about x")
- Trivial observations or fleeting thoughts
- Temporary activities
- Sarcastic remarks or obvious jokes
- Non-literal statements or hyperbole
</what_not_to_extract>

<actions_to_take>
Based on your analysis, return a list of actions:

**ADD**: Create new memory when:
- New information not covered by existing memories
- Distinct facts even if related to existing topics
- User explicitly requests to remember something

**UPDATE**: Modify existing memory when:
- User provides updated/corrected information about the same fact
- Consolidating small, inseparable or closely related facts into one memory
- User explicitly asks to update something
- New information refines but doesn't fundamentally change existing memory

**DELETE**: Remove existing memory when:
- User explicitly requests to forget something
- User's statement directly contradicts an existing memory
- Consolidating memories (update the oldest, delete the rest)
- Memory is completely obsolete due to new information
- Duplicate memories exist (keep oldest based on `created_at` timestamp)

When updating or deleting, ONLY use the memory ID from the related memories list.
</actions_to_take>

<consolidation_rules>
**Core Principle**: Default to keeping memories separate and granular for precise retrieval. Only consolidate when it meaningfully improves memory quality and coherence.

**When to CONSOLIDATE** (merge existing memories):

- **Exact Duplicates** - Same fact, different wording
    - Action: Delete the newer duplicate, keep the oldest (based on `created_at` timestamp)
    - Example: "User prefers Python for scripting" + "User likes Python for scripting tasks" → Keep oldest, delete duplicate

- **Direct Conflicts** - Contradictory facts about the same subject
    - Action: Update the older memory to reflect the latest information, or delete if completely obsolete
    - Example: "User lives in San Francisco" conflicts with "User moved to Mountain View" → Update or delete old info

- **Inseparable Facts** - Multiple facts about the same entity that would be incomplete or confusing if retrieved separately
    - Action: Merge into the oldest memory as a single self-contained statement, then delete the redundant memories
    - Test: Would retrieving one fact without the other create confusion or require additional context?
    - Example: "User's cat is named Luna" + "User's cat is a Siamese" → "User has a Siamese cat named Luna"
    - Counter-example: "User works at Google" + "User started at Google in 2023" → Keep separate (start date is distinct from employment)

- **Small, better retrieved together** - Closely related facts that enhance understanding when combined
    - Action: Merge into the oldest memory, delete the others
    - Test: Would I prefer to retrieve these facts together every time, rather than separately?
    - Example: "User loves Italian food" + "User loves Indian food" → "User loves Italian and Indian food"

**When to keep SEPARATE** (or split if wrongly combined):

Facts should remain separate when they represent distinct, independently-retrievable information:

- **Similar but distinct facts** - Related information representing different aspects or time periods
    - Example: "User works at Google" vs "User got promoted to team lead" (employment vs career progression)
  
- **Past events as journal entries** - Historical facts that provide temporal context
    - Example: "User bought a Samsung TV" and "User's Samsung TV broke" (separate events in time)

- **Related but separable facts** - Facts about the same topic that are meaningful independently
    - Example: "User loves dogs" vs "User has a golden retriever named Max" (general preference vs specific pet)

- **Too long or complex** - Merging would create an overly long memory that contains too many distinct facts

If an existing memory wrongly combines separable facts: UPDATE the existing memory to contain one fact (preserves timestamp), then ADD new memories for the other facts. Deleting the original would lose the timestamp.

**Guiding Question**: If vector search retrieves only one of these memories, would the user experience be degraded? If yes, consider merging. If no, keep separate.
</consolidation_rules>

<examples>
**Example 1 - Store new memories when no related found**
Conversation:
-2. user: ```I work as a senior data scientist at Tesla and my favorite programming language is Rust```
-1. assistant: ```That's impressive! Working at Tesla must be exciting, and Rust is a great choice for systems programming```

Related Memories:
[
  {"mem_id": "1", "created_at": "2024-01-05T10:00:00", "update_at": "2024-01-05T10:00:00", "content": "User enjoys electric vehicles"},
  {"mem_id": "2", "created_at": "2024-02-10T14:00:00", "update_at": "2024-02-10T14:00:00", "content": "User has experience with Python and data analysis"},
  {"mem_id": "3", "created_at": "2024-01-20T09:30:00", "update_at": "2024-01-20T09:30:00", "content": "User likes reading science fiction novels"}
]

**Analysis**
- Existing memories might be tangentially related (electric vehicles/Tesla, data analysis) but don't actually cover the specific facts mentioned
- User provides two distinct new facts: job/company and programming preference
- Each should be stored as a separate new memory

Output:
{
  "actions": [
    {"action": "add", "content": "User works as a senior data scientist at Tesla"},
    {"action": "add", "content": "User's favorite programming language is Rust"}
  ]
}

**Example 2 - Consolidate similar memories while retaining context**
Conversation:
-2. user: ```Actually I prefer TypeScript over JavaScript for frontend work these days```
-1. assistant: ```TypeScript's type safety definitely makes frontend development more maintainable!```

Related Memories:
[
  {"mem_id": "123", "created_at": "2024-01-15T10:00:00", "update_at": "2024-01-15T10:00:00", "content": "User likes JavaScript for web development"},
  {"mem_id": "456", "created_at": "2024-02-20T14:30:00", "update_at": "2024-02-20T14:30:00", "content": "User prefers JavaScript for frontend projects"},
  {"mem_id": "789", "created_at": "2024-03-01T09:00:00", "update_at": "2024-03-01T09:00:00", "content": "User is learning React"}
]

**Analysis**
- Two existing similar memories about JavaScript preference
- User said they now prefer TypeScript, but it doesn't mean they don't *like* JavaScript anymore
- Update one memory to reflect the new preference, leave all other memories untouched

Output:
{
  "actions": [
    {"action": "update", "id": "456", "new_content": "User prefers TypeScript for frontend work"}
  ]
}

**Example 3 - Delete conflicting memory while retaining others**
Conversation:
-2. user: ```I'm joking! I didn't actually buy the iPhone!```
-1. assistant: ```Ahh, you got me there! No worries.```

Related Memories:
[
  {"mem_id": "789", "created_at": "2024-03-01T09:00:00", "update_at": "2024-03-01T09:00:00", "content": "User just bought a new iPhone"},
  {"mem_id": "012", "created_at": "2024-03-02T11:00:00", "update_at": "2024-03-02T11:00:00", "content": "User likes Apple products"},
  {"mem_id": "345", "created_at": "2024-03-02T11:00:00", "update_at": "2024-03-02T11:00:00", "content": "User is considering buying a new iPad"}
]

**Analysis**
- User negates a previous statement about buying an iPhone
- We should delete the memory about the iPhone purchase
- The other memories about liking Apple products and considering an iPad remain valid

Output:
{
  "actions": [
    {"action": "delete", "id": "789"}
  ]
}

**Example 4 - Handling multiple updates while retaining context**
Conversation:
-4. user: ```I'm thinking of switching from my current role```
-3. assistant: ```What's motivating you to consider a change?```
-2. user: ```Well, I got promoted to team lead last month, but I'm also interviewing at Google next week. The commute would be better since I just moved to Mountain View```
-1. assistant: ```Congratulations on the promotion! That's interesting timing with the Google interview```

Related Memories:
[
  {"mem_id": "345", "created_at": "2024-02-15T10:00:00", "update_at": "2024-02-15T10:00:00", "content": "User lives in San Francisco"},
  {"mem_id": "678", "created_at": "2024-01-10T08:00:00", "update_at": "2024-01-10T08:00:00", "content": "User works as a software engineer"}
]

**Analysis**
- User reveals: promoted to team lead (updates role), moved to Mountain View (conflicts with SF), interviewing at Google (new info)
- We don't want to forget any of the user's life details, unless there is a conflict. So we create a new memory, and update the legacy ones.
- Add new memory about Google interview as it's distinct future event

Output:
{
  "actions": [
    {"action": "update", "id": "345", "new_content": "User used to live in San Francisco"},
    {"action": "update", "id": "678", "new_content": "User works as a team lead software engineer"},
    {"action": "add", "content": "User got promoted to team lead"},
    {"action": "add", "content": "User has just moved to Mountain View"},
    {"action": "add", "content": "User lives in Mountain View"},
    {"action": "add", "content": "User has an interview at Google"}
  ]
}

**Example 5 - Handling sarcasm and non-literal language**
Conversation:
-3. assistant: ```As an AI assistant, I can perform extremely complex calculations in seconds.```
-2. user: ```Oh yeah? I can do that with my eyes closed! I'm basically a human calculator!```
-1. assistant: ```😂 Sure you can!```

Related Memories:
[]

**Analysis**
- The User's message is clearly sarcastic/joking - they're not literally claiming to be a human calculator
- This is hyperbole used for humorous effect, not a factual statement about their abilities
- No memories should be created from obvious sarcasm or jokes

Output:
{
  "actions": []
}

**Example 6 - Cross-message context linking**
Conversation:
-5. assistant: ```How's your new TV working out?```
-4. user: ```Remember how I bought that Samsung OLED TV last week?```
-3. assistant: ```Yes, I remember that. What about it?```
-2. user: ```Well, it broke down today! The screen just went black.```
-1. assistant: ```Oh no! That's terrible for such a new TV!```

Related Memories:
[
  {"mem_id": "101", "created_at": "2024-03-15T10:00:00", "update_at": "2024-03-15T10:00:00", "content": "User bought a Samsung OLED TV"}
]

**Analysis**
- The User's latest message provides new information about the TV breaking
- We need to create a self-contained memory that includes context from earlier messages
- The new memory should reference the Samsung OLED TV specifically, not just "it" or "the TV"
- This helps semantically link to the existing memory about the purchase

Output:
{
  "actions": [
    {"action": "add", "content": "User's Samsung OLED TV, that was recently purchased, just broke down with a black screen"}
  ]
}

**Example 7 - Memory maintenance: merging and deleting duplicates and bad memories**
Conversation:
-2. user: ```Can you help me write a Python function to sort a list?```
-1. assistant: ```Of course! Here's a simple example using sorted()...```

Related Memories:
[
  {"mem_id": "234", "created_at": "2024-02-10T09:00:00", "update_at": "2024-02-10T09:00:00", "content": "User prefers Python for scripting"},
  {"mem_id": "567", "created_at": "2024-03-15T14:30:00", "update_at": "2024-03-15T14:30:00", "content": "User likes Python for scripting tasks"},
  {"mem_id": "890", "created_at": "2024-01-05T10:00:00", "update_at": "2024-01-05T10:00:00", "content": "User knows Python programming"},
  {"mem_id": "123", "created_at": "2024-01-10T11:00:00", "update_at": "2024-01-10T11:00:00", "content": "User's name is Jake"},
  {"mem_id": "456", "created_at": "2024-01-15T08:00:00", "update_at": "2024-01-15T08:00:00", "content": "User's cat is named Luna"},
  {"mem_id": "789", "created_at": "2024-02-20T10:00:00", "update_at": "2024-02-20T10:00:00", "content": "User's cat is a Siamese"}
]

**Analysis**
- The current conversation is just a technical question about Python - no new personal information
- However, the related memories show issues that need maintenance. We apply the relevant Memory rules:
  1. **Delete bad memory**: Memory 123 contains the user's name, which violates the rule "never store user/assistant names" - should be deleted
  2. **Delete duplicate**: Memory 234 and 567 express essentially the same preference (Python for scripting) - keep older (234), delete newer duplicate (567)
  3. **Merge inseparable facts**: Memory 456 and 789 are about the same cat and should ALWAYS be retrieved together (cat's name + breed) - merge into oldest memory (456)
- Memory 890 is distinct (knowledge vs preference) so it should remain

Output:
{
  "actions": [
    {"action": "delete", "id": "123"},
    {"action": "delete", "id": "567"},
    {"action": "update", "id": "456", "new_content": "User has a Siamese cat named Luna"},
    {"action": "delete", "id": "789"}
  ]
}

**Example 8 - Explicit memory request**
Conversation:
-4. user: ```Hey, do you remember what my dog's name is?```
-3. assistant: ```I don't have that information. Could you tell me?```
-2. user: ```Sure! His name is Max and he's a golden retriever.```
-1. assistant: ```What a lovely name! Max sounds like a wonderful companion. I'll remember that.```

Related Memories:
[
  {"mem_id": "111", "created_at": "2024-01-20T10:00:00", "update_at": "2024-01-20T10:00:00", "content": "User loves dogs"}
]

**Analysis**
- Assistant explicitly expresses intent to remember something. We ALWAYS honor explicit memory requests.
- User provides info about his dog's name and breed these can be stored as a single memory as they are closely related
- The existing memory about loving dogs is related but doesn't conflict

Output:
{
  "actions": [
    {"action": "add", "content": "User has a golden retriever named Max"}
  ]
}

**Example 9 - Memory maintenance: splitting and adding context**
Conversation:
-2. user: ```Sadie invited me to her birthday party next week, I'm excited!```
-1. assistant: ```That's wonderful! I hope you have a great time at Sadie's party.```

Related Memories:
[
  {"mem_id": "555", "created_at": "2024-02-10T10:00:00", "update_at": "2024-02-10T10:00:00", "content": "User has an old time friend named Sadie who they grew up with, and whose mother is a long time friend of User's mother"},
  {"mem_id": "666", "created_at": "2024-02-12T14:00:00", "update_at": "2024-02-12T14:00:00", "content": "The two mothers also did their english courses together"}
]

**Analysis**
- User mentions Sadie's party (new event to store)
- Memory 555 combines two separable facts: User's friendship with Sadie (including growing up together), and the mothers' friendship
- Memory 666 lacks clear context - "the two mothers" is ambiguous without memory 555
- This is a **passive maintenance scenario**: even though the conversation doesn't directly discuss the memory issues, we should fix them
- Actions: update 555 to remove the mothers' friendship, add new memory for mothers' relationship, add context to 666

Output:
{
  "actions": [
    {"action": "add", "content": "User is invited to Sadie's birthday party next week"},
    {"action": "update", "id": "555", "new_content": "User has an old friend named Sadie who they grew up with"},
    {"action": "add", "content": "User's mother and Sadie's mother are long time friends"},
    {"action": "update", "id": "666", "new_content": "User's mother and Sadie's mother did their english courses together"}
  ]
}
</examples>\
"""


async def emit_status(
    description: str,
    emitter: Any,
    status: Literal["in_progress", "complete", "error"] = "complete",
    extra_data: Optional[dict] = None,
):
    if not emitter:
        raise ValueError("Emitter is required to emit status updates")

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


class MemoryAddAction(BaseModel):
    action: Literal["add"] = Field(..., description="Action type (add)")
    content: str = Field(..., description="Content of the memory to add")


class MemoryUpdateAction(BaseModel):
    action: Literal["update"] = Field(..., description="Action type (update)")
    id: str = Field(..., description="ID of the memory to update")
    new_content: str = Field(..., description="New content for the memory")


class MemoryDeleteAction(BaseModel):
    action: Literal["delete"] = Field(..., description="Action type (delete)")
    id: str = Field(..., description="ID of the memory to delete")


class MemoryActionRequestStub(BaseModel):
    """This is a stub model to correctly type parameters. Not used directly."""

    actions: list[Union[MemoryAddAction, MemoryUpdateAction, MemoryDeleteAction]] = (
        Field(
            default_factory=list,
            description="List of actions to perform on memories",
            max_length=20,
        )
    )


class Memory(BaseModel):
    """Single memory entry with metadata."""

    mem_id: str = Field(..., description="ID of the memory")
    created_at: datetime = Field(..., description="Creation timestamp")
    update_at: datetime = Field(..., description="Last update timestamp")
    content: str = Field(..., description="Content of the memory")
    similarity_score: Optional[float] = Field(
        None,
        description="Similarity score (0 to 1 - higher is **more similar** to user query) if available",
    )


def build_actions_request_model(existing_ids: list[str]):
    """Dynamically build versions of the Update/Delete action models whose `id` fields
    are Literal[...] constrained to the provided existing_ids. Returns a tuple:

        (DynamicMemoryUpdateAction, DynamicMemoryDeleteAction, DynamicMemoryUpdateRequest)

    If existing_ids is empty, we still return permissive forms (falls back to str) so that
    add-only flows still parse.
    """
    if not existing_ids:
        # No IDs to constrain, so no relevant memories = can only create new memories
        allowed_actions = MemoryAddAction
    else:
        id_literal_type = Literal[tuple(existing_ids)]

        DynamicMemoryUpdateAction = create_model(
            "MemoryUpdateAction",
            id=(id_literal_type, ...),
            __base__=MemoryUpdateAction,
        )

        DynamicMemoryDeleteAction = create_model(
            "MemoryDeleteAction",
            id=(id_literal_type, ...),
            __base__=MemoryDeleteAction,
        )

        allowed_actions = Union[
            MemoryAddAction, DynamicMemoryUpdateAction, DynamicMemoryDeleteAction
        ]

    return create_model(
        "MemoriesActionRequest",
        actions=(
            list[allowed_actions],
            Field(
                default_factory=list,
                description="List of actions to perform on memories",
                max_length=20,
            ),
        ),
        __base__=BaseModel,
    )


def searchresults_to_memories(results: SearchResult) -> list[Memory]:
    memories = []

    if not results.ids or not results.documents or not results.metadatas:
        raise ValueError("SearchResult must contain ids, documents, and metadatas")

    for batch_idx, (ids_batch, docs_batch, metas_batch) in enumerate(
        zip(results.ids, results.documents, results.metadatas)
    ):
        distances_batch = results.distances[batch_idx] if results.distances else None

        for doc_idx, (mem_id, content, meta) in enumerate(
            zip(ids_batch, docs_batch, metas_batch)
        ):
            if not meta:
                raise ValueError(f"Missing metadata for memory id={mem_id}")
            if "created_at" not in meta:
                raise ValueError(
                    f"Missing 'created_at' in metadata for memory id={mem_id}"
                )
            if "updated_at" not in meta:
                # If updated_at is missing, default to created_at
                meta["updated_at"] = meta["created_at"]

            created_at = datetime.fromtimestamp(meta["created_at"])
            updated_at = datetime.fromtimestamp(meta["updated_at"])

            # Extract similarity score if available
            similarity_score = None
            if distances_batch is not None and doc_idx < len(distances_batch):
                similarity_score = round(distances_batch[doc_idx], 3)

            mem = Memory(
                mem_id=mem_id,
                created_at=created_at,
                update_at=updated_at,
                content=content,
                similarity_score=similarity_score,
            )
            memories.append(mem)

    return memories


def _run_detached(coro):
    """Helper to run coroutine in detached thread"""

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()


R = TypeVar("R", bound=BaseModel)
ValveType = TypeVar("ValveType", str, int)


class Filter:
    class Valves(BaseModel):
        openai_api_url: str = Field(
            default="https://api.openai.com/v1",
            description="openai compatible endpoint",
        )
        model: str = Field(
            default="gpt-5-mini",
            description="model to use to determine memory. an intelligent model is highly recommended, as it will be able to better understand the context of the conversation.",
        )
        api_key: str = Field(
            default="", description="API key for OpenAI compatible endpoint"
        )
        messages_to_consider: int = Field(
            default=4,
            description="global default number of recent messages to consider for memory extraction (user override can supply a different value).",
        )
        related_memories_n: int = Field(
            default=5,
            description="number of related memories to consider when updating memories",
        )
        minimum_memory_similarity: Optional[float] = Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="minimum similarity of memories to consider for updates. higher is more similar to user query. if not set, no filtering is applied.",
        )
        allow_unsafe_user_overrides: bool = Field(
            default=False,
            description="SECURITY WARNING: allow users to override API URL/model without providing their own API key. this could allow users to steal your API key or use expensive models at your expense. only enable if you trust all users.",
        )
        override_memory_context: bool = Field(
            default=False,
            description="intercept and override memory context injection in system prompts. when enabled, allows customization of how memories are presented to the model.",
        )
        debug_mode: bool = Field(
            default=False,
            description="enable debug logging",
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="whether to enable Auto Memory for this user",
        )
        show_status: bool = Field(
            default=True, description="show status of the action."
        )
        openai_api_url: Optional[str] = Field(
            default=None,
            description="user-specific openai compatible endpoint (overrides global)",
        )
        model: Optional[str] = Field(
            default=None,
            description="user-specific model to use (overrides global). an intelligent model is highly recommended, as it will be able to better understand the context of the conversation.",
        )
        api_key: Optional[str] = Field(
            default=None, description="user-specific API key (overrides global)"
        )
        messages_to_consider: Optional[int] = Field(
            default=None,
            description="override for number of recent messages to consider (falls back to global if null). includes assistant responses.",
        )

    def log(self, message: str, level: LogLevel = "info"):
        if level == "debug" and not self.valves.debug_mode:
            return
        if level not in {"debug", "info", "warning", "error"}:
            level = "info"

        logger = logging.getLogger()
        getattr(logger, level, logger.info)(message)

    def messages_to_string(self, messages: list[dict[str, Any]]) -> str:
        stringified_messages: list[str] = []

        effective_messages_to_consider = self.get_restricted_user_valve(
            user_valve_value=self.user_valves.messages_to_consider,
            admin_fallback=self.valves.messages_to_consider,
            authorization_check=bool(
                self.user_valves.api_key and self.user_valves.api_key.strip()
            ),
            valve_name="messages_to_consider",
        )

        self.log(
            f"using last {effective_messages_to_consider} messages",
            level="debug",
        )

        for i in range(1, effective_messages_to_consider + 1):
            if i > len(messages):
                break
            try:
                message = messages[-i]
                stringified_messages.append(
                    STRINGIFIED_MESSAGE_TEMPLATE.format(
                        index=i,
                        role=message.get("role", "user"),
                        content=message.get("content", ""),
                    )
                )
            except Exception as e:
                self.log(f"error stringifying message {i}: {e}", level="warning")

        return "\n".join(stringified_messages)

    @overload
    async def query_openai_sdk(
        self,
        system_prompt: str,
        user_message: str,
        response_model: Type[R],
    ) -> R: ...

    @overload
    async def query_openai_sdk(
        self,
        system_prompt: str,
        user_message: str,
        response_model: None = None,
    ) -> str: ...

    async def query_openai_sdk(
        self,
        system_prompt: str,
        user_message: str,
        response_model: Optional[Type[R]] = None,
    ) -> Union[str, R]:
        """Generic wrapper around OpenAI chat completions.
        - Uses SDK for api.openai.com only
        - Structured outputs when official domain and response_model provided
        - Returns: model instance or raw string
        """

        user_has_own_key = bool(
            self.user_valves.api_key and self.user_valves.api_key.strip()
        )

        api_url = self.get_restricted_user_valve(
            user_valve_value=self.user_valves.openai_api_url,
            admin_fallback=self.valves.openai_api_url,
            authorization_check=user_has_own_key,
            valve_name="openai_api_url",
        ).rstrip("/")

        model_name = self.get_restricted_user_valve(
            user_valve_value=self.user_valves.model,
            admin_fallback=self.valves.model,
            authorization_check=user_has_own_key,
            valve_name="model",
        )
        api_key = self.user_valves.api_key or self.valves.api_key

        hostname = urlparse(api_url).hostname or ""
        enable_structured_outputs = (
            hostname == "api.openai.com" and response_model is not None
        )

        if "gpt-5" in model_name:
            temperature = 1.0
            extra_args = {"reasoning_effort": "medium"}
        else:
            temperature = 0.3
            extra_args = {}

        client = OpenAI(api_key=api_key, base_url=api_url)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        if enable_structured_outputs:
            response_model = cast(Type[R], response_model)
            self.log(
                f"using structured outputs with {response_model.__name__}",
                level="debug",
            )

            response = client.chat.completions.parse(
                model=model_name,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                response_format=response_model,
                **extra_args,  # pyright: ignore[reportArgumentType]
            )

            message = response.choices[0].message
            if message.parsed is None:
                raise ValueError(
                    f"unable to parse structured response. message={message}"
                )

            return cast(R, message.parsed)

        else:
            self.log("not using structured outputs", level="debug")

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                **extra_args,  # pyright: ignore[reportArgumentType]
            )
            self.log(f"sdk response: {response}", level="debug")

            text_response = response.choices[0].message.content
            if text_response is None:
                raise ValueError(f"no text response from LLM. message={text_response}")

            if response_model:
                try:
                    return response_model.model_validate_json(text_response)
                except ValidationError as e:
                    self.log(f"response model validation error: {e}", level="warning")
                    raise

            return text_response

    def __init__(self):
        self.valves = self.Valves()

    def extract_memory_context(self, content: str) -> Optional[tuple[str, list[dict]]]:
        """
        Extract memory context from system message content.

        Returns:
            tuple of (full_match_string, parsed_memories_list) if found, None otherwise
        """
        # Open WebUI uses this standard format
        pattern = r"<memory_user_context>\s*(\[[\s\S]*?\])\s*</memory_user_context>"
        match = re.search(pattern, content)

        if not match:
            self.log("no memory context found in system message", level="debug")
            return None

        try:
            memories_json = match.group(1)
            memories_list = json.loads(memories_json)
            self.log(
                f"extracted {len(memories_list)} memories from context", level="debug"
            )
            return (match.group(0), memories_list)
        except json.JSONDecodeError as e:
            self.log(
                f"failed to parse memory context JSON: {e}. raw content: {match.group(1)[:200]}...",
                level="error",
            )
            return None

    def format_memory_context(self, memories: list[dict]) -> str:
        """
        Format memories into the memory context string.
        Override this method to customize how memories are presented.

        Args:
            memories: List of memory objects with 'content', 'created_at', 'updated_at', 'similarity_score'

        Returns:
            Formatted memory context string to inject into system prompt
        """
        # Remove similarity_score from each memory
        memories = [
            {k: v for k, v in mem.items() if k != "similarity_score"}
            for mem in memories
        ]

        # Format with custom XML tag
        memories_json = json.dumps(memories, indent=2, ensure_ascii=False)
        return f"<long_term_memory>\n{memories_json}\n</long_term_memory>"

    def process_memory_context_in_messages(self, messages: list[dict]) -> list[dict]:
        """
        Process messages to intercept and optionally override memory context.

        Args:
            messages: List of message dicts from the body

        Returns:
            Modified messages list
        """
        found_any_memory_context = False

        # Find system message(s)
        for i, message in enumerate(messages):
            if message.get("role") != "system":
                continue

            content = message.get("content", "")
            if not content:
                continue

            # Try to extract existing memory context
            extraction_result = self.extract_memory_context(content)

            if extraction_result:
                found_any_memory_context = True
                full_match, memories_list = extraction_result

                # Override: format the memories using custom method
                new_context = self.format_memory_context(memories_list)

                # Replace in content
                messages[i]["content"] = content.replace(full_match, new_context)

                # Log successful override
                self.log(
                    f"overrode memory context in system message {i}: {len(memories_list)} memories processed, "
                    f"similarity scores removed, XML tag changed to <long_term_memory>",
                    level="info",
                )
            else:
                self.log(f"no memory context in system message {i}", level="debug")

        # If valve is enabled and we didn't find any memory context, that's unusual
        if not found_any_memory_context:
            self.log(
                "memory context override is enabled but no <memory_user_context> found in any system message",
                level="warning",
            )

        return messages

    def get_restricted_user_valve(
        self,
        user_valve_value: Optional[ValveType],
        admin_fallback: ValveType,
        authorization_check: Optional[bool] = None,
        valve_name: Optional[str] = None,
    ) -> ValveType:
        """
        Get user valve value with security checks.

        Args:
            user_valve_value: The user's valve value to check
            admin_fallback: Admin's fallback value
            authorization_check: The valve value to check for authorization (e.g., user's API key)
            valve_name: Name of the valve being checked (for logging)

        Returns user's value only if:
        1. authorization_check is provided and non-empty, OR
        2. User is an admin, OR
        3. Admin allows unsafe overrides

        Otherwise returns admin fallback.
        """
        if authorization_check is None:
            authorization_check = False

        if authorization_check:
            if user_valve_value is not None:
                self.log(
                    f"'{valve_name or 'unknown'}' override authorized (user has own API key)",
                    level="debug",
                )
            return user_valve_value if user_valve_value is not None else admin_fallback

        # Allow admins to override without providing their own API key
        if hasattr(self, "current_user") and self.current_user.get("role") == "admin":
            if user_valve_value is not None:
                self.log(
                    f"'{valve_name or 'unknown'}' override allowed for admin user",
                    level="info",
                )
            return user_valve_value if user_valve_value is not None else admin_fallback

        if self.valves.allow_unsafe_user_overrides:
            if user_valve_value is not None:
                self.log(
                    f"'{valve_name or 'unknown'}' override allowed (unsafe overrides enabled)",
                    level="warning",
                )
            return user_valve_value if user_valve_value is not None else admin_fallback

        if user_valve_value is not None:
            self.log(
                f"'{valve_name or 'unknown'}' override blocked - user attempted override without authorization, using admin defaults for security",
                level="warning",
            )
        return admin_fallback

    def build_memory_query(self, messages: list[dict[str, Any]]) -> str:
        """
        Build a query string for memory retrieval from recent messages.

        Strategy:
        - Always include: last user message + last assistant response
        - If user message is short (≤8 words), also include the previous assistant message

        This gives embeddings enough context without overwhelming with noise.
        """
        query_parts = []

        # Find last user message and its index
        last_user_idx = None
        last_user_msg = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_idx = idx
                last_user_msg = messages[idx].get("content", "")
                break

        if last_user_msg is None or last_user_idx is None:
            raise ValueError("no user message found in messages")

        # Count words in last user message
        user_word_count = len(last_user_msg.split())

        # Check if we should include extra context for short messages
        include_extra_context = user_word_count <= 8

        # Build query from most recent to older messages
        # Add last assistant response (if exists)
        if last_user_idx + 1 < len(messages):
            last_assistant_msg = messages[last_user_idx + 1].get("content", "")
            if last_assistant_msg:
                query_parts.append(f"Assistant: {last_assistant_msg}")

        # Add last user message
        query_parts.append(f"User: {last_user_msg}")

        # If short message, add previous assistant context
        if include_extra_context and last_user_idx > 0:
            prev_assistant_msg = messages[last_user_idx - 1].get("content", "")
            if (
                prev_assistant_msg
                and messages[last_user_idx - 1].get("role") == "assistant"
            ):
                query_parts.append(f"Assistant: {prev_assistant_msg}")

        # Reverse to get chronological order and join
        query_parts.reverse()
        query = "\n".join(query_parts)

        self.log(
            f"built memory query with {len(query_parts)} messages (user message: {user_word_count} words)",
            level="debug",
        )
        self.log(f"memory query: {query}", level="debug")

        return query

    async def get_related_memories(
        self,
        messages: list[dict[str, Any]],
        user: UserModel,
    ) -> list[Memory]:
        memory_query = self.build_memory_query(messages)

        # Query related memories
        try:
            results = await query_memory(
                request=Request(scope={"type": "http", "app": webui_app}),
                form_data=QueryMemoryForm(
                    content=memory_query, k=self.valves.related_memories_n
                ),
                user=user,
            )
        except HTTPException as e:
            if e.status_code == 404:
                self.log("no related memories found", level="info")
                results = None
            else:
                self.log(
                    f"failed to query memories due to HTTP error {e.status_code}: {e.detail}",
                    level="error",
                )
                raise RuntimeError("failed to query memories") from e
        except Exception as e:
            self.log(f"failed to query memories: {e}", level="error")
            raise RuntimeError("failed to query memories") from e

        related_memories = searchresults_to_memories(results) if results else []
        self.log(
            f"found {len(related_memories)} related memories before filtering",
            level="info",
        )

        # Filter by minimum similarity if configured
        if self.valves.minimum_memory_similarity is not None:
            filtered_memories = [
                mem
                for mem in related_memories
                if mem.similarity_score is not None
                and mem.similarity_score >= self.valves.minimum_memory_similarity
            ]
            filtered_count = len(related_memories) - len(filtered_memories)
            if filtered_count > 0:
                self.log(
                    f"filtered out {filtered_count} memories below similarity threshold {self.valves.minimum_memory_similarity}",
                    level="info",
                )
            related_memories = filtered_memories

        self.log(f"using {len(related_memories)} related memories", level="info")
        self.log(f"related memories: {related_memories}", level="debug")

        return related_memories

    async def auto_memory(
        self,
        messages: list[dict[str, Any]],
        user: UserModel,
        emitter: Callable[[Any], Awaitable[None]],
    ) -> None:
        """Execute the auto-memory extraction and update flow."""

        if len(messages) < 2:
            self.log("need at least 2 messages for context", level="debug")
            return
        self.log(f"flow started. user ID: {user.id}", level="debug")

        related_memories = await self.get_related_memories(messages=messages, user=user)

        stringified_memories = json.dumps(
            [memory.model_dump(mode="json") for memory in related_memories]
        )
        conversation_str = self.messages_to_string(messages)

        try:
            action_plan = await self.query_openai_sdk(
                system_prompt=UNIFIED_SYSTEM_PROMPT,
                user_message=f"Conversation snippet:\n{conversation_str}\n\nRelated Memories:\n{stringified_memories}",
                response_model=build_actions_request_model(
                    [m.mem_id for m in related_memories]
                ),
            )
            self.log(f"action plan: {action_plan}", level="debug")

            await self.apply_memory_actions(
                action_plan=action_plan,  # pyright: ignore[reportArgumentType]
                user=user,
                emitter=emitter,
            )

        except Exception as e:
            self.log(f"LLM query failed: {e}", level="error")
            if self.user_valves.show_status:
                await emit_status(
                    "memory processing failed", emitter=emitter, status="error"
                )
            return None

    async def apply_memory_actions(
        self,
        action_plan: MemoryActionRequestStub,
        user: UserModel,
        emitter: Callable[[Any], Awaitable[None]],
    ) -> None:
        """
        Execute memory actions from the plan.
        Order: delete -> update -> add (prevents conflicts)
        """
        self.log("started apply_memory_actions", level="debug")
        actions = action_plan.actions

        # Show processing status
        if emitter and len(actions) > 0:
            self.log(f"processing {len(actions)} memory actions", level="debug")
            await emit_status(
                f"processing {len(actions)} memory actions",
                emitter=emitter,
                status="in_progress",
            )
        if self.valves.debug_mode:
            self.log(f"memory actions to apply: {actions}", level="debug")

        # Group actions and define handlers
        operations = {
            "delete": {
                "actions": [a for a in actions if a.action == "delete"],
                "handler": lambda a: delete_memory_by_id(memory_id=a.id, user=user),
                "log_msg": lambda a: f"deleted memory. id={a.id}",
                "error_msg": lambda a, e: f"failed to delete memory {a.id}: {e}",
                "skip_empty": lambda a: False,
                "status_verb": "deleted",
            },
            "update": {
                "actions": [a for a in actions if a.action == "update"],
                "handler": lambda a: update_memory_by_id(
                    memory_id=a.id,
                    request=Request(scope={"type": "http", "app": webui_app}),
                    form_data=MemoryUpdateModel(content=a.new_content),
                    user=user,
                ),
                "log_msg": lambda a: f"updated memory. id={a.id}",
                "error_msg": lambda a, e: f"failed to update memory {a.id}: {e}",
                "skip_empty": lambda a: not a.new_content.strip(),
                "status_verb": "updated",
            },
            "add": {
                "actions": [a for a in actions if a.action == "add"],
                "handler": lambda a: add_memory(
                    request=Request(scope={"type": "http", "app": webui_app}),
                    form_data=AddMemoryForm(content=a.content),
                    user=user,
                ),
                "log_msg": lambda a: f"added memory. content={a.content}",
                "error_msg": lambda a, e: f"failed to add memory: {e}",
                "skip_empty": lambda a: not a.content.strip(),
                "status_verb": "saved",
            },
        }

        # Process all operations in order
        counts = {}
        for op_name, op_config in operations.items():
            counts[op_name] = 0
            for action in op_config["actions"]:
                if op_config["skip_empty"](action):
                    continue
                try:
                    await op_config["handler"](action)
                    self.log(op_config["log_msg"](action))
                    counts[op_name] += 1
                except Exception as e:
                    raise RuntimeError(op_config["error_msg"](action, e))

        # Build status message
        status_parts = []
        for op_name, op_config in operations.items():
            count = counts[op_name]
            if count > 0:
                memory_word = "memory" if count == 1 else "memories"
                status_parts.append(f"{op_config['status_verb']} {count} {memory_word}")

        status_message = ", ".join(status_parts)
        self.log(status_message or "no changes", level="info")

        if status_message and self.user_valves.show_status:
            await emit_status(status_message, emitter=emitter, status="complete")

    def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        self.log(f"inlet: {__name__}", level="info")
        self.log(
            f"inlet: user ID: {__user__.get('id') if __user__ else 'no user'}",
            level="debug",
        )

        # Process memory context interception if enabled
        if self.valves.override_memory_context and "messages" in body:
            try:
                body["messages"] = self.process_memory_context_in_messages(
                    body["messages"]
                )
            except Exception as e:
                self.log(f"error processing memory context: {e}", level="error")

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:

        self.log("outlet invoked")
        if __user__ is None:
            raise ValueError("user information is required")

        user = Users.get_user_by_id(__user__["id"])
        if user is None:
            raise ValueError("user not found")
        self.current_user = __user__

        self.log(f"input user type = {type(__user__)}", level="debug")
        self.log(
            f"user.id = {user.id} user.name = {user.name} user.email = {user.email}",
            level="debug",
        )

        user_memory_enabled = False
        if hasattr(user, "settings") and user.settings:
            user_memory_enabled = user.settings.model_dump().get("ui", {}).get("memory", False)
        
        if not user_memory_enabled:
            self.log(
                "memory is disabled in user's personalization settings, skipping",
                level="info",
            )
            return body

        self.user_valves = __user__.get("valves", self.UserValves())
        if not isinstance(self.user_valves, self.UserValves):
            raise ValueError("invalid user valves")
        self.user_valves = cast(Filter.UserValves, self.user_valves)
        self.log(f"user valves = {self.user_valves}", level="debug")

        if not self.user_valves.enabled:
            self.log("component was disabled by user, skipping", level="info")
            return body

        _run_detached(
            self.auto_memory(
                body.get("messages", []), user=user, emitter=__event_emitter__
            )
        )

        return body
