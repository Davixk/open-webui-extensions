"""
title: Auto Memory
author: @nokodo
description: Automatically identify and store valuable information from chats as Memories.
author_email: nokodo@nokodo.net
author_url: https://nokodo.net
repository_url: https://nokodo.net/github/open-webui-extensions
version: 1.0.0-alpha4
required_open_webui_version: >= 0.5.0
funding_url: https://ko-fi.com/nokodo
"""

import asyncio
import json
import logging
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

from fastapi.requests import Request
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

LEGACY_IDENTIFY_MEMORIES_PROMPT = """\
You are helping maintain a collection of Memories— individual “journal entries”, each automatically timestamped upon creation or update.
You will be provided with the last several messages from a conversation (displayed with negative indices; -1 is the most recent overall message). Your job is to decide which details within the last User message (-2) are worth saving long-term as Memory entries.

<key_instructions>
1. Identify new or changed personal details from the User's **latest** message (-2) only. Older user messages may appear for context; do not re-store older facts unless explicitly repeated or modified in the last User message (-2).
2. If the User’s newest message contradicts an older statement (e.g., message -4 says “I love oranges” vs. message -2 says “I hate oranges”), extract only the updated info (“User hates oranges”).
3. Think of each Memory as a single “fact” or statement. Never combine multiple facts into one Memory. If the User mentions multiple distinct items, break them into separate entries.
4. Link related Memories together by including brief, minimal references to other Memories when relevant, to help semantically connect them. For example, if the User mentions a new detail about a previously noted event or preference, include a short reference to that earlier Memory to maintain context.
5. Your goal is to capture anything that might be valuable for the "assistant" to remember about the User, to personalize and enrich future interactions.
6. If the User explicitly requests to “remember” or note down something in their latest message (-2), always include it.
7. Avoid storing short-term or trivial details (e.g. user: “I’m reading this question right now”, user: "I just woke up!", user: "Oh yeah, I saw that on TV the other day").
8. Return your result as a Python list of strings, **each string representing a separate Memory**. If no relevant info is found, **only** return an empty list (`[]`). No explanations, just the list.
</key_instructions>

<what_to_extract>
- Personal preferences, opinions, and feelings about topics/things/people
- Information that will likely remain true for months or years
- Anything with future-oriented phrases: "from now on", "going forward", "in the future"
- Direct memory requests: "remember that", "note this", "add to memory", "store this"
- Hobbies, interests, skills, and long-term activities
- Important life details (job, education, relationships, location, etc.)
- Personal goals, plans, or aspirations
- Recurring patterns or habits
- Strong likes/dislikes that could affect future conversations
- "Forget" requests (store as "Forget that User...")
</what_to_extract>

<what_not_to_extract>
- User names, since these are already in profile info and this would only create confusion
- Assistant names, since Memories are assistant-agnostic and can be used across different assistants
- Short-lived facts that won't matter soon (e.g., "I'm reading this right now", "I just woke up")
- Random details that lack clear future relevance
- Redundant information already known about the User (e.g., when the assistant replies with "Yes, I remember that" to a User message, it means the info is already stored)
- Information from text the User is asking to translate or rewrite
- Trivial observations or fleeting thoughts
- Current temporary states or activities
</what_not_to_extract>

---

<examples>
**Example 1 - Only storing Memories from the latest user message**
-4. user: ```I love oranges 😍```
-3. assistant: ```That's great! 🍊 I love oranges too!```
-2. user: ```Actually, I hate oranges 😂```
-1. assistant: ```omg you LIAR 😡```

**Analysis**  
- The last user message states a new personal fact: “User hates oranges.”  
- This replaces the older statement about loving oranges.
- We only extract Memories from the latest user message (-2).

Output:
```
["User hates oranges"]
```

**Example 2 - Explicit and Implicit Memories**
-2. user: ```I work as a junior data analyst. Please remember that my big presentation is on March 15.```
-1. assistant: ```Got it! I'll make a note of that.```

**Analysis**
- The user provides two new pieces of information: their profession and the date of their presentation.
- These are both distinct facts that should be remembered separately.
- We extract both the explicit request to remember the presentation date and the implicit fact about their occupation.

Output:
```
["User works as a junior data analyst", "User has a big presentation on March 15"]
```

**Example 3 - Memory linking via context**
-5. assistant: ```Nutella is amazing! 😍```
-4. user: ```Soo, remember how a week ago I had bought a new TV?```
-3. assistant: ```Yes, I remember that. What about it?```
-2. user: ```well, today it broke down 😭```
-1. assistant: ```Oh no! That's terrible!```

**Analysis**
- The only relevant message is the last User message (-2), which provides new information about the TV breaking down.
- The previous messages (-3, -4) provide context over what the user was talking about.
- The remaining message (-5) is irrelevant.
- When extracting the memory, we include the context of the TV purchase to make the memory meaningful. This will help semantically link it to the prior fact about buying the TV.
- We assume there might be a prior memory about the TV purchase, so we phrase this new memory to connect to that earlier fact.

Output:
```
["User's TV they bought a week ago broke down today"]
```

**Example 4 - Sarcasm use**
-3. assistant: ```As an AI assistant, I can perform extremely complex calculations in seconds.```
-2. user: ```Oh yeah? I can do that with my eyes closed!```
-1. assistant: ```😂 Sure you can, Joe!```

**Analysis**
- The User message (-2) is clearly sarcastic and not meant to be taken literally. It does not contain any relevant information to store.
- The other messages (-3, -1) are not relevant as they're not about the User.

Output:
```
[]
```

**Example 5 - Multiple complex linked Memories**
-2. user: ```I am following a 30-day program to improve my fitness and health. If I send you the details, could you be my personal trainer for day 12?```
-1. assistant: ```Absolutely! Please send me the details of your program, and I'll be happy to assist you as your personal trainer for day 12.```

**Analysis**
- The User message (-2) contains two distinct pieces of information:
  1. The User is following a 30-day fitness program.
  2. The User is on day 12 of that program.
- We have to store both facts as separate Memories, and we have to link them logically so they can be understood both individually and in relation to each other.
- To link them logically, we phrase the second memory to reference the first, indicating that day 12 is part of the 30-day program.
- We can't phrase them like "User is on a 30-day program to improve fitness and health" and "User is on day 12 of that program", because *that program* will have no meaning without context.
- We don't need to add dates, as all Memories are automatically timestamped upon creation and update.

Output:
```
["User is following a 30-day program to improve fitness and health", "User is on day 12 of their 30-day fitness program"
```
</examples>\
"""

LEGACY_CONSOLIDATE_MEMORIES_PROMPT = """You are maintaining a set of “Memories” for a user, similar to journal entries. Each memory has:
- A "fact" (a string describing something about the user or a user-related event).
- A "created_at" timestamp (an integer or float representing when it was stored/updated).

**What You’re Doing**
1. You’re given a list of such Memories that the system believes might be related or overlapping.
2. Your goal is to produce a cleaned-up list of final facts, making sure we:
   - Only combine Memories if they are exact duplicates or direct conflicts about the same topic.
   - In case of duplicates, keep only the one with the latest (most recent) `created_at`.
   - In case of a direct conflict (e.g., the user’s favorite color stated two different ways), keep only the most recent one.
   - If Memories are partially similar but not truly duplicates or direct conflicts, preserve them both. We do NOT want to lose details or unify “User likes oranges” and “User likes ripe oranges” into a single statement—those remain separate.
3. Return the final list as a simple Python list of strings—**each string is one separate memory/fact**—with no extra commentary.

**Remember**  
- This is a journaling system meant to give the user a clear, time-based record of who they are and what they’ve done.  
- We do not want to clump multiple distinct pieces of info into one memory.  
- We do not throw out older facts unless they are direct duplicates or in conflict with a newer statement.  
- If there is a conflict (e.g., “User’s favorite color is red” vs. “User’s favorite color is teal”), keep the more recent memory only.

---

### **Extended Example**

Below is an example list of 15 “Memories.” Notice the variety of scenarios:
- Potential duplicates
- Partial overlaps
- Direct conflicts
- Ephemeral/past events

**Input** (a JSON-like array):

```
[
  {"fact": "User visited Paris for a business trip", "created_at": 1631000000},
  {"fact": "User visited Paris for a personal trip with their girlfriend", "created_at": 1631500000},
  {"fact": "User visited Paris for a personal trip with their girlfriend", "created_at": 1631600000}, 
  {"fact": "User works as a junior data analyst", "created_at": 1633000000},
  {"fact": "User's meeting with the project team is scheduled for Friday at 10 AM", "created_at": 1634000000},
  {"fact": "User's meeting with the project team is scheduled for Friday at 11 AM", "created_at": 1634050000}, 
  {"fact": "User likes to eat oranges", "created_at": 1635000000},
  {"fact": "User likes to eat ripe oranges", "created_at": 1635100000},
  {"fact": "User used to like red color, but not anymore", "created_at": 1635200000},
  {"fact": "User's favorite color is teal", "created_at": 1635500000},
  {"fact": "User's favorite color is red", "created_at": 1636000000},
  {"fact": "User traveled to Japan last year", "created_at": 1637000000},
  {"fact": "User traveled to Japan this month", "created_at": 1637100000},
  {"fact": "User also works part-time as a painter", "created_at": 1637200000},
  {"fact": "User had a dentist appointment last Tuesday", "created_at": 1637300000}
]
```

**Analysis**:
1. **Paris trips**  
   - "User visited Paris for a personal trip with their girlfriend" appears **twice** (`created_at`: 1631500000 and 1631600000). They are exact duplicates but have different timestamps, so we keep only the most recent. The business trip is different, so keep it too.

2. **Meeting time**  
   - There's a direct conflict about the meeting time (10 AM vs 11 AM). We keep the more recent statement.

3. **Likes oranges / ripe oranges**  
   - These are partially similar, but not exactly the same or in conflict, so we keep both.

4. **Color**  
   - We have “User used to like red,” “User’s favorite color is teal,” and “User’s favorite color is red.” 
   - The statement “User used to like red color, but not anymore” is not actually a direct conflict with “favorite color is teal.” We keep them both. 
   - The newest color memory is “User’s favorite color is red” (timestamp 1636000000) which conflicts with the older “User’s favorite color is teal” (timestamp 1635500000). We keep the more recent red statement.

5. **Japan**  
   - “User traveled to Japan last year” vs “User traveled to Japan this month.” They’re not contradictory; one is old, one is new. Keep them both.

6. **Past events**  
   - Dentist appointment is ephemeral, but we keep it since each memory is a separate time-based journal entry.

**Correct Output** (the final consolidated list of facts as strings):

```
[
  "User visited Paris for a business trip",
  "User visited Paris for a personal trip with their girlfriend",  <-- keep only the most recent from duplicates
  "User works as a junior data analyst",
  "User's meeting with the project team is scheduled for Friday at 11 AM", 
  "User likes to eat oranges",
  "User likes to eat ripe oranges",
  "User used to like red color, but not anymore",
  "User's favorite color is red",  <-- overrides teal
  "User traveled to Japan last year",
  "User traveled to Japan this month",
  "User also works part-time as a painter",
  "User had a dentist appointment last Tuesday"
]
```

Make sure your final answer is just the array, with no added commentary.

---

### **Final Reminder**
- You’re only seeing these Memories because our system guessed they might overlap. If they’re not exact duplicates or direct conflicts, keep them all.  
- Always produce a **Python list of strings**—each string is a separate memory/fact.  
- Do not add any explanation or disclaimers—just the final list.\
"""

UNIFIED_SYSTEM_PROMPT = """\
You are maintaining a collection of Memories - individual "journal entries" about a user, each automatically timestamped upon creation or update.

You will be provided with:
1. Recent messages from a conversation (displayed with negative indices; -1 is the most recent overall message)
2. Any existing related memories that might potentially be relevant

Your job is to determine what actions to take on the memory collection based on the User's **latest** message (-2).

<key_instructions>
## Instructions
1. Focus ONLY on the User's most recent message (-2). Older messages provide context but should not generate new memories unless explicitly referenced in the latest message.
2. Each Memory should represent a single fact or statement. Never combine multiple facts into one Memory.
3. When the User's latest message contradicts existing memories, update the existing memory rather than creating a conflicting new one.
4. If memories are exact duplicates or direct conflicts about the same topic, consolidate them by updating or deleting as appropriate.
5. Link related Memories by including brief references when relevant to maintain semantic connections.
6. Capture anything valuable for personalizing future interactions with the User.
7. Honor explicit user requests to "remember", "forget", or "update" information.
</key_instructions>

<what_to_extract>
## What you WANT to extract
- Personal preferences, opinions, and feelings
- Long-term information (likely true for months/years)
- Future-oriented statements ("from now on", "going forward")
- Direct memory requests ("remember that", "note this", "forget that")
- Hobbies, interests, skills, activities
- Important life details (job, education, relationships, location)
- Goals, plans, aspirations
- Recurring patterns or habits
- Strong likes/dislikes affecting future conversations
</what_to_extract>

<what_not_to_extract>
## What you do NOT want to extract
- User/assistant names (already in profile)
- Ephemeral states ("I'm reading this now", "I just woke up")
- Information the assistant confirms is already known
- Content from translation/rewrite requests
- Trivial observations or fleeting thoughts
- Temporary activities
</what_not_to_extract>

<actions_to_take>
Based on your analysis, return a list of actions:

**ADD**: Create new memory when:
- New information not covered by existing memories
- Distinct facts even if related to existing topics
- User explicitly requests to remember something

**UPDATE**: Modify existing memory when:
- User provides updated/corrected information about the same fact
- User explicitly asks to update something
- New information refines but doesn't fundamentally change existing memory

**DELETE**: Remove existing memory when:
- User explicitly requests to forget something
- User's statement directly contradicts an existing memory
- Memory is completely obsolete due to new information
- Duplicate memories exist (keep most recent)

When updating or deleting, ONLY use the memory ID from the related memories list.
</actions_to_take>

<consolidation_rules>
- Only combine memories if they are exact duplicates or direct conflicts about the same topic
- For duplicates: keep only the most recent
- For conflicts: update to reflect the latest information
- For similar but distinct facts: keep them separate (e.g., "likes oranges" vs "likes ripe oranges")
- Past events remain as separate journal entries unless explicitly contradicted
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
    {"action": "add", "content": "User got promoted to team lead last month"},
    {"action": "add", "content": "User has just moved to Mountain View"},
    {"action": "add", "content": "User lives in Mountain View"},
    {"action": "add", "content": "User has an interview at Google next week"}
  ]
}
</examples>\
"""


LOG_FORMAT = "[Auto Memory][{level}] {message}"
LOGGER_NAME = "open_webui.extensions.auto_memory"


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


class MemoryExtract(BaseModel):
    """Single extracted memory fact."""

    content: str = Field(..., description="Memory fact string")


class MemoryExtractResponse(BaseModel):
    """Structured extraction response (list of new memory facts)."""

    memories: list[MemoryExtract] = Field(
        default_factory=list, description="List of extracted memory facts"
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


def searchresult_to_memories(result: SearchResult) -> list[Memory]:
    memories = []

    if not result.ids or not result.documents or not result.metadatas:
        raise ValueError("SearchResult must contain ids, documents, and metadatas")

    # iterate over each query batch
    for ids_batch, docs_batch, metas_batch in zip(
        result.ids, result.documents, result.metadatas
    ):
        for mem_id, content, meta in zip(ids_batch, docs_batch, metas_batch):
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

            mem = Memory(
                mem_id=mem_id,
                created_at=created_at,
                update_at=updated_at,
                content=content,
            )
            memories.append(mem)

    return memories


R = TypeVar("R", bound=BaseModel)


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
        related_memories_dist: float = Field(
            default=0.75,
            description="distance of memories to consider for updates. Smaller number will be more closely related.",
        )
        debug_mode: bool = Field(
            default=False,
            description="enable debug logging",
        )

    class UserValves(BaseModel):
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

        logger = logging.getLogger(LOGGER_NAME)
        getattr(logger, level, logger.info)(message)

        print(LOG_FORMAT.format(level=level, message=message))

    def messages_to_string(self, messages: list[dict[str, Any]]) -> str:
        stringified_messages: list[str] = []
        effective_messages_to_consider = (
            self.user_valves.messages_to_consider
            if self.user_valves.messages_to_consider is not None
            else self.valves.messages_to_consider
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

        api_url = (
            self.user_valves.openai_api_url or self.valves.openai_api_url
        ).rstrip("/")
        hostname = urlparse(api_url).hostname or ""
        enable_structured_outputs = (
            hostname == "api.openai.com" and response_model is not None
        )

        model_name = self.user_valves.model or self.valves.model
        api_key = self.user_valves.api_key or self.valves.api_key
        temperature = 0.3 if "gpt-5" not in model_name else 1

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

        # Extract latest user message for finding related memories
        latest_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                latest_user_msg = msg.get("content", "")
                break

        if not latest_user_msg:
            self.log("no user message found", level="debug")
            return

        # Query related memories
        try:
            results = await query_memory(
                request=Request(scope={"type": "http", "app": webui_app}),
                form_data=QueryMemoryForm(
                    content=latest_user_msg, k=self.valves.related_memories_n
                ),
                user=user,
            )
        except Exception as e:
            self.log(f"failed to query memories: {e}", level="error")
            raise RuntimeError("failed to query memories") from e

        related_memories = searchresult_to_memories(results) if results else []
        self.log(f"found {len(related_memories)} related memories", level="info")
        self.log(f"related memories: {related_memories}", level="debug")
        stringified_memories = json.dumps(
            [memory.model_dump(mode="json") for memory in related_memories]
        )
        conversation_str = self.messages_to_string(messages)

        try:
            action_plan = await self.query_openai_sdk(
                system_prompt=UNIFIED_SYSTEM_PROMPT,
                user_message=f"Conversation:\n{conversation_str}\n\nRelated Memories:\n{stringified_memories}",
                response_model=build_actions_request_model(
                    [m.mem_id for m in related_memories]
                ),
            )
            self.log(f"action plan: {action_plan}", level="debug")

            # Apply the actions
            await self.apply_memory_actions(
                action_plan=action_plan, user=user, emitter=emitter
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

        delete_actions = [a for a in actions if a.action == "delete"]
        update_actions = [a for a in actions if a.action == "update"]
        add_actions = [a for a in actions if a.action == "add"]

        for action in delete_actions:
            try:
                await delete_memory_by_id(memory_id=action.id, user=user)
                self.log(f"deleted memory. id={action.id}")
            except Exception as e:
                raise RuntimeError(f"failed to delete memory {action.id}: {e}")

        for action in update_actions:
            try:
                if action.new_content.strip():
                    await update_memory_by_id(
                        memory_id=action.id,
                        request=Request(scope={"type": "http", "app": webui_app}),
                        form_data=MemoryUpdateModel(content=action.new_content),
                        user=user,
                    )
                    self.log(f"updated memory. id={action.id}")
            except Exception as e:
                raise RuntimeError(f"failed to update memory {action.id}: {e}")

        for action in add_actions:
            try:
                if action.content.strip():
                    await add_memory(
                        request=Request(scope={"type": "http", "app": webui_app}),
                        form_data=AddMemoryForm(content=action.content),
                        user=user,
                    )
                    self.log(f"added memory. content={action.content}")
            except Exception as e:
                raise RuntimeError(f"failed to add memory: {e}")

        deleted_message = (
            f"deleted {len(delete_actions)} memories" if delete_actions else ""
        )
        updated_message = (
            f"updated {len(update_actions)} memories" if update_actions else ""
        )
        added_message = f"added {len(add_actions)} memories" if add_actions else ""
        status_parts = [
            part for part in [deleted_message, updated_message, added_message] if part
        ]
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
        self.log(f"inlet: {__name__}", level="debug")
        self.log(
            f"inlet: user ID: {__user__.get('id') if __user__ else 'no user'}",
            level="debug",
        )
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

        self.log(f"input user type = {type(__user__)}", level="debug")
        self.log(
            f"user.id = {user.id} user.name = {user.name} user.email = {user.email}",
            level="debug",
        )

        self.user_valves = __user__.get("valves", self.UserValves())
        if not isinstance(self.user_valves, self.UserValves):
            raise ValueError("invalid user valves")
        self.user_valves = cast(Filter.UserValves, self.user_valves)
        self.log(f"user valves = {self.user_valves}", level="debug")

        asyncio.create_task(
            self.auto_memory(
                body.get("messages", []), user=user, emitter=__event_emitter__
            )
        )

        return body
