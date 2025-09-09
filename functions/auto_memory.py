"""
title: Auto Memory
author: @nokodo
description: Automatically identify and store valuable information from chats as Memories.
author_email: nokodo@nokodo.net
author_url: https://nokodo.net
repository_url: https://nokodo.net/github/open-webui-extensions
version: 0.5.3
required_open_webui_version: >= 0.5.0
funding_url: https://ko-fi.com/nokodo
"""

import ast
import json
import time
from typing import Any, Awaitable, Callable, Literal, Optional, cast

import aiohttp
from aiohttp import ClientError
from fastapi.requests import Request
from open_webui.main import app as webui_app
from open_webui.models.users import UserModel, Users
from open_webui.routers.memories import (
    AddMemoryForm,
    QueryMemoryForm,
    add_memory,
    delete_memory_by_id,
    query_memory,
)
from pydantic import BaseModel, Field

LogLevel = Literal["debug", "info", "warning", "error"]

STRINGIFIED_MESSAGE_TEMPLATE = "-{index}. {role}: ```{content}```"

IDENTIFY_MEMORIES_PROMPT = """\
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

CONSOLIDATE_MEMORIES_PROMPT = """You are maintaining a set of “Memories” for a user, similar to journal entries. Each memory has:
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


LOG_FORMAT = "[Auto Memory][{level}] {message}"


class Filter:
    class Valves(BaseModel):
        openai_api_url: str = Field(
            default="https://api.openai.com",
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
        save_assistant_response: bool = Field(
            default=False,
            description="automatically save assistant responses as memories",
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
        """Unified logger.

        Args:
            message: Text to log
            level: LogLevel
        """
        if level == "debug" and not self.valves.debug_mode:
            return
        # Simple normalization
        if level not in {"debug", "info", "warning", "error"}:
            level = "info"
        print(LOG_FORMAT.format(level=level, message=message))

    def __init__(self):
        self.valves = self.Valves()

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
        # --- Initialization & validation ---
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

        # --- Message handling ---
        messages = body.get("messages", [])
        self.log(
            f"debug user={'yes' if user else 'no'} messages={len(messages)} api_url={self.user_valves.openai_api_url or self.valves.openai_api_url} model={self.user_valves.model or self.valves.model}",
            level="debug",
        )

        if len(messages) == 0:
            self.log("skipping (no messages)")
        elif len(messages) < 2:
            self.log("skipping (need >=2 messages for context)")
        elif not (self.user_valves.api_key or self.valves.api_key):
            self.log("skipping (no API key configured)")
        else:
            # Require at least 2 messages (one user + one prior context) to attempt extraction
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
            prompt_string = "\n".join(stringified_messages)
            self.log("calling identify_memories", level="debug")
            try:
                memories = await self.identify_memories(prompt_string)
            except Exception as e:
                self.log(f"identify_memories error: {e}", level="error")
                memories = "[]"
            self.log(f"raw identify response: {memories[:200]}", level="debug")
            if (
                memories.startswith("[")
                and memories.endswith("]")
                and len(memories) != 2
            ):
                result = await self.process_memories(memories, user)
                if self.user_valves.show_status:
                    desc = (
                        f"added memory: {memories}"
                        if result
                        else f"memory failed: {result}"
                    )
                    await __event_emitter__(
                        {"type": "status", "data": {"description": desc, "done": True}}
                    )
            else:
                self.log("no new memories identified")
                self.log(f"raw response: {memories[:120]}...", level="debug")

        # --- Optional assistant response memory ---
        if (
            user
            and self.valves.save_assistant_response
            and len(body.get("messages", [])) > 0
        ):
            last_assistant_message = body["messages"][-1]
            try:
                memory_obj = await add_memory(
                    request=Request(scope={"type": "http", "app": webui_app}),
                    form_data=AddMemoryForm(content=last_assistant_message["content"]),
                    user=user,
                )
                self.log(f"assistant memory added: {memory_obj}")
                if self.user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "memory saved", "done": True},
                        }
                    )
            except Exception as e:
                self.log(f"error adding assistant memory {str(e)}", level="error")
                if self.user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "error saving memory",
                                "done": True,
                            },
                        }
                    )

        return body

    async def identify_memories(self, input_text: str) -> str:
        memories = await self.query_openai_api(
            system_prompt=IDENTIFY_MEMORIES_PROMPT,
            prompt=input_text,
        )
        return memories

    async def query_openai_api(self, system_prompt: str, prompt: str) -> str:
        api_url = self.user_valves.openai_api_url or self.valves.openai_api_url
        model = self.user_valves.model or self.valves.model
        api_key = self.user_valves.api_key or self.valves.api_key

        url = f"{api_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            self.log(f"sending LLM request model={model} url={url}", level="debug")
            async with aiohttp.ClientSession(timeout=timeout) as session:
                response = await session.post(url, headers=headers, json=payload)
                response.raise_for_status()
                json_content = await response.json()
            self.log("received LLM response", level="debug")
            return json_content["choices"][0]["message"]["content"]
        except ClientError as e:
            # Fixed error handling
            error_msg = str(
                e
            )  # Convert the error to string instead of trying to access .response
            raise Exception(f"http error: {error_msg}")
        except Exception as e:
            raise Exception(f"unexpected error: {str(e)}")

    async def process_memories(
        self,
        memories: str,
        user: UserModel,
    ) -> bool:
        """
        given a list of memories as a string, go through each memory, check for duplicates, then store the remaining memories
        """
        try:
            memory_list = ast.literal_eval(memories)
            self.log(f"identified {len(memory_list)} new memories", level="info")
            for memory in memory_list:
                await self.store_memory(memory, user)
            return True
        except Exception as e:
            self.log(f"error processing memories: {e}")
            return False

    async def store_memory(
        self,
        memory: str,
        user: UserModel,
    ) -> str:
        """
        given a memory, retrieve related memories. update conflicting memories and consolidate memories as needed. then store remaining memories
        """
        try:
            related_memories = await query_memory(
                request=Request(scope={"type": "http", "app": webui_app}),
                form_data=QueryMemoryForm(
                    content=memory, k=self.valves.related_memories_n
                ),
                user=user,
            )
            if related_memories is None:
                related_memories = [
                    ["ids", [["123"]]],
                    ["documents", [["blank"]]],
                    ["metadatas", [[{"created_at": 999}]]],
                    ["distances", [[100]]],
                ]
        except Exception as e:
            return f"unable to query related memories: {e}"
        try:
            # Make a more useable format
            related_list = [obj for obj in related_memories]
            ids = related_list[0][1][0]
            documents = related_list[1][1][0]
            metadatas = related_list[2][1][0]
            distances = related_list[3][1][0]
            # Combine each document and its associated data into a list of dictionaries
            structured_data = [
                {
                    "id": ids[i],
                    "fact": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i],
                }
                for i in range(len(documents))
            ]
            # Filter for distance within threshhold
            filtered_data = [
                item
                for item in structured_data
                if item["distance"] < self.valves.related_memories_dist
            ]
            # Limit to relevant data to minimize tokens
            self.log(f"filtered data: {filtered_data}", level="debug")
            fact_list = [
                {"fact": item["fact"], "created_at": item["metadata"]["created_at"]}
                for item in filtered_data
            ]
            fact_list.append({"fact": memory, "created_at": time.time()})
        except Exception as e:
            return f"unable to restructure and filter related memories: {e}"
        # Consolidate conflicts or overlaps
        try:
            consolidated_memories = await self.query_openai_api(
                system_prompt=CONSOLIDATE_MEMORIES_PROMPT,
                prompt=json.dumps(fact_list),
            )
        except Exception as e:
            return f"Unable to consolidate related memories: {e}"
        try:
            # Add the new memories
            memory_list = ast.literal_eval(consolidated_memories)
            for item in memory_list:
                try:
                    await add_memory(
                        request=Request(scope={"type": "http", "app": webui_app}),
                        form_data=AddMemoryForm(content=item),
                        user=user,
                    )
                except Exception as inner:
                    self.log(f"failed adding memory '{item}': {inner}", level="error")
        except Exception as e:
            return f"unable to add consolidated memories: {e}"
        try:
            # Delete the old memories
            if len(filtered_data) > 0:
                for id in [item["id"] for item in filtered_data]:
                    await delete_memory_by_id(id, user)
        except Exception as e:
            return f"unable to delete related memories: {e}"
        return "ok"
