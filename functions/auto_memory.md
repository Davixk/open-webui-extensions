# Auto Memory

Automatically identify and store relevant information from chats as Memories in Open WebUI.  
Based on the original function by [@devve](https://openwebui.com/u/devve).

<br>

## ✨ What It Does

**Auto Memory** listens in on your conversations and detects facts, preferences, key moments, or anything useful for the assistant to remember about you.  
It stores these as separate memories, so future AI interactions stay personal and context-aware—_without you micromanaging_.

You get:

-   Seamless journaling of your important info
-   Smarter, context-rich AI assistance
-   No more “please remember X” (unless you _want_ to!)

<br>

## 💾 How It Works

-   **Auto-extracts** new or changed "facts" from recent user messages
-   **Stores each fact** separately in your Memory database
-   **Consolidates/conflicts** are resolved: more recent info replaces the old
-   Optionally can save assistant responses as memories (`save_assistant_response`)
-   Uses advanced LLMs (like GPT-4o) to understand context and nuance

---

## 🚀 Installation

1. Make sure your Open WebUI is version `0.5.0` or newer.
2. Click on _GET_ to add the extension to your Open WebUI deployment
3. Configure API keys and model (see below).

---

## ⚙️ Configuration

Configure via the Open WebUI extension settings or directly in code:

| Setting                   | Description                                                        | Default                  |
| ------------------------- | ------------------------------------------------------------------ | ------------------------ |
| `openai_api_url`          | OpenAI-compatible API endpoint                                     | `https://api.openai.com` |
| `model`                   | LLM model for memory identification                                | `gpt-4o`                 |
| `api_key`                 | API key for the chosen endpoint                                    | _(empty)_                |
| `related_memories_n`      | Number of related memories to check for consolidation              | `5`                      |
| `related_memories_dist`   | Similarity distance threshold for related memories                 | `0.75`                   |
| `save_assistant_response` | Also auto-save assistant replies as memories                       | `false`                  |
| `use_legacy_mode`         | Use simplified old prompts & only last user message for extraction | `false`                  |
| `messages_to_consider`    | How many recent messages to consider (user+assistant)              | `4`                      |
| `show_status`             | Display memory save status on UI                                   | `true`                   |

Supports per-user overrides.

---

## 🧠 Memory Extraction Logic

-   New or changed facts from User's latest message are saved.
-   Explicit "please remember..." requests always create a Memory.
-   Avoids duplicates & merges conflicts by keeping only the latest.
-   Filters out ephemeral/trivial/short-term details.

### Example

Conversation:

```
-4. user: I love oranges 😍
-3. assistant: That's great!
-2. user: Actually, I hate oranges 😂
-1. assistant: omg you LIAR 😡
```

Memory stored:

```python
["User hates oranges"]
```

See full logic and more cases in code (`IDENTIFY_MEMORIES_PROMPT`, etc).

---

## 🧰 Extension Metadata

```
title: Auto Memory (post 0.5)
author: nokodo, based on devve
version: 0.4.8
required_open_webui_version: >= 0.5.0
repository_url: https://nokodo.net/github/open-webui-extensions
funding_url: https://ko-fi.com/nokodo
```

---

## 🙌 Credits

-   Created by [nokodo](https://nokodo.net)
-   Based on [@devve](https://openwebui.com/u/devve)’s original design

---

## 💖 Support & Feedback

-   [Open an Issue / Suggest Improvements](https://nokodo.net/github/open-webui-extensions)
-   [Buy me a coffee ☕](https://ko-fi.com/nokodo)

---

## 📜 License

MIT @ nokodo

---

_Keep your AI tuned in to who you really are—automatically!_
