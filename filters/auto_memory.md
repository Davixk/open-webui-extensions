# Auto Memory

> Automatically identify and store relevant information from chats as Memories in Open WebUI.

<br>

## ✨ What It Does

**Auto Memory** listens in on your conversations and detects facts, preferences, key moments, or anything useful for the assistant to remember about you.
It stores these as separate memories, so future AI interactions stay personal and context-aware—_without you micromanaging_.

You get:

- Seamless journaling of your important info
- Smarter, context-rich AI assistance
- No “please remember X” (unless you _want_ to!)

> **Note:** Make sure to enable the Memory feature in your user profile settings (Profile → Settings → Personalization) to allow models to access your memories!

<br>

## 💾 How It Works

- **Auto-extracts** new or changed "facts" from recent user messages
- **Stores each fact** separately in your Memory database
- **Auto-maintains** Memories: merges duplicates, resolves conflicts, and prunes old/irrelevant ones
- Uses advanced LLMs to understand context and nuance

---

## 🚀 Installation

1. Make sure your Open WebUI is version `0.5.0` or newer
2. Click on _GET_ to add the extension to your Open WebUI deployment
3. Configure API keys and model (see below)

---

## ⚙️ Configuration

Configure via the Open WebUI extension settings or directly in code:

| Setting                 | Description                                           | Default                  |
| ----------------------- | ----------------------------------------------------- | ------------------------ |
| `openai_api_url`        | OpenAI-compatible API endpoint                        | `https://api.openai.com` |
| `model`                 | LLM model for memory identification                   | `gpt-5-mini`             |
| `api_key`               | API key for the chosen endpoint                       | _(empty)_                |
| `related_memories_n`    | Number of related memories to check for consolidation | `5`                      |
| `related_memories_dist` | Similarity distance threshold for related memories    | `0.75`                   |
| `messages_to_consider`  | How many recent messages to consider (user+assistant) | `4`                      |
| `show_status`           | Display memory save status on UI                      | `true`                   |
| `debug_mode`            | Enable detailed logging for troubleshooting           | `false`                  |

Supports per-user overrides.

---

## 🧠 Memory Extraction Logic

- New or changed facts from User's latest message are saved.
- Explicit "please remember..." requests always create a Memory.
- Avoids duplicates & merges conflicts.
- Automatically deletes and maintains Memories over time.

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

See full logic and more cases in code.

---

## 🧰 Extension Metadata

```
title: Auto Memory
author: nokodo
version: 1.0.0-alpha9
required_open_webui_version: >= 0.5.0
repository_url: https://nokodo.net/github/open-webui-extensions
funding_url: https://ko-fi.com/nokodo
```

---

## 🙌 Credits

- Created by [nokodo](https://nokodo.net)

---

## 💖 Support & Feedback

- [Open an Issue / Suggest Improvements](https://nokodo.net/github/open-webui-extensions)
- [Buy me a coffee ☕](https://ko-fi.com/nokodo)

---

## 📜 License

Source-Available – No Redistribution Without Permission
Copyright (c) 2025 nokodo

You are free to use, run, and modify this extension for personal or internal purposes.
You may NOT redistribute, publish, sublicense or sell this extension or any modified version without prior explicit written consent from the author.
All copies must retain this notice. Provided “AS IS” without warranty.
Earlier pre-release versions may have been available under different terms.

---

_Keep your AI tuned in to who you really are—automatically!_
