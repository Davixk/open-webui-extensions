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

| Setting                       | Description                                                                                                                                                                                                                | Default                  |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `openai_api_url`              | OpenAI-compatible API endpoint                                                                                                                                                                                             | `https://api.openai.com` |
| `model`                       | LLM model for memory identification                                                                                                                                                                                        | `gpt-5-mini`             |
| `api_key`                     | API key for the chosen endpoint                                                                                                                                                                                            | _(empty)_                |
| `related_memories_n`          | Number of related memories to check for consolidation                                                                                                                                                                      | `5`                      |
| `related_memories_dist`       | Similarity distance threshold for related memories                                                                                                                                                                         | `0.75`                   |
| `messages_to_consider`        | How many recent messages to consider (user+assistant)                                                                                                                                                                      | `4`                      |
| `allow_unsafe_user_overrides` | **SECURITY WARNING**: Allow users to override API URL/model without providing their own API key. This could allow users to steal your API key or use expensive models at your expense. Only enable if you trust all users. | `false`                  |
| `debug_mode`                  | Enable detailed logging for troubleshooting                                                                                                                                                                                | `false`                  |

### User-Level Settings

Users can override settings individually:

| Setting                | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `enabled`              | Whether to enable Auto Memory for this user    |
| `show_status`          | Display memory save status on UI               |
| `openai_api_url`       | User-specific API endpoint (overrides global)  |
| `model`                | User-specific model (overrides global)         |
| `api_key`              | User-specific API key (overrides global)       |
| `messages_to_consider` | User-specific message count (overrides global) |

**Security Note**: User overrides for `openai_api_url` and `model` require the user to provide their own `api_key`. Without it, the admin's global settings are used for security. Admins can enable `allow_unsafe_user_overrides` to bypass this check, but this is not recommended.

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
version: 1.0.0-alpha16
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
