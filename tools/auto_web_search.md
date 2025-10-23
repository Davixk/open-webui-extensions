# 🔎 Auto Web Search

A lightweight, native-first web search tool for Open WebUI. It plugs into your chats to fetch current, factual information and returns clean, citeable results — no external services required.

- ⚡ Fast, native search via Open WebUI’s built-in retrieval pipeline
- 🔗 Clear citations (source + content) streamed back to the chat
- 🧰 Sensible defaults with simple configuration
- 🧩 Optional compatibility path for a separate Perplexica mode (off by default)

## ✨ Features

- 🔎 Native web search through Open WebUI’s native `Web Search`
- ➕ Accepts multiple queries in one call
- 📡 Emits status updates and citeable snippets as results stream in
- 🧾 Structured summary of results

## 🧭 How it works

Ask a question in chat that requires current or factual information. Under the default `native` mode, the tool uses Open WebUI’s built-in search and retrieval pipeline and streams:

1. ⏳ A short status message (e.g., “searching the web for: …”)
2. 🔗 One citation event per matched document (source + snippet)
3. ✅ A completion status with a compact summary you can reuse or display

## 🚀 Installation

1. Make sure your Open WebUI is version `0.6.0` or newer.
2. Click on _GET_ to add the extension to your Open WebUI deployment.
3. Configure settings (see below).

## ▶️ Usage

Once installed and enabled, this tool is available inside chats. Simply ask your question — the tool will:

- Search the web using Open WebUI’s native pipeline
- Stream status updates and citations back into the chat
- Summarize results when complete

### 🔔 What you'll see

- Progress updates (e.g., “searching the web for…”) and a completion message
- Citations for each result with source and snippet, inline in the chat

## ⚙️ Configuration

Configure everything from the Open WebUI extension settings — no code required.

| Setting                        | Description                                               | Default                             |
| ------------------------------ | --------------------------------------------------------- | ----------------------------------- |
| `SEARCH_MODE`                  | Search engine to use. Native is recommended.              | `native`                            |
| `PERPLEXICA_BASE_URL`          | Base URL of your Perplexica server (legacy/optional).     | `http://host.docker.internal:3001`  |
| `PERPLEXICA_OPTIMIZATION_MODE` | Perplexica strategy: `speed` or `balanced`.               | `balanced`                          |
| `PERPLEXICA_CHAT_MODEL`        | Chat model used by the Perplexica pipeline.               | `gpt-5-chat-latest`                 |
| `PERPLEXICA_EMBEDDING_MODEL`   | Embedding model used by the Perplexica pipeline.          | `bge-m3:latest`                     |
| `OLLAMA_BASE_URL`              | Base URL for the Ollama provider when used by Perplexica. | `http://host.docker.internal:11434` |

Tip: Keep `SEARCH_MODE` set to `native` for the cleanest, lowest-latency experience fully within Open WebUI.

Supports per-user overrides.

## 🎯 Good query patterns

- Ask specific, factual questions
- Include entities, dates, or locations
- Provide 1–3 related variations to broaden coverage

Examples:

- "What changed in Python 3.13 release notes?"
- "Latest CVE advisories for OpenSSL 3.2"
- ["NVIDIA earnings Q2 2025", "NVIDIA revenue Q2 2025"]

## 🛟 Troubleshooting

- No results returned
  - Try narrower queries or add a second query variation
  - Check that your Open WebUI has network access from its host
- Permission or user issues
  - The tool expects a valid user context in the UI; ensure you’re signed in and have permission to use extensions

## 🤝 Why this tool

The goal is to make web search inside Open WebUI feel native: minimal setup, helpful streaming feedback, and clear citations you can trust. If your use case needs external search stacks, you can switch `SEARCH_MODE`, but the native path remains the recommended default.

## 📜 License

Source-Available – No Redistribution Without Permission
Copyright (c) 2025 nokodo

You are free to use, run, and modify this extension for personal or internal purposes.
You may NOT redistribute, publish, sublicense or sell this extension or any modified version without prior explicit written consent from the author.
All copies must retain this notice. Provided "AS IS" without warranty.
Earlier pre-release versions may have been available under different terms.
