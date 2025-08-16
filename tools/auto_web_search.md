# 🔎 Auto Web Search (Open WebUI Tool)

A lightweight, native-first web search tool for Open WebUI. It plugs into your chats to fetch current, factual information and returns clean, citeable results — no external services required.

-   ⚡ Fast, native search via Open WebUI’s built-in retrieval pipeline
-   🔗 Clear citations (source + content) streamed back to the chat
-   🧰 Sensible defaults with simple configuration
-   🧩 Optional compatibility path for a separate Perplexica mode (off by default)

## ✨ Features

-   🔎 Native web search through Open WebUI’s native `Web Search`
-   ➕ Accepts multiple queries in one call
-   📡 Emits status updates and citeable snippets as results stream in
-   🧾 Structured JSON response

## 🧭 How it works

The tool exposes a single function, `web_search`, that takes an array of search queries. Under the default `native` mode, it uses Open WebUI’s built-in search and retrieval pipeline and streams:

1. ⏳ A short status message (e.g., “searching the web for: …”)
2. 🔗 One citation event per matched document (source + snippet)
3. ✅ A completion status with a compact JSON payload you can reuse or display

## 🛠️ Installation

This tool is part of the extension repo and relies on Open WebUI APIs. Ensure your Open WebUI instance is running (>= 0.6.0). The only Python dependency added by this tool is `aiohttp`.

### 📦 Requirements

-   Open WebUI >= 0.6.0
-   Python environment consistent with your Open WebUI deployment
-   Python package: `aiohttp`

### ⚙️ Quick setup

-   Place `auto_web_search.py` under `tools/` in your Open WebUI extensions folder (already done in this repo).
-   Make sure your environment can import from `open_webui`.

If you need to install `aiohttp` in your environment:

```powershell
pip install aiohttp
```

## ▶️ Usage

The tool registers itself with a single function you can call from your assistant runtime.

-   Function name: `web_search`
-   Parameter: `search_queries` (array of 1–5 strings)
-   Returns: JSON string with status, result_count, and results (each result includes `source` and `content`)

Example shape:

```json
{
	"status": "web search completed successfully!",
	"result_count": 3,
	"results": [
		{ "source": "https://example.com/post", "content": "Snippet text…" },
		{ "source": "https://another.com/page", "content": "Snippet text…" }
	]
}
```

### 🔔 Events emitted

-   Status events: progress updates and completion
-   Citation events: one per result, containing source metadata and a content snippet

This makes it easy to render live search activity and include citations in model responses.

## ⚙️ Configuration

You can adjust behavior via the `Valves` model on the tool:

-   `SEARCH_MODE`: `"native"` | `"perplexica"` (default: `native`)

Legacy, optional parameters for the separate Perplexica flow (not needed for native mode):

-   `PERPLEXICA_BASE_URL` (default: `http://host.docker.internal:3001`)
-   `PERPLEXICA_OPTIMIZATION_MODE`: `"speed" | "balanced"` (default: `balanced`)
-   `PERPLEXICA_CHAT_MODEL` (default: `gpt-5-chat-latest`)
-   `PERPLEXICA_EMBEDDING_MODEL` (default: `bge-m3:latest`)

> Tip: Keep `SEARCH_MODE` as `native` for the cleanest, lowest-latency experience fully within Open WebUI.

## 🎯 Good query patterns

-   Ask specific, factual questions
-   Include entities, dates, or locations
-   Provide 1–3 related variations to broaden coverage

Examples:

-   "What changed in Python 3.13 release notes?"
-   "Latest CVE advisories for OpenSSL 3.2"
-   ["NVIDIA earnings Q2 2025", "NVIDIA revenue Q2 2025"]

## 🛟 Troubleshooting

-   No results returned
    -   Try narrower queries or add a second query variation
    -   Check that your Open WebUI has network access from its host
-   ImportError: cannot import `open_webui.*`
    -   Ensure you are running inside the Open WebUI app environment or have its packages available
-   Permission or user issues
    -   The tool expects a valid `__user__` with an `id`; ensure your caller supplies it

## 🤝 Why this tool

The goal is to make web search inside Open WebUI feel native: minimal setup, helpful streaming feedback, and clear citations you can trust. If your use case needs external search stacks, you can switch `SEARCH_MODE`, but the native path remains the recommended default.

## ⚖️ License

MIT © nokodo
