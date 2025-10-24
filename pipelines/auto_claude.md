# Auto Claude

> Clean, plug-and-play Claude manifold pipeline with support for all the latest features from Anthropic.

<br>

## ✨ What It Does

**Auto Claude** provides seamless integration with Anthropic's Claude models in Open WebUI, supporting advanced features like:

- **Extended Thinking**: Enable Claude's reasoning capabilities with configurable budget tokens
- **Vision Support**: Process images and PDFs in your conversations
- **Tool Calling**: Full support for function calling with automatic execution loops
- **Multiple Model Variants**: Access to Claude Sonnet 4.5, Opus 4.1, and Haiku 4.5
- **Streaming Responses**: Real-time streaming with proper tool call handling

You get:

- Production-ready Claude integration with minimal setup
- Native support for both Anthropic and OpenAI SDK patterns
- Automatic tool execution without manual intervention
- Seamless multimodal conversations

<br>

## 💾 How It Works

- **Manifold Pipeline**: Exposes multiple Claude models through a single pipeline
- **Smart Token Management**: Automatically calculates and validates token budgets for extended thinking
- **Tool Execution Loop**: Handles multi-turn tool calling automatically
- **Format Conversion**: Seamlessly converts between OpenAI and Anthropic message formats
- **Multimodal Processing**: Validates and processes images and PDFs with size limits

---

## 🚀 Installation

1. Make sure your Open WebUI is version `0.5.0` or newer
2. Click on _GET_ to add the extension to your Open WebUI deployment
3. Add your Anthropic API key to the configuration

---

## ⚙️ Configuration

Configure via the Open WebUI extension settings:

| Setting             | Description             | Default                       |
| ------------------- | ----------------------- | ----------------------------- |
| `ANTHROPIC_API_KEY` | Your Anthropic API key  | _(from environment variable)_ |
| `debug_mode`        | Enable detailed logging | `false`                       |

---

## 🤖 Available Models

All models released after Claude Sonnet 4 are available in both standard and extended thinking variants.

---

## 🧠 Extended Thinking

Enable Claude's reasoning capabilities by using `-thinking` model variants or setting the `reasoning_effort` parameter:

### Reasoning Effort Levels

| Level    | Budget Tokens | Use Case                         |
| -------- | ------------- | -------------------------------- |
| `none`   | 0             | Standard responses (no thinking) |
| `low`    | 4,000         | Quick reasoning tasks            |
| `medium` | 16,000        | Moderate complexity problems     |
| `high`   | 32,000        | Complex reasoning and analysis   |
| `max`    | 48,000        | Most difficult reasoning tasks   |

You can also specify custom budget tokens as an integer value.

**Token Limit**: Combined `budget_tokens` + `max_tokens` cannot exceed 128,000 tokens.

### Example

```python
{
  "model": "claude-sonnet-4-5-20250929-thinking",
  "reasoning_effort": "high",
  "max_tokens": 8000
}
# This uses 32,000 thinking tokens + 8,000 output tokens = 40,000 total
```

---

## 🖼️ Multimodal Support

### Images

- **Supported formats**: JPEG, PNG, GIF, WebP
- **Max size**: 5 MB
- **Methods**: Base64 data URIs or direct URLs

### PDFs

- **Max size**: 32 MB
- **Methods**: Base64 data URIs or direct URLs

Images and PDFs are automatically validated for size and format before processing.

---

## 🛠️ Tool Calling

Auto Claude supports full tool calling with automatic execution loops:

1. **Automatic Detection**: Recognizes when Claude requests tool use
2. **Execution**: Runs the tool with provided arguments
3. **Result Injection**: Sends results back to Claude
4. **Multi-turn Support**: Continues until task completion
5. **Error Handling**: Gracefully handles tool execution failures

The pipeline handles both streaming and non-streaming tool calls seamlessly.

---

## 🔧 Advanced Features

### Dual SDK Support

- **Anthropic SDK**: Native support for Anthropic's message format and features
- **OpenAI SDK**: Compatible with OpenAI-style requests for easier migration

### Smart Message Processing

- Automatically converts between message formats
- Handles tool results and tool use blocks
- Preserves conversation context across tool calls

---

## 🙌 Credits

- Created by [nokodo](https://nokodo.net)
- Built with the official [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)

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
All copies must retain this notice. Provided "AS IS" without warranty.  
Earlier pre-release versions may have been available under different terms.

---

_Experience Claude's full potential in Open WebUI—automatically!_
