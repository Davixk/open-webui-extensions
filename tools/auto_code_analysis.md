# 🐍 Auto Code Analysis

> Execute Python code to perform any analysis, file operations, or data processing tasks — directly within Open WebUI.

<br>

## ✨ What It Does

**Auto Code Analysis** gives your Open WebUI assistant the ability to write and execute Python code. This allows it to perform complex calculations, analyze data, process files, and generate visualizations on the fly.

You get:

- 🐍 **Python Execution**: Run Python code securely in a sandboxed environment
- 📊 **Data Analysis**: Process and analyze data from files or conversation context
- 📈 **Visualizations**: Generate charts and graphs that are displayed directly in the chat
- 📁 **File Operations**: Read, write, and manipulate files
- 🛡️ **Sandboxed Environments**: Choose between self-hosted Jupyter or cloud-based E2B sandboxes

<br>

## 💾 How It Works

- **Code Generation**: The assistant writes Python code to solve your problem
- **Execution**: The code is executed in a secure sandbox (Jupyter or E2B)
- **Output Handling**: Standard output, errors, and results are captured and returned to the assistant
- **File Management**: Generated files (like charts) are automatically uploaded and displayed in the chat

---

## 🚀 Installation

1. Make sure your Open WebUI is version `0.6.0` or newer
2. Click on _GET_ to add the extension to your Open WebUI deployment
3. Configure your execution engine (Jupyter or E2B) in the settings

---

## ⚙️ Configuration

Configure via the Open WebUI extension settings:

| Setting                   | Description                                                               | Default                            |
| ------------------------- | ------------------------------------------------------------------------- | ---------------------------------- |
| `ENGINE`                  | Execution engine to use: `jupyter` (self-hosted) or `e2b` (cloud sandbox) | `jupyter`                          |
| `JUPYTER_URL`             | Jupyter server URL (if using Jupyter engine)                              | `http://host.docker.internal:8888` |
| `JUPYTER_AUTH`            | Jupyter authentication method (`token`, `password`, `none`)               | `token`                            |
| `JUPYTER_TOKEN`           | Jupyter authentication token                                              | _(empty)_                          |
| `JUPYTER_PASSWORD`        | Jupyter password                                                          | _(empty)_                          |
| `JUPYTER_RETENTION_HOURS` | Retention period for Jupyter session directories in hours                 | `72`                               |
| `E2B_API_KEY`             | E2B API Key (if using E2B engine)                                         | _(empty)_                          |
| `TIMEOUT`                 | Execution timeout in seconds                                              | `60`                               |

---

## 🙌 Credits

- Created by [nokodo](https://nokodo.net)
- Built for Open WebUI with ❤️

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

_Empower your AI with the ability to code—automatically!_
