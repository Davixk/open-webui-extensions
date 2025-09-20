# Open WebUI Extensions

This repository contains a collection of extensions for [Open WebUI](https://github.com/open-webui/open-webui), enhancing its functionality and user experience.

You can find the latest version of each extension on [openwebui.com](https://openwebui.com/f/nokodo/).

## Available Extensions

### Auto Memory Function

Automatically identifies and stores valuable information from chats as Memories in Open WebUI. This extension analyzes conversations and extracts key details to remember for future interactions.

#### Features:

- Automatically captures relevant information from user messages
- Consolidates similar or overlapping memories to avoid duplication
- Updates conflicting information with the most recent data
- Optionally saves assistant responses as memories
- Configurable memory processing parameters

#### Configuration Options:

- **OpenAI API URL**: Configure a compatible endpoint for memory processing
- **Model**: Select which model to use for memory identification
- **API Key**: Set your API key for the chosen endpoint
- **Related Memories**: Control how many previous memories to consider when updating
- **Memory Distance**: Adjust the threshold for determining related memories
- **Save Assistant Responses**: Option to automatically save assistant messages as memories
- **Legacy Mode**: Use simpler memory processing for older setups
- **Messages to Consider**: Set how many recent messages to analyze for memory extraction

### Easy Installation (Recommended)

1. Navigate to the [Open WebUI Extensions page](https://openwebui.com/f/nokodo/auto_memory)
2. Click on the "Get" button to automatically install the extension
3. Configure the extension through the Open WebUI settings interface

### Manual Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/davixk/open-webui-extensions.git
   ```
2. Copy the desired extension(s) to your Open WebUI functions directory:
   ```bash
   cp -r functions/auto_memory_function.py /path/to/open-webui/functions/
   ```
3. Restart Open WebUI to load the new extensions.

## Usage

Each extension can be configured through the Open WebUI settings interface under the "Extensions" section. Enable the desired extensions and adjust their settings as needed.

## Developing Extensions

To create new extensions for Open WebUI:

1. Use the existing extensions as templates
2. Follow the function structure with `inlet` and `outlet` methods
3. Create appropriate configuration models using Pydantic
4. Add comprehensive documentation in the extension's docstring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See individual extension documentation for specific licensing terms.

## Support

For questions, issues, or feature requests, please [open an issue](https://github.com/davixk/open-webui-extensions/issues) on GitHub.
