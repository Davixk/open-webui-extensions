# 🎨 Auto Image

> Automated image generation and editing tools using AI — seamlessly integrated into your Open WebUI conversations.

<br>

## ✨ What It Does

**Auto Image** brings powerful AI image generation and editing capabilities directly into your Open WebUI chats. Create stunning visuals from text descriptions or modify existing images with simple prompts — no external tools or complex workflows required.

You get:

- 🎨 **Text-to-Image Generation**: Create original images from detailed text descriptions
- ✏️ **AI-Powered Image Editing**: Modify and transform existing images with natural language
- 🔄 **Multiple Variations**: Generate multiple images or edits in a single call
- 📐 **Flexible Sizing**: Customize output dimensions to fit your needs
- 🎯 **Negative Prompts**: Specify what you don't want in your images for better control

<br>

## 💾 How It Works

- **Text-to-Image**: Converts your detailed text descriptions into images using Open WebUI's configured image generation engine
- **Image Editing**: Accesses images from your conversation context and applies AI-driven modifications based on your prompts
- **Streaming Feedback**: Displays real-time status updates and automatically embeds generated images in the chat
- **Smart Context**: Automatically finds and uses images from your current conversation for editing tasks

---

## 🚀 Installation

1. Make sure your Open WebUI is version `0.6.0` or newer
2. Configure an image generation engine in Open WebUI (Settings → Images)
3. Click on _GET_ to add the extension to your Open WebUI deployment
4. Enable the tool in your chat settings

---

## ⚙️ Configuration

Configure via the Open WebUI extension settings:

| Setting            | Description                                                                  | Default     |
| ------------------ | ---------------------------------------------------------------------------- | ----------- |
| `IMAGE_SIZE`       | Default image dimensions in WIDTHxHEIGHT format (e.g., 512x512, 1024x1024)   | `1024x1024` |
| `IMAGE_STEPS`      | Number of generation steps (if supported by your image engine)               | _(none)_    |
| `DEFAULT_N`        | Default number of images to generate per prompt (1-10)                       | `1`         |
| `CHECK_CHAT_FILES` | Whether to check for chat-level files in addition to message-embedded images | `false`     |

---

## 🎨 Image Generation

Create images from scratch using text descriptions. The tool uses Open WebUI's configured image generation engine.

### Usage Pattern

Simply ask the assistant to create or generate an image:

```
"Create an image of a serene mountain landscape at sunset"
"Generate a futuristic city with flying cars"
"Make me a picture of a cute robot reading a book"
```

### Advanced Options

You can specify additional parameters:

- **Multiple Images**: "Generate 3 variations of..."
- **Custom Size**: "Create a 512x512 image of..."
- **Negative Prompts**: Tell the assistant what to avoid (e.g., "no people, no text")

### Example

```
User: Create a detailed image of a magical forest with glowing mushrooms and fireflies.
      Make it mystical and enchanting. Generate 2 variations.
Assistant: I'll create two mystical forest images for you.
```

**What happens:**

1. The assistant identifies the image generation request
2. Constructs a detailed prompt from your description
3. Calls the configured image engine with `n=2` for two variations
4. Streams status updates ("creating image...")
5. Displays both generated images in the chat

---

## ✏️ Image Editing

Modify existing images in your conversation using natural language prompts.

### Usage Pattern

Upload or paste an image into the chat, then ask for modifications:

```
"Edit this image to add a starry night sky"
"Change the colors to warm autumn tones"
"Remove the background and make it transparent"
"Add more detail to the foreground"
```

### How It Works

1. **Automatic Image Detection**: The tool scans your conversation for images
2. **Smart Indexing**: Uses the most recent image by default, or you can specify which one
3. **Context-Aware Editing**: Preserves the original image structure while applying your changes

### Image Selection

- **Most Recent** (default): Automatically uses the last image in the conversation
- **Specific Image**: The assistant can reference earlier images when needed
- **Multiple Sources**: Works with both uploaded files and inline images

### Example

```
User: [uploads image of a beach scene]
      Edit this to make it look like sunset with dramatic clouds
Assistant: I'll edit your beach image to add sunset and dramatic clouds.
            [Edited image appears in chat]
```

---

## 🎯 Tips for Best Results

### Writing Good Prompts

- **Be Specific**: Include details about style, mood, colors, and composition
- **Use Descriptive Language**: "dramatic lighting" > "nice lighting"
- **Mention Medium**: "oil painting", "photograph", "digital art", "watercolor"
- **Include Context**: Time of day, weather, atmosphere, perspective

### Effective Negative Prompts

Use negative prompts to avoid common issues:

- Quality problems: "blurry, low quality, pixelated, distorted"
- Unwanted elements: "text, watermark, people, animals"
- Style constraints: "cartoon, anime, 3D render" (if you want photorealism)

### Size Recommendations

- **1024x1024**: Best for general use, square images
- **512x512**: Faster generation, smaller files
- **1024x768** or **768x1024**: Landscape or portrait orientations
- Check your image engine's supported sizes for optimal results

---

## 🛟 Troubleshooting

### No Images Generated

- Verify that an image generation engine is configured in Open WebUI (Settings → Images)
- Check that the engine is running and accessible
- Review image engine logs for specific error messages

### Image Editing Not Working

- Make sure `CHECK_CHAT_FILES` is enabled if using chat-level attachments
- Verify that the image is visible in the conversation
- Try explicitly mentioning "edit the image I just uploaded"
- Check that your image engine supports editing (not all do)

### Poor Quality Results

- Provide more detailed, specific prompts
- Adjust `IMAGE_STEPS` if supported by your engine
- Try adding negative prompts to exclude unwanted elements
- Experiment with different sizes and aspect ratios

### Tool Not Appearing in Chat

- Ensure the extension is enabled in your chat settings
- Check that Open WebUI version is 0.6.0 or newer
- Verify the tool is not disabled in user or admin settings

---

## 🎨 Supported Image Engines

Auto Image works with any image generation engine configured in Open WebUI, including:

- **Automatic1111 / Stable Diffusion WebUI**
- **ComfyUI**
- **DALL-E (via OpenAI API)**
- **Midjourney (via API)**
- **Local Stable Diffusion models**
- Any other OpenAI-compatible image generation API

Configuration is handled through Open WebUI's native image settings—no additional setup required!

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

_Bring your imagination to life—automatically!_
