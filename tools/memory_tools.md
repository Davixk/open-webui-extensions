# 🧠 Memory Tools

> Native tools for AI models to interact with and query Open WebUI's built-in long-term memory system.

<br>

## ✨ What It Does

**Memory Tools** empowers AI models with direct access to the user's long-term memory vectorstore. Instead of relying on automatic memory injection in system prompts, models can now proactively search and retrieve relevant memories when needed, enabling more contextual and personalized interactions.

You get:

- 🔍 **Direct Memory Access**: Models can query the memory vectorstore on-demand
- 🎯 **Semantic Search**: Natural language queries find relevant memories via vector similarity
- ⚡ **Real-time Retrieval**: Fetch memories instantly during conversation
- 📊 **Configurable Results**: Control how many memories to retrieve and filter by relevance
- 🧹 **Clean Output**: Timestamps and content returned in a structured format

<br>

## 💾 How It Works

- **Vector Search**: Uses Open WebUI's native memory vectorstore to find semantically similar memories
- **Smart Filtering**: Applies similarity thresholds to return only relevant results
- **Structured Response**: Returns memories with creation/update timestamps in JSON format
- **Status Updates**: Provides real-time feedback during memory retrieval

---

## 🚀 Installation

1. Make sure your Open WebUI is version `0.5.0` or newer
2. Ensure the memory system is enabled and configured in Open WebUI
3. Click on _GET_ to add the extension to your Open WebUI deployment
4. Enable the tool in your chat settings

---

## ⚙️ Configuration

Configure via the Open WebUI extension settings:

| Setting                        | Description                                            | Default |
| ------------------------------ | ------------------------------------------------------ | ------- |
| `memory_max_k`                 | Maximum number of memories to retrieve (1-50)          | `15`    |
| `minimum_similarity_threshold` | Minimum similarity score to include memories (0.0-1.0) | `0.65`  |
| `show_status`                  | Show status messages during memory retrieval           | `true`  |

---

## 🔍 Memory Retrieval

The AI model can search for relevant memories at any time during conversation using natural language queries.

### Usage Pattern

The model will autonomously decide when to search memories based on conversation context:

```
User: What's my favorite programming language?
Model: [searches memories for "favorite programming language"]
       Based on your memories, your favorite programming language is Python.
```

### Query Examples

Models can query memories with natural language:

- "user's favorite programming languages"
- "user's job and work details"
- "user's pets and animals"
- "user's hobbies and interests"
- "user's location and living situation"

### Response Format

The tool returns a JSON structure:

```json
{
	"query": "user's favorite programming languages",
	"relevant_memories": 2,
	"memories": [
		{
			"content": "User's favorite programming language is Python",
			"created_at": "2025-01-15T10:30:00",
			"updated_at": "2025-01-15T10:30:00"
		},
		{
			"content": "User enjoys using Rust for systems programming",
			"created_at": "2025-02-10T14:20:00",
			"updated_at": "2025-02-10T14:20:00"
		}
	],
	"message": "found relevant memories matching your query"
}
```

---

## 🎯 Use Cases

### Personalized Recommendations

Models can retrieve user preferences to provide tailored suggestions:

```
User: Recommend me a good book
Model: [searches memories for "user's reading preferences and favorite genres"]
       Based on your love for science fiction, I'd recommend...
```

### Contextual Conversations

Recall past discussions and details:

```
User: How's my project going?
Model: [searches memories for "user's current projects and work"]
       Last time you mentioned working on the web app...
```

### Preference Awareness

Remember user settings and choices:

```
User: Set up my development environment
Model: [searches memories for "user's preferred tools and setup"]
       I'll configure it with Python and VS Code as you prefer...
```

---

## 🎯 Tips for Best Results

### Similarity Threshold

- **0.65** (default): Good balance between precision and recall
- **0.50-0.64**: More permissive, returns broader results
- **0.66-0.80**: Stricter matching, only highly relevant memories
- **0.81-1.00**: Very strict, requires near-exact semantic matches

### Memory Limit

- **10-15**: Good for focused queries on specific topics
- **20-30**: Broader context, multiple related topics
- **40-50**: Comprehensive retrieval, research purposes

### Query Patterns

The model should use descriptive queries:

- ✅ "user's favorite foods and dietary preferences"
- ✅ "user's work history and current job"
- ✅ "user's family members and relationships"
- ❌ "food" (too vague)
- ❌ "stuff about user" (not specific enough)

---

## 🛟 Troubleshooting

### No Memories Found

- **Check the query**: Make sure it's specific and descriptive
- **Lower threshold**: Reduce `minimum_similarity_threshold` to 0.5 or below
- **Verify memories exist**: Ensure Auto Memory or manual memory creation is working
- **Check memory count**: Increase `memory_max_k` to retrieve more results

### Too Many Irrelevant Results

- **Raise threshold**: Increase `minimum_similarity_threshold` to 0.7 or higher
- **Refine query**: Use more specific, detailed queries
- **Reduce limit**: Lower `memory_max_k` to get only top results

### Permission Issues

- Ensure the memory system is enabled for the user
- Check that the user has proper permissions in Open WebUI

---

## 🤝 Why This Tool

Traditional memory systems inject memories into system prompts automatically, which can:

- Consume tokens unnecessarily
- Include irrelevant context
- Provide no control over retrieval timing

**Memory Tools** gives AI models:

- **Autonomy**: Decide when to search memories
- **Precision**: Query for exactly what's needed
- **Efficiency**: Retrieve only relevant information
- **Transparency**: Users see when memories are accessed

---

## 📜 License

Source-Available – No Redistribution Without Permission  
Copyright (c) 2025 nokodo

You are free to use, run, and modify this extension for personal or internal purposes.  
You may NOT redistribute, publish, sublicense or sell this extension or any modified version without prior explicit written consent from the author.  
All copies must retain this notice. Provided "AS IS" without warranty.  
Earlier pre-release versions may have been available under different terms.
