# Memuri Chat CLI Example

This example demonstrates a real-time chat interface with memory persistence using the Memuri SDK. It shows how to integrate memory capabilities into a conversational application.

## Features

- Interactive terminal-based chat interface
- Memory persistence with intelligent filtering (memory gating)
- Contextual responses based on past interactions
- Memory retrieval for relevant information
- OpenAI integration for chat responses

## Requirements

- OpenAI API key
- Running PostgreSQL database with pgvector extension
- Python 3.10+ with the Memuri package installed

## Setup

1. Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Set your PostgreSQL connection (if not using default):

   ```bash
   export MEMURI_DATABASE__POSTGRES_URL="postgresql://username:password@localhost:5432/database_name" 
   ```

3. Navigate to the examples directory:

   ```bash
   cd docs/examples
   ```

4. Run the chat CLI:

   ```bash
   python memuri_chat_cli.py
   ```

## Usage

Once running, the CLI provides a simple chat interface:

- Type messages and press Enter to chat
- Type `memory` to see what has been stored in memory
- Type `exit` or `quit` to end the conversation

### Example Conversation

```
You: Hi, my name is Alex and I'm a software developer.
Assistant: Hello Alex! It's nice to meet you. What kind of software development do you do?
[Memory: ✅ Stored in memory - Contains explicit keep phrase]

You: I mainly work with Python and JavaScript.
Assistant: That's great! Python and JavaScript are versatile languages. Are you working on any particular projects right now?
[Memory: ✅ Stored in memory - Passed all checks]

You: I'm working on a chat application that can remember information.
Assistant: That sounds fascinating! Building a chat application with memory capabilities is a complex but rewarding project. Are you implementing any specific memory mechanisms like vector databases or semantic search?
[Memory: ✅ Stored in memory - Classified as project_information with score 0.68]

You: memory
Stored Memories:
1. Hi, my name is Alex and I'm a software developer.
2. I mainly work with Python and JavaScript.
3. I'm working on a chat application that can remember information.
4. That sounds fascinating! Building a chat application with memory capabilities is a complex but rewarding project. Are you implementing any specific memory mechanisms like vector databases or semantic search?

You: quit
Ending conversation. Goodbye!
```

## How It Works

1. **Memory Initialization**: Creates a unique collection in the vector database for each chat session.

2. **Memory Gating**: Intelligently filters messages to store only important information:
   - Rejects short or common phrases ("ok", "thanks", etc.)
   - Prioritizes statements with key phrases ("remember", "my name is", etc.)
   - Filters out messages too similar to already stored content

3. **Context Retrieval**: For each user message, the system:
   - Searches for relevant memories using semantic similarity
   - Includes these memories as context for the LLM
   - Uses both conversation history and memories to form responses

4. **Dual Storage**: Both user messages and important assistant responses are stored, creating a comprehensive memory.

## Customization

You can modify the chat behavior by changing:

- Memory gating parameters in the `setup_memuri()` function
- The LLM model used in `get_llm_response()`
- The memory search strategy in `get_relevant_memories()`
- The special commands in the main chat loop

## Additional Notes

- Each chat session uses a unique collection in the database
- For production use, you might want to use a fixed collection name or user ID
- The terminal interface uses ANSI color codes for better readability 