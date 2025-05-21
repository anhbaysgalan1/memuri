import asyncio
import os
import uuid
import datetime
import readline  # For input editing
import unicodedata
from typing import List, Dict, Tuple, Optional, Any

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from memuri.factory import (
    ClassifierFactory,
    EmbedderFactory,
    GatingFactory,
    VectorStoreFactory,
)
from memuri.domain.models import MemorySource, SearchQuery
from memuri.core.config import get_settings
from memuri.services.memory import MemoryOrchestrator
from memuri.core.text_utils import (
    clean_text, 
    normalize_text_for_embedding, 
    process_text_async,
    safe_process_batched_text
)

# Initialize Memuri LLM client
api_key = os.environ.get("MEMURI_LLM_API_KEY")
client = AsyncOpenAI(api_key=api_key) if api_key else None

# Conversation history
conversation_history: List[Dict[str, str]] = []


def normalize_text(text: str) -> str:
    """Normalize text using the memuri text_utils.
    This is a compatibility wrapper around the core text_utils module."""
    return clean_text(text)


def print_colored(msg: str, color: str = 'white', bold: bool = False) -> None:
    codes = {
        'red': '\033[91m', 'green': '\033[92m',
        'yellow': '\033[93m', 'blue': '\033[94m',
        'magenta': '\033[95m', 'cyan': '\033[96m', 'white': '\033[97m',
    }
    b = '\033[1m' if bold else ''
    r = '\033[0m'
    prefix = codes.get(color, codes['white']) + b
    print(f"{prefix}{msg}{r}")


async def setup_memuri() -> Tuple[MemoryOrchestrator, str]:
    """Initialize Memuri services, loading DB URL from env vars."""
    global api_key, client
    
    if not api_key:
        print_colored("WARNING: MEMURI_LLM_API_KEY not set, running offline.", 'yellow')
        print_colored("Please set the MEMURI_LLM_API_KEY environment variable to use the LLM features.", 'yellow')
        api_key = "dummy-key-for-offline-mode"
    os.environ['MEMURI_LLM_API_KEY'] = api_key
    os.environ['MEMURI_EMBEDDING_API_KEY'] = api_key
    if not client and api_key != "dummy-key-for-offline-mode":
        client = AsyncOpenAI(api_key=api_key)
    db_url = os.environ.get('MEMURI_DATABASE_POSTGRES_URL') or 'postgresql://memuri:memuri@localhost:5432/memuri'
    if 'postgresql://' not in os.environ.get('MEMURI_DATABASE_POSTGRES_URL', ''):
        print_colored(f"Using default DB URL: {db_url}", 'yellow')
    else:
        print_colored(f"Using DB URL from env: {db_url}", 'green')
    os.environ['MEMURI_DATABASE_POSTGRES_URL'] = db_url
    collection = f"chat_session_{uuid.uuid4().hex[:8]}"
    emb = EmbedderFactory.create(provider='openai')
    vec_settings = get_settings().vector_store
    vec_settings.collection_name = collection
    vec_provider = os.environ.get('MEMURI_VECTOR_STORE_PROVIDER', 'pgvector')
    mem_svc = VectorStoreFactory.create(provider=vec_provider, settings=vec_settings)
    clf = ClassifierFactory.create(provider='keyword')
    gate = GatingFactory.create(
        embedding_service=emb,
        memory_service=mem_svc,
        classifier_service=clf,
        similarity_threshold=0.90,
        confidence_threshold=0.3,
        min_content_length=10,
        skip_words=["ok","I see","thanks","thank you","got it","sure"],
        keep_phrases=["remember","important","note","don't forget","my name is","I am","I'm","I like"],
    )
    orchestrator = MemoryOrchestrator(
        memory_service=mem_svc,
        embedding_service=emb,
        reranking_service=None,
        classifier_service=clf,
        feedback_service=None,
        memory_gate=gate,
    )
    return orchestrator, collection


async def get_relevant_memories(orchestrator, query: str, limit: int = 3) -> List[str]:
    try:
        # Use the enhanced text normalization specifically for embeddings
        q = normalize_text_for_embedding(query)
        sq = SearchQuery(query=q, top_k=limit, min_score=0.7)
        res = await orchestrator.search_memory(sq)
        return [m.memory.content for m in res.memories]
    except Exception as e:
        print_colored(f"Mem search error: {e}", 'red')
        return []


async def get_llm_response(msg: str, memories: List[str]) -> str:
    global client, api_key
    
    if not client or api_key == "dummy-key-for-offline-mode":
        return "I'm in offline mode. Please set MEMURI_LLM_API_KEY to enable AI responses."
    
    # Prepare system message
    sys = "You are a friendly AI with memory context."
    if memories:
        sys += "\nRelevant memories:"
        for i, mem in enumerate(memories,1): 
            # Also clean memory content
            sys += f"\n{i}. {normalize_text(mem)}"
    
    # Create message list for the API call
    msgs = [{"role":"system","content":sys}] + conversation_history[-8:] + [{"role":"user","content":msg}]
    
    try:
        resp = await client.chat.completions.create(model='gpt-3.5-turbo', messages=msgs)
        return resp.choices[0].message.content
    except Exception as e:
        print_colored(f"LLM error: {e}", 'red')
        if "authentication" in str(e).lower():
            return "Error: Invalid API key. Please check your MEMURI_LLM_API_KEY."
        return f"Error: LLM call failed. {str(e)}"


async def process_user_input(orchestrator, user_input: str) -> str:
    """Process user input in the background and return response."""
    
    # Define the processor function
    async def process_text(text: str) -> str:
        try:
            # Clean text
            txt = normalize_text(text)
            
            # Add to conversation history
            conversation_history.append({"role":"user","content":txt})
            
            # Add to memory
            await orchestrator.add_memory(
                content=txt, 
                source=MemorySource.USER, 
                metadata={"ts":datetime.datetime.now().isoformat()}
            )
            
            # Get relevant memories
            mems = await get_relevant_memories(orchestrator, txt)
            
            # Get LLM response
            resp = await get_llm_response(txt, mems)
            
            # Add to conversation
            conversation_history.append({"role":"assistant","content":resp})
            
            # Add assistant response to memory
            await orchestrator.add_memory(
                content=normalize_text(resp), 
                source=MemorySource.SYSTEM, 
                metadata={"ts":datetime.datetime.now().isoformat()}
            )
            
            return resp
        except Exception as e:
            print_colored(f"Error processing input: {e}", 'red')
            return f"Error processing your input: {e}"
    
    # Define callbacks
    def on_start():
        print_colored("Processing your input...", 'blue')
    
    def on_error(e):
        print_colored(f"An error occurred: {e}", 'red')
    
    # Process based on input length
    if len(user_input) > 5000:
        print_colored("Processing long text in batches...", 'blue')
        return await process_text_async(
            user_input, 
            process_text,
            on_start=on_start,
            on_error=on_error
        )
    else:
        return await process_text(user_input)


async def main():
    print_colored("Memuri Chat CLI", 'cyan', True)
    orchestrator, collection = await setup_memuri()
    print_colored(f"Memory initialized – Collection: {collection}", 'green')
    greeting = "Hello! I'm your assistant with memory capabilities."
    print_colored(f"Assistant: {greeting}", 'green')
    conversation_history.append({"role":"assistant","content":greeting})
    
    while True:
        try:
            # Get user input with asyncio to avoid blocking
            ui = (await asyncio.to_thread(input, "You: ")).strip()
            
            # Handle exit commands
            if ui.lower() in ['exit','quit']:
                print_colored("Goodbye!", 'cyan')
                break
                
            # Handle memory command
            if ui.lower() == 'memory':
                mems = await get_relevant_memories(orchestrator, 'All important info', 10)
                print_colored("Stored Memories:", 'magenta', True)
                if mems:
                    for i, m in enumerate(mems,1): print_colored(f"{i}. {m}", 'magenta')
                else:
                    print_colored("No memories yet.", 'magenta')
                continue
                
            # Handle API key command
            if ui.lower() in ['key','apikey']:
                new_key = (await asyncio.to_thread(input, "Enter your API key: ")).strip()
                if new_key.startswith("sk-"):
                    api_key = new_key
                    os.environ['MEMURI_LLM_API_KEY'] = api_key
                    os.environ['MEMURI_EMBEDDING_API_KEY'] = api_key
                    client = AsyncOpenAI(api_key=api_key)
                    print_colored("API key updated successfully.", 'green')
                else:
                    print_colored("Invalid API key format. It should start with 'sk-'.", 'red')
                continue
                
            # Check if input is empty
            if not ui.strip():
                print_colored("Please enter some text.", 'yellow')
                continue
            
            # Process normal input with background handling
            response = await process_user_input(orchestrator, ui)
            print_colored(f"Assistant: {response}", 'green')
            
        except KeyboardInterrupt:
            print_colored("\nInterrupted. Type 'exit' to quit.", 'cyan')
            continue
        except Exception as e:
            print_colored(f"Unexpected error: {e}", 'red')
            print_colored("Please try again.", 'yellow')


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_colored("\nConversation ended.", 'cyan')
