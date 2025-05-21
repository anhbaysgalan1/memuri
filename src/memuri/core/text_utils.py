"""
Text processing utilities for Memuri.

This module provides functions for processing and sanitizing text inputs
before they are used in embedding or LLM services, ensuring compatibility
with external APIs and preventing common errors.
"""

import re
import unicodedata
import logging
import asyncio
from typing import Optional, List, Dict, Any, Callable, Awaitable, Tuple

logger = logging.getLogger(__name__)

# Common patterns to identify log lines and other problematic content
LOG_PATTERN = re.compile(r'^\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}[,.]\d{3}.*?INFO|DEBUG|ERROR|WARNING')
REPEATED_FRAGMENT_PATTERN = re.compile(r'(.{15,}?)\1{2,}')  # Detect 3+ repetitions of 15+ char fragments
REPEATED_PHRASE_PATTERN = re.compile(r'((?:\b\w+\b\W+){2,5})\1{2,}')  # Detect 3+ repetitions of 2-5 word phrases
CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
URL_PATTERN = re.compile(r'https?://\S+')
PROBLEMATIC_CHARS = re.compile(r'[\uFFFE\uFFFF\uFEFF\u0000-\u0008\u000B\u000C\u000E-\u001F\uD800-\uDFFF]')

def is_valid_utf8(text: str) -> bool:
    """Check if a string can be encoded as valid UTF-8.
    
    Args:
        text: String to check
        
    Returns:
        True if valid UTF-8, False otherwise
    """
    try:
        text.encode('utf-8').decode('utf-8')
        return True
    except UnicodeError:
        return False

def remove_surrogate_chars(text: str) -> str:
    """Remove surrogate characters that cause encoding errors.
    
    Args:
        text: Input text
        
    Returns:
        Text with surrogate characters removed
    """
    # Handle surrogate pairs and isolated surrogates
    result = ""
    for c in text:
        if not (0xD800 <= ord(c) <= 0xDFFF):  # Skip surrogate range
            result += c
    return result

def clean_text(text: str, max_length: int = 8000, preserve_utf8: bool = True) -> str:
    """
    Clean and sanitize text for API compatibility and improved processing.
    
    Args:
        text: The input text to clean
        max_length: Maximum length to truncate to (default: 8000)
        preserve_utf8: Whether to preserve valid UTF-8 characters (default: True)
        
    Returns:
        Cleaned and sanitized text string
    """
    if not text:
        return ""
    
    # Remove problematic characters
    text = PROBLEMATIC_CHARS.sub('', text)
    
    # Remove surrogate characters
    try:
        text = remove_surrogate_chars(text)
    except Exception as e:
        logger.warning(f"Error removing surrogate characters: {e}")
    
    # Remove control characters
    text = CONTROL_CHARS_PATTERN.sub('', text)
    
    # Remove log lines that might be pasted accidentally
    cleaned_lines = []
    for line in text.split('\n'):
        if not LOG_PATTERN.search(line):
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    
    # Normalize Unicode (handling errors)
    try:
        text = unicodedata.normalize('NFKD', text)
    except Exception as e:
        logger.warning(f"Unicode normalization error: {e}")
        # Try to extract valid characters
        valid_chars = []
        for c in text:
            try:
                unicodedata.normalize('NFKD', c)
                valid_chars.append(c)
            except:
                pass
        text = ''.join(valid_chars)
    
    # Ensure the text is valid UTF-8
    if not is_valid_utf8(text):
        logger.warning("Text contains invalid UTF-8 sequences, attempting to fix")
        # Try to encode and decode to fix UTF-8 issues
        try:
            # Encode with 'replace' to replace invalid sequences with ? char
            # Then decode back to string
            text = text.encode('utf-8', 'replace').decode('utf-8')
        except UnicodeError as e:
            logger.error(f"Failed to fix UTF-8 encoding: {e}")
            # Fallback: keep only ASCII
            text = ''.join(c for c in text if ord(c) < 128)
    
    # Replace common Unicode quotes and other special characters
    replacements = {
        '"': '"', '"': '"',
        ''': "'", ''': "'",
        '–': '-', '—': '--', 
        '…': '...', '•': '*',
        '\u200b': '',  # Zero-width space
        '\u200e': '',  # Left-to-right mark
        '\u200f': '',  # Right-to-left mark
        '\u2028': ' ', # Line separator
        '\u2029': ' ', # Paragraph separator
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Fix repeated fragments that might be from copy-paste errors
    def replace_repetition(match):
        return match.group(1)
    
    # Apply repetition fixes multiple times
    prev_text = ""
    while prev_text != text:
        prev_text = text
        # Fix repetitions of longer fragments
        text = REPEATED_FRAGMENT_PATTERN.sub(replace_repetition, text)
        # Fix repetitions of phrases (2-5 words)
        text = REPEATED_PHRASE_PATTERN.sub(replace_repetition, text)
    
    # Fix repeating punctuation sequences
    text = re.sub(r'([!?.;,_\-*]{2,})', lambda m: m.group(1)[0], text)
    
    # Fix extreme spaces and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Replace 3+ newlines with double newline
    text = re.sub(r' {3,}', ' ', text)     # Replace 3+ spaces with single space
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "... [text truncated due to length]"
    
    # Final strip of whitespace
    text = text.strip()
    
    return text

def normalize_text_for_embedding(text: str) -> str:
    """
    Normalize text specifically for embedding models.

    This is a specialized version of clean_text that ensures the text
    is compatible with embedding APIs while preserving meaningful UTF-8.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string suitable for embedding
    """
    text = clean_text(text, preserve_utf8=True)
    
    # Additional processing specific to embedding APIs
    
    # Some APIs have issues with very long URLs
    text = URL_PATTERN.sub(lambda m: m.group(0)[:60] + "..." if len(m.group(0)) > 60 else m.group(0), text)
    
    # Convert multiple spaces/newlines to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove only problematic characters rather than all non-ASCII
    # This keeps valid UTF-8 characters that APIs can handle
    text = PROBLEMATIC_CHARS.sub('', text)
    
    # Handle empty result
    if not text or text.strip() == "":
        return "Empty content after sanitization"
    
    # Final UTF-8 validation
    if not is_valid_utf8(text):
        # Fallback to ASCII-only if UTF-8 validation fails
        logger.warning("Embedding text failed UTF-8 validation, falling back to ASCII-only")
        text = ''.join(c for c in text if ord(c) < 128)
    
    return text

def batch_text(text: str, max_tokens_per_chunk: int = 1000) -> List[str]:
    """
    Split text into smaller chunks for processing with token limits.
    
    Args:
        text: Text to split into chunks
        max_tokens_per_chunk: Approximate maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    # Simple approximation: ~4 chars per token for English
    max_chars = max_tokens_per_chunk * 4
    
    # If text is already small enough, return as single chunk
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed limit, save current chunk and start new one
        if len(current_chunk) + len(para) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extract potential metadata from text.
    
    Args:
        text: Text to analyze for metadata
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}
    
    # Extract emails
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if emails:
        metadata['emails'] = emails
    
    # Extract dates
    dates = re.findall(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', text)
    if dates:
        metadata['dates'] = dates
    
    # Extract potential company names (simple heuristic)
    company_pattern = re.compile(r'\b([A-Z][a-z]+\s?)+\b\s*(\([A-Za-z\s,]+\))?')
    companies = company_pattern.findall(text)
    if companies:
        metadata['potential_organizations'] = [c[0].strip() for c in companies]
    
    return metadata

async def process_text_async(
    text: str, 
    processor: Callable[[str], Awaitable[Any]], 
    on_start: Optional[Callable[[], None]] = None,
    on_complete: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None
) -> Any:
    """
    Process text asynchronously in the background.
    
    Args:
        text: Text to process
        processor: Async function to process the text
        on_start: Optional callback when processing starts
        on_complete: Optional callback when processing completes
        on_error: Optional callback when processing errors
        
    Returns:
        Result from the processor
    """
    # Clean the text first
    clean_content = clean_text(text)
    
    if on_start:
        on_start()
    
    try:
        # Process the text
        result = await processor(clean_content)
        
        if on_complete:
            on_complete(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in background text processing: {e}")
        if on_error:
            on_error(e)
        raise

async def safe_process_batched_text(
    text: str,
    processor: Callable[[str], Awaitable[Any]],
    max_tokens_per_batch: int = 1000,
    combine_results: Callable[[List[Any]], Any] = lambda results: results
) -> Any:
    """
    Process long text in batches to avoid token limits.
    
    Args:
        text: Text to process
        processor: Async function to process each batch
        max_tokens_per_batch: Maximum tokens per batch
        combine_results: Function to combine batch results
        
    Returns:
        Combined result from processing batches
    """
    # Clean the text
    clean_content = clean_text(text)
    
    # Split into batches
    batches = batch_text(clean_content, max_tokens_per_batch)
    
    # Process each batch concurrently
    batch_results = await asyncio.gather(
        *[processor(batch) for batch in batches],
        return_exceptions=True
    )
    
    # Handle errors
    successful_results = []
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.error(f"Error processing batch {i}: {result}")
        else:
            successful_results.append(result)
    
    # Combine results
    if successful_results:
        return combine_results(successful_results)
    else:
        raise ValueError("All text batches failed to process") 