"""
Text chunking module for the RAG pipeline.
Splits extracted article text into overlapping chunks optimized for embedding + retrieval.

Chunking Strategy:
  1. Split by paragraphs first (preserves semantic boundaries)
  2. Merge small paragraphs into chunks up to CHUNK_SIZE
  3. Apply CHUNK_OVERLAP to maintain context across chunk boundaries
  4. Attach source metadata (URL, title) to each chunk
"""

from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger("RAG")


@dataclass
class TextChunk:
    """A single chunk of text with source metadata."""
    text: str
    source_url: str
    source_title: str
    chunk_index: int
    total_chunks: int
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "word_count": self.word_count,
        }


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Algorithm:
      1. Split text into paragraphs (double newline)
      2. Greedily merge paragraphs into chunks ≤ chunk_size characters
      3. When a chunk is full, start a new one with `chunk_overlap` chars
         carried over from the end of the previous chunk

    Args:
        text: The raw text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters of overlap between consecutive chunks

    Returns:
        List of text chunk strings
    """
    if not text or not text.strip():
        return []

    # Step 1: Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # If no paragraph breaks, split by single newlines then sentences
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    if not paragraphs:
        return []

    # Step 2: Merge paragraphs into chunks
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If a single paragraph exceeds chunk_size, split it by sentences
        if len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            sentence_chunks = _split_long_paragraph(para, chunk_size, chunk_overlap)
            chunks.extend(sentence_chunks)
            continue

        # Check if adding this paragraph would exceed chunk_size
        test_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Apply overlap: carry end of previous chunk into new chunk
            if chunk_overlap > 0 and current_chunk:
                overlap_text = current_chunk[-chunk_overlap:]
                # Find a clean word boundary for overlap
                space_idx = overlap_text.find(" ")
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]
                current_chunk = f"{overlap_text}\n\n{para}"
            else:
                current_chunk = para

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_long_paragraph(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Split a very long paragraph into sentence-level chunks."""
    # Simple sentence splitting on period + space
    sentences = []
    for part in text.replace(". ", ".|").split("|"):
        part = part.strip()
        if part:
            sentences.append(part)

    chunks = []
    current = ""

    for sentence in sentences:
        test = f"{current} {sentence}" if current else sentence
        if len(test) <= chunk_size:
            current = test
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_articles(
    articles: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """
    Chunk multiple articles and attach source metadata.

    Args:
        articles: List of dicts with keys: url, title, text
        chunk_size: Max chars per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of TextChunk objects with metadata
    """
    all_chunks = []

    for article in articles:
        url = article.get("url", "unknown")
        title = article.get("title", "Untitled")
        text = article.get("text", "")

        if not text:
            continue

        raw_chunks = chunk_text(text, chunk_size, chunk_overlap)

        for i, chunk_text_str in enumerate(raw_chunks):
            chunk = TextChunk(
                text=chunk_text_str,
                source_url=url,
                source_title=title,
                chunk_index=i,
                total_chunks=len(raw_chunks),
            )
            all_chunks.append(chunk)

    logger.info(
        f"Chunked {len(articles)} articles into {len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks
