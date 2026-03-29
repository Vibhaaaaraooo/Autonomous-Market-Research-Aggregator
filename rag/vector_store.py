"""
Vector store module for the RAG pipeline.
Uses ChromaDB for persistent vector storage with sentence-transformers embeddings.

Flow:
  1. Receive TextChunks from the chunker
  2. Embed each chunk using sentence-transformers (all-MiniLM-L6-v2)
  3. Store embeddings + metadata in ChromaDB collection
  4. Retrieve top-K relevant chunks for a given query
"""

import uuid
import chromadb
from chromadb.utils import embedding_functions
from rag.chunker import TextChunk
from utils.logger import get_logger

logger = get_logger("RAG")


class VectorStore:
    """
    ChromaDB-backed vector store for document chunks.

    Each research session creates a new collection so results don't bleed
    across different research topics.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.

        Args:
            embedding_model: HuggingFace model name for sentence-transformers
        """
        self.embedding_model = embedding_model

        # Use in-memory ChromaDB (no persistence needed across sessions)
        self.client = chromadb.Client()

        # Sentence-transformers embedding function
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self.collection = None
        self.collection_name = None
        logger.info(f"VectorStore initialized with model: {embedding_model}")

    def create_collection(self, topic: str) -> None:
        """
        Create a new ChromaDB collection for a research topic.

        Args:
            topic: Research topic (used to name the collection)
        """
        # Sanitize topic to valid collection name
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic)
        safe_name = safe_name[:50]  # ChromaDB name length limit
        self.collection_name = f"research_{safe_name}_{uuid.uuid4().hex[:8]}"

        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fn,
            metadata={"topic": topic},
        )
        logger.info(f"Created collection: {self.collection_name}")

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        """
        Embed and store text chunks in the vector database.

        Args:
            chunks: List of TextChunk objects to store

        Returns:
            Number of chunks successfully stored
        """
        if not self.collection:
            raise RuntimeError("No collection created. Call create_collection() first.")

        if not chunks:
            return 0

        # Prepare data for ChromaDB batch insert
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}_{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append({
                "source_url": chunk.source_url,
                "source_title": chunk.source_title,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "word_count": chunk.word_count,
            })

        # Batch insert (ChromaDB handles embedding automatically)
        batch_size = 100
        total_added = 0

        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            total_added += len(ids[start:end])

        logger.info(f"Stored {total_added} chunks in vector database")
        return total_added

    def query(self, query_text: str, top_k: int = 20) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query_text: The search query
            top_k: Number of top results to return

        Returns:
            List of dicts with keys: text, source_url, source_title, relevance_score
        """
        if not self.collection:
            raise RuntimeError("No collection created. Call create_collection() first.")

        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Parse ChromaDB results into clean format
        retrieved = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                retrieved.append({
                    "text": doc,
                    "source_url": meta.get("source_url", ""),
                    "source_title": meta.get("source_title", ""),
                    "relevance_score": round(1 - dist, 4),  # Convert distance to similarity
                })

        logger.info(
            f"Retrieved {len(retrieved)} chunks for query: '{query_text[:60]}...'"
        )
        return retrieved

    def multi_query(self, queries: list[str], top_k_per_query: int = 10) -> list[dict]:
        """
        Query with multiple search terms and deduplicate results.
        This improves recall by searching from different angles.

        Args:
            queries: List of query strings
            top_k_per_query: Results per query

        Returns:
            Deduplicated list of relevant chunks, sorted by relevance
        """
        seen_texts = set()
        all_results = []

        for query in queries:
            results = self.query(query, top_k=top_k_per_query)
            for r in results:
                # Deduplicate by text content (first 200 chars)
                text_key = r["text"][:200]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    all_results.append(r)

        # Sort by relevance score descending
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        logger.info(
            f"Multi-query with {len(queries)} queries returned "
            f"{len(all_results)} unique chunks"
        )
        return all_results

    def get_stats(self) -> dict:
        """Return collection statistics."""
        if not self.collection:
            return {"status": "no collection"}

        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "embedding_model": self.embedding_model,
        }
