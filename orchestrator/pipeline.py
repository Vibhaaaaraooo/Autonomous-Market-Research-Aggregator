"""
Orchestrator Pipeline - Agent Coordinator

This is the brain of the system. It coordinates the three agents (Planner, Searcher, Writer)
and the RAG pipeline in the correct sequence, passing data between them.

Full Pipeline Flow:
  ┌──────────────────────────────────────────────────────────────────┐
  │  USER INPUT: "Electric Vehicle startups 2026"                   │
  │                          │                                       │
  │                          ▼                                       │
  │  ┌─────────────────────────────────────┐                        │
  │  │  STEP 1: PLANNER AGENT             │                        │
  │  │  • Receives topic                   │                        │
  │  │  • Calls Groq LLM                   │                        │
  │  │  • Outputs: search_queries,         │                        │
  │  │    report_sections, dimensions       │                        │
  │  └──────────────┬──────────────────────┘                        │
  │                 │                                                │
  │                 ▼                                                │
  │  ┌─────────────────────────────────────┐                        │
  │  │  STEP 2: SEARCH AGENT              │                        │
  │  │  • Takes search_queries             │                        │
  │  │  • Calls Serper.dev API (loop)      │                        │
  │  │  • Scrapes article URLs             │                        │
  │  │  • Outputs: article texts           │                        │
  │  └──────────────┬──────────────────────┘                        │
  │                 │                                                │
  │                 ▼                                                │
  │  ┌─────────────────────────────────────┐                        │
  │  │  STEP 3: RAG PIPELINE              │                        │
  │  │  • Chunks article texts             │                        │
  │  │  • Embeds via sentence-transformers │                        │
  │  │  • Stores in ChromaDB               │                        │
  │  │  • Ready for semantic retrieval     │                        │
  │  └──────────────┬──────────────────────┘                        │
  │                 │                                                │
  │                 ▼                                                │
  │  ┌─────────────────────────────────────┐                        │
  │  │  STEP 4: WRITER AGENT              │                        │
  │  │  • Retrieves relevant chunks (RAG)  │                        │
  │  │  • Calls Groq LLM per section       │                        │
  │  │  • Synthesizes markdown report      │                        │
  │  │  • Outputs: complete report.md      │                        │
  │  └──────────────┬──────────────────────┘                        │
  │                 │                                                │
  │                 ▼                                                │
  │  ┌─────────────────────────────────────┐                        │
  │  │  STEP 5: MCP SERVER                │                        │
  │  │  • Saves report to local disk       │                        │
  │  │  • Reports file path to user        │                        │
  │  └─────────────────────────────────────┘                        │
  └──────────────────────────────────────────────────────────────────┘
"""

import time
from dataclasses import dataclass, field
from config import Settings
from agents.planner import PlannerAgent
from agents.searcher import SearchAgent
from agents.writer import WriterAgent
from rag.chunker import chunk_articles
from rag.vector_store import VectorStore
from utils.logger import get_logger, log_step

logger = get_logger("ORCHESTRATOR")


@dataclass
class PipelineResult:
    """Complete result of a research pipeline run."""
    topic: str
    report_markdown: str
    research_plan: dict
    articles_found: int
    chunks_stored: int
    sources_used: int
    total_time_seconds: float
    steps: list[dict] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.report_markdown)


class ResearchPipeline:
    """
    Orchestrates the full research pipeline from topic → report.
    Coordinates all agents and the RAG system.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # Initialize RAG components
        self.vector_store = VectorStore(embedding_model=settings.embedding_model)

        # Initialize agents
        self.planner = PlannerAgent(settings)
        self.searcher = SearchAgent(settings)
        self.writer = WriterAgent(settings, self.vector_store)

    async def run(self, topic: str) -> PipelineResult:
        """
        Execute the full research pipeline.

        Args:
            topic: Research topic string

        Returns:
            PipelineResult with report and metadata
        """
        start_time = time.time()
        steps = []

        log_step("ORCHESTRATOR", "Pipeline started", f'Topic: "{topic}"')
        log_step("ORCHESTRATOR", "=" * 50)

        # ══════════════════════════════════════════════════
        # STEP 1: PLANNER AGENT
        # ══════════════════════════════════════════════════
        step_start = time.time()
        log_step("ORCHESTRATOR", "STEP 1/5", "Planner Agent - Decomposing topic")

        research_plan = await self.planner.create_research_plan(topic)
        search_queries = research_plan.get("search_queries", [])

        steps.append({
            "step": "planning",
            "queries_generated": len(search_queries),
            "sections_planned": len(research_plan.get("report_sections", [])),
            "time_seconds": round(time.time() - step_start, 2),
        })

        log_step(
            "ORCHESTRATOR", "STEP 1 COMPLETE",
            f"{len(search_queries)} queries, "
            f"{len(research_plan.get('report_sections', []))} sections planned"
        )

        # ══════════════════════════════════════════════════
        # STEP 2: SEARCH AGENT
        # ══════════════════════════════════════════════════
        step_start = time.time()
        log_step("ORCHESTRATOR", "STEP 2/5", "Search Agent - Gathering articles")

        articles = await self.searcher.search_and_extract(search_queries)

        steps.append({
            "step": "searching",
            "articles_found": len(articles),
            "total_words": sum(a.get("word_count", 0) for a in articles),
            "time_seconds": round(time.time() - step_start, 2),
        })

        log_step(
            "ORCHESTRATOR", "STEP 2 COMPLETE",
            f"{len(articles)} articles extracted, "
            f"{sum(a.get('word_count', 0) for a in articles):,} total words"
        )

        if not articles:
            logger.error("No articles found. Cannot generate report.")
            return PipelineResult(
                topic=topic,
                report_markdown="",
                research_plan=research_plan,
                articles_found=0,
                chunks_stored=0,
                sources_used=0,
                total_time_seconds=round(time.time() - start_time, 2),
                steps=steps,
            )

        # ══════════════════════════════════════════════════
        # STEP 3: RAG PIPELINE (Chunk → Embed → Store)
        # ══════════════════════════════════════════════════
        step_start = time.time()
        log_step("ORCHESTRATOR", "STEP 3/5", "RAG Pipeline - Chunking & embedding")

        # Create fresh collection for this topic
        self.vector_store.create_collection(topic)

        # Chunk all articles
        chunks = chunk_articles(
            articles,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        # Store in vector database (embeds automatically)
        chunks_stored = self.vector_store.add_chunks(chunks)

        steps.append({
            "step": "rag_indexing",
            "chunks_created": len(chunks),
            "chunks_stored": chunks_stored,
            "time_seconds": round(time.time() - step_start, 2),
        })

        log_step(
            "ORCHESTRATOR", "STEP 3 COMPLETE",
            f"{chunks_stored} chunks embedded and stored in ChromaDB"
        )

        # ══════════════════════════════════════════════════
        # STEP 4: WRITER AGENT
        # ══════════════════════════════════════════════════
        step_start = time.time()
        log_step("ORCHESTRATOR", "STEP 4/5", "Writer Agent - Synthesizing report")

        report = await self.writer.write_report(topic, research_plan)

        steps.append({
            "step": "writing",
            "report_length_chars": len(report),
            "report_length_words": len(report.split()),
            "time_seconds": round(time.time() - step_start, 2),
        })

        log_step(
            "ORCHESTRATOR", "STEP 4 COMPLETE",
            f"Report generated: {len(report.split()):,} words"
        )

        # ══════════════════════════════════════════════════
        # STEP 5 is handled by the caller (MCP save or direct save)
        # ══════════════════════════════════════════════════

        total_time = round(time.time() - start_time, 2)

        log_step("ORCHESTRATOR", "=" * 50)
        log_step(
            "ORCHESTRATOR", "PIPELINE COMPLETE",
            f"Total time: {total_time}s"
        )

        return PipelineResult(
            topic=topic,
            report_markdown=report,
            research_plan=research_plan,
            articles_found=len(articles),
            chunks_stored=chunks_stored,
            sources_used=len(set(a.get("url", "") for a in articles)),
            total_time_seconds=total_time,
            steps=steps,
        )
