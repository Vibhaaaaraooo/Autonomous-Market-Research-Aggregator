"""
Writer Agent - Report Synthesis

Role: Takes the RAG-retrieved chunks + research plan and synthesizes
a structured, professional markdown research report using Groq LLM.

Flow:
  1. Receive research plan (sections) + relevant chunks from vector store
  2. For each report section, retrieve the most relevant chunks
  3. Feed section context + chunks to Groq LLM
  4. LLM writes that section with citations
  5. Assemble all sections into final markdown report
  6. Add metadata header, table of contents, and source list
"""

import json
from datetime import datetime
from groq import Groq
from config import Settings
from rag.vector_store import VectorStore
from utils.logger import get_logger, log_step

logger = get_logger("WRITER")

SECTION_WRITER_PROMPT = """You are an expert market research analyst writing a section of a comprehensive research report.

REPORT TOPIC: {topic}
SECTION TITLE: {section_title}
SECTION OBJECTIVE: {section_description}

Below are relevant research excerpts gathered from recent articles and sources. Use these to write a detailed, insightful section.

RESEARCH DATA:
{context_chunks}

INSTRUCTIONS:
- Write in professional, analytical tone suitable for business executives
- Include specific data points, numbers, percentages, and company names when available
- Cite sources using [Source: title] format when referencing specific facts
- Be comprehensive but concise — aim for 300-600 words for this section
- Use bullet points or sub-sections where appropriate for readability
- DO NOT fabricate data — only use information from the provided research excerpts
- If the research data doesn't cover an aspect well, note it as an area needing further research
- Write ONLY the section content (no section title — that's added separately)
- Use markdown formatting: **bold** for emphasis, bullet lists, sub-headers (###) where helpful"""

EXECUTIVE_SUMMARY_PROMPT = """You are an expert market research analyst writing an executive summary for a comprehensive report.

REPORT TOPIC: {topic}

Below are all the sections that have been written for this report:

{all_sections}

Write a compelling executive summary (200-400 words) that:
- Highlights the most critical findings and insights
- Provides key data points and market figures
- Notes important trends and their implications
- Gives a clear top-level view that a busy executive can read in 2 minutes
- Uses professional, authoritative tone
- Write ONLY the summary content, no title"""


class WriterAgent:
    """
    Synthesizes research data into a structured markdown report.
    Uses Groq LLM for each section with RAG-retrieved context.
    """

    def __init__(self, settings: Settings, vector_store: VectorStore):
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model
        self.vector_store = vector_store
        self.top_k = settings.top_k_chunks

    async def write_report(self, topic: str, research_plan: dict) -> str:
        """
        Generate the complete markdown research report.

        Args:
            topic: Original research topic
            research_plan: Plan from PlannerAgent with report_sections

        Returns:
            Complete markdown report string

        Steps:
            1. Write each section using RAG context
            2. Generate executive summary from all sections
            3. Assemble report with header, TOC, sections, sources
        """
        log_step("WRITER", "Writing report", f'Topic: "{topic}"')

        sections = research_plan.get("report_sections", [])
        written_sections = []
        all_sources = set()

        # Step 1: Write each section
        for i, section in enumerate(sections):
            title = section.get("title", f"Section {i+1}")
            description = section.get("description", "")
            retrieval_queries = section.get("retrieval_queries", [title])

            log_step("WRITER", f"Writing section {i+1}/{len(sections)}", title)

            # Retrieve relevant chunks for this section
            chunks = self.vector_store.multi_query(
                retrieval_queries, top_k_per_query=self.top_k // 2
            )

            # Limit context window
            context_text = self._format_chunks_for_context(chunks[:self.top_k])

            # Track sources
            for chunk in chunks:
                source_url = chunk.get("source_url", "")
                source_title = chunk.get("source_title", "")
                if source_url:
                    all_sources.add((source_title, source_url))

            # Generate section content via Groq
            section_content = await self._write_section(
                topic, title, description, context_text
            )

            written_sections.append({
                "title": title,
                "content": section_content,
            })

        # Step 2: Generate executive summary
        log_step("WRITER", "Writing executive summary")
        exec_summary = await self._write_executive_summary(topic, written_sections)

        # Step 3: Assemble the full report
        report = self._assemble_report(
            topic, research_plan, exec_summary, written_sections, all_sources
        )

        logger.info(f"Report complete: {len(report)} characters, {len(written_sections)} sections")
        return report

    async def _write_section(
        self, topic: str, title: str, description: str, context: str
    ) -> str:
        """Write a single report section using Groq LLM."""
        prompt = SECTION_WRITER_PROMPT.format(
            topic=topic,
            section_title=title,
            section_description=description,
            context_chunks=context,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional market research report writer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to write section '{title}': {e}")
            return f"*Section generation failed: {e}*"

    async def _write_executive_summary(
        self, topic: str, sections: list[dict]
    ) -> str:
        """Generate executive summary from all written sections."""
        all_sections_text = ""
        for s in sections:
            all_sections_text += f"\n## {s['title']}\n{s['content']}\n"

        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            topic=topic,
            all_sections=all_sections_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional market research report writer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to write executive summary: {e}")
            return "*Executive summary generation failed.*"

    def _format_chunks_for_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source_title", "Unknown")
            text = chunk.get("text", "")
            score = chunk.get("relevance_score", 0)
            parts.append(
                f"--- Excerpt {i} (Relevance: {score:.2f}) [Source: {source}] ---\n{text}\n"
            )
        return "\n".join(parts)

    def _assemble_report(
        self,
        topic: str,
        plan: dict,
        exec_summary: str,
        sections: list[dict],
        sources: set,
    ) -> str:
        """Assemble all parts into the final markdown report."""
        now = datetime.now()
        date_str = now.strftime("%B %d, %Y")

        # Header
        report = f"""---
title: "Market Research Report: {topic}"
date: {date_str}
generated_by: Autonomous Market Research Aggregator
model: {self.model}
---

# {topic}
### Comprehensive Market Research Report

**Generated:** {date_str}
**Research Topic:** {topic}
**AI Model:** {self.model}
**Sources Analyzed:** {len(sources)} articles

---

## Table of Contents

1. [Executive Summary](#executive-summary)
"""

        # TOC entries
        for i, s in enumerate(sections, 2):
            anchor = s["title"].lower().replace(" ", "-").replace("&", "and")
            report += f"{i}. [{s['title']}](#{anchor})\n"

        report += f"{len(sections) + 2}. [Sources & References](#sources--references)\n"
        report += "\n---\n\n"

        # Executive Summary
        report += f"## Executive Summary\n\n{exec_summary}\n\n---\n\n"

        # Main Sections
        for section in sections:
            report += f"## {section['title']}\n\n{section['content']}\n\n---\n\n"

        # Sources
        report += "## Sources & References\n\n"
        report += "The following sources were analyzed to compile this report:\n\n"

        for i, (title, url) in enumerate(sorted(sources), 1):
            display_title = title if title else url
            report += f"{i}. [{display_title}]({url})\n"

        report += f"\n---\n\n*This report was automatically generated by the Autonomous Market Research Aggregator on {date_str}.*\n"

        return report
