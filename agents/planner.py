"""
Planner Agent - Research Decomposition

Role: Takes a user's research topic and breaks it into:
  1. A set of targeted search queries for Serper.dev
  2. Key research dimensions/angles to cover
  3. The report structure outline

Uses Groq LLM to intelligently decompose any topic into searchable sub-queries
that maximize information coverage while minimizing redundancy.
"""

import json
from groq import Groq
from config import Settings
from utils.logger import get_logger, log_step

logger = get_logger("PLANNER")

PLANNER_SYSTEM_PROMPT = """You are a senior market research analyst specializing in creating comprehensive research plans.

Given a research topic, you must produce a detailed research plan as a JSON object with these exact keys:

{{
  "topic_summary": "A clear 1-2 sentence summary of what we're researching",
  "research_dimensions": [
    "List of 4-6 key angles/dimensions to investigate"
  ],
  "search_queries": [
    "List of specific, targeted search queries optimized for Google/news search",
    "Each query should be 3-8 words, specific enough to find relevant results",
    "Include a mix of: industry overview, key players, recent news, financial data, trends, challenges"
  ],
  "report_sections": [
    {{
      "title": "Section title for the final report",
      "description": "Brief description of what this section should cover",
      "retrieval_queries": ["1-2 specific queries for RAG retrieval for this section"]
    }}
  ]
}}

Guidelines:
- Generate between 5 and {max_queries} search queries
- Make queries specific and time-relevant (include year if appropriate)
- Cover different facets: market size, competitors, technology, regulation, funding, trends
- Design report sections that tell a coherent story
- Prioritize recent/current information
- Include queries for both broad overview and specific details
- Always output ONLY valid JSON, no markdown formatting or extra text"""


class PlannerAgent:
    """
    Decomposes a research topic into a structured research plan
    using Groq LLM for intelligent query generation.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    async def create_research_plan(self, topic: str) -> dict:
        """
        Generate a comprehensive research plan for the given topic.

        Args:
            topic: The research topic (e.g., "Electric Vehicle startups 2026")

        Returns:
            Dict with keys: topic_summary, research_dimensions,
            search_queries, report_sections

        Flow:
            1. Format the system prompt with config values
            2. Send topic to Groq LLM
            3. Parse JSON response
            4. Validate structure
            5. Return research plan
        """
        log_step("PLANNER", "Creating research plan", f'Topic: "{topic}"')

        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            max_queries=self.settings.max_search_queries
        )

        user_prompt = f"""Create a detailed research plan for the following topic:

TOPIC: {topic}

Current date context: March 2026

Generate a comprehensive research plan with targeted search queries that will help build a thorough market research report. Focus on finding the most current and relevant information."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Low temp for structured output
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            raw_content = response.choices[0].message.content
            plan = json.loads(raw_content)

            # Validate required keys
            required_keys = ["topic_summary", "search_queries", "report_sections"]
            for key in required_keys:
                if key not in plan:
                    raise ValueError(f"Missing required key in plan: {key}")

            # Ensure search_queries is a list with reasonable count
            if not isinstance(plan["search_queries"], list):
                raise ValueError("search_queries must be a list")

            # Cap queries at max
            plan["search_queries"] = plan["search_queries"][:self.settings.max_search_queries]

            # Log plan summary
            logger.info(
                f"Plan created: {len(plan['search_queries'])} search queries, "
                f"{len(plan.get('report_sections', []))} report sections"
            )

            for i, q in enumerate(plan["search_queries"], 1):
                logger.info(f"  Query {i}: {q}")

            return plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Return a basic fallback plan
            return self._fallback_plan(topic)

        except Exception as e:
            logger.error(f"Planner agent error: {e}")
            return self._fallback_plan(topic)

    def _fallback_plan(self, topic: str) -> dict:
        """Generate a basic fallback plan if LLM fails."""
        logger.warning("Using fallback research plan")
        return {
            "topic_summary": f"Research report on: {topic}",
            "research_dimensions": [
                "Market overview",
                "Key players",
                "Recent developments",
                "Future outlook",
            ],
            "search_queries": [
                f"{topic} market overview 2026",
                f"{topic} latest news",
                f"{topic} key companies",
                f"{topic} market size revenue",
                f"{topic} trends forecast",
                f"{topic} challenges risks",
            ],
            "report_sections": [
                {
                    "title": "Executive Summary",
                    "description": "High-level overview of the topic",
                    "retrieval_queries": [f"{topic} overview summary"],
                },
                {
                    "title": "Market Landscape",
                    "description": "Current state of the market",
                    "retrieval_queries": [f"{topic} market landscape"],
                },
                {
                    "title": "Key Players",
                    "description": "Major companies and competitors",
                    "retrieval_queries": [f"{topic} companies competitors"],
                },
                {
                    "title": "Recent Developments",
                    "description": "Latest news and events",
                    "retrieval_queries": [f"{topic} recent news developments"],
                },
                {
                    "title": "Outlook & Trends",
                    "description": "Future predictions and emerging trends",
                    "retrieval_queries": [f"{topic} future trends forecast"],
                },
            ],
        }
