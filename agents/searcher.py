"""
Search Agent - Web Search & Content Collection

Role: Executes the search queries from the Planner Agent via Serper.dev API,
then scrapes and extracts article content from the discovered URLs.

Flow:
  1. Receive search queries from Planner
  2. Loop through each query → call Serper.dev Google Search API
  3. Collect unique URLs from all search results
  4. Scrape and extract full article text from top URLs
  5. Return structured article data for the RAG pipeline
"""

import asyncio
import httpx
from config import Settings
from utils.logger import get_logger, log_step
from utils.web_scraper import extract_multiple_articles

logger = get_logger("SEARCHER")

SERPER_API_URL = "https://google.serper.dev/search"
SERPER_NEWS_URL = "https://google.serper.dev/news"


class SearchAgent:
    """
    Searches the web using Serper.dev and extracts article content.
    Implements deduplication, rate limiting, and parallel extraction.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.serper_api_key
        self.results_per_query = settings.search_results_per_query
        self.max_articles = settings.max_articles_to_scrape

    async def search_and_extract(self, search_queries: list[str]) -> list[dict]:
        """
        Execute all search queries and extract article content.

        Args:
            search_queries: List of search query strings from the Planner

        Returns:
            List of article dicts with keys: url, title, text, word_count

        Pipeline:
            1. Run all search queries against Serper.dev (web + news)
            2. Deduplicate URLs across all results
            3. Rank URLs by relevance
            4. Scrape top N articles
            5. Return extracted content
        """
        log_step("SEARCHER", "Starting web search", f"{len(search_queries)} queries")

        # Step 1: Execute all search queries
        all_search_results = await self._execute_searches(search_queries)

        # Step 2: Deduplicate and rank URLs
        unique_urls = self._deduplicate_urls(all_search_results)
        logger.info(f"Found {len(unique_urls)} unique URLs across all queries")

        # Step 3: Limit to max articles
        urls_to_scrape = unique_urls[:self.max_articles]
        log_step(
            "SEARCHER",
            "Scraping articles",
            f"{len(urls_to_scrape)} URLs to extract",
        )

        # Step 4: Extract article content in parallel
        articles = await extract_multiple_articles(urls_to_scrape, max_concurrent=5)

        # Step 5: Filter out low-quality extractions
        quality_articles = [
            a for a in articles
            if a.get("word_count", 0) >= 50  # Min 50 words
        ]

        log_step(
            "SEARCHER",
            "Search complete",
            f"{len(quality_articles)} quality articles extracted",
        )

        return quality_articles

    async def _execute_searches(
        self, queries: list[str]
    ) -> list[dict]:
        """
        Run all search queries against Serper.dev API.
        Uses both web search and news search for comprehensive coverage.
        """
        all_results = []

        async with httpx.AsyncClient(timeout=20.0) as client:
            for i, query in enumerate(queries):
                logger.info(f"  Searching [{i+1}/{len(queries)}]: {query}")

                # Web search
                web_results = await self._serper_request(
                    client, query, SERPER_API_URL
                )
                all_results.extend(web_results)

                # News search (for recent articles)
                news_results = await self._serper_request(
                    client, query, SERPER_NEWS_URL
                )
                all_results.extend(news_results)

                # Small delay between queries to respect rate limits
                if i < len(queries) - 1:
                    await asyncio.sleep(0.3)

        return all_results

    async def _serper_request(
        self, client: httpx.AsyncClient, query: str, endpoint: str
    ) -> list[dict]:
        """
        Make a single request to the Serper.dev API.

        Args:
            client: httpx async client
            query: Search query string
            endpoint: API endpoint (web or news)

        Returns:
            List of result dicts with keys: url, title, snippet
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "q": query,
            "num": self.results_per_query,
        }

        try:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            results = []

            # Parse organic/web results
            for item in data.get("organic", []):
                results.append({
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "web",
                })

            # Parse news results
            for item in data.get("news", []):
                results.append({
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "news",
                    "date": item.get("date", ""),
                })

            return results

        except httpx.HTTPStatusError as e:
            logger.warning(f"Serper API error ({e.response.status_code}): {query}")
            return []
        except Exception as e:
            logger.warning(f"Serper request failed: {e}")
            return []

    def _deduplicate_urls(self, results: list[dict]) -> list[str]:
        """
        Deduplicate URLs and prioritize by source type.
        News results get slight priority over web results.
        """
        seen = set()
        url_scores: dict[str, float] = {}

        # Domains to skip (not useful for research content)
        skip_domains = {
            "youtube.com", "twitter.com", "x.com", "facebook.com",
            "instagram.com", "tiktok.com", "reddit.com", "pinterest.com",
        }

        for result in results:
            url = result.get("url", "").strip()
            if not url:
                continue

            # Skip social media and video sites
            if any(domain in url.lower() for domain in skip_domains):
                continue

            if url not in seen:
                seen.add(url)
                # Score: news gets +0.5, web gets +1.0 base
                score = 1.5 if result.get("source") == "news" else 1.0
                url_scores[url] = score

        # Sort by score descending
        sorted_urls = sorted(url_scores.keys(), key=lambda u: url_scores[u], reverse=True)
        return sorted_urls
