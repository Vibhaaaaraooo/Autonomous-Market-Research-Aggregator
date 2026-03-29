"""
Web content extractor for the Search Agent.
Uses trafilatura as primary extractor with BeautifulSoup fallback.
Handles timeouts, encoding issues, and content cleaning.
"""

import asyncio
import httpx
from bs4 import BeautifulSoup
import trafilatura
from utils.logger import get_logger

logger = get_logger("SEARCHER")

# Headers to mimic a real browser request
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


async def extract_article_content(url: str, timeout: float = 15.0) -> dict | None:
    """
    Fetch and extract main article text from a URL.

    Returns dict with keys: url, title, text, word_count
    Returns None if extraction fails.
    """
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=HEADERS,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text

        # Primary extraction via trafilatura (best quality)
        text = trafilatura.extract(
            html,
            include_links=False,
            include_images=False,
            include_tables=True,
            favor_recall=True,
        )

        # Fallback to BeautifulSoup if trafilatura fails
        if not text or len(text) < 100:
            text = _bs4_fallback(html)

        if not text or len(text) < 100:
            return None

        # Extract title
        title = _extract_title(html)

        return {
            "url": url,
            "title": title,
            "text": text.strip(),
            "word_count": len(text.split()),
        }

    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching: {url}")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP {e.response.status_code} for: {url}")
        return None
    except Exception as e:
        logger.warning(f"Failed to extract {url}: {e}")
        return None


def _bs4_fallback(html: str) -> str | None:
    """Fallback text extraction using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Try article tag first, then main, then body
    for selector in ["article", "main", "[role='main']", "body"]:
        container = soup.select_one(selector)
        if container:
            paragraphs = container.find_all("p")
            text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            if len(text) > 100:
                return text

    return None


def _extract_title(html: str) -> str:
    """Extract page title from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Try og:title first (most reliable for articles)
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    # Fall back to <title> tag
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    # Fall back to first h1
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return "Untitled"


async def extract_multiple_articles(
    urls: list[str], max_concurrent: int = 5
) -> list[dict]:
    """
    Extract content from multiple URLs concurrently with rate limiting.
    Returns list of successfully extracted articles.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _limited_extract(url: str) -> dict | None:
        async with semaphore:
            return await extract_article_content(url)

    tasks = [_limited_extract(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
    for result in results:
        if isinstance(result, dict) and result is not None:
            articles.append(result)

    logger.info(f"Successfully extracted {len(articles)}/{len(urls)} articles")
    return articles
