"""
Search client for gathering external information to support forecasting.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx
import structlog

from ..config.settings import Settings

logger = structlog.get_logger(__name__)


class SearchClient(ABC):
    """Abstract base class for search clients."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional config parameter for test compatibility."""
        self.config = config

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for information and return results."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the search service is available."""
        pass


class NoOpSearchClient(SearchClient):
    """No-op search client: external web search disabled per provider policy.

    Research is handled by AskNews-first and free-model synthesis in the domain pipeline.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        logger.info(
            "External search disabled; returning no results (AskNews-first via pipeline)",
            query=query,
            max_results=max_results,
        )
        return []

    async def health_check(self) -> bool:
        # Yield control to satisfy async linters
        await asyncio.sleep(0)
        return True


class DuckDuckGoSearchClient(SearchClient):
    """Search client using DuckDuckGo instant answer API."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.base_url = "https://api.duckduckgo.com"

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo API with query simplification + fallback variants.

        Strategy:
        - Derive concise keyword query (strip auxiliaries, punctuation, limit tokens)
        - Try concise form first; if zero results, fallback to original question
        - If still zero, attempt keyword-only variant (drop stopwords)
        """
        original_query = query.strip()
        simplified = self._simplify_query(original_query)
        keyword_only = self._keywords_only(original_query)
        tried: List[str] = []
        for variant in [simplified, original_query, keyword_only]:
            if not variant or variant in tried:
                continue
            tried.append(variant)
            logger.info("Performing DuckDuckGo search", query=variant, original=original_query, max_results=max_results)
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    params: Dict[str, str | int | float | bool | Any | None] = {
                        "q": variant,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1",
                    }
                    response = await client.get(f"{self.base_url}/", params=params)
                    response.raise_for_status()
                    data = response.json()
                    results = self._parse_duckduckgo_response(data, max_results)
                    if results:
                        logger.info("DuckDuckGo search completed", results_count=len(results), variant=variant)
                        return results
                    logger.debug("DuckDuckGo returned zero results", variant=variant)
            except Exception as e:  # pragma: no cover
                logger.warning("DuckDuckGo variant failed", variant=variant, error=str(e))
        logger.info("DuckDuckGo search exhausted variants with no results", original=original_query)
        return []

    def _simplify_query(self, q: str) -> str:
        lowers = q.lower().strip().rstrip('?')
        for prefix in (
            "what will be the ",
            "what will be the result of the ",
            "what will be the",
            "what will",
            "what is",
            "will ",
            "what ",
            "who ",
            "how ",
        ):
            if lowers.startswith(prefix):
                lowers = lowers[len(prefix):]
                break
        # Keep first 8 words for brevity
        return " ".join([w for w in lowers.split()[:8]])

    def _keywords_only(self, q: str) -> str:
        import re
        tokens = re.findall(r"[A-Za-z0-9]+", q.lower())
        stop = {
            "the",
            "of",
            "a",
            "an",
            "will",
            "be",
            "is",
            "what",
            "result",
            "between",
            "vs",
            "and",
            "in",
            "on",
            "at",
            "to",
        }
        keep = [t for t in tokens if t not in stop]
        return " ".join(keep[:10])

    def _parse_duckduckgo_response(
        self, data: Dict[str, Any], max_results: int
    ) -> List[Dict[str, Any]]:
        """Parse DuckDuckGo API response."""
        results = []

        # Abstract (instant answer)
        if data.get("Abstract"):
            results.append(
                {
                    "title": data.get("AbstractText", "Instant Answer"),
                    "snippet": data.get("Abstract", ""),
                    "url": data.get("AbstractURL", ""),
                    "source": "DuckDuckGo Instant Answer",
                }
            )

        # Related topics
        for topic in data.get("RelatedTopics", [])[: max_results - len(results)]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(
                    {
                        "title": topic.get("Text", "").split(" - ")[0],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo Related Topic",
                    }
                )

        return results[:max_results]

    async def health_check(self) -> bool:
        """Check DuckDuckGo API health."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/?q=test&format=json")
                return response.status_code == 200
        except Exception:
            return False


class SerpAPISearchClient(SearchClient):
    """Search client using SerpAPI for Google search results."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.search.serpapi_key
        self.base_url = "https://serpapi.com/search"

        if not self.api_key:
            logger.warning("SerpAPI key not configured")

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using SerpAPI."""
        if not self.api_key:
            logger.warning("SerpAPI key not available, skipping search")
            return []

        logger.info("Performing SerpAPI search", query=query, max_results=max_results)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                params: Dict[str, str | int | float | bool | None] = {
                    "q": query,
                    "api_key": self.api_key,
                    "engine": "google",
                    "num": min(max_results, 10),  # SerpAPI limit
                    "output": "json",
                }

                response = await client.get(self.base_url, params=params)
                response.raise_for_status()

                data = response.json()
                results = self._parse_serpapi_response(data, max_results)

                logger.info("SerpAPI search completed", results_count=len(results))
                return results

        except Exception as e:
            logger.error("SerpAPI search failed", query=query, error=str(e))
            return []

    def _parse_serpapi_response(
        self, data: Dict[str, Any], max_results: int
    ) -> List[Dict[str, Any]]:
        """Parse SerpAPI response."""
        results = []

        # Organic results
        for result in data.get("organic_results", [])[:max_results]:
            results.append(
                {
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "url": result.get("link", ""),
                    "source": "Google Search",
                    "position": result.get("position", 0),
                }
            )

        # Knowledge graph
        if data.get("knowledge_graph"):
            kg = data["knowledge_graph"]
            results.insert(
                0,
                {
                    "title": kg.get("title", "Knowledge Graph"),
                    "snippet": kg.get("description", ""),
                    "url": kg.get("website", ""),
                    "source": "Google Knowledge Graph",
                },
            )

        return results[:max_results]

    async def health_check(self) -> bool:
        """Check SerpAPI health."""
        if not self.api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                params: Dict[str, str | int | float | bool | None] = {
                    "q": "test",
                    "api_key": self.api_key,
                    "engine": "google",
                    "output": "json",
                }
                response = await client.get(self.base_url, params=params)
                return response.status_code == 200
        except Exception:
            return False


class WikipediaSearchClient(SearchClient):
    """Search client for Wikipedia articles."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.base_url = "https://en.wikipedia.org/api/rest_v1"

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Wikipedia articles (guard against poorly-formed question queries).

        We skip if the query looks like a long interrogative sentence ( >10 words & ends with '?')
        or contains verbs unlikely to map to article titles, to avoid 404 spam.
        """
        q = query.strip()
        if (q.endswith("?") and len(q.split()) > 10) or q.lower().startswith(
            "what will be the result"
        ):
            logger.info("Skipping Wikipedia search for interrogative question form", query=q)
            return []
        logger.info("Performing Wikipedia search", query=q, max_results=max_results)
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                # Properly encode query in URL path to avoid 404s
                from urllib.parse import quote
                encoded_query = quote(q, safe='')
                search_url = f"{self.base_url}/page/search/{encoded_query}"
                params = {"limit": min(max_results, 10)}
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                search_data = response.json()
                pages = search_data.get("pages", [])
                results: List[Dict[str, Any]] = []
                for page in pages[:max_results]:
                    try:
                        summary_url = f"{self.base_url}/page/summary/{page['key']}"
                        summary_response = await client.get(summary_url)
                        if summary_response.status_code == 200:
                            sd = summary_response.json()
                            results.append(
                                {
                                    "title": sd.get("title", page.get("title", "")),
                                    "snippet": sd.get("extract", ""),
                                    "url": sd.get("content_urls", {})
                                    .get("desktop", {})
                                    .get("page", ""),
                                    "source": "Wikipedia",
                                }
                            )
                    except Exception as e:  # pragma: no cover
                        logger.debug(
                            "Wikipedia summary fetch failed",
                            page=page.get("title"),
                            error=str(e),
                        )
                logger.info("Wikipedia search completed", results_count=len(results))
                return results
        except Exception as e:
            logger.debug("Wikipedia search failed", query=q, error=str(e))
            return []

    async def health_check(self) -> bool:
        """Check Wikipedia API health."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/page/search/test")
                return response.status_code == 200
        except Exception:
            return False


class MultiSourceSearchClient(SearchClient):
    """Search client that combines multiple search sources."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.clients: List[SearchClient] = []

        # Environment overrides for quick disable without changing config
        ddg_env = os.getenv("SEARCH_DUCKDUCKGO_ENABLED", "").lower()
        wiki_env = os.getenv("SEARCH_WIKIPEDIA_ENABLED", "").lower()
        serp_env = os.getenv("SEARCH_SERPAPI_ENABLED", "").lower()

        ddg_enabled = (
            settings.search.duckduckgo_enabled
            and ddg_env not in ("0", "false", "no")
        ) or ddg_env in ("1", "true", "yes")
        wiki_enabled = (
            settings.search.wikipedia_enabled
            and wiki_env not in ("0", "false", "no")
        ) or wiki_env in ("1", "true", "yes")
        serp_enabled = (
            bool(settings.search.serpapi_key)
            and serp_env not in ("0", "false", "no")
        ) or serp_env in ("1", "true", "yes")

        # Initialize available search clients according to flags
        if ddg_enabled:
            self.clients.append(DuckDuckGoSearchClient(settings))
        if wiki_enabled:
            self.clients.append(WikipediaSearchClient(settings))
        if serp_enabled:
            self.clients.append(SerpAPISearchClient(settings))

        logger.info(
            "Initialized multi-source search",
            client_count=len(self.clients),
            ddg_enabled=ddg_enabled,
            wikipedia_enabled=wiki_enabled,
            serpapi_enabled=serp_enabled,
        )

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search across multiple sources and combine results."""
        logger.info(
            "Performing multi-source search", query=query, max_results=max_results
        )

        results_per_source = max(1, max_results // len(self.clients))
        search_tasks = [
            client.search(query, results_per_source) for client in self.clients
        ]
        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        combined_results = self._merge_results(all_results, max_results)
        logger.info(
            "Multi-source search completed",
            total_results=len(combined_results),
            sources_used=len([r for r in all_results if not isinstance(r, Exception)]),
        )
        return combined_results

    def _merge_results(
        self, all_results: List[Any], max_results: int
    ) -> List[Dict[str, Any]]:
        combined_results: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        for source_results in all_results:
            if isinstance(source_results, Exception) or not isinstance(
                source_results, list
            ):
                continue
            for result in source_results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    combined_results.append(result)
                    if len(combined_results) >= max_results:
                        return combined_results[:max_results]
        return combined_results[:max_results]

    async def health_check(self) -> bool:
        """Check health of all search clients."""
        health_checks = await asyncio.gather(
            *[client.health_check() for client in self.clients], return_exceptions=True
        )

        # Return True if at least one client is healthy
        return any(
            check is True for check in health_checks if not isinstance(check, Exception)
        )

    def get_available_sources(self) -> List[str]:
        """Get list of available search sources."""
        return [client.__class__.__name__ for client in self.clients]


def create_search_client(settings: Settings) -> SearchClient:
    """Factory selecting multi-source search if enabled, else NoOp.

    Environment overrides:
        SEARCH_DISABLE_ALL=true  -> force NoOp
        SEARCH_DUCKDUCKGO_ENABLED / SERPAPI_KEY presence / Wikipedia always on
    """
    import os

    if os.getenv("SEARCH_DISABLE_ALL", "").lower() in ("1", "true", "yes"):
        logger.info("Search globally disabled by SEARCH_DISABLE_ALL env var")
        return NoOpSearchClient(settings)
    logger.info("Creating MultiSourceSearchClient for research fallback")
    return MultiSourceSearchClient(settings)
