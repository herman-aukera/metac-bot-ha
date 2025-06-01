"""
Search client for gathering external information to support forecasting.
"""
import asyncio
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import structlog
import httpx

from ..config.settings import Settings

logger = structlog.get_logger(__name__)


class SearchClient(ABC):
    """Abstract base class for search clients."""
    
    def __init__(self, config=None):
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


class DuckDuckGoSearchClient(SearchClient):
    """Search client using DuckDuckGo instant answer API."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.base_url = "https://api.duckduckgo.com"
        
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo API."""
        logger.info("Performing DuckDuckGo search", query=query, max_results=max_results)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # DuckDuckGo instant answer API
                params = {
                    'q': query,
                    'format': 'json',
                    'no_html': '1',
                    'skip_disambig': '1'
                }
                
                response = await client.get(f"{self.base_url}/", params=params)
                response.raise_for_status()
                
                data = response.json()
                results = self._parse_duckduckgo_response(data, max_results)
                
                logger.info("DuckDuckGo search completed", results_count=len(results))
                return results
                
        except Exception as e:
            logger.error("DuckDuckGo search failed", query=query, error=str(e))
            return []
    
    def _parse_duckduckgo_response(self, data: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Parse DuckDuckGo API response."""
        results = []
        
        # Abstract (instant answer)
        if data.get('Abstract'):
            results.append({
                'title': data.get('AbstractText', 'Instant Answer'),
                'snippet': data.get('Abstract', ''),
                'url': data.get('AbstractURL', ''),
                'source': 'DuckDuckGo Instant Answer'
            })
        
        # Related topics
        for topic in data.get('RelatedTopics', [])[:max_results-len(results)]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append({
                    'title': topic.get('Text', '').split(' - ')[0],
                    'snippet': topic.get('Text', ''),
                    'url': topic.get('FirstURL', ''),
                    'source': 'DuckDuckGo Related Topic'
                })
        
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
                params = {
                    'q': query,
                    'api_key': self.api_key,
                    'engine': 'google',
                    'num': min(max_results, 10),  # SerpAPI limit
                    'output': 'json'
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
    
    def _parse_serpapi_response(self, data: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Parse SerpAPI response."""
        results = []
        
        # Organic results
        for result in data.get('organic_results', [])[:max_results]:
            results.append({
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'url': result.get('link', ''),
                'source': 'Google Search',
                'position': result.get('position', 0)
            })
        
        # Knowledge graph
        if data.get('knowledge_graph'):
            kg = data['knowledge_graph']
            results.insert(0, {
                'title': kg.get('title', 'Knowledge Graph'),
                'snippet': kg.get('description', ''),
                'url': kg.get('website', ''),
                'source': 'Google Knowledge Graph'
            })
        
        return results[:max_results]
    
    async def health_check(self) -> bool:
        """Check SerpAPI health."""
        if not self.api_key:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                params = {
                    'q': 'test',
                    'api_key': self.api_key,
                    'engine': 'google',
                    'output': 'json'
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
        """Search Wikipedia articles."""
        logger.info("Performing Wikipedia search", query=query, max_results=max_results)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Search for articles
                search_url = f"{self.base_url}/page/search/{query}"
                params = {
                    'limit': min(max_results, 10)
                }
                
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                
                search_data = response.json()
                results = []
                
                # Get summaries for top results
                for page in search_data.get('pages', [])[:max_results]:
                    try:
                        summary_url = f"{self.base_url}/page/summary/{page['key']}"
                        summary_response = await client.get(summary_url)
                        
                        if summary_response.status_code == 200:
                            summary_data = summary_response.json()
                            results.append({
                                'title': summary_data.get('title', page.get('title', '')),
                                'snippet': summary_data.get('extract', ''),
                                'url': summary_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                                'source': 'Wikipedia'
                            })
                    except Exception as e:
                        logger.warning("Failed to get Wikipedia summary", page=page.get('title'), error=str(e))
                        continue
                
                logger.info("Wikipedia search completed", results_count=len(results))
                return results
                
        except Exception as e:
            logger.error("Wikipedia search failed", query=query, error=str(e))
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
        self.clients = []
        
        # Initialize available search clients
        self.clients.append(DuckDuckGoSearchClient(settings))
        self.clients.append(WikipediaSearchClient(settings))
        
        if settings.search.serpapi_key:
            self.clients.append(SerpAPISearchClient(settings))
        
        logger.info("Initialized multi-source search", client_count=len(self.clients))
    
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search across multiple sources and combine results."""
        logger.info("Performing multi-source search", query=query, max_results=max_results)
        
        # Calculate results per source
        results_per_source = max(1, max_results // len(self.clients))
        
        # Search all sources concurrently
        search_tasks = [
            client.search(query, results_per_source)
            for client in self.clients
        ]
        
        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine and deduplicate results
        combined_results = []
        seen_urls = set()
        
        for source_results in all_results:
            if isinstance(source_results, Exception):
                logger.warning("Search source failed", error=str(source_results))
                continue
                
            for result in source_results:
                url = result.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    combined_results.append(result)
                    
                    if len(combined_results) >= max_results:
                        break
            
            if len(combined_results) >= max_results:
                break
        
        logger.info("Multi-source search completed", 
                   total_results=len(combined_results),
                   sources_used=len([r for r in all_results if not isinstance(r, Exception)]))
        
        return combined_results[:max_results]
    
    async def health_check(self) -> bool:
        """Check health of all search clients."""
        health_checks = await asyncio.gather(
            *[client.health_check() for client in self.clients],
            return_exceptions=True
        )
        
        # Return True if at least one client is healthy
        return any(
            check is True for check in health_checks
            if not isinstance(check, Exception)
        )
    
    def get_available_sources(self) -> List[str]:
        """Get list of available search sources."""
        return [client.__class__.__name__ for client in self.clients]


def create_search_client(settings: Settings) -> SearchClient:
    """Factory function to create appropriate search client based on settings."""
    if settings.search.serpapi_key:
        logger.info("Creating multi-source search client with SerpAPI")
        return MultiSourceSearchClient(settings)
    else:
        logger.info("Creating DuckDuckGo search client (no API keys configured)")
        return DuckDuckGoSearchClient(settings)
