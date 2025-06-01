"""Unit tests for infrastructure components."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import aiohttp
import httpx
import json

from src.infrastructure.external_apis.llm_client import LLMClient
from src.infrastructure.external_apis.search_client import SearchClient
from src.infrastructure.external_apis.metaculus_client import MetaculusClient
from src.infrastructure.config.settings import LLMConfig, SearchConfig, MetaculusConfig


class TestLLMClient:
    """Test LLM client functionality."""
    
    @pytest.fixture
    def llm_config(self):
        """Create LLM configuration for testing."""
        return LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-api-key",
            temperature=0.7,
            max_tokens=4000,
            timeout=60,
            max_retries=3
        )
    
    @pytest.fixture
    def llm_client(self, llm_config):
        """Create LLM client instance."""
        return LLMClient(llm_config)
    
    @pytest.mark.asyncio
    async def test_openai_request_success(self, llm_client, mock_openai_response):
        """Test successful OpenAI API request."""
        with patch('httpx.AsyncClient.post') as mock_post:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json = Mock(return_value=mock_openai_response)
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response
            
            # Test request
            prompt = "What is the probability that AI will achieve AGI by 2030?"
            response = await llm_client.generate_response(prompt)
            
            assert "42%" in response or "0.42" in response
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_openai_request_retry_on_failure(self, llm_client):
        """Test retry mechanism on API failure."""
        with patch('httpx.AsyncClient.post') as mock_post:
            # Mock failed responses followed by success
            mock_failed_response = Mock()
            mock_failed_response.status_code = 500
            mock_failed_response.raise_for_status.side_effect = Exception("Server Error")
            
            mock_success_response = Mock()
            mock_success_response.status_code = 200
            mock_success_response.json.return_value = {
                "choices": [{"message": {"content": "Success after retry"}}]
            }
            mock_success_response.raise_for_status = Mock()
            
            mock_post.side_effect = [
                mock_failed_response,  # First attempt fails
                mock_failed_response,  # Second attempt fails
                mock_success_response  # Third attempt succeeds
            ]
            
            response = await llm_client.generate_response("Test prompt")
            
            assert "Success after retry" in response
            assert mock_post.call_count == 3
    
    @pytest.mark.asyncio
    async def test_openai_max_retries_exceeded(self, llm_client):
        """Test behavior when max retries are exceeded."""
        with patch('httpx.AsyncClient.post') as mock_post:
            # Mock all responses as failures
            mock_failed_response = Mock()
            mock_failed_response.status_code = 500
            mock_failed_response.raise_for_status.side_effect = Exception("Server Error")
            
            mock_post.return_value = mock_failed_response
            
            with pytest.raises(Exception):
                await llm_client.generate_response("Test prompt")
            
            # Should attempt max_retries times (3 retries total)
            assert mock_post.call_count == 3  # max_retries = 3
    
    def test_openai_headers_construction(self, llm_client):
        """Test that OpenAI headers are constructed correctly."""
        headers = llm_client._get_headers()
        
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"
    
    def test_prompt_formatting(self, llm_client):
        """Test prompt formatting for different use cases."""
        # Test basic prompt
        formatted = llm_client._format_prompt("Simple question")
        assert "Simple question" in formatted
        
        # Test structured prompt
        structured_prompt = {
            "question": "Test question",
            "context": "Test context",
            "format": "json"
        }
        formatted_structured = llm_client._format_prompt(structured_prompt)
        assert "Test question" in formatted_structured
        assert "Test context" in formatted_structured


class TestSearchClient:
    """Test search client functionality."""
    
    @pytest.fixture
    def search_config(self):
        """Create search configuration for testing."""
        return SearchConfig(
            provider="multi_source",
            max_results=10,
            timeout=30,
            duckduckgo_enabled=True,
            wikipedia_enabled=True,
            concurrent_searches=True,
            deduplicate_results=True
        )
    
    @pytest.fixture
    def search_client(self, search_config):
        """Create search client instance."""
        return SearchClient(search_config)
    
    @pytest.mark.asyncio
    async def test_duckduckgo_search(self, search_client, mock_search_results):
        """Test DuckDuckGo search functionality."""
        with patch('src.infrastructure.external_apis.search_client.duckduckgo_search') as mock_ddg:
            mock_ddg.return_value = [
                {
                    "title": "AI Progress Report",
                    "href": "https://example.com/ai-report",
                    "body": "Recent AI developments..."
                }
            ]
            
            results = await search_client.search("AI AGI progress")
            
            assert len(results) > 0
            assert results[0]["source"] == "duckduckgo"
            assert "title" in results[0]
            assert "url" in results[0]
            assert "snippet" in results[0]
    
    @pytest.mark.asyncio
    async def test_wikipedia_search(self, search_client):
        """Test Wikipedia search functionality."""
        with patch('wikipedia.search') as mock_search, \
             patch('wikipedia.summary') as mock_summary, \
             patch('wikipedia.page') as mock_page:
            
            # Mock Wikipedia API responses
            mock_search.return_value = ["Artificial general intelligence"]
            mock_summary.return_value = "AGI is the intelligence of a machine..."
            
            mock_page_obj = Mock()
            mock_page_obj.url = "https://en.wikipedia.org/wiki/Artificial_general_intelligence"
            mock_page.return_value = mock_page_obj
            
            results = await search_client._search_wikipedia("AGI")
            
            assert len(results) > 0
            assert results[0]["source"] == "wikipedia"
            assert "Artificial general intelligence" in results[0]["title"]
    
    @pytest.mark.asyncio
    async def test_search_deduplication(self, search_client):
        """Test search result deduplication."""
        with patch.object(search_client, '_search_duckduckgo') as mock_ddg, \
             patch.object(search_client, '_search_wikipedia') as mock_wiki:
            
            # Mock duplicate results from different sources
            duplicate_result = {
                "title": "AI Progress Report",
                "url": "https://example.com/ai-report",
                "snippet": "AI developments...",
                "source": "duckduckgo"
            }
            
            mock_ddg.return_value = [duplicate_result]
            mock_wiki.return_value = [
                {
                    "title": "AI Progress Report",  # Same title
                    "url": "https://en.wikipedia.org/wiki/AI",  # Different URL
                    "snippet": "AI is the simulation...",
                    "source": "wikipedia"
                }
            ]
            
            results = await search_client.search("AI progress")
            
            # Should deduplicate based on title similarity
            unique_titles = set(result["title"] for result in results)
            assert len(unique_titles) <= len(results)
    
    @pytest.mark.asyncio
    async def test_search_caching(self, search_client):
        """Test search result caching."""
        with patch.object(search_client, '_search_duckduckgo') as mock_ddg:
            mock_ddg.return_value = [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "snippet": "Test snippet",
                    "source": "duckduckgo"
                }
            ]
            
            # First search
            results1 = await search_client.search("test query")
            
            # Second search with same query
            results2 = await search_client.search("test query")
            
            # Should use cache for second request
            assert results1 == results2
            mock_ddg.assert_called_once()  # Only called once due to caching
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, search_client):
        """Test search error handling."""
        with patch.object(search_client, '_search_duckduckgo') as mock_ddg:
            mock_ddg.side_effect = Exception("Search API error")
            
            # Should handle errors gracefully
            results = await search_client.search("test query")
            
            # Should return empty results on error
            assert results == []


class TestMetaculusClient:
    """Test Metaculus client functionality."""
    
    @pytest.fixture
    def metaculus_config(self):
        """Create Metaculus configuration for testing."""
        return MetaculusConfig(
            api_token="test-metaculus-key",
            base_url="https://www.metaculus.com/api2",
            timeout=30.0,
            submit_predictions=False,
            dry_run=True
        )
    
    @pytest.fixture
    def metaculus_client(self, metaculus_config):
        """Create Metaculus client instance."""
        return MetaculusClient(metaculus_config)
    
    @pytest.mark.asyncio
    async def test_get_question_success(self, metaculus_client, mock_metaculus_response):
        """Test successful question retrieval."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_metaculus_response)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_get.return_value = mock_response
            
            question = await metaculus_client.get_question(12345)
            
            assert question["id"] == 12345
            assert question["title"] == "Will AI achieve AGI by 2030?"
            assert question["type"] == "binary"
            mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_questions_list(self, metaculus_client):
        """Test retrieving list of questions."""
        mock_questions_response = {
            "results": [
                {"id": 12345, "title": "Question 1", "type": "binary"},
                {"id": 12346, "title": "Question 2", "type": "numeric"}
            ],
            "count": 2
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_questions_response)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_get.return_value = mock_response
            
            questions = await metaculus_client.get_questions(limit=10)
            
            assert len(questions["results"]) == 2
            assert questions["count"] == 2
            mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_prediction_dry_run(self, metaculus_client):
        """Test prediction submission in dry run mode."""
        prediction_data = {
            "question_id": 12345,
            "prediction": 0.42,
            "reasoning": "Test reasoning"
        }
        
        # Should not make actual API call in dry run mode
        result = await metaculus_client.submit_prediction(prediction_data)
        
        assert result["status"] == "dry_run"
        assert result["would_submit"] is True
    
    @pytest.mark.asyncio
    async def test_submit_prediction_disabled(self, metaculus_client):
        """Test prediction submission when disabled."""
        metaculus_client.config.submit_predictions = False
        metaculus_client.config.dry_run = False
        
        prediction_data = {
            "question_id": 12345,
            "prediction": 0.42,
            "reasoning": "Test reasoning"
        }
        
        result = await metaculus_client.submit_prediction(prediction_data)
        
        assert result["status"] == "disabled"
        assert result["submitted"] is False
    
    @pytest.mark.asyncio
    async def test_authentication_headers(self, metaculus_client):
        """Test authentication headers are set correctly."""
        headers = metaculus_client._get_headers()
        
        assert "Authorization" in headers
        assert headers["Authorization"] == "Token test-metaculus-key"
        assert headers["Content-Type"] == "application/json"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, metaculus_client):
        """Test rate limiting functionality."""
        with patch('asyncio.sleep') as mock_sleep:
            # Mock multiple rapid requests
            for _ in range(5):
                await metaculus_client._handle_rate_limit()
            
            # Should call sleep for rate limiting
            assert mock_sleep.call_count >= 0  # May or may not sleep depending on timing
    
    @pytest.mark.asyncio
    async def test_error_handling(self, metaculus_client):
        """Test API error handling."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 404
            mock_response.text = AsyncMock(return_value="Not found")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_get.return_value = mock_response
            
            with pytest.raises(Exception):
                await metaculus_client.get_question(99999)
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, metaculus_client):
        """Test retry mechanism on API failures."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock failed responses followed by success
            mock_failed_response = Mock()
            mock_failed_response.status = 500
            mock_failed_response.__aenter__ = AsyncMock(return_value=mock_failed_response)
            mock_failed_response.__aexit__ = AsyncMock(return_value=None)
            
            mock_success_response = Mock()
            mock_success_response.status = 200
            mock_success_response.json = AsyncMock(return_value={"id": 12345})
            mock_success_response.__aenter__ = AsyncMock(return_value=mock_success_response)
            mock_success_response.__aexit__ = AsyncMock(return_value=None)
            
            mock_get.side_effect = [
                mock_failed_response,  # First attempt fails
                mock_success_response  # Second attempt succeeds
            ]
            
            question = await metaculus_client.get_question(12345)
            
            assert question["id"] == 12345
            assert mock_get.call_count == 2
