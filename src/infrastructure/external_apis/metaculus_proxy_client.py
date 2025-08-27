"""Metaculus proxy API client for free credits with fallback to OpenRouter."""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .llm_client import LLMClient, LLMConfig
from ..config.settings import Config


logger = logging.getLogger(__name__)


class ProxyModelType(Enum):
    """Metaculus proxy model types."""
    CLAUDE_3_5_SONNET = "metaculus/claude-3-5-sonnet"
    GPT_4O = "metaculus/gpt-4o"
    GPT_4O_MINI = "metaculus/gpt-4o-mini"


@dataclass
class ProxyUsageStats:
    """Track proxy API usage statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    fallback_requests: int = 0
    estimated_credits_used: float = 0.0

    def add_request(self, success: bool, used_fallback: bool = False, credits_used: float = 0.0):
        """Add a request to the statistics."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        if used_fallback:
            self.fallback_requests += 1
        self.estimated_credits_used += credits_used

    def get_success_rate(self) -> float:
        """Get the success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def get_fallback_rate(self) -> float:
        """Get the fallback rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.fallback_requests / self.total_requests) * 100


class MetaculusProxyClient:
    """
    Client for Metaculus proxy API with automatic fallback to OpenRouter.

    Supports free credit models:
    - metaculus/claude-3-5-sonnet
    - metaculus/gpt-4o
    - metaculus/gpt-4o-mini

    Falls back to OpenRouter when:
    - Proxy credits are exhausted
    - Proxy API is unavailable
    - Proxy models fail
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the Metaculus proxy client."""
        self.config = config or Config()
        self.usage_stats = ProxyUsageStats()
        self.logger = logging.getLogger(__name__)

        # Proxy model configuration
        self.proxy_models = {
            "default": os.getenv("METACULUS_DEFAULT_MODEL", ProxyModelType.CLAUDE_3_5_SONNET.value),
            "summarizer": os.getenv("METACULUS_SUMMARIZER_MODEL", ProxyModelType.GPT_4O_MINI.value),
            "research": os.getenv("METACULUS_RESEARCH_MODEL", ProxyModelType.GPT_4O.value)
        }

        # Fallback model configuration
        self.fallback_models = {
            "default": "openrouter/anthropic/claude-3-5-sonnet",
            "summarizer": "openai/gpt-4o-mini",
            "research": "openrouter/openai/gpt-4o"
        }

        # Credit management
        self.proxy_credits_enabled = os.getenv("ENABLE_PROXY_CREDITS", "true").lower() == "true"
        self.max_proxy_requests = int(os.getenv("MAX_PROXY_REQUESTS", "1000"))  # Conservative limit
        self.proxy_exhausted = False

        self.logger.info(f"Initialized Metaculus proxy client (credits_enabled: {self.proxy_credits_enabled})")

    def get_llm_client(self, model_type: str = "default", purpose: str = "general") -> LLMClient:
        """
        Get an LLM client with proxy support and fallback.

        Args:
            model_type: Type of model to use ("default", "summarizer", "research")
            purpose: Purpose description for logging

        Returns:
            LLMClient configured with proxy or fallback model
        """
        try:
            # Try proxy first if enabled and not exhausted
            if self.proxy_credits_enabled and not self.proxy_exhausted:
                proxy_model = self.proxy_models.get(model_type, self.proxy_models["default"])

                # Check if we should try proxy
                if self._should_use_proxy():
                    try:
                        proxy_config = self._create_proxy_config(proxy_model, purpose)
                        proxy_client = LLMClient(proxy_config)

                        # Test the proxy client with a simple request
                        if self._test_proxy_client(proxy_client):
                            self.logger.info(f"Using Metaculus proxy model: {proxy_model} for {purpose}")
                            return self._wrap_proxy_client(proxy_client, model_type)
                        else:
                            self.logger.warning(f"Proxy model {proxy_model} test failed, falling back")

                    except Exception as e:
                        self.logger.warning(f"Failed to create proxy client for {proxy_model}: {e}")

            # Fall back to regular models
            fallback_model = self.fallback_models.get(model_type, self.fallback_models["default"])
            fallback_config = self._create_fallback_config(fallback_model, purpose)
            fallback_client = LLMClient(fallback_config)

            self.logger.info(f"Using fallback model: {fallback_model} for {purpose}")
            return self._wrap_fallback_client(fallback_client, model_type)

        except Exception as e:
            self.logger.error(f"Failed to create LLM client for {model_type}: {e}")
            # Return a basic fallback client
            basic_config = LLMConfig(
                provider="openrouter",
                model="openrouter/anthropic/claude-3-5-sonnet",
                api_key=self.config.llm.openrouter_api_key,
                temperature=0.3,
                max_retries=2,
                timeout=60.0
            )
            return LLMClient(basic_config)

    def _should_use_proxy(self) -> bool:
        """Check if we should attempt to use the proxy."""
        if self.proxy_exhausted:
            return False

        if self.usage_stats.total_requests >= self.max_proxy_requests:
            self.logger.warning("Proxy request limit reached, disabling proxy")
            self.proxy_exhausted = True
            return False

        # If failure rate is too high, temporarily disable proxy
        if (self.usage_stats.total_requests > 10 and
            self.usage_stats.get_success_rate() < 50):
            self.logger.warning("Proxy success rate too low, temporarily disabling")
            return False

        return True

    def _create_proxy_config(self, model: str, purpose: str) -> LLMConfig:
        """Create LLM configuration for proxy model."""
        return LLMConfig(
            provider="openrouter",  # Proxy models use OpenRouter format
            model=model,
            api_key=self.config.llm.openrouter_api_key,  # Use same API key
            temperature=0.3 if "summarizer" not in purpose else 0.0,
            max_retries=2,
            timeout=60.0,
            max_tokens=4000 if "research" in purpose else 2000
        )

    def _create_fallback_config(self, model: str, purpose: str) -> LLMConfig:
        """Create LLM configuration for fallback model."""
        # Determine provider from model name
        if model.startswith("openrouter/"):
            provider = "openrouter"
            api_key = self.config.llm.openrouter_api_key
        elif model.startswith("openai/"):
            provider = "openai"
            api_key = self.config.llm.openai_api_key
        elif model.startswith("anthropic/"):
            provider = "anthropic"
            api_key = self.config.llm.anthropic_api_key
        else:
            provider = "openrouter"
            api_key = self.config.llm.openrouter_api_key

        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.3 if "summarizer" not in purpose else 0.0,
            max_retries=3,
            timeout=90.0,
            max_tokens=4000 if "research" in purpose else 2000
        )

    def _test_proxy_client(self, client: LLMClient) -> bool:
        """Test if proxy client is working with a simple request."""
        try:
            # Simple test prompt
            test_response = client.generate_text("Say 'OK' if you can respond.", max_tokens=10)
            return test_response and len(test_response.strip()) > 0
        except Exception as e:
            self.logger.debug(f"Proxy client test failed: {e}")
            return False

    def _wrap_proxy_client(self, client: LLMClient, model_type: str) -> LLMClient:
        """Wrap proxy client to track usage statistics."""
        original_generate = client.generate_text

        def tracked_generate(*args, **kwargs):
            try:
                result = original_generate(*args, **kwargs)
                # Estimate credits used (rough approximation)
                credits_used = self._estimate_credits_used(args, kwargs, result)
                self.usage_stats.add_request(success=True, credits_used=credits_used)
                return result
            except Exception as e:
                self.usage_stats.add_request(success=False)
                self.logger.warning(f"Proxy request failed: {e}")
                raise

        client.generate_text = tracked_generate
        return client

    def _wrap_fallback_client(self, client: LLMClient, model_type: str) -> LLMClient:
        """Wrap fallback client to track usage statistics."""
        original_generate = client.generate_text

        def tracked_generate(*args, **kwargs):
            try:
                result = original_generate(*args, **kwargs)
                self.usage_stats.add_request(success=True, used_fallback=True)
                return result
            except Exception as e:
                self.usage_stats.add_request(success=False, used_fallback=True)
                raise

        client.generate_text = tracked_generate
        return client

    def _estimate_credits_used(self, args: tuple, kwargs: dict, result: str) -> float:
        """Estimate credits used for a request (rough approximation)."""
        # This is a rough estimation - actual credit usage depends on Metaculus pricing
        prompt_length = len(str(args[0]) if args else "")
        response_length = len(result or "")

        # Rough token estimation (4 chars per token)
        input_tokens = prompt_length / 4
        output_tokens = response_length / 4

        # Rough credit estimation (this would need to be calibrated with actual usage)
        credits = (input_tokens * 0.001) + (output_tokens * 0.002)
        return credits

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "total_requests": self.usage_stats.total_requests,
            "successful_requests": self.usage_stats.successful_requests,
            "failed_requests": self.usage_stats.failed_requests,
            "fallback_requests": self.usage_stats.fallback_requests,
            "success_rate": self.usage_stats.get_success_rate(),
            "fallback_rate": self.usage_stats.get_fallback_rate(),
            "estimated_credits_used": self.usage_stats.estimated_credits_used,
            "proxy_exhausted": self.proxy_exhausted,
            "proxy_credits_enabled": self.proxy_credits_enabled
        }

    def reset_proxy_status(self):
        """Reset proxy status (useful for testing or manual recovery)."""
        self.proxy_exhausted = False
        self.usage_stats = ProxyUsageStats()
        self.logger.info("Proxy status reset")

    def disable_proxy(self):
        """Manually disable proxy (useful for testing fallback)."""
        self.proxy_exhausted = True
        self.logger.info("Proxy manually disabled")

    def get_available_models(self) -> Dict[str, Dict[str, str]]:
        """Get available proxy and fallback models."""
        return {
            "proxy_models": self.proxy_models.copy(),
            "fallback_models": self.fallback_models.copy(),
            "proxy_enabled": self.proxy_credits_enabled and not self.proxy_exhausted
        }
