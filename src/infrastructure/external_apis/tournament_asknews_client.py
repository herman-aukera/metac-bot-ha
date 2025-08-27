"""Tournament-optimized AskNews client with quota management and monitoring."""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio

from ..config.tournament_config import get_tournament_config


logger = logging.getLogger(__name__)


@dataclass
class AskNewsUsageStats:
    """Track AskNews API usage statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    quota_exhausted_requests: int = 0
    fallback_requests: int = 0
    estimated_quota_used: int = 0
    last_request_time: Optional[datetime] = None
    daily_request_count: int = 0
    last_reset_date: Optional[str] = None

    def add_request(self, success: bool, used_fallback: bool = False, quota_exhausted: bool = False):
        """Add a request to the statistics."""
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        # Reset daily count if it's a new day
        if self.last_reset_date != today:
            self.daily_request_count = 0
            self.last_reset_date = today

        self.total_requests += 1
        self.daily_request_count += 1
        self.last_request_time = now

        if success:
            self.successful_requests += 1
            self.estimated_quota_used += 1  # Rough estimate: 1 quota per successful request
        else:
            self.failed_requests += 1

        if used_fallback:
            self.fallback_requests += 1

        if quota_exhausted:
            self.quota_exhausted_requests += 1

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

    def get_quota_usage_percentage(self, quota_limit: int) -> float:
        """Get quota usage as a percentage."""
        if quota_limit == 0:
            return 0.0
        return (self.estimated_quota_used / quota_limit) * 100


class TournamentAskNewsClient:
    """
    Tournament-optimized AskNews client with quota management and fallback.

    Features:
    - 9,000 free call quota management
    - Automatic fallback to other search providers
    - Usage monitoring and alerting
    - Tournament-specific optimizations
    """

    def __init__(self):
        """Initialize the tournament AskNews client."""
        self.config = get_tournament_config()
        self.usage_stats = AskNewsUsageStats()
        self.logger = logging.getLogger(__name__)

        # AskNews credentials
        self.client_id = os.getenv("ASKNEWS_CLIENT_ID")
        self.client_secret = os.getenv("ASKNEWS_SECRET")

        # Quota management
        self.quota_limit = self.config.asknews_quota_limit
        self.quota_exhausted = False
        self.daily_limit = int(os.getenv("ASKNEWS_DAILY_LIMIT", "500"))  # Conservative daily limit

        # Fallback providers
        self.fallback_providers = [
            "perplexity",
            "exa",
            "duckduckgo"
        ]

        # Initialize AskNews client if credentials are available
        self.asknews_available = bool(self.client_id and self.client_secret)
        if self.asknews_available:
            try:
                # Import AskNews here to avoid dependency issues if not installed
                from forecasting_tools import AskNewsSearcher
                self.asknews_searcher = AskNewsSearcher()
                self.logger.info("AskNews client initialized successfully")
            except ImportError as e:
                self.logger.warning(f"AskNews SDK not available: {e}")
                self.asknews_available = False
        else:
            self.logger.warning("AskNews credentials not found, will use fallback providers")

    async def get_news_research(self, question: str, max_retries: int = 2) -> str:
        """
        Get news research with quota management and fallback.

        Args:
            question: Question to research
            max_retries: Maximum number of retries for AskNews

        Returns:
            Research results as formatted string
        """
        # Check if we should use AskNews
        if self._should_use_asknews():
            for attempt in range(max_retries + 1):
                try:
                    research = await self._call_asknews(question)
                    if research and len(research.strip()) > 0:
                        self.usage_stats.add_request(success=True)
                        self.logger.info(f"AskNews research successful (attempt {attempt + 1})")
                        return research
                    else:
                        self.logger.warning(f"AskNews returned empty result (attempt {attempt + 1})")

                except Exception as e:
                    self.logger.warning(f"AskNews request failed (attempt {attempt + 1}): {e}")

                    # Check if it's a quota exhaustion error
                    if "quota" in str(e).lower() or "limit" in str(e).lower():
                        self.usage_stats.add_request(success=False, quota_exhausted=True)
                        self.quota_exhausted = True
                        self.logger.error("AskNews quota exhausted, switching to fallback providers")
                        break
                    else:
                        self.usage_stats.add_request(success=False)

                # Wait before retry
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

        # Fall back to other providers
        return await self._call_fallback_providers(question)

    def _should_use_asknews(self) -> bool:
        """Check if we should attempt to use AskNews."""
        if not self.asknews_available:
            return False

        if self.quota_exhausted:
            return False

        # Check quota limits
        if self.usage_stats.estimated_quota_used >= self.quota_limit:
            self.logger.warning("AskNews quota limit reached")
            self.quota_exhausted = True
            return False

        # Check daily limits
        if self.usage_stats.daily_request_count >= self.daily_limit:
            self.logger.warning("AskNews daily limit reached")
            return False

        # Check if failure rate is too high
        if (self.usage_stats.total_requests > 10 and
            self.usage_stats.get_success_rate() < 70):
            self.logger.warning("AskNews success rate too low, temporarily disabling")
            return False

        return True

    async def _call_asknews(self, question: str) -> str:
        """Call AskNews API with tournament optimizations."""
        if not self.asknews_available:
            raise Exception("AskNews not available")

        try:
            # Use the forecasting_tools AskNewsSearcher
            research = await self.asknews_searcher.get_formatted_news_async(question)

            # Log successful usage
            self.logger.info(f"AskNews research completed for question: {question[:100]}...")

            return research or ""

        except Exception as e:
            self.logger.error(f"AskNews API call failed: {e}")
            raise

    async def _call_fallback_providers(self, question: str) -> str:
        """Call fallback search providers when AskNews is unavailable."""
        self.logger.info("Using fallback search providers")

        # Try Perplexity first
        if os.getenv("PERPLEXITY_API_KEY"):
            try:
                research = await self._call_perplexity(question)
                if research:
                    self.usage_stats.add_request(success=True, used_fallback=True)
                    return research
            except Exception as e:
                self.logger.warning(f"Perplexity fallback failed: {e}")

        # Try Exa
        if os.getenv("EXA_API_KEY"):
            try:
                research = await self._call_exa(question)
                if research:
                    self.usage_stats.add_request(success=True, used_fallback=True)
                    return research
            except Exception as e:
                self.logger.warning(f"Exa fallback failed: {e}")

        # Try OpenRouter Perplexity
        if os.getenv("OPENROUTER_API_KEY"):
            try:
                research = await self._call_perplexity(question, use_open_router=True)
                if research:
                    self.usage_stats.add_request(success=True, used_fallback=True)
                    return research
            except Exception as e:
                self.logger.warning(f"OpenRouter Perplexity fallback failed: {e}")

        # If all fallbacks fail
        self.usage_stats.add_request(success=False, used_fallback=True)
        self.logger.error("All search providers failed")
        return ""

    async def _call_perplexity(self, question: str, use_open_router: bool = False) -> str:
        """Call Perplexity API for research."""
        try:
            from forecasting_tools import GeneralLlm, clean_indents

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question}
                """
            )

            if use_open_router:
                model_name = "openrouter/perplexity/sonar-reasoning"
            else:
                model_name = "perplexity/sonar-pro"

            model = GeneralLlm(
                model=model_name,
                temperature=0.1,
            )
            response = await model.invoke(prompt)
            return response

        except Exception as e:
            self.logger.error(f"Perplexity call failed: {e}")
            raise

    async def _call_exa(self, question: str) -> str:
        """Call Exa API for research."""
        try:
            from forecasting_tools import SmartSearcher, GeneralLlm

            # This would need a proper LLM client - simplified for now
            searcher = SmartSearcher(
                model=GeneralLlm(model="openrouter/anthropic/claude-3-5-sonnet"),
                temperature=0,
                num_searches_to_run=2,
                num_sites_per_search=10,
            )

            prompt = (
                "You are an assistant to a superforecaster. The superforecaster will give"
                "you a question they intend to forecast on. To be a great assistant, you generate"
                "a concise but detailed rundown of the most relevant news, including if the question"
                "would resolve Yes or No based on current information. You do not produce forecasts yourself."
                f"\n\nThe question is: {question}"
            )

            response = await searcher.invoke(prompt)
            return response

        except Exception as e:
            self.logger.error(f"Exa call failed: {e}")
            raise

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "total_requests": self.usage_stats.total_requests,
            "successful_requests": self.usage_stats.successful_requests,
            "failed_requests": self.usage_stats.failed_requests,
            "fallback_requests": self.usage_stats.fallback_requests,
            "quota_exhausted_requests": self.usage_stats.quota_exhausted_requests,
            "success_rate": self.usage_stats.get_success_rate(),
            "fallback_rate": self.usage_stats.get_fallback_rate(),
            "estimated_quota_used": self.usage_stats.estimated_quota_used,
            "quota_limit": self.quota_limit,
            "quota_usage_percentage": self.usage_stats.get_quota_usage_percentage(self.quota_limit),
            "daily_request_count": self.usage_stats.daily_request_count,
            "daily_limit": self.daily_limit,
            "quota_exhausted": self.quota_exhausted,
            "asknews_available": self.asknews_available,
            "last_request_time": self.usage_stats.last_request_time.isoformat() if self.usage_stats.last_request_time else None
        }

    def reset_quota_status(self):
        """Reset quota status (useful for testing or manual recovery)."""
        self.quota_exhausted = False
        self.usage_stats = AskNewsUsageStats()
        self.logger.info("AskNews quota status reset")

    def get_quota_alert_level(self) -> str:
        """Get current quota alert level."""
        usage_percentage = self.usage_stats.get_quota_usage_percentage(self.quota_limit)

        if usage_percentage >= 95:
            return "CRITICAL"
        elif usage_percentage >= 80:
            return "HIGH"
        elif usage_percentage >= 60:
            return "MEDIUM"
        else:
            return "LOW"

    def should_alert_quota_usage(self) -> bool:
        """Check if quota usage should trigger an alert."""
        alert_level = self.get_quota_alert_level()
        return alert_level in ["HIGH", "CRITICAL"]

    def get_fallback_providers_status(self) -> Dict[str, bool]:
        """Get status of fallback providers."""
        return {
            "perplexity": bool(os.getenv("PERPLEXITY_API_KEY")),
            "exa": bool(os.getenv("EXA_API_KEY")),
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
            "duckduckgo": True  # Always available
        }
