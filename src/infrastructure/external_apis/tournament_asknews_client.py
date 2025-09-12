"""Tournament-optimized AskNews client with quota management and monitoring."""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

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

    def add_request(
        self, success: bool, used_fallback: bool = False, quota_exhausted: bool = False
    ) -> None:
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
            self.estimated_quota_used += (
                1  # Rough estimate: 1 quota per successful request
            )
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

    # Class-level semaphore for cross-instance concurrency limiting
    _global_semaphore: Optional[asyncio.Semaphore] = None

    def __init__(self) -> None:
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
        self.daily_limit = int(
            os.getenv("ASKNEWS_DAILY_LIMIT", "500")
        )  # Conservative daily limit
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until: Optional[datetime] = None
        self._failure_threshold = int(os.getenv("ASKNEWS_FAILURE_THRESHOLD", "5"))
        self._cooldown_seconds = int(os.getenv("ASKNEWS_COOLDOWN_SECONDS", "120"))

        # Fallback providers (re-enabled DuckDuckGo lightweight public source)
        self.fallback_providers = ["duckduckgo"]

        # Initialize AskNews SDK if credentials are available
        self.asknews_available = bool(self.client_id and self.client_secret)
        self.ask = None

        # Initialize global semaphore lazily
        if TournamentAskNewsClient._global_semaphore is None:
            max_conc = int(os.getenv("ASKNEWS_MAX_CONCURRENCY", "1"))
            TournamentAskNewsClient._global_semaphore = asyncio.Semaphore(max_conc)
            self.logger.info(f"AskNews global concurrency set to {max_conc}")
        if self.asknews_available:
            try:
                from asknews_sdk import AskNewsSDK  # Provided by 'asknews' package

                scopes = {"news"}  # minimal scope needed for research
                self.ask = AskNewsSDK(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    scopes=scopes,
                )
                self.logger.info("AskNewsSDK initialized successfully")
            except Exception as e:
                self.logger.warning(f"AskNews SDK init failed: {e}")
                self.asknews_available = False
        else:
            self.logger.warning(
                "AskNews credentials not found; deferring to pipeline free-model synthesis"
            )

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
                result, should_break = await self._asknews_attempt(question, attempt)
                if result:
                    return result
                if should_break:
                    break
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

        # Fall back: defer to pipeline free-model synthesis (Kimi K2 / GPT-OSS via OpenRouter)
        self.usage_stats.add_request(success=False, used_fallback=True)
        return ""

    def _should_use_asknews(self) -> bool:
        """Check if we should attempt to use AskNews."""
        if not self.asknews_available:
            return False

        # Circuit breaker check
        if self._breaker_open_until and datetime.now(timezone.utc) < self._breaker_open_until:
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
        if (
            self.usage_stats.total_requests > 10
            and self.usage_stats.get_success_rate() < 70
        ):
            self.logger.warning("AskNews success rate too low, temporarily disabling")
            return False

        return True

    async def _call_asknews(self, question: str) -> str:
        """Call AskNews API with tournament optimizations."""
        if not self.asknews_available or self.ask is None:
            raise RuntimeError("AskNews SDK not available")

        try:
            # Stagger start to reduce burst (global small jitter)
            await asyncio.sleep(float(os.getenv("ASKNEWS_STAGGER_SECONDS", "0.15")))
            # Acquire global semaphore
            assert TournamentAskNewsClient._global_semaphore is not None
            async with TournamentAskNewsClient._global_semaphore:
                def _search() -> Any:  # run sync SDK in thread
                    assert self.ask is not None
                    return self.ask.news.search_news(
                        query=question,
                        n_articles=10,
                        return_type="string",
                        method="nl",
                    )
                resp = await asyncio.to_thread(_search)
            research = getattr(resp, "as_string", None) or str(resp)

            # Log successful usage
            self.logger.info(
                f"AskNews research completed for question: {question[:100]}..."
            )

            return research or ""

        except Exception as e:
            self.logger.error(f"AskNews API call failed: {e}")
            raise

    async def _asknews_attempt(self, question: str, attempt: int) -> Tuple[Optional[str], bool]:
        """Perform a single AskNews attempt.

        Returns
        -------
        tuple
            (research_text_or_none, should_break)
        """
        try:
            # Simple rate limit guard: small async sleep between attempts to reduce burst 429s
            if attempt > 0:
                await asyncio.sleep(0.75)
            research = await self._call_asknews(question)
            if research and len(research.strip()) > 0:
                # reset failure counter on success
                self._consecutive_failures = 0
                self.usage_stats.add_request(success=True)
                self.logger.info(
                    f"AskNews research successful (attempt {attempt + 1})"
                )
                return research, False
            else:
                self.logger.warning(
                    f"AskNews returned empty result (attempt {attempt + 1})"
                )
                self.usage_stats.add_request(success=False)
                self._consecutive_failures += 1
                return None, False
        except Exception as e:
            self.logger.warning(
                f"AskNews request failed (attempt {attempt + 1}): {e}"
            )
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                self.usage_stats.add_request(success=False, quota_exhausted=True)
                self.quota_exhausted = True
                self.logger.error(
                    "AskNews quota exhausted, switching to fallback providers"
                )
                return None, True
            else:
                self.usage_stats.add_request(success=False)
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failure_threshold:
                    self._breaker_open_until = datetime.now(timezone.utc) + timedelta(seconds=self._cooldown_seconds)
                    self.logger.error(
                        f"AskNews circuit breaker OPEN for {self._cooldown_seconds}s after {self._consecutive_failures} consecutive failures"
                    )
                return None, False

    async def _call_fallback_providers(self, question: str) -> str:
        """Use DuckDuckGo instant answer as lightweight fallback, then synthesize."""
        try:
            from .search_client import DuckDuckGoSearchClient
            from ..config.settings import get_settings
            settings = get_settings()
            ddg = DuckDuckGoSearchClient(settings)
            results = await ddg.search(question, max_results=5)
            if not results:
                return ""
            bullet_lines = [
                f"- {r.get('title', '')}: {r.get('url', '')[:120]}" for r in results
            ]
            summary = "DuckDuckGo fallback results:\n" + "\n".join(bullet_lines)
            return summary
        except Exception as e:
            self.logger.warning(f"DuckDuckGo fallback failed: {e}")
            return ""

    async def _call_perplexity(self, question: str, use_open_router: bool = False) -> str:
        """Deprecated: Perplexity usage removed per provider policy."""
        raise RuntimeError("Perplexity fallback disabled")

    async def _call_exa(self, question: str) -> str:
        """Deprecated: Exa usage removed per provider policy."""
        raise RuntimeError("Exa fallback disabled")

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
            "quota_usage_percentage": self.usage_stats.get_quota_usage_percentage(
                self.quota_limit
            ),
            "daily_request_count": self.usage_stats.daily_request_count,
            "daily_limit": self.daily_limit,
            "quota_exhausted": self.quota_exhausted,
            "asknews_available": self.asknews_available,
            "last_request_time": (
                self.usage_stats.last_request_time.isoformat()
                if self.usage_stats.last_request_time
                else None
            ),
        }

    def reset_quota_status(self) -> None:
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
        """Get status of (now disabled) external fallback providers."""
        return {
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        }

    async def aclose(self) -> None:  # pragma: no cover - graceful shutdown hook
        """Graceful async close for interface parity (future async SDK)."""
        try:
            ask_obj = getattr(self, "ask", None)
            if ask_obj and hasattr(ask_obj, "close"):
                maybe = ask_obj.close()
                if asyncio.iscoroutine(maybe):  # type: ignore[arg-type]
                    await maybe  # type: ignore[misc]
        except Exception:
            pass
