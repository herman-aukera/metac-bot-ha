"""Community prediction fetcher for anchoring strategies."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

from ...domain.services.tournament_calibration_service import CommunityPredictionData


class CommunityPredictionFetcher:
    """Fetches community prediction data for anchoring strategies."""

    def __init__(self, metaculus_client=None):
        self.logger = logging.getLogger(__name__)
        self.metaculus_client = metaculus_client
        self.cache = {}  # Simple cache for community data
        self.cache_duration = timedelta(hours=1)  # Cache for 1 hour

    async def fetch_community_predictions(
        self, question_id: str
    ) -> Optional[CommunityPredictionData]:
        """Fetch community prediction data for a question."""

        # Check cache first
        cache_key = f"community_{question_id}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.utcnow() - cached_time < self.cache_duration:
                self.logger.debug(
                    f"Using cached community data for question {question_id}"
                )
                return cached_data

        try:
            # Fetch from Metaculus API if client is available
            if self.metaculus_client:
                community_data = await self._fetch_from_metaculus_api(question_id)
            else:
                # Fallback to public API
                community_data = await self._fetch_from_public_api(question_id)

            # Cache the result
            if community_data:
                self.cache[cache_key] = (community_data, datetime.utcnow())
                self.logger.info(
                    f"Fetched community predictions for question {question_id}",
                    median=community_data.median_prediction,
                    count=community_data.prediction_count,
                )

            return community_data

        except Exception as e:
            self.logger.error(
                f"Error fetching community predictions for question {question_id}: {e}"
            )
            return None

    async def _fetch_from_metaculus_api(
        self, question_id: str
    ) -> Optional[CommunityPredictionData]:
        """Fetch community data using authenticated Metaculus API."""

        if not self.metaculus_client or not self.metaculus_client.session_token:
            return None

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Token {self.metaculus_client.session_token}",
                    "Content-Type": "application/json",
                }

                # Fetch question details with community predictions
                response = await client.get(
                    f"{self.metaculus_client.base_url}/questions/{question_id}/",
                    headers=headers,
                )

                if response.status_code == 200:
                    question_data = response.json()
                    return self._parse_community_data(question_data)
                else:
                    self.logger.warning(
                        f"Failed to fetch question data from API: {response.status_code}"
                    )
                    return None

        except Exception as e:
            self.logger.error(f"Error in authenticated API fetch: {e}")
            return None

    async def _fetch_from_public_api(
        self, question_id: str
    ) -> Optional[CommunityPredictionData]:
        """Fetch community data using public Metaculus API."""

        try:
            async with httpx.AsyncClient() as client:
                # Try public API endpoint
                response = await client.get(
                    f"https://www.metaculus.com/api2/questions/{question_id}/"
                )

                if response.status_code == 200:
                    question_data = response.json()
                    return self._parse_community_data(question_data)
                else:
                    self.logger.warning(
                        f"Failed to fetch from public API: {response.status_code}"
                    )
                    return None

        except Exception as e:
            self.logger.error(f"Error in public API fetch: {e}")
            return None

    def _parse_community_data(
        self, question_data: Dict
    ) -> Optional[CommunityPredictionData]:
        """Parse community prediction data from Metaculus API response."""

        try:
            # Extract community prediction statistics
            community_prediction = question_data.get("community_prediction")
            if not community_prediction:
                return None

            # Handle different question types
            if question_data.get("type") == "binary":
                median_prediction = community_prediction.get("full", {}).get("q2")
                mean_prediction = community_prediction.get("full", {}).get("mean")

                # Get confidence interval
                q1 = community_prediction.get("full", {}).get("q1")
                q3 = community_prediction.get("full", {}).get("q3")
                confidence_interval = (
                    (q1, q3) if q1 is not None and q3 is not None else None
                )

            elif question_data.get("type") == "continuous":
                # For continuous questions, use the median and mean
                median_prediction = community_prediction.get("full", {}).get("q2")
                mean_prediction = community_prediction.get("full", {}).get("mean")

                q1 = community_prediction.get("full", {}).get("q1")
                q3 = community_prediction.get("full", {}).get("q3")
                confidence_interval = (
                    (q1, q3) if q1 is not None and q3 is not None else None
                )

            else:
                # For other types, try to extract what we can
                median_prediction = community_prediction.get("full", {}).get("q2")
                mean_prediction = community_prediction.get("full", {}).get("mean")
                confidence_interval = None

            # Get prediction count
            prediction_count = question_data.get("number_of_predictions", 0)

            # Get last update time
            last_activity = question_data.get("last_activity_time")
            last_updated = None
            if last_activity:
                try:
                    last_updated = datetime.fromisoformat(
                        last_activity.replace("Z", "+00:00")
                    )
                except:
                    pass

            return CommunityPredictionData(
                median_prediction=median_prediction,
                mean_prediction=mean_prediction,
                prediction_count=prediction_count,
                confidence_interval=confidence_interval,
                last_updated=last_updated,
            )

        except Exception as e:
            self.logger.error(f"Error parsing community data: {e}")
            return None

    def get_cached_community_data(
        self, question_id: str
    ) -> Optional[CommunityPredictionData]:
        """Get cached community data if available and fresh."""

        cache_key = f"community_{question_id}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.utcnow() - cached_time < self.cache_duration:
                return cached_data

        return None

    def clear_cache(self):
        """Clear the community prediction cache."""
        self.cache.clear()
        self.logger.info("Cleared community prediction cache")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        current_time = datetime.utcnow()
        fresh_entries = 0
        stale_entries = 0

        for _, (_, cached_time) in self.cache.items():
            if current_time - cached_time < self.cache_duration:
                fresh_entries += 1
            else:
                stale_entries += 1

        return {
            "total_entries": len(self.cache),
            "fresh_entries": fresh_entries,
            "stale_entries": stale_entries,
        }

    async def fetch_multiple_community_predictions(
        self, question_ids: List[str]
    ) -> Dict[str, Optional[CommunityPredictionData]]:
        """Fetch community predictions for multiple questions concurrently."""

        tasks = [
            self.fetch_community_predictions(question_id)
            for question_id in question_ids
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        community_data = {}
        for question_id, result in zip(question_ids, results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Error fetching community data for {question_id}: {result}"
                )
                community_data[question_id] = None
            else:
                community_data[question_id] = result

        return community_data

    def estimate_community_prediction(
        self, question_type: str, base_rate: Optional[float] = None
    ) -> Optional[CommunityPredictionData]:
        """Estimate community prediction when actual data is unavailable."""

        if question_type == "binary":
            # For binary questions, use base rate or default to 0.5
            estimated_median = base_rate if base_rate is not None else 0.5

            return CommunityPredictionData(
                median_prediction=estimated_median,
                mean_prediction=estimated_median,
                prediction_count=0,  # Indicate this is estimated
                confidence_interval=(
                    max(0.1, estimated_median - 0.2),
                    min(0.9, estimated_median + 0.2),
                ),
                last_updated=datetime.utcnow(),
            )

        return None
