"""
Metaculus API client for fetching questions and submitting predictions.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import structlog

from ...domain.entities.prediction import Prediction
from ...domain.entities.question import Question, QuestionType
from ...domain.value_objects.probability import Probability
from ..config.settings import Settings
from .reasoning_comment_formatter import ReasoningCommentFormatter

logger = structlog.get_logger(__name__)


class MetaculusClient:
    """Client for interacting with the Metaculus API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = "https://www.metaculus.com/api2"
        self.session_token = None
        self.user_id = None
        self.reasoning_formatter = ReasoningCommentFormatter()

    @property
    def config(self):
        """Access to metaculus configuration."""
        return self.settings.metaculus

    async def authenticate(
        self, username: Optional[str] = None, password: Optional[str] = None
    ) -> bool:
        """Authenticate with Metaculus API."""
        username = username or self.settings.metaculus_username
        password = password or self.settings.metaculus_password

        if not username or not password:
            logger.warning("Metaculus credentials not provided")
            return False

        try:
            async with httpx.AsyncClient() as client:
                auth_data = {"username": username, "password": password}

                response = await client.post(
                    f"{self.base_url}/accounts/login/",
                    json=auth_data,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    data = response.json()
                    self.session_token = data.get("session_token")
                    self.user_id = data.get("user_id")

                    logger.info(
                        "Successfully authenticated with Metaculus",
                        user_id=self.user_id,
                    )
                    return True
                else:
                    logger.error(
                        "Metaculus authentication failed",
                        status_code=response.status_code,
                    )
                    return False

        except Exception as e:
            logger.error("Metaculus authentication error", error=str(e))
            return False

    async def fetch_questions(
        self,
        status: str = "open",
        limit: int = 20,
        offset: int = 0,
        categories: Optional[List[str]] = None,
    ) -> List[Question]:
        """Fetch questions from Metaculus."""
        logger.info(
            "Fetching Metaculus questions", status=status, limit=limit, offset=offset
        )

        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "limit": limit,
                    "offset": offset,
                    "status": status,
                    "order_by": "-publish_time",
                }

                if categories:
                    params["categories"] = ",".join(categories)

                response = await client.get(
                    f"{self.base_url}/questions/", params=params
                )
                response.raise_for_status()

                data = response.json()
                questions = []

                for q_data in data.get("results", []):
                    try:
                        question = self._parse_question(q_data)
                        questions.append(question)
                    except Exception as e:
                        logger.warning(
                            "Failed to parse question",
                            question_id=q_data.get("id"),
                            error=str(e),
                        )
                        continue

                logger.info("Fetched Metaculus questions", count=len(questions))
                return questions

        except Exception as e:
            logger.error("Failed to fetch Metaculus questions", error=str(e))
            return []

    async def fetch_question(self, question_id: int) -> Optional[Question]:
        """Fetch a specific question by ID."""
        logger.info("Fetching Metaculus question", question_id=question_id)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/questions/{question_id}/")
                response.raise_for_status()

                question_data = response.json()
                question = self._parse_question(question_data)

                logger.info(
                    "Fetched Metaculus question",
                    question_id=question_id,
                    title=question.title,
                )
                return question

        except Exception as e:
            logger.error(
                "Failed to fetch Metaculus question",
                question_id=question_id,
                error=str(e),
            )
            return None

    async def submit_prediction(
        self,
        prediction_data_or_question_id,
        prediction: Optional[Prediction] = None,
        comment: Optional[str] = None,
    ):
        """Submit a prediction to Metaculus."""
        # Handle both new dict interface and old separate parameters interface
        if isinstance(prediction_data_or_question_id, dict):
            # New interface: submit_prediction(prediction_data)
            prediction_data = prediction_data_or_question_id
            question_id = prediction_data["question_id"]
            prediction_value = prediction_data["prediction"]
            raw_reasoning = prediction_data.get("reasoning", "")

            # For dict interface, we need to create a minimal prediction object for formatting
            # if we don't have the full prediction object
            if prediction is None:
                # Create a minimal prediction for formatting purposes
                from uuid import uuid4

                from ...domain.entities.prediction import (
                    Prediction,
                    PredictionConfidence,
                    PredictionMethod,
                    PredictionResult,
                )

                temp_prediction = Prediction(
                    id=uuid4(),
                    question_id=uuid4(),  # Will be overridden
                    research_report_id=uuid4(),
                    result=PredictionResult(binary_probability=prediction_value),
                    confidence=PredictionConfidence.MEDIUM,
                    method=PredictionMethod.ENSEMBLE,
                    reasoning=raw_reasoning,
                    reasoning_steps=[],
                    created_at=datetime.utcnow(),
                    created_by="api_submission",
                )

                reasoning = self.reasoning_formatter.format_prediction_comment(
                    temp_prediction, question_title=f"Question {question_id}"
                )
            else:
                reasoning = self.reasoning_formatter.format_prediction_comment(
                    prediction, question_title=f"Question {question_id}"
                )

            # Validate the formatted comment
            validation_prediction = prediction if prediction else temp_prediction
            validation_result = (
                self.reasoning_formatter.validate_comment_before_submission(
                    reasoning, validation_prediction
                )
            )

            if not validation_result["is_valid"]:
                logger.warning(
                    "Reasoning comment validation issues",
                    question_id=question_id,
                    issues=validation_result["issues"],
                )
                reasoning = validation_result["formatted_comment"]
        else:
            # Old interface: submit_prediction(question_id, prediction, comment)
            question_id = prediction_data_or_question_id
            if prediction is None:
                raise ValueError("prediction is required when using old interface")

            # Extract prediction value from PredictionResult
            if prediction.result.binary_probability is not None:
                prediction_value = prediction.result.binary_probability
            elif prediction.result.numeric_value is not None:
                prediction_value = prediction.result.numeric_value
            else:
                raise ValueError(
                    "Prediction must have either binary_probability or numeric_value"
                )

            # Format reasoning comment for tournament compliance
            raw_reasoning = comment or prediction.reasoning
            reasoning = self.reasoning_formatter.format_prediction_comment(
                prediction, question_title=f"Question {question_id}"
            )

            # Validate the formatted comment
            validation_result = (
                self.reasoning_formatter.validate_comment_before_submission(
                    reasoning, prediction
                )
            )

            if not validation_result["is_valid"]:
                logger.warning(
                    "Reasoning comment validation issues",
                    question_id=question_id,
                    issues=validation_result["issues"],
                )
                # Use the formatted version from validation
                reasoning = validation_result["formatted_comment"]

            logger.info(
                "Formatted reasoning comment for tournament compliance",
                question_id=question_id,
                original_length=len(raw_reasoning) if raw_reasoning else 0,
                formatted_length=len(reasoning),
                validation_score=validation_result.get("score", "N/A"),
            )

        # Check dry_run mode
        if hasattr(self.config, "dry_run") and self.config.dry_run:
            logger.info(
                "Dry run mode - would submit prediction",
                question_id=question_id,
                prediction=prediction_value,
            )
            return {
                "status": "dry_run",
                "would_submit": True,
                "question_id": question_id,
                "prediction": prediction_value,
            }

        # Check if submissions are disabled
        if (
            hasattr(self.config, "submit_predictions")
            and not self.config.submit_predictions
        ):
            logger.info("Prediction submission disabled", question_id=question_id)
            return {
                "status": "disabled",
                "submitted": False,
                "question_id": question_id,
            }

        # Check authentication
        if not self.session_token:
            logger.error("Not authenticated with Metaculus")
            return {"status": "error", "error": "Not authenticated", "submitted": False}

        logger.info(
            "Submitting prediction to Metaculus",
            question_id=question_id,
            prediction=prediction_value,
        )

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Token {self.session_token}",
                    "Content-Type": "application/json",
                }

                submit_data = {"prediction": prediction_value, "comment": reasoning}

                response = await client.post(
                    f"{self.base_url}/questions/{question_id}/predict/",
                    json=submit_data,
                    headers=headers,
                )

                if response.status_code in [200, 201]:
                    logger.info(
                        "Successfully submitted prediction", question_id=question_id
                    )
                    return {
                        "status": "success",
                        "submitted": True,
                        "question_id": question_id,
                        "prediction": prediction_value,
                    }
                else:
                    logger.error(
                        "Failed to submit prediction",
                        question_id=question_id,
                        status_code=response.status_code,
                        response=response.text,
                    )
                    return {
                        "status": "error",
                        "submitted": False,
                        "error": f"HTTP {response.status_code}",
                        "question_id": question_id,
                    }

        except Exception as e:
            logger.error(
                "Error submitting prediction", question_id=question_id, error=str(e)
            )
            return {
                "status": "error",
                "submitted": False,
                "error": str(e),
                "question_id": question_id,
            }

    async def fetch_user_predictions(
        self, user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch user's predictions."""
        user_id = user_id or self.user_id

        if not user_id:
            logger.error("No user ID available")
            return []

        try:
            async with httpx.AsyncClient() as client:
                headers = {}
                if self.session_token:
                    headers["Authorization"] = f"Token {self.session_token}"

                response = await client.get(
                    f"{self.base_url}/users/{user_id}/predictions/", headers=headers
                )
                response.raise_for_status()

                data = response.json()
                predictions = data.get("results", [])

                logger.info("Fetched user predictions", count=len(predictions))
                return predictions

        except Exception as e:
            logger.error(
                "Failed to fetch user predictions", user_id=user_id, error=str(e)
            )
            return []

    async def fetch_question_comments(self, question_id: int) -> List[Dict[str, Any]]:
        """Fetch comments for a question."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/questions/{question_id}/comments/"
                )
                response.raise_for_status()

                data = response.json()
                comments = data.get("results", [])

                logger.info(
                    "Fetched question comments",
                    question_id=question_id,
                    count=len(comments),
                )
                return comments

        except Exception as e:
            logger.error(
                "Failed to fetch question comments",
                question_id=question_id,
                error=str(e),
            )
            return []

    async def health_check(self) -> bool:
        """Check if Metaculus API is accessible."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/questions/?limit=1")
                return response.status_code == 200
        except Exception:
            return False

    def _parse_question(self, data: Dict[str, Any]) -> Question:
        """Parse Metaculus question data into Question entity."""
        # Determine question type
        question_type = QuestionType.BINARY
        if data.get("type") == "continuous":
            question_type = QuestionType.CONTINUOUS
        elif data.get("type") == "multiple_choice":
            question_type = QuestionType.MULTIPLE_CHOICE

        # Parse dates
        created_at = self._parse_datetime(data.get("created_time"))
        close_time = self._parse_datetime(data.get("close_time"))
        resolve_time = self._parse_datetime(data.get("resolve_time"))

        # Extract additional metadata
        metadata = {
            "metaculus_id": data.get("id"),
            "url": f"https://www.metaculus.com/questions/{data.get('id')}/",
            "author": data.get("author_name"),
            "category": data.get("category"),
            "tags": data.get("tags", []),
            "prediction_count": data.get("number_of_predictions", 0),
            "comment_count": data.get("comment_count", 0),
            "community_prediction": data.get("community_prediction"),
            "status": data.get("status"),
            "points": data.get("points"),
            "resolution": data.get("resolution"),
        }

        return Question.create(
            title=data.get("title", ""),
            description=data.get("description", ""),
            question_type=question_type,
            resolution_criteria=data.get("resolution_criteria"),
            close_time=close_time,
            resolve_time=resolve_time,
            created_at=created_at,
            metadata=metadata,
            choices=(
                data.get("choices")
                if question_type == QuestionType.MULTIPLE_CHOICE
                else None
            ),
            min_value=(
                data.get("min_value")
                if question_type == QuestionType.CONTINUOUS
                else None
            ),
            max_value=(
                data.get("max_value")
                if question_type == QuestionType.CONTINUOUS
                else None
            ),
        )

    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string to datetime object."""
        if not date_str:
            return None

        try:
            # Metaculus uses ISO format
            if date_str.endswith("Z"):
                date_str = date_str[:-1] + "+00:00"

            return datetime.fromisoformat(date_str)
        except Exception as e:
            logger.warning("Failed to parse datetime", date_str=date_str, error=str(e))
            return None

    async def fetch_benchmark_questions(self, limit: int = 50) -> List[Question]:
        """Fetch questions suitable for benchmarking."""
        logger.info("Fetching benchmark questions", limit=limit)

        # Fetch resolved questions for benchmarking
        resolved_questions = await self.fetch_questions(
            status="resolved",
            limit=limit,
            categories=["Technology", "Science", "Economics", "Politics"],
        )

        # Filter for binary questions (easier to benchmark)
        binary_questions = [
            q
            for q in resolved_questions
            if q.question_type == QuestionType.BINARY
            and q.metadata.get("resolution") is not None
        ]

        logger.info(
            "Fetched benchmark questions",
            total=len(resolved_questions),
            binary=len(binary_questions),
        )

        return binary_questions[:limit]

    async def batch_fetch_questions(
        self, question_ids: List[int]
    ) -> List[Optional[Question]]:
        """Fetch multiple questions by ID concurrently."""
        logger.info("Batch fetching questions", count=len(question_ids))

        tasks = [self.fetch_question(qid) for qid in question_ids]
        questions = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        valid_questions = []
        for i, result in enumerate(questions):
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to fetch question",
                    question_id=question_ids[i],
                    error=str(result),
                )
                valid_questions.append(None)
            else:
                valid_questions.append(result)

        logger.info(
            "Batch fetch completed",
            requested=len(question_ids),
            successful=len([q for q in valid_questions if q is not None]),
        )

        return valid_questions

    async def get_question(self, question_id: int) -> Optional[Question]:
        """
        Get a single question by ID - alias for fetch_question.

        Args:
            question_id: Metaculus question ID

        Returns:
            Question object if found, None otherwise
        """
        return await self.fetch_question(question_id)

    async def get_questions(self, limit: int = 20, **kwargs) -> List[Question]:
        """
        Get multiple questions - alias for fetch_questions.

        Args:
            limit: Maximum number of questions to fetch
            **kwargs: Additional filters

        Returns:
            List of Question objects
        """
        return await self.fetch_questions(limit=limit, **kwargs)

    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.

        Returns:
            Dictionary of headers for the API request
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MetaculusForecastingBot/1.0",
        }

        if self.session_token:
            headers["Authorization"] = f"Token {self.session_token}"
        elif hasattr(self.settings, "metaculus") and hasattr(
            self.settings.metaculus, "api_token"
        ):
            if self.settings.metaculus.api_token:
                headers["Authorization"] = f"Token {self.settings.metaculus.api_token}"

        return headers

    async def _handle_rate_limit(self) -> None:
        """
        Handle rate limiting by waiting if necessary.
        """
        # Simple rate limiting - wait 1 second between requests
        await asyncio.sleep(1.0)

    async def health_check(self) -> bool:
        """
        Check if the Metaculus API is available.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/questions/", params={"limit": 1}
                )
                return response.status_code == 200
        except Exception as e:
            logger.error("Metaculus health check failed", error=str(e))
            return False
