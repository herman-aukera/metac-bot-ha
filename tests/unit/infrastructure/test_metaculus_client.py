"""Comprehensive unit tests for MetaculusClient."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx
import pytest

from src.domain.entities.prediction import (
    Prediction,
    PredictionConfidence,
    PredictionMethod,
    PredictionResult,
)
from src.domain.entities.question import Question, QuestionType
from src.infrastructure.config.settings import MetaculusConfig, Settings
from src.infrastructure.external_apis.metaculus_client import MetaculusClient


class TestMetaculusClientComprehensive:
    """Comprehensive test cases for MetaculusClient."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        settings = Settings()
        settings.metaculus = MetaculusConfig(
            api_token="test-api-token",
            base_url="https://www.metaculus.com/api2",
            timeout=30.0,
            submit_predictions=True,
            dry_run=False,
        )
        settings.metaculus_username = "test_user"
        settings.metaculus_password = "test_pass"
        return settings

    @pytest.fixture
    def metaculus_client(self, settings):
        """Create MetaculusClient instance."""
        return MetaculusClient(settings)

    @pytest.fixture
    def sample_question_data(self):
        """Sample Metaculus question data."""
        return {
            "id": 12345,
            "title": "Will AGI be achieved by 2030?",
            "description": "This question asks about the development of AGI...",
            "type": "binary",
            "status": "open",
            "created_time": "2024-01-01T00:00:00Z",
            "close_time": "2030-01-01T00:00:00Z",
            "resolve_time": "2030-12-31T23:59:59Z",
            "resolution_criteria": "AGI is defined as...",
            "author_name": "test_author",
            "category": "Technology",
            "tags": ["ai", "technology"],
            "number_of_predictions": 150,
            "comment_count": 25,
            "community_prediction": 0.42,
            "points": 100,
            "resolution": None,
        }

    @pytest.fixture
    def sample_prediction(self):
        """Sample prediction for testing."""
        return Prediction.create_binary_prediction(
            question_id=uuid4(),
            research_report_id=uuid4(),
            probability=0.65,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.ENSEMBLE,
            reasoning="Strong evidence suggests this outcome",
            created_by="test_agent",
        )

    def test_init_sets_correct_attributes(self, settings):
        """Test client initialization."""
        client = MetaculusClient(settings)

        assert client.settings == settings
        assert client.base_url == "https://www.metaculus.com/api2"
        assert client.session_token is None
        assert client.user_id is None
        assert client.config == settings.metaculus

    @pytest.mark.asyncio
    async def test_authenticate_success(self, metaculus_client):
        """Test successful authentication."""
        mock_response_data = {"session_token": "test_session_token", "user_id": 12345}

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value = mock_response

            result = await metaculus_client.authenticate()

            assert result is True
            assert metaculus_client.session_token == "test_session_token"
            assert metaculus_client.user_id == 12345

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            # Check that URL is in the first positional argument (URL string)
            assert "accounts/login/" in call_args[0][0]
            assert call_args[1]["json"]["username"] == "test_user"
            assert call_args[1]["json"]["password"] == "test_pass"

    @pytest.mark.asyncio
    async def test_authenticate_failure(self, metaculus_client):
        """Test authentication failure."""
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_post.return_value = mock_response

            result = await metaculus_client.authenticate()

            assert result is False
            assert metaculus_client.session_token is None
            assert metaculus_client.user_id is None

    @pytest.mark.asyncio
    async def test_authenticate_with_provided_credentials(self, metaculus_client):
        """Test authentication with provided credentials."""
        mock_response_data = {"session_token": "custom_token", "user_id": 54321}

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value = mock_response

            result = await metaculus_client.authenticate("custom_user", "custom_pass")

            assert result is True
            call_args = mock_post.call_args
            assert call_args[1]["json"]["username"] == "custom_user"
            assert call_args[1]["json"]["password"] == "custom_pass"

    @pytest.mark.asyncio
    async def test_authenticate_no_credentials(self, settings):
        """Test authentication without credentials."""
        settings.metaculus_username = None
        settings.metaculus_password = None
        client = MetaculusClient(settings)

        result = await client.authenticate()

        assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_exception_handling(self, metaculus_client):
        """Test authentication exception handling."""
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.side_effect = httpx.HTTPError("Connection failed")

            result = await metaculus_client.authenticate()

            assert result is False

    @pytest.mark.asyncio
    async def test_fetch_questions_success(
        self, metaculus_client, sample_question_data
    ):
        """Test successful question fetching."""
        mock_response_data = {
            "results": [
                sample_question_data,
                {**sample_question_data, "id": 12346, "title": "Another question"},
            ]
        }

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            questions = await metaculus_client.fetch_questions(limit=10, status="open")

            assert len(questions) == 2
            assert questions[0].metadata["metaculus_id"] == 12345
            assert questions[1].metadata["metaculus_id"] == 12346

            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "questions/" in call_args[0][0]
            assert call_args[1]["params"]["limit"] == 10
            assert call_args[1]["params"]["status"] == "open"

    @pytest.mark.asyncio
    async def test_fetch_questions_with_categories(
        self, metaculus_client, sample_question_data
    ):
        """Test fetching questions with category filter."""
        mock_response_data = {"results": [sample_question_data]}

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            questions = await metaculus_client.fetch_questions(
                categories=["Technology", "Science"]
            )

            assert len(questions) == 1
            call_args = mock_get.call_args
            assert call_args[1]["params"]["categories"] == "Technology,Science"

    @pytest.mark.asyncio
    async def test_fetch_questions_parse_error_handling(self, metaculus_client):
        """Test handling of question parsing errors."""
        invalid_question_data = {"id": 12345}  # Missing required fields
        mock_response_data = {"results": [invalid_question_data]}

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            questions = await metaculus_client.fetch_questions()

            # The actual implementation creates a question with default values
            # but validates it, so it does get created successfully
            assert len(questions) == 1  # Question gets created with defaults
            assert questions[0].metaculus_id == 12345
            assert questions[0].title == ""  # Default empty title

    @pytest.mark.asyncio
    async def test_fetch_questions_http_error(self, metaculus_client):
        """Test fetch questions HTTP error handling."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = httpx.HTTPError("Network error")

            questions = await metaculus_client.fetch_questions()

            assert questions == []

    @pytest.mark.asyncio
    async def test_fetch_question_success(self, metaculus_client, sample_question_data):
        """Test successful single question fetching."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_question_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            question = await metaculus_client.fetch_question(12345)

            assert question is not None
            assert question.metadata["metaculus_id"] == 12345
            assert question.title == "Will AGI be achieved by 2030?"

            mock_get.assert_called_once()
            assert "questions/12345/" in mock_get.call_args[0][0]

    @pytest.mark.asyncio
    async def test_fetch_question_not_found(self, metaculus_client):
        """Test fetch question when not found."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=mock_response
            )
            mock_get.return_value = mock_response

            question = await metaculus_client.fetch_question(99999)

            assert question is None

    @pytest.mark.asyncio
    async def test_submit_prediction_new_interface(self, metaculus_client):
        """Test prediction submission with new dict interface."""
        metaculus_client.session_token = "test_token"

        prediction_data = {
            "question_id": 12345,
            "prediction": 0.75,
            "reasoning": "Strong evidence for this outcome",
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            result = await metaculus_client.submit_prediction(prediction_data)

            assert result["status"] == "success"
            assert result["submitted"] is True
            assert result["question_id"] == 12345
            assert result["prediction"] == 0.75

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "questions/12345/predict/" in call_args[0][0]
            assert call_args[1]["json"]["prediction"] == 0.75
            assert call_args[1]["json"]["comment"] == "Strong evidence for this outcome"

    @pytest.mark.asyncio
    async def test_submit_prediction_old_interface(
        self, metaculus_client, sample_prediction
    ):
        """Test prediction submission with old interface."""
        metaculus_client.session_token = "test_token"

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_post.return_value = mock_response

            result = await metaculus_client.submit_prediction(
                12345, sample_prediction, "Custom comment"
            )

            assert result["status"] == "success"
            assert result["submitted"] is True

            call_args = mock_post.call_args
            assert call_args[1]["json"]["prediction"] == 0.65
            assert call_args[1]["json"]["comment"] == "Custom comment"

    @pytest.mark.asyncio
    async def test_submit_prediction_dry_run_mode(self, settings, sample_prediction):
        """Test prediction submission in dry run mode."""
        settings.metaculus.dry_run = True
        client = MetaculusClient(settings)

        result = await client.submit_prediction(12345, sample_prediction)

        assert result["status"] == "dry_run"
        assert result["would_submit"] is True
        assert result["question_id"] == 12345
        assert result["prediction"] == 0.65

    @pytest.mark.asyncio
    async def test_submit_prediction_disabled(self, settings, sample_prediction):
        """Test prediction submission when disabled."""
        settings.metaculus.submit_predictions = False
        client = MetaculusClient(settings)

        result = await client.submit_prediction(12345, sample_prediction)

        assert result["status"] == "disabled"
        assert result["submitted"] is False

    @pytest.mark.asyncio
    async def test_submit_prediction_not_authenticated(
        self, metaculus_client, sample_prediction
    ):
        """Test prediction submission without authentication."""
        # No session token set

        result = await metaculus_client.submit_prediction(12345, sample_prediction)

        assert result["status"] == "error"
        assert result["error"] == "Not authenticated"
        assert result["submitted"] is False

    @pytest.mark.asyncio
    async def test_submit_prediction_http_error(
        self, metaculus_client, sample_prediction
    ):
        """Test prediction submission HTTP error."""
        metaculus_client.session_token = "test_token"

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            mock_post.return_value = mock_response

            result = await metaculus_client.submit_prediction(12345, sample_prediction)

            assert result["status"] == "error"
            assert result["submitted"] is False
            assert "HTTP 400" in result["error"]

    @pytest.mark.asyncio
    async def test_submit_prediction_exception(
        self, metaculus_client, sample_prediction
    ):
        """Test prediction submission exception handling."""
        metaculus_client.session_token = "test_token"

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.side_effect = httpx.HTTPError("Connection failed")

            result = await metaculus_client.submit_prediction(12345, sample_prediction)

            assert result["status"] == "error"
            assert result["submitted"] is False
            assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_submit_prediction_old_interface_missing_prediction(
        self, metaculus_client
    ):
        """Test old interface with missing prediction parameter."""
        with pytest.raises(
            ValueError, match="prediction is required when using old interface"
        ):
            await metaculus_client.submit_prediction(12345, None)

    @pytest.mark.asyncio
    async def test_submit_prediction_invalid_prediction_type(self, metaculus_client):
        """Test prediction submission with invalid prediction type."""
        metaculus_client.session_token = "test_token"

        # Create prediction with no binary or numeric value
        invalid_prediction = Prediction(
            id=uuid4(),
            question_id=uuid4(),
            research_report_id=uuid4(),
            result=PredictionResult(),  # No values set
            confidence=PredictionConfidence.MEDIUM,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Test",
            reasoning_steps=[],
            created_at=datetime.now(timezone.utc),
            created_by="test",
        )

        with pytest.raises(
            ValueError,
            match="Prediction must have either binary_probability or numeric_value",
        ):
            await metaculus_client.submit_prediction(12345, invalid_prediction)

    @pytest.mark.asyncio
    async def test_fetch_user_predictions_success(self, metaculus_client):
        """Test successful user predictions fetching."""
        metaculus_client.user_id = 12345
        metaculus_client.session_token = "test_token"

        mock_predictions = [
            {"id": 1, "question": 12345, "prediction": 0.6},
            {"id": 2, "question": 12346, "prediction": 0.4},
        ]
        mock_response_data = {"results": mock_predictions}

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            predictions = await metaculus_client.fetch_user_predictions()

            assert len(predictions) == 2
            assert predictions[0]["id"] == 1

            mock_get.assert_called_once()
            assert "users/12345/predictions/" in mock_get.call_args[0][0]

    @pytest.mark.asyncio
    async def test_fetch_user_predictions_no_user_id(self, metaculus_client):
        """Test user predictions fetching without user ID."""
        predictions = await metaculus_client.fetch_user_predictions()

        assert predictions == []

    @pytest.mark.asyncio
    async def test_fetch_user_predictions_custom_user_id(self, metaculus_client):
        """Test user predictions fetching with custom user ID."""
        mock_response_data = {"results": []}

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            predictions = await metaculus_client.fetch_user_predictions(54321)

            assert predictions == []
            assert "users/54321/predictions/" in mock_get.call_args[0][0]

    @pytest.mark.asyncio
    async def test_fetch_question_comments_success(self, metaculus_client):
        """Test successful question comments fetching."""
        mock_comments = [
            {"id": 1, "text": "Great question!", "author": "user1"},
            {"id": 2, "text": "I disagree", "author": "user2"},
        ]
        mock_response_data = {"results": mock_comments}

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            comments = await metaculus_client.fetch_question_comments(12345)

            assert len(comments) == 2
            assert comments[0]["text"] == "Great question!"

            mock_get.assert_called_once()
            assert "questions/12345/comments/" in mock_get.call_args[0][0]

    @pytest.mark.asyncio
    async def test_fetch_question_comments_error(self, metaculus_client):
        """Test question comments fetching error handling."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = httpx.HTTPError("Network error")

            comments = await metaculus_client.fetch_question_comments(12345)

            assert comments == []

    @pytest.mark.asyncio
    async def test_health_check_success(self, metaculus_client):
        """Test successful health check."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = await metaculus_client.health_check()

            assert result is True
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, metaculus_client):
        """Test health check failure."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection failed")

            result = await metaculus_client.health_check()

            assert result is False

    def test_parse_question_binary(self, metaculus_client, sample_question_data):
        """Test parsing binary question data."""
        question = metaculus_client._parse_question(sample_question_data)

        assert isinstance(question, Question)
        assert question.title == "Will AGI be achieved by 2030?"
        assert question.question_type == QuestionType.BINARY
        assert question.metadata["metaculus_id"] == 12345
        assert question.metadata["author"] == "test_author"
        assert question.metadata["tags"] == ["ai", "technology"]
        assert question.metadata["prediction_count"] == 150

    def test_parse_question_continuous(self, metaculus_client, sample_question_data):
        """Test parsing continuous question data."""
        sample_question_data["type"] = "continuous"

        question = metaculus_client._parse_question(sample_question_data)

        assert question.question_type == QuestionType.CONTINUOUS

    def test_parse_question_multiple_choice(
        self, metaculus_client, sample_question_data
    ):
        """Test parsing multiple choice question data."""
        sample_question_data["type"] = "multiple_choice"
        sample_question_data["choices"] = ["Choice A", "Choice B", "Choice C"]

        question = metaculus_client._parse_question(sample_question_data)

        assert question.question_type == QuestionType.MULTIPLE_CHOICE
        assert question.choices == ["Choice A", "Choice B", "Choice C"]

        assert question.question_type == QuestionType.MULTIPLE_CHOICE

    def test_parse_datetime_valid(self, metaculus_client):
        """Test parsing valid datetime strings."""
        # Test Z suffix
        dt1 = metaculus_client._parse_datetime("2024-01-01T12:00:00Z")
        assert dt1 is not None
        assert dt1.year == 2024

        # Test timezone offset
        dt2 = metaculus_client._parse_datetime("2024-01-01T12:00:00+00:00")
        assert dt2 is not None

    def test_parse_datetime_invalid(self, metaculus_client):
        """Test parsing invalid datetime strings."""
        assert metaculus_client._parse_datetime(None) is None
        assert metaculus_client._parse_datetime("invalid") is None
        assert metaculus_client._parse_datetime("") is None

    @pytest.mark.asyncio
    async def test_fetch_benchmark_questions(
        self, metaculus_client, sample_question_data
    ):
        """Test fetching benchmark questions."""
        # Mock resolved question
        resolved_question_data = {
            **sample_question_data,
            "status": "resolved",
            "resolution": "yes",
        }
        mock_response_data = {"results": [resolved_question_data]}

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            questions = await metaculus_client.fetch_benchmark_questions(limit=10)

            assert len(questions) == 1
            assert questions[0].metadata["status"] == "resolved"
            assert questions[0].metadata["resolution"] == "yes"

    @pytest.mark.asyncio
    async def test_batch_fetch_questions_success(
        self, metaculus_client, sample_question_data
    ):
        """Test successful batch question fetching."""
        question_ids = [12345, 12346, 12347]

        async def mock_fetch_question(qid):
            return Question.create(
                title=f"Question {qid}",
                description="Test question",
                question_type=QuestionType.BINARY,
                metadata={"metaculus_id": qid},
            )

        with patch.object(
            metaculus_client, "fetch_question", side_effect=mock_fetch_question
        ):
            questions = await metaculus_client.batch_fetch_questions(question_ids)

            assert len(questions) == 3
            assert all(q is not None for q in questions)
            assert questions[0].metadata["metaculus_id"] == 12345

    @pytest.mark.asyncio
    async def test_batch_fetch_questions_with_exceptions(self, metaculus_client):
        """Test batch fetch with some failing requests."""
        question_ids = [12345, 12346, 12347]

        async def mock_fetch_question(qid):
            if qid == 12346:
                raise httpx.HTTPError("Failed to fetch")
            return Question.create(
                title=f"Question {qid}",
                description="Test question",
                question_type=QuestionType.BINARY,
                metadata={"metaculus_id": qid},
            )

        with patch.object(
            metaculus_client, "fetch_question", side_effect=mock_fetch_question
        ):
            questions = await metaculus_client.batch_fetch_questions(question_ids)

            assert len(questions) == 3
            assert questions[0] is not None  # 12345 success
            assert questions[1] is None  # 12346 failed
            assert questions[2] is not None  # 12347 success

    @pytest.mark.asyncio
    async def test_get_question_alias(self, metaculus_client):
        """Test get_question method is alias for fetch_question."""
        with patch.object(
            metaculus_client, "fetch_question", return_value=Mock()
        ) as mock_fetch:
            await metaculus_client.get_question(12345)

            mock_fetch.assert_called_once_with(12345)

    @pytest.mark.asyncio
    async def test_get_questions_alias(self, metaculus_client):
        """Test get_questions method is alias for fetch_questions."""
        with patch.object(
            metaculus_client, "fetch_questions", return_value=[]
        ) as mock_fetch:
            await metaculus_client.get_questions(limit=5, status="open")

            mock_fetch.assert_called_once_with(limit=5, status="open")

    def test_get_headers_with_session_token(self, metaculus_client):
        """Test header generation with session token."""
        metaculus_client.session_token = "session_token_123"

        headers = metaculus_client._get_headers()

        assert headers["Authorization"] == "Token session_token_123"
        assert headers["Content-Type"] == "application/json"
        assert "User-Agent" in headers

    def test_get_headers_with_api_token(self, metaculus_client):
        """Test header generation with API token."""
        # No session token, should use API token from settings
        headers = metaculus_client._get_headers()

        assert headers["Authorization"] == "Token test-api-token"

    def test_get_headers_no_auth(self, settings):
        """Test header generation without authentication."""
        settings.metaculus.api_token = None
        client = MetaculusClient(settings)

        headers = client._get_headers()

        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_handle_rate_limit(self, metaculus_client):
        """Test rate limiting functionality."""
        with patch("asyncio.sleep") as mock_sleep:
            await metaculus_client._handle_rate_limit()

            mock_sleep.assert_called_once_with(1.0)

    def test_config_property(self, metaculus_client, settings):
        """Test config property access."""
        assert metaculus_client.config == settings.metaculus
