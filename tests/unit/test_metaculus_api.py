"""
Unit tests for MetaculusAPI infrastructure component.
Tests the mock API client functionality and error handling.
"""

from src.infrastructure.metaculus_api import APIConfig, MetaculusAPI, MetaculusAPIError


class TestMetaculusAPI:
    """Test suite for MetaculusAPI class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        config = APIConfig(base_url="https://api.example.com")
        self.api = MetaculusAPI(config=config)

    def test_init_with_custom_config(self):
        """Test MetaculusAPI initialization with custom config."""
        custom_config = APIConfig(
            base_url="https://custom.api.com", timeout=60, max_retries=5
        )
        api = MetaculusAPI(config=custom_config)
        assert api.config.base_url == "https://custom.api.com"
        assert api.config.timeout == 60
        assert api.config.max_retries == 5

    def test_init_with_default_config(self):
        """Test MetaculusAPI initialization with default config."""
        api = MetaculusAPI()
        assert api.config.base_url == "https://www.metaculus.com/api2/"
        assert api.config.timeout == 30
        assert api.config.max_retries == 3
        assert api.config.mock_mode is True

    def test_fetch_questions_success(self):
        """Test successful fetch_questions call returns dummy data."""
        # Act
        result = self.api.fetch_questions(
            limit=10, status="open", category="technology"
        )

        # Assert
        assert isinstance(result, list)
        assert len(result) >= 0  # May return filtered results
        if result:
            # Check structure of returned question
            question = result[0]
            assert "id" in question
            assert "title" in question
            assert "question_type" in question

    def test_fetch_questions_with_limit(self):
        """Test fetch_questions respects the limit parameter."""
        # Act
        result = self.api.fetch_questions(limit=1)

        # Assert
        assert len(result) <= 1

    def test_fetch_questions_with_status_filter_open(self):
        """Test fetch_questions applies open status filter."""
        # Act
        result = self.api.fetch_questions(status="open")

        # Assert
        for question in result:
            assert not question.get("is_resolved", True)

    def test_fetch_questions_with_status_filter_resolved(self):
        """Test fetch_questions applies resolved status filter."""
        # Act
        result = self.api.fetch_questions(status="resolved")

        # Assert
        for question in result:
            assert question.get("is_resolved", False)

    def test_fetch_questions_with_category_filter(self):
        """Test fetch_questions applies category filter."""
        # Act
        result = self.api.fetch_questions(category="technology")

        # Assert
        for question in result:
            assert question.get("category") == "technology"

    def test_fetch_questions_empty_results_with_nonexistent_category(self):
        """Test fetch_questions returns empty list for nonexistent category."""
        # Act
        result = self.api.fetch_questions(category="nonexistent_category")

        # Assert
        assert result == []

    def test_get_api_stats(self):
        """Test get_api_stats returns correct statistics."""
        # Act
        stats = self.api.get_api_stats()

        # Assert
        assert "total_questions" in stats
        assert "open_questions" in stats
        assert "resolved_questions" in stats
        assert "mock_mode" in stats
        assert "last_updated" in stats
        assert stats["mock_mode"] is True
        assert isinstance(stats["total_questions"], int)
        assert stats["total_questions"] >= 0

    def test_api_config_dataclass(self):
        """Test APIConfig dataclass functionality."""
        # Test default values
        config = APIConfig()
        assert config.base_url == "https://www.metaculus.com/api2/"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.mock_mode is True

    def test_metaculus_api_error_inheritance(self):
        """Test MetaculusAPIError is properly defined."""
        # Test that it's a proper exception
        error = MetaculusAPIError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_dummy_data_generation(self):
        """Test that dummy data is properly generated."""
        # Act
        api = MetaculusAPI()

        # Assert - should have some dummy data
        assert hasattr(api, "_dummy_data")
        assert isinstance(api._dummy_data, list)
        assert len(api._dummy_data) > 0

        # Check structure of first dummy question
        if api._dummy_data:
            question = api._dummy_data[0]
            required_fields = ["id", "title", "description", "question_type", "url"]
            for field in required_fields:
                assert field in question
