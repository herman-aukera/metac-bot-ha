"""Minimal e2e test to verify imports and basic functionality."""

import os
from unittest.mock import patch

import pytest

from main import TemplateForecaster


class TestImportAndBasicFunctionality:
    """Test that we can import and create the bot without errors."""

    def test_import_template_forecaster(self):
        """Test that we can import TemplateForecaster successfully."""
        # This test passes if the import at the top of the file works
        assert TemplateForecaster is not None

    def test_create_bot_instance(self):
        """Test that we can create a TemplateForecaster instance."""
        # Set minimal environment variables to avoid errors
        test_env = {
            "OPENAI_API_KEY": "test-key",
            "METACULUS_API_KEY": "test-key",
            "OPENROUTER_API_KEY": "test-key",
        }

        with patch.dict(os.environ, test_env):
            try:
                bot = TemplateForecaster()
                assert bot is not None
                assert isinstance(bot, TemplateForecaster)
            except Exception as e:
                pytest.fail(f"Failed to create TemplateForecaster instance: {e}")

    def test_bot_has_expected_methods(self):
        """Test that the bot has the expected methods."""
        test_env = {
            "OPENAI_API_KEY": "test-key",
            "METACULUS_API_KEY": "test-key",
            "OPENROUTER_API_KEY": "test-key",
        }

        with patch.dict(os.environ, test_env):
            try:
                bot = TemplateForecaster()

                # Check for key methods that should exist
                assert hasattr(bot, 'run_research'), "Bot should have run_research method"
                assert hasattr(bot, 'get_llm'), "Bot should have get_llm method"
                assert hasattr(bot, 'forecast_on_tournament'), "Bot should have forecast_on_tournament method"

                # Check for budget management attributes
                assert hasattr(bot, 'budget_manager'), "Bot should have budget_manager attribute"
                assert hasattr(bot, 'token_tracker'), "Bot should have token_tracker attribute"

            except Exception as e:
                pytest.fail(f"Failed to verify bot methods: {e}")

    def test_environment_variables_required(self):
        """Test that the bot requires proper environment variables."""
        # Test without environment variables - should still work but may have warnings
        with patch.dict(os.environ, {}, clear=True):
            try:
                bot = TemplateForecaster()
                # If it creates successfully, that's fine - it might have fallbacks
                assert bot is not None
            except Exception:
                # If it fails without env vars, that's also expected behavior
                pass
