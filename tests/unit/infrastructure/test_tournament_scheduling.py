"""Tests for tournament scheduling configuration."""

import os
from unittest.mock import patch

import pytest

from src.infrastructure.config.tournament_config import TournamentConfig, TournamentMode


class TestTournamentScheduling:
    """Test tournament scheduling functionality."""

    def test_default_scheduling_configuration(self):
        """Test default scheduling configuration values."""
        config = TournamentConfig()

        assert config.scheduling_interval_hours == 4
        assert config.deadline_aware_scheduling is True
        assert config.critical_period_frequency_hours == 2
        assert config.final_24h_frequency_hours == 1
        assert config.tournament_scope == "seasonal"

    def test_environment_variable_configuration(self):
        """Test scheduling configuration from environment variables."""
        env_vars = {
            "SCHEDULING_FREQUENCY_HOURS": "6",
            "DEADLINE_AWARE_SCHEDULING": "false",
            "CRITICAL_PERIOD_FREQUENCY_HOURS": "3",
            "FINAL_24H_FREQUENCY_HOURS": "2",
            "TOURNAMENT_SCOPE": "daily",
        }

        with patch.dict(os.environ, env_vars):
            config = TournamentConfig.from_environment()

            assert config.scheduling_interval_hours == 6
            assert config.deadline_aware_scheduling is False
            assert config.critical_period_frequency_hours == 3
            assert config.final_24h_frequency_hours == 2
            assert config.tournament_scope == "daily"

    def test_deadline_aware_frequency_calculation(self):
        """Test deadline-aware frequency calculation."""
        config = TournamentConfig(
            scheduling_interval_hours=4,
            deadline_aware_scheduling=True,
            critical_period_frequency_hours=2,
            final_24h_frequency_hours=1,
        )

        # Normal period (more than 72 hours)
        assert config.get_deadline_aware_frequency(100) == 4

        # Critical period (72 hours or less)
        assert config.get_deadline_aware_frequency(48) == 2
        assert config.get_deadline_aware_frequency(72) == 2

        # Final 24 hours
        assert config.get_deadline_aware_frequency(12) == 1
        assert config.get_deadline_aware_frequency(24) == 1

    def test_deadline_aware_frequency_disabled(self):
        """Test deadline-aware frequency when disabled."""
        config = TournamentConfig(
            scheduling_interval_hours=4,
            deadline_aware_scheduling=False,
            critical_period_frequency_hours=2,
            final_24h_frequency_hours=1,
        )

        # Should always return base frequency when disabled
        assert config.get_deadline_aware_frequency(100) == 4
        assert config.get_deadline_aware_frequency(48) == 4
        assert config.get_deadline_aware_frequency(12) == 4

    def test_should_run_now_logic(self):
        """Test the should_run_now decision logic."""
        config = TournamentConfig(
            scheduling_interval_hours=4,
            deadline_aware_scheduling=True,
            critical_period_frequency_hours=2,
            final_24h_frequency_hours=1,
        )

        # Normal period - should run after 4 hours
        assert config.should_run_now(100, 4) is True
        assert config.should_run_now(100, 3) is False

        # Critical period - should run after 2 hours
        assert config.should_run_now(48, 2) is True
        assert config.should_run_now(48, 1) is False

        # Final 24 hours - should run after 1 hour
        assert config.should_run_now(12, 1) is True
        assert config.should_run_now(12, 0.5) is False

    def test_cron_schedule_generation(self):
        """Test cron schedule generation for different modes."""
        # Tournament mode
        config = TournamentConfig(
            mode=TournamentMode.TOURNAMENT, scheduling_interval_hours=4
        )
        assert config.get_cron_schedule() == "0 */4 * * *"

        # Development mode
        config = TournamentConfig(
            mode=TournamentMode.DEVELOPMENT, scheduling_interval_hours=4
        )
        assert config.get_cron_schedule() == "0 */6 * * *"

        # Quarterly cup mode
        config = TournamentConfig(
            mode=TournamentMode.QUARTERLY_CUP, scheduling_interval_hours=4
        )
        assert config.get_cron_schedule() == "0 0 */2 * *"

    def test_scheduling_strategy_output(self):
        """Test scheduling strategy configuration output."""
        config = TournamentConfig(
            scheduling_interval_hours=4,
            deadline_aware_scheduling=True,
            critical_period_frequency_hours=2,
            final_24h_frequency_hours=1,
            tournament_scope="seasonal",
        )

        strategy = config.get_scheduling_strategy()

        assert strategy["base_frequency_hours"] == 4
        assert strategy["deadline_aware"] is True
        assert strategy["critical_period_frequency_hours"] == 2
        assert strategy["final_24h_frequency_hours"] == 1
        assert strategy["tournament_scope"] == "seasonal"
        assert "cron_schedule" in strategy

    def test_to_dict_includes_scheduling_fields(self):
        """Test that to_dict includes all scheduling configuration fields."""
        config = TournamentConfig(
            scheduling_interval_hours=4,
            deadline_aware_scheduling=True,
            critical_period_frequency_hours=2,
            final_24h_frequency_hours=1,
            tournament_scope="seasonal",
        )

        config_dict = config.to_dict()

        assert config_dict["scheduling_interval_hours"] == 4
        assert config_dict["deadline_aware_scheduling"] is True
        assert config_dict["critical_period_frequency_hours"] == 2
        assert config_dict["final_24h_frequency_hours"] == 1
        assert config_dict["tournament_scope"] == "seasonal"

    @pytest.mark.parametrize(
        "hours_until_deadline,expected_frequency",
        [
            (200, 4),  # Normal period
            (100, 4),  # Normal period
            (72, 2),  # Critical period boundary
            (48, 2),  # Critical period
            (25, 2),  # Critical period
            (24, 1),  # Final 24h boundary
            (12, 1),  # Final 24h
            (1, 1),  # Final hour
        ],
    )
    def test_deadline_frequency_boundaries(
        self, hours_until_deadline, expected_frequency
    ):
        """Test deadline-aware frequency calculation at boundaries."""
        config = TournamentConfig(
            scheduling_interval_hours=4,
            deadline_aware_scheduling=True,
            critical_period_frequency_hours=2,
            final_24h_frequency_hours=1,
        )

        assert (
            config.get_deadline_aware_frequency(hours_until_deadline)
            == expected_frequency
        )
