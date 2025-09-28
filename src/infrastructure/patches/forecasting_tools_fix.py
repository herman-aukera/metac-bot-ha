"""
Monkey patch for forecasting-tools library to fix tournament question filtering.

The library uses incorrect API parameter names:
- Uses 'tournaments' instead of 'tournament'
- Uses 'statuses' instead of 'status'

This causes the Metaculus API to return 0 questions even when valid questions exist.
"""

from typing import Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from forecasting_tools.forecast_helpers.metaculus_api import ApiFilter

logger = logging.getLogger(__name__)


def apply_forecasting_tools_patch() -> None:
    """Apply the monkey patch to fix forecasting-tools library issues."""
    from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi

    logger.info("Applying forecasting-tools library patch for correct API parameters")

    # Store the original method for potential restoration
    original_method = MetaculusApi._grab_filtered_questions_with_offset

    def patched_grab_filtered_questions_with_offset(
        cls,
        api_filter: "ApiFilter",
        offset: int = 0,
    ):
        """
        Patched version that uses correct API parameter names.
        """
        url_params: dict[str, Any] = {
            "limit": cls.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST,
            "offset": offset,
            "order_by": "-published_at",
            "with_cp": "true",
        }

        # Fix 1: Don't use forecast_type parameter - it excludes valid questions

        # Fix 2: Use correct parameter names (singular, not plural)
        if api_filter.allowed_statuses:
            if isinstance(api_filter.allowed_statuses, list) and api_filter.allowed_statuses:
                url_params["status"] = api_filter.allowed_statuses[0]
            else:
                url_params["status"] = api_filter.allowed_statuses

        if api_filter.allowed_tournaments:
            if isinstance(api_filter.allowed_tournaments, list) and api_filter.allowed_tournaments:
                url_params["tournament"] = api_filter.allowed_tournaments[0]
            else:
                url_params["tournament"] = api_filter.allowed_tournaments

        # Keep original date filtering logic
        if api_filter.scheduled_resolve_time_gt:
            url_params["scheduled_resolve_time__gt"] = (
                api_filter.scheduled_resolve_time_gt.strftime("%Y-%m-%d")
            )
        if api_filter.scheduled_resolve_time_lt:
            url_params["scheduled_resolve_time__lt"] = (
                api_filter.scheduled_resolve_time_lt.strftime("%Y-%m-%d")
            )

        if api_filter.publish_time_gt:
            url_params["published_at__gt"] = (
                api_filter.publish_time_gt.strftime("%Y-%m-%d")
            )
        if api_filter.publish_time_lt:
            url_params["published_at__lt"] = (
                api_filter.publish_time_lt.strftime("%Y-%m-%d")
            )

        if api_filter.open_time_gt:
            url_params["open_time__gt"] = api_filter.open_time_gt.strftime("%Y-%m-%d")
        if api_filter.open_time_lt:
            url_params["open_time__lt"] = api_filter.open_time_lt.strftime("%Y-%m-%d")

        logger.info(f"Patched API call with params: {url_params}")
        questions = cls._get_questions_from_api(url_params)
        questions_were_found_before_local_filter = len(questions) > 0

        logger.info(f"Retrieved {len(questions)} questions before local filtering")

        # Apply original local filters
        if api_filter.num_forecasters_gte is not None:
            questions = cls._filter_questions_by_forecasters(
                questions, api_filter.num_forecasters_gte
            )

        if api_filter.close_time_gt or api_filter.close_time_lt:
            questions = cls._filter_questions_by_close_time(
                questions, api_filter.close_time_gt, api_filter.close_time_lt
            )

        if api_filter.includes_bots_in_aggregates is not None:
            questions = cls._filter_questions_by_includes_bots_in_aggregates(
                questions, api_filter.includes_bots_in_aggregates
            )

        if api_filter.cp_reveal_time_gt or api_filter.cp_reveal_time_lt:
            questions = cls._filter_questions_by_cp_reveal_time(
                questions,
                api_filter.cp_reveal_time_gt,
                api_filter.cp_reveal_time_lt,
            )

        logger.info(f"Returning {len(questions)} questions after local filtering")
        return questions, questions_were_found_before_local_filter    # Apply the patch - replace the classmethod
    MetaculusApi._grab_filtered_questions_with_offset = classmethod(patched_grab_filtered_questions_with_offset)

    logger.info("✅ Forecasting-tools patch applied successfully")
