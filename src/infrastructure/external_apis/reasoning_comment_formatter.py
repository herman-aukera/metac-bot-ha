"""Reasoning comment formatter for tournament compliance and transparency."""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

from ...domain.entities.forecast import Forecast
from ...domain.entities.prediction import Prediction
from ...domain.services.tournament_compliance_validator import (
    TournamentComplianceValidator,
)


class ReasoningCommentFormatter:
    """Formats reasoning comments for tournament submission with compliance validation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compliance_validator = TournamentComplianceValidator()

        # Tournament-specific formatting rules
        self.max_comment_length = 2000  # Metaculus comment limit
        self.min_comment_length = 100  # Tournament transparency requirement

    def format_prediction_comment(
        self, prediction: Prediction, question_title: str = ""
    ) -> str:
        """Format a prediction's reasoning into a tournament-compliant comment."""
        try:
            # Validate compliance first
            compliance_report = self.compliance_validator.validate_reasoning_comment(
                prediction
            )

            if not compliance_report.is_compliant:
                self.logger.warning(
                    f"Prediction reasoning has compliance issues (score: {compliance_report.score:.2f})",
                    extra={
                        "prediction_id": str(prediction.id),
                        "issues": [issue.message for issue in compliance_report.issues],
                    },
                )

            # Format the reasoning for tournament submission
            formatted_comment = (
                self.compliance_validator.format_reasoning_for_tournament(prediction)
            )

            # Add tournament-specific enhancements
            enhanced_comment = self._enhance_comment_for_tournament(
                formatted_comment, prediction, question_title
            )

            # Ensure length constraints
            final_comment = self._enforce_length_constraints(enhanced_comment)

            # Final validation
            if len(final_comment.strip()) < self.min_comment_length:
                self.logger.warning(
                    f"Final comment too short ({len(final_comment)} chars), padding with transparency info"
                )
                final_comment = self._pad_comment_with_transparency(
                    final_comment, prediction
                )

            return final_comment

        except Exception as e:
            self.logger.error(f"Error formatting reasoning comment: {e}")
            return self._generate_fallback_comment(prediction)

    def format_forecast_comment(
        self, forecast: Forecast, question_title: str = ""
    ) -> str:
        """Format a forecast's ensemble reasoning into a tournament-compliant comment."""
        try:
            # Use the primary prediction's reasoning or ensemble summary
            if forecast.predictions:
                primary_prediction = forecast.predictions[
                    0
                ]  # Use first prediction as primary
                base_comment = self.format_prediction_comment(
                    primary_prediction, question_title
                )

                # Add ensemble information if multiple predictions
                if len(forecast.predictions) > 1:
                    ensemble_info = self._format_ensemble_information(forecast)
                    base_comment = f"{base_comment}\n\n{ensemble_info}"

                return self._enforce_length_constraints(base_comment)
            else:
                return self._generate_fallback_comment_for_forecast(forecast)

        except Exception as e:
            self.logger.error(f"Error formatting forecast comment: {e}")
            return self._generate_fallback_comment_for_forecast(forecast)

    def validate_comment_before_submission(
        self, comment: str, prediction: Prediction
    ) -> Dict[str, any]:
        """Validate comment meets all tournament requirements before submission."""
        validation_result = {
            "is_valid": True,
            "issues": [],
            "formatted_comment": comment,
        }

        # Length validation
        if len(comment.strip()) < self.min_comment_length:
            validation_result["is_valid"] = False
            validation_result["issues"].append(
                f"Comment too short: {len(comment)} chars (min: {self.min_comment_length})"
            )

        if len(comment) > self.max_comment_length:
            validation_result["is_valid"] = False
            validation_result["issues"].append(
                f"Comment too long: {len(comment)} chars (max: {self.max_comment_length})"
            )
            # Auto-truncate if possible
            validation_result["formatted_comment"] = (
                self._truncate_comment_intelligently(comment)
            )

        # Content validation
        content_issues = self._validate_comment_content(comment)
        if content_issues:
            validation_result["issues"].extend(content_issues)
            if any("prohibited" in issue.lower() for issue in content_issues):
                validation_result["is_valid"] = False

        # Tournament compliance validation
        temp_prediction = prediction
        temp_prediction.reasoning = comment
        compliance_report = self.compliance_validator.validate_reasoning_comment(
            temp_prediction
        )

        if not compliance_report.is_compliant:
            validation_result["issues"].extend(
                [issue.message for issue in compliance_report.issues]
            )
            if any(issue.severity == "error" for issue in compliance_report.issues):
                validation_result["is_valid"] = False

        return validation_result

    def _enhance_comment_for_tournament(
        self, comment: str, prediction: Prediction, question_title: str
    ) -> str:
        """Add tournament-specific enhancements to the comment."""
        enhanced = comment

        # Add confidence and method transparency if not already present
        if "confidence:" not in enhanced.lower() and "method:" not in enhanced.lower():
            transparency_section = f"\n\nForecast Details:\n"
            transparency_section += (
                f"• Method: {prediction.method.value.replace('_', ' ').title()}\n"
            )
            transparency_section += (
                f"• Confidence: {prediction.confidence.value.replace('_', ' ').title()}"
            )

            if prediction.method_metadata:
                key_metadata = {
                    k: v
                    for k, v in prediction.method_metadata.items()
                    if k in ["base_probability", "variation", "component_predictions"]
                }
                if key_metadata:
                    metadata_str = ", ".join(
                        f"{k}: {v}" for k, v in key_metadata.items()
                    )
                    transparency_section += f"\n• Details: {metadata_str}"

            enhanced += transparency_section

        # Add timestamp for transparency
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        if "generated:" not in enhanced.lower():
            enhanced += f"\n\nGenerated: {timestamp}"

        return enhanced

    def _format_ensemble_information(self, forecast: Forecast) -> str:
        """Format ensemble information for transparency."""
        ensemble_info = f"Ensemble Analysis:\n"
        ensemble_info += f"• Combined {len(forecast.predictions)} predictions\n"

        if forecast.ensemble_method:
            ensemble_info += f"• Method: {forecast.ensemble_method}\n"

        if forecast.weight_distribution:
            weights_str = ", ".join(
                f"{k}: {v:.2f}" for k, v in forecast.weight_distribution.items()
            )
            ensemble_info += f"• Weights: {weights_str}\n"

        if forecast.reasoning_summary:
            ensemble_info += f"• Summary: {forecast.reasoning_summary}"

        return ensemble_info

    def _enforce_length_constraints(self, comment: str) -> str:
        """Ensure comment meets length constraints."""
        if len(comment) <= self.max_comment_length:
            return comment

        # Intelligent truncation
        return self._truncate_comment_intelligently(comment)

    def _truncate_comment_intelligently(self, comment: str) -> str:
        """Truncate comment while preserving important information."""
        if len(comment) <= self.max_comment_length:
            return comment

        # Try to preserve structure by truncating sections
        sections = comment.split("\n\n")

        # Keep the most important sections (analysis, conclusion, transparency)
        important_keywords = [
            "analysis",
            "conclusion",
            "forecast details",
            "method:",
            "confidence:",
        ]
        important_sections = []
        other_sections = []

        for section in sections:
            if any(keyword in section.lower() for keyword in important_keywords):
                important_sections.append(section)
            else:
                other_sections.append(section)

        # Start with important sections
        truncated = "\n\n".join(important_sections)

        # Add other sections if space allows
        for section in other_sections:
            test_length = len(truncated) + len(section) + 2  # +2 for \n\n
            if test_length <= self.max_comment_length - 50:  # Leave some buffer
                truncated += "\n\n" + section
            else:
                break

        # If still too long, truncate the last section
        if len(truncated) > self.max_comment_length:
            truncated = truncated[: self.max_comment_length - 20] + "... [truncated]"

        return truncated

    def _pad_comment_with_transparency(
        self, comment: str, prediction: Prediction
    ) -> str:
        """Pad short comment with transparency information."""
        padded = comment

        # Add method and confidence information
        if len(padded) < self.min_comment_length:
            transparency_padding = f"\n\nMethodology: This forecast was generated using {prediction.method.value.replace('_', ' ')} approach with {prediction.confidence.value} confidence level."

            if prediction.reasoning_steps:
                transparency_padding += f" The analysis involved {len(prediction.reasoning_steps)} reasoning steps."

            if prediction.method_metadata:
                transparency_padding += f" Additional parameters: {', '.join(f'{k}={v}' for k, v in prediction.method_metadata.items())}."

            padded += transparency_padding

        return padded

    def _validate_comment_content(self, comment: str) -> List[str]:
        """Validate comment content for tournament compliance."""
        issues = []

        # Check for prohibited AI self-references
        ai_patterns = [
            r"I am an AI",
            r"As an AI",
            r"I cannot",
            r"I don't have access",
            r"I'm not able to",
            r"as a language model",
            r"I'm an AI",
        ]

        for pattern in ai_patterns:
            if re.search(pattern, comment, re.IGNORECASE):
                issues.append(f"Contains prohibited AI self-reference: {pattern}")

        # Check for required elements
        required_elements = ["analysis", "evidence", "conclusion"]
        missing_elements = [
            elem for elem in required_elements if elem not in comment.lower()
        ]

        if len(missing_elements) > 1:  # Allow some flexibility
            issues.append(
                f"Missing key reasoning elements: {', '.join(missing_elements)}"
            )

        return issues

    def _generate_fallback_comment(self, prediction: Prediction) -> str:
        """Generate a minimal compliant comment when formatting fails."""
        fallback = f"Forecast Analysis:\n\n"
        fallback += f"This prediction was generated using {prediction.method.value.replace('_', ' ')} methodology "
        fallback += f"with {prediction.confidence.value} confidence level.\n\n"

        if prediction.result.binary_probability is not None:
            fallback += f"The assessed probability reflects analysis of available information and uncertainty factors. "
        elif prediction.result.numeric_value is not None:
            fallback += f"The predicted value is based on quantitative analysis and trend assessment. "

        fallback += f"Confidence level indicates the degree of certainty in this assessment.\n\n"
        fallback += f"Method: {prediction.method.value} | Confidence: {prediction.confidence.value}\n"
        fallback += f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"

        return fallback

    def _generate_fallback_comment_for_forecast(self, forecast: Forecast) -> str:
        """Generate fallback comment for forecast when no predictions available."""
        fallback = f"Forecast Summary:\n\n"
        fallback += f"This forecast represents an ensemble analysis "

        if forecast.ensemble_method:
            fallback += f"using {forecast.ensemble_method} methodology. "
        else:
            fallback += f"combining multiple prediction approaches. "

        if forecast.reasoning_summary:
            fallback += f"\n\nReasoning: {forecast.reasoning_summary}\n\n"

        fallback += f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"

        return fallback
