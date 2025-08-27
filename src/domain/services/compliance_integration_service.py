"""Integration service for tournament compliance monitoring."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..entities.prediction import Prediction
from ..entities.forecast import Forecast
from .tournament_rule_compliance_monitor import TournamentRuleComplianceMonitor, ComplianceReport


class ComplianceIntegrationService:
    """Integrates compliance monitoring with forecasting pipeline."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compliance_monitor = TournamentRuleComplianceMonitor()

        # Integration settings
        self.enforce_compliance = True
        self.block_non_compliant_submissions = True
        self.auto_remediate_violations = True

    def validate_and_enhance_prediction(self, prediction: Prediction) -> tuple[Prediction, ComplianceReport]:
        """Validate prediction compliance and enhance with automation markers."""

        # Add automation markers to ensure compliance
        enhanced_prediction = self._add_automation_markers(prediction)

        # Validate compliance
        compliance_report = self.compliance_monitor.validate_prediction_compliance(enhanced_prediction)

        # Log the validation
        self.compliance_monitor.log_automated_decision(
            component="compliance_validator",
            decision_type="prediction_validation",
            metadata={
                "prediction_id": str(prediction.id),
                "compliance_score": compliance_report.compliance_score,
                "is_compliant": compliance_report.is_compliant
            }
        )

        # Handle non-compliance
        if not compliance_report.is_compliant and self.enforce_compliance:
            if self.auto_remediate_violations:
                remediated_prediction = self._remediate_prediction_violations(
                    enhanced_prediction, compliance_report
                )
                # Re-validate after remediation
                final_report = self.compliance_monitor.validate_prediction_compliance(remediated_prediction)
                return remediated_prediction, final_report
            else:
                self.logger.error(
                    f"Non-compliant prediction blocked: {compliance_report.violations}",
                    extra={"compliance_violation": True}
                )

        return enhanced_prediction, compliance_report

    def validate_and_enhance_forecast(self, forecast: Forecast) -> tuple[Forecast, ComplianceReport]:
        """Validate forecast compliance and enhance with automation markers."""

        # Enhance all predictions in the forecast
        enhanced_predictions = []
        for prediction in forecast.predictions:
            enhanced_pred, _ = self.validate_and_enhance_prediction(prediction)
            enhanced_predictions.append(enhanced_pred)

        # Create enhanced forecast
        enhanced_forecast = Forecast(
            id=forecast.id,
            question_id=forecast.question_id,
            predictions=enhanced_predictions,
            research_reports=forecast.research_reports,
            created_at=forecast.created_at,
            updated_at=datetime.utcnow(),
            ensemble_method=self._ensure_automated_ensemble_method(forecast.ensemble_method),
            weight_distribution=forecast.weight_distribution,
            reasoning_summary=self._sanitize_reasoning_for_compliance(forecast.reasoning_summary),
            tournament_strategy=forecast.tournament_strategy,
            reasoning_traces=forecast.reasoning_traces
        )

        # Validate enhanced forecast
        compliance_report = self.compliance_monitor.validate_forecast_compliance(enhanced_forecast)

        # Log forecast validation
        self.compliance_monitor.log_automated_decision(
            component="compliance_validator",
            decision_type="forecast_validation",
            metadata={
                "forecast_id": str(forecast.id),
                "predictions_count": len(enhanced_predictions),
                "compliance_score": compliance_report.compliance_score,
                "is_compliant": compliance_report.is_compliant
            }
        )

        return enhanced_forecast, compliance_report

    def _add_automation_markers(self, prediction: Prediction) -> Prediction:
        """Add automation markers to prediction for compliance."""

        # Ensure method metadata has automation markers
        updated_metadata = prediction.method_metadata.copy() if prediction.method_metadata else {}

        # Add automation markers
        automation_markers = updated_metadata.get("automation_markers", [])
        if not isinstance(automation_markers, list):
            automation_markers = []

        # Add required markers
        required_markers = ["automated_prediction", "ai_generated", "no_human_intervention"]
        for marker in required_markers:
            if marker not in automation_markers:
                automation_markers.append(marker)

        updated_metadata["automation_markers"] = automation_markers
        updated_metadata["compliance_enhanced"] = True
        updated_metadata["enhancement_timestamp"] = datetime.utcnow().isoformat()

        # Create enhanced prediction
        enhanced = Prediction(
            id=prediction.id,
            question_id=prediction.question_id,
            research_report_id=prediction.research_report_id,
            result=prediction.result,
            confidence=prediction.confidence,
            method=prediction.method,
            reasoning=prediction.reasoning,
            reasoning_steps=prediction.reasoning_steps,
            created_at=prediction.created_at,
            created_by=prediction.created_by,
            method_metadata=updated_metadata
        )

        # Copy other attributes
        for attr in ['lower_bound', 'upper_bound', 'confidence_interval',
                     'internal_consistency_score', 'evidence_strength',
                     'reasoning_trace', 'bias_checks_performed',
                     'uncertainty_quantification', 'calibration_data']:
            if hasattr(prediction, attr):
                setattr(enhanced, attr, getattr(prediction, attr))

        return enhanced

    def _remediate_prediction_violations(self, prediction: Prediction, compliance_report: ComplianceReport) -> Prediction:
        """Remediate compliance violations in a prediction."""

        remediated = prediction

        for violation in compliance_report.violations:
            if violation.violation_type.value == "human_intervention":
                # Remove human intervention indicators
                remediated = self._remove_human_intervention_indicators(remediated, violation)

            elif violation.violation_type.value == "non_automated_decision":
                # Add missing automation markers
                remediated = self._add_automation_markers(remediated)

            elif violation.violation_type.value == "manual_override":
                # Remove manual override flags
                remediated = self._remove_manual_override_flags(remediated)

        # Log remediation
        self.compliance_monitor.log_automated_decision(
            component="compliance_remediator",
            decision_type="violation_remediation",
            metadata={
                "prediction_id": str(prediction.id),
                "violations_remediated": len(compliance_report.violations),
                "violation_types": [v.violation_type.value for v in compliance_report.violations]
            }
        )

        return remediated

    def _remove_human_intervention_indicators(self, prediction: Prediction, violation) -> Prediction:
        """Remove human intervention indicators from prediction."""

        # Clean reasoning text
        cleaned_reasoning = self._sanitize_reasoning_for_compliance(prediction.reasoning)

        # Clean reasoning steps
        cleaned_steps = []
        for step in prediction.reasoning_steps:
            cleaned_step = self._sanitize_reasoning_for_compliance(step)
            if cleaned_step and cleaned_step != step:
                cleaned_steps.append(cleaned_step)
            elif not any(phrase in step.lower() for phrase in ["manual", "human", "i think", "i believe"]):
                cleaned_steps.append(step)

        # Update method metadata
        updated_metadata = prediction.method_metadata.copy() if prediction.method_metadata else {}
        updated_metadata.pop("human_reviewed", None)
        updated_metadata.pop("manual_override", None)
        updated_metadata["human_intervention_removed"] = True

        # Create cleaned prediction
        return Prediction(
            id=prediction.id,
            question_id=prediction.question_id,
            research_report_id=prediction.research_report_id,
            result=prediction.result,
            confidence=prediction.confidence,
            method=prediction.method,
            reasoning=cleaned_reasoning,
            reasoning_steps=cleaned_steps,
            created_at=prediction.created_at,
            created_by="ai_forecasting_system",  # Ensure automated creator
            method_metadata=updated_metadata
        )

    def _remove_manual_override_flags(self, prediction: Prediction) -> Prediction:
        """Remove manual override flags from prediction metadata."""

        updated_metadata = prediction.method_metadata.copy() if prediction.method_metadata else {}

        # Remove manual override indicators
        flags_to_remove = ["manual_override", "human_reviewed", "manual_adjustment"]
        for flag in flags_to_remove:
            updated_metadata.pop(flag, None)

        # Add automation confirmation
        updated_metadata["manual_flags_removed"] = True
        updated_metadata["fully_automated"] = True

        return Prediction(
            id=prediction.id,
            question_id=prediction.question_id,
            research_report_id=prediction.research_report_id,
            result=prediction.result,
            confidence=prediction.confidence,
            method=prediction.method,
            reasoning=prediction.reasoning,
            reasoning_steps=prediction.reasoning_steps,
            created_at=prediction.created_at,
            created_by=prediction.created_by,
            method_metadata=updated_metadata
        )

    def _sanitize_reasoning_for_compliance(self, reasoning: Optional[str]) -> str:
        """Sanitize reasoning text to remove human intervention language."""

        if not reasoning:
            return ""

        # Replace human intervention phrases
        replacements = {
            "i think": "analysis suggests",
            "i believe": "evidence indicates",
            "in my opinion": "based on analysis",
            "personally": "analytically",
            "manually adjusted": "algorithmically adjusted",
            "human review": "automated validation",
            "expert judgment": "analytical assessment"
        }

        sanitized = reasoning.lower()
        for phrase, replacement in replacements.items():
            sanitized = sanitized.replace(phrase, replacement)

        # Capitalize first letter of sentences
        sentences = sanitized.split('. ')
        capitalized_sentences = [s.capitalize() if s else s for s in sentences]

        return '. '.join(capitalized_sentences)

    def _ensure_automated_ensemble_method(self, ensemble_method: Optional[str]) -> str:
        """Ensure ensemble method indicates automation."""

        if not ensemble_method:
            return "automated_ensemble"

        if "manual" in ensemble_method.lower():
            return ensemble_method.replace("manual", "automated")

        if "automated" not in ensemble_method.lower():
            return f"automated_{ensemble_method}"

        return ensemble_method
