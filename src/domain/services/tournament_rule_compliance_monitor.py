"""Tournament rule compliance monitor for automated decision-making validation."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..entities.forecast import Forecast
from ..entities.prediction import Prediction


class ComplianceViolationType(Enum):
    """Types of tournament rule compliance violations."""

    HUMAN_INTERVENTION = "human_intervention"
    MANUAL_OVERRIDE = "manual_override"
    NON_AUTOMATED_DECISION = "non_automated_decision"
    EXTERNAL_INPUT = "external_input"
    RULE_VIOLATION = "rule_violation"
    LATE_SUBMISSION = "late_submission"


@dataclass
class ComplianceViolation:
    """Represents a tournament rule compliance violation."""

    violation_type: ComplianceViolationType
    severity: str  # "critical", "major", "minor", "warning"
    description: str
    timestamp: datetime
    component: str  # Which component detected the violation
    metadata: Dict[str, Any]
    remediation_required: bool = True


@dataclass
class ComplianceReport:
    """Tournament rule compliance report."""

    is_compliant: bool
    violations: List[ComplianceViolation]
    compliance_score: float  # 0.0 to 1.0
    automated_decisions_count: int
    human_interventions_count: int
    report_timestamp: datetime
    monitoring_period: timedelta


class TournamentRuleComplianceMonitor:
    """Monitors tournament rule compliance for automated decision-making."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Compliance tracking
        self.violations = []
        self.decision_log = []
        self.human_intervention_indicators = set()
        self.automated_decision_markers = set()

        # Tournament rule parameters
        self.max_human_interventions = 0  # Tournament requires full automation
        self.required_automation_markers = {
            "automated_research",
            "automated_prediction",
            "automated_submission",
        }

        # Monitoring configuration
        self.monitoring_enabled = True
        self.strict_mode = True  # Strict tournament compliance
        self.violation_threshold = 0.95  # 95% compliance required

    def validate_prediction_compliance(
        self, prediction: Prediction
    ) -> ComplianceReport:
        """Validate a prediction for tournament rule compliance."""

        violations = []
        automated_decisions = 0
        human_interventions = 0

        # Check for automated decision markers
        if self._has_automation_markers(prediction):
            automated_decisions += 1
        else:
            violations.append(
                ComplianceViolation(
                    violation_type=ComplianceViolationType.NON_AUTOMATED_DECISION,
                    severity="critical",
                    description="Prediction lacks required automation markers",
                    timestamp=datetime.utcnow(),
                    component="prediction_validator",
                    metadata={"prediction_id": str(prediction.id)},
                    remediation_required=True,
                )
            )

        # Check for human intervention indicators
        human_indicators = self._detect_human_intervention(prediction)
        if human_indicators:
            human_interventions += len(human_indicators)
            for indicator in human_indicators:
                violations.append(
                    ComplianceViolation(
                        violation_type=ComplianceViolationType.HUMAN_INTERVENTION,
                        severity="critical",
                        description=f"Human intervention detected: {indicator}",
                        timestamp=datetime.utcnow(),
                        component="human_intervention_detector",
                        metadata={
                            "indicator": indicator,
                            "prediction_id": str(prediction.id),
                        },
                        remediation_required=True,
                    )
                )

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(
            violations, automated_decisions, human_interventions
        )
        is_compliant = (
            compliance_score >= self.violation_threshold and human_interventions == 0
        )

        return ComplianceReport(
            is_compliant=is_compliant,
            violations=violations,
            compliance_score=compliance_score,
            automated_decisions_count=automated_decisions,
            human_interventions_count=human_interventions,
            report_timestamp=datetime.utcnow(),
            monitoring_period=timedelta(seconds=0),  # Single prediction check
        )

    def validate_forecast_compliance(self, forecast: Forecast) -> ComplianceReport:
        """Validate a forecast for tournament rule compliance."""

        all_violations = []
        total_automated_decisions = 0
        total_human_interventions = 0

        # Validate each prediction in the forecast
        for prediction in forecast.predictions:
            pred_report = self.validate_prediction_compliance(prediction)
            all_violations.extend(pred_report.violations)
            total_automated_decisions += pred_report.automated_decisions_count
            total_human_interventions += pred_report.human_interventions_count

        # Check forecast-level compliance
        forecast_violations = self._validate_forecast_level_compliance(forecast)
        all_violations.extend(forecast_violations)

        # Calculate overall compliance
        compliance_score = self._calculate_compliance_score(
            all_violations, total_automated_decisions, total_human_interventions
        )
        is_compliant = (
            compliance_score >= self.violation_threshold
            and total_human_interventions == 0
        )

        return ComplianceReport(
            is_compliant=is_compliant,
            violations=all_violations,
            compliance_score=compliance_score,
            automated_decisions_count=total_automated_decisions,
            human_interventions_count=total_human_interventions,
            report_timestamp=datetime.utcnow(),
            monitoring_period=timedelta(seconds=0),
        )

    def log_automated_decision(
        self, component: str, decision_type: str, metadata: Dict[str, Any]
    ):
        """Log an automated decision for compliance tracking."""

        decision_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "component": component,
            "decision_type": decision_type,
            "metadata": metadata,
            "automated": True,
        }

        self.decision_log.append(decision_entry)
        self.automated_decision_markers.add(f"{component}:{decision_type}")

        self.logger.debug(
            f"Logged automated decision: {component}:{decision_type}",
            extra={"compliance_tracking": True},
        )

    def detect_human_intervention(
        self, component: str, intervention_type: str, metadata: Dict[str, Any]
    ):
        """Detect and log human intervention for compliance monitoring."""

        intervention_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "component": component,
            "intervention_type": intervention_type,
            "metadata": metadata,
            "automated": False,
        }

        self.decision_log.append(intervention_entry)
        self.human_intervention_indicators.add(f"{component}:{intervention_type}")

        # Create violation
        violation = ComplianceViolation(
            violation_type=ComplianceViolationType.HUMAN_INTERVENTION,
            severity="critical",
            description=f"Human intervention detected in {component}: {intervention_type}",
            timestamp=datetime.utcnow(),
            component=component,
            metadata=metadata,
            remediation_required=True,
        )

        self.violations.append(violation)

        self.logger.warning(
            f"COMPLIANCE VIOLATION: Human intervention detected: {component}:{intervention_type}",
            extra={"compliance_violation": True},
        )

    def _has_automation_markers(self, prediction: Prediction) -> bool:
        """Check if prediction has required automation markers."""

        # Check method metadata for automation markers
        if prediction.method_metadata:
            automation_markers = prediction.method_metadata.get(
                "automation_markers", []
            )
            if isinstance(automation_markers, list):
                return any(
                    marker in self.required_automation_markers
                    for marker in automation_markers
                )

        # Check reasoning steps for automation indicators
        automation_keywords = ["automated", "algorithm", "model", "ai", "generated"]
        if prediction.reasoning_steps:
            reasoning_text = " ".join(prediction.reasoning_steps).lower()
            if any(keyword in reasoning_text for keyword in automation_keywords):
                return True

        # Check created_by field
        if prediction.created_by and "ai" in prediction.created_by.lower():
            return True

        return False

    def _detect_human_intervention(self, prediction: Prediction) -> List[str]:
        """Detect human intervention indicators in a prediction."""

        indicators = []

        # Check for manual override indicators
        if prediction.method_metadata:
            if prediction.method_metadata.get("manual_override", False):
                indicators.append("manual_override_flag")

            if prediction.method_metadata.get("human_reviewed", False):
                indicators.append("human_review_flag")

        # Check reasoning for human intervention language
        human_phrases = [
            "manually adjusted",
            "human review",
            "expert opinion",
            "manual override",
            "human judgment",
            "personally think",
            "in my opinion",
            "i believe",
            "i think",
        ]

        reasoning_text = (prediction.reasoning or "").lower()
        for phrase in human_phrases:
            if phrase in reasoning_text:
                indicators.append(f"human_language: {phrase}")

        # Check reasoning steps for human intervention
        if prediction.reasoning_steps:
            steps_text = " ".join(prediction.reasoning_steps).lower()
            for phrase in human_phrases:
                if phrase in steps_text:
                    indicators.append(f"human_reasoning_step: {phrase}")

        return indicators

    def _validate_forecast_level_compliance(
        self, forecast: Forecast
    ) -> List[ComplianceViolation]:
        """Validate forecast-level compliance requirements."""

        violations = []

        # Check ensemble method for automation
        if forecast.ensemble_method:
            if "manual" in forecast.ensemble_method.lower():
                violations.append(
                    ComplianceViolation(
                        violation_type=ComplianceViolationType.MANUAL_OVERRIDE,
                        severity="major",
                        description="Manual ensemble method detected",
                        timestamp=datetime.utcnow(),
                        component="ensemble_validator",
                        metadata={"ensemble_method": forecast.ensemble_method},
                        remediation_required=True,
                    )
                )

        # Check reasoning summary for human intervention
        if forecast.reasoning_summary:
            human_indicators = self._detect_human_language(forecast.reasoning_summary)
            for indicator in human_indicators:
                violations.append(
                    ComplianceViolation(
                        violation_type=ComplianceViolationType.HUMAN_INTERVENTION,
                        severity="major",
                        description=f"Human intervention in reasoning: {indicator}",
                        timestamp=datetime.utcnow(),
                        component="reasoning_validator",
                        metadata={"indicator": indicator},
                        remediation_required=True,
                    )
                )

        return violations

    def _detect_human_language(self, text: str) -> List[str]:
        """Detect human intervention language in text."""

        indicators = []
        text_lower = text.lower()

        human_phrases = [
            "i think",
            "i believe",
            "in my opinion",
            "personally",
            "manually",
            "human review",
            "expert judgment",
        ]

        for phrase in human_phrases:
            if phrase in text_lower:
                indicators.append(phrase)

        return indicators

    def _calculate_compliance_score(
        self,
        violations: List[ComplianceViolation],
        automated_decisions: int,
        human_interventions: int,
    ) -> float:
        """Calculate compliance score based on violations and decision types."""

        if human_interventions > 0:
            return 0.0  # Any human intervention = non-compliant

        if not violations:
            return 1.0  # Perfect compliance

        # Weight violations by severity
        severity_weights = {
            "critical": 0.4,
            "major": 0.2,
            "minor": 0.1,
            "warning": 0.05,
        }

        total_penalty = sum(
            severity_weights.get(violation.severity, 0.1) for violation in violations
        )

        # Calculate score
        score = max(0.0, 1.0 - total_penalty)

        # Bonus for automated decisions
        if automated_decisions > 0:
            automation_bonus = min(0.1, automated_decisions * 0.02)
            score = min(1.0, score + automation_bonus)

        return score

    def generate_compliance_report(
        self, monitoring_period: timedelta
    ) -> ComplianceReport:
        """Generate comprehensive compliance report for monitoring period."""

        # Filter violations by monitoring period
        cutoff_time = datetime.utcnow() - monitoring_period
        recent_violations = [v for v in self.violations if v.timestamp >= cutoff_time]

        # Count decisions in monitoring period
        recent_decisions = [
            d
            for d in self.decision_log
            if datetime.fromisoformat(d["timestamp"]) >= cutoff_time
        ]

        automated_count = sum(1 for d in recent_decisions if d["automated"])
        human_count = sum(1 for d in recent_decisions if not d["automated"])

        # Calculate compliance
        compliance_score = self._calculate_compliance_score(
            recent_violations, automated_count, human_count
        )
        is_compliant = compliance_score >= self.violation_threshold and human_count == 0

        return ComplianceReport(
            is_compliant=is_compliant,
            violations=recent_violations,
            compliance_score=compliance_score,
            automated_decisions_count=automated_count,
            human_interventions_count=human_count,
            report_timestamp=datetime.utcnow(),
            monitoring_period=monitoring_period,
        )

    def check_human_intervention(self, prediction_metadata: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check for human intervention violations in prediction metadata."""

        # Check for human review flag
        if prediction_metadata.get("human_review", False):
            return ComplianceViolation(
                violation_type=ComplianceViolationType.HUMAN_INTERVENTION,
                severity="critical",
                description="Human review detected in prediction process",
                timestamp=datetime.utcnow(),
                component="human_intervention_checker",
                metadata=prediction_metadata,
                remediation_required=True
            )

        # Check for manual adjustments
        manual_adjustments = prediction_metadata.get("manual_adjustments", [])
        if manual_adjustments:
            return ComplianceViolation(
                violation_type=ComplianceViolationType.MANUAL_OVERRIDE,
                severity="critical",
                description=f"Manual adjustments detected: {', '.join(manual_adjustments)}",
                timestamp=datetime.utcnow(),
                component="manual_adjustment_checker",
                metadata=prediction_metadata,
                remediation_required=True
            )

        # Check for intervention flags
        intervention_flags = prediction_metadata.get("intervention_flags", [])
        if intervention_flags:
            return ComplianceViolation(
                violation_type=ComplianceViolationType.HUMAN_INTERVENTION,
                severity="critical",
                description=f"Intervention flags detected: {', '.join(intervention_flags)}",
                timestamp=datetime.utcnow(),
                component="intervention_flag_checker",
                metadata=prediction_metadata,
                remediation_required=True
            )

        # Check agent type
        agent_type = prediction_metadata.get("agent_type", "unknown")
        if agent_type != "automated":
            return ComplianceViolation(
                violation_type=ComplianceViolationType.NON_AUTOMATED_DECISION,
                severity="major",
                description=f"Non-automated agent type: {agent_type}",
                timestamp=datetime.utcnow(),
                component="agent_type_checker",
                metadata=prediction_metadata,
                remediation_required=True
            )

        return None  # No violations detected

    def check_submission_timing(self, submission_metadata: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check for submission timing compliance violations."""

        question_close_time = submission_metadata.get("question_close_time")
        submission_time = submission_metadata.get("submission_time")

        if not question_close_time or not submission_time:
            return ComplianceViolation(
                violation_type=ComplianceViolationType.RULE_VIOLATION,
                severity="major",
                description="Missing timing information for compliance check",
                timestamp=datetime.utcnow(),
                component="timing_checker",
                metadata=submission_metadata,
                remediation_required=True
            )

        # Check if submission was made after question close time
        if submission_time > question_close_time:
            return ComplianceViolation(
                violation_type=ComplianceViolationType.LATE_SUBMISSION,
                severity="critical",
                description="Submission made after close time",
                timestamp=datetime.utcnow(),
                component="timing_checker",
                metadata={
                    "submission_time": submission_time.isoformat() if hasattr(submission_time, 'isoformat') else str(submission_time),
                    "close_time": question_close_time.isoformat() if hasattr(question_close_time, 'isoformat') else str(question_close_time),
                    **submission_metadata
                },
                remediation_required=True
            )

        return None  # No timing violations detected
