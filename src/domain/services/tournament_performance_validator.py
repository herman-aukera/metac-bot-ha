"""
Tournament performance validation and competitive optimization service.

This service implements log-based scoring optimization strategies, tournament-specific
calibration adjustments, compliance monitoring, and competitive performance validation.
"""

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
from uuid import UUID
import json
from pathlib import Path

from ..entities.forecast import Forecast
from ..entities.prediction import Prediction
from .performance_tracking_service import PerformanceTrackingService, MetricType
from .calibration_service import CalibrationTracker, CalibrationDriftSeverity
from .tournament_analytics import TournamentAnalytics
from ...infrastructure.config.tournament_config import get_tournament_config
from ...infrastructure.logging.reasoning_logger import get_reasoning_logger


class ComplianceStatus(Enum):
    """Tournament compliance status levels."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


class OptimizationStrategy(Enum):
    """Log-based scoring optimization strategies."""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
    TOURNAMENT_SPECIFIC = "tournament_specific"


@dataclass
class LogScoreOptimization:
    """Log score optimization configuration and results."""
    strategy: OptimizationStrategy
    target_log_score: float
    confidence_adjustment: float
    probability_bounds: Tuple[float, float]
    expected_improvement: float
    risk_tolerance: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TournamentCalibration:
    """Tournament-specific calibration adjustments."""
    base_calibration_error: float
    tournament_adjustment: float
    confidence_multiplier: float
    probability_shift: float
    competitive_pressure_factor: float
    time_pressure_factor: float
    calibration_confidence: float
    last_updated: datetime


@dataclass
class ComplianceViolation:
    """Tournament compliance violation record."""
    violation_type: str
    severity: ComplianceStatus
    description: str
    detected_at: datetime
    question_id: Optional[UUID]
    forecast_id: Optional[UUID]
    resolution_required: bool
    resolution_deadline: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceValidationResult:
    """Tournament performance validation result."""
    is_valid: bool
    validation_score: float
    compliance_status: ComplianceStatus
    optimization_recommendations: List[str]
    calibration_adjustments: Optional[TournamentCalibration]
    log_score_optimization: Optional[LogScoreOptimization]
    violations: List[ComplianceViolation]
    competitive_position: Dict[str, float]
    validation_timestamp: datetime


class TournamentPerformanceValidator:
    """
    Tournament performance validation and competitive optimization service.

    Provides comprehensive tournament performance validation including:
    - Log-based scoring optimization strategies
    - Tournament-specific calibration and confidence adjustment
    - Tournament compliance monitoring and alerting
    - Competitive performance optimization
    """

    def __init__(
        self,
        performance_tracker: Optional[PerformanceTrackingService] = None,
        calibration_tracker: Optional[CalibrationTracker] = None,
        tournament_analytics: Optional[TournamentAnalytics] = None
    ):
        """Initialize tournament performance validator."""
        self.logger = logging.getLogger(__name__)
        self.reasoning_logger = get_reasoning_logger()
        self.tournament_config = get_tournament_config()

        # Service dependencies
        self.performance_tracker = performance_tracker or PerformanceTrackingService()
        self.calibration_tracker = calibration_tracker or CalibrationTracker()
        self.tournament_analytics = tournament_analytics or TournamentAnalytics()

        # Validation state
        self.compliance_violations: List[ComplianceViolation] = []
        self.optimization_history: List[LogScoreOptimization] = []
        self.calibration_history: List[TournamentCalibration] = []

        # Optimization parameters
        self.log_score_targets = {
            OptimizationStrategy.CONSERVATIVE: 0.4,
            OptimizationStrategy.BALANCED: 0.3,
            OptimizationStrategy.AGGRESSIVE: 0.25,
            OptimizationStrategy.ADAPTIVE: 0.35,
            OptimizationStrategy.TOURNAMENT_SPECIFIC: 0.28
        }

        # Compliance thresholds
        self.compliance_thresholds = {
            "max_log_score": 0.5,
            "min_calibration_score": 0.7,
            "max_response_time": 300.0,  # 5 minutes
            "min_confidence_threshold": 0.1,
            "max_confidence_threshold": 0.95,
            "max_prediction_variance": 0.3
        }

        self.logger.info("Tournament performance validator initialized")

    def validate_tournament_performance(
        self,
        forecast: Forecast,
        tournament_context: Optional[Dict[str, Any]] = None
    ) -> PerformanceValidationResult:
        """
        Validate tournament performance for a forecast.

        Args:
            forecast: Forecast to validate
            tournament_context: Additional tournament context

        Returns:
            Performance validation result with optimization recommendations
        """
        try:
            validation_timestamp = datetime.utcnow()

            # Initialize validation components
            violations = []
            optimization_recommendations = []
            competitive_position = {}

            # Validate compliance
            compliance_status, compliance_violations = self._validate_compliance(
                forecast, tournament_context
            )
            violations.extend(compliance_violations)

            # Calculate validation score
            validation_score = self._calculate_validation_score(forecast, violations)

            # Generate log score optimization
            log_score_optimization = self._optimize_log_score(
                forecast, tournament_context
            )

            # Generate tournament-specific calibration
            calibration_adjustments = self._calculate_tournament_calibration(
                forecast, tournament_context
            )

            # Analyze competitive position
            competitive_position = self._analyze_competitive_position(
                forecast, tournament_context
            )

            # Generate optimization recommendations
            optimization_recommendations = self._generate_optimization_recommendations(
                forecast, violations, log_score_optimization, calibration_adjustments
            )

            # Create validation result
            result = PerformanceValidationResult(
                is_valid=compliance_status != ComplianceStatus.CRITICAL,
                validation_score=validation_score,
                compliance_status=compliance_status,
                optimization_recommendations=optimization_recommendations,
                calibration_adjustments=calibration_adjustments,
                log_score_optimization=log_score_optimization,
                violations=violations,
                competitive_position=competitive_position,
                validation_timestamp=validation_timestamp
            )

            # Log validation result
            self._log_validation_result(forecast, result)

            # Store violations for monitoring
            self.compliance_violations.extend(violations)

            # Store optimization history
            if log_score_optimization:
                self.optimization_history.append(log_score_optimization)

            # Store calibration history
            if calibration_adjustments:
                self.calibration_history.append(calibration_adjustments)

            self.logger.info(
                f"Tournament performance validation completed for forecast {forecast.id}: "
                f"Score={validation_score:.3f}, Status={compliance_status.value}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error validating tournament performance: {e}")
            # Return minimal validation result on error
            return PerformanceValidationResult(
                is_valid=False,
                validation_score=0.0,
                compliance_status=ComplianceStatus.CRITICAL,
                optimization_recommendations=["Error in validation - manual review required"],
                calibration_adjustments=None,
                log_score_optimization=None,
                violations=[ComplianceViolation(
                    violation_type="validation_error",
                    severity=ComplianceStatus.CRITICAL,
                    description=f"Validation error: {str(e)}",
                    detected_at=datetime.utcnow(),
                    question_id=forecast.question_id,
                    forecast_id=forecast.id,
                    resolution_required=True,
                    resolution_deadline=datetime.utcnow() + timedelta(hours=1)
                )],
                competitive_position={},
                validation_timestamp=datetime.utcnow()
            )
    def _validate_compliance(
        self,
        forecast: Forecast,
        tournament_context: Optional[Dict[str, Any]]
    ) -> Tuple[ComplianceStatus, List[ComplianceViolation]]:
        """Validate tournament compliance for a forecast."""
        violations = []
        overall_status = ComplianceStatus.COMPLIANT

        # Validate prediction bounds
        if forecast.prediction < 0.01 or forecast.prediction > 0.99:
            violations.append(ComplianceViolation(
                violation_type="prediction_bounds",
                severity=ComplianceStatus.WARNING,
                description=f"Prediction {forecast.prediction:.3f} near extreme bounds",
                detected_at=datetime.utcnow(),
                question_id=forecast.question_id,
                forecast_id=forecast.id,
                resolution_required=False
            ))

        # Validate confidence thresholds
        if forecast.confidence_score < self.compliance_thresholds["min_confidence_threshold"]:
            violations.append(ComplianceViolation(
                violation_type="low_confidence",
                severity=ComplianceStatus.VIOLATION,
                description=f"Confidence {forecast.confidence_score:.3f} below minimum threshold",
                detected_at=datetime.utcnow(),
                question_id=forecast.question_id,
                forecast_id=forecast.id,
                resolution_required=True,
                resolution_deadline=datetime.utcnow() + timedelta(hours=2)
            ))

        # Validate prediction variance (ensemble disagreement)
        prediction_variance = forecast.calculate_prediction_variance()
        if prediction_variance > self.compliance_thresholds["max_prediction_variance"]:
            violations.append(ComplianceViolation(
                violation_type="high_variance",
                severity=ComplianceStatus.WARNING,
                description=f"Prediction variance {prediction_variance:.3f} indicates high disagreement",
                detected_at=datetime.utcnow(),
                question_id=forecast.question_id,
                forecast_id=forecast.id,
                resolution_required=False
            ))

        # Validate reasoning quality
        if not forecast.reasoning_summary or len(forecast.reasoning_summary) < 100:
            violations.append(ComplianceViolation(
                violation_type="insufficient_reasoning",
                severity=ComplianceStatus.WARNING,
                description="Insufficient reasoning documentation for tournament submission",
                detected_at=datetime.utcnow(),
                question_id=forecast.question_id,
                forecast_id=forecast.id,
                resolution_required=False
            ))

        # Validate tournament-specific requirements
        if tournament_context:
            # Check deadline compliance
            deadline = tournament_context.get("deadline")
            if deadline:
                try:
                    deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                    if deadline_dt.tzinfo:
                        deadline_dt = deadline_dt.replace(tzinfo=None)

                    time_to_deadline = (deadline_dt - datetime.utcnow()).total_seconds()
                    if time_to_deadline < 3600:  # Less than 1 hour
                        violations.append(ComplianceViolation(
                            violation_type="deadline_pressure",
                            severity=ComplianceStatus.WARNING,
                            description=f"Submission close to deadline ({time_to_deadline/60:.1f} minutes remaining)",
                            detected_at=datetime.utcnow(),
                            question_id=forecast.question_id,
                            forecast_id=forecast.id,
                            resolution_required=False
                        ))
                except (ValueError, AttributeError):
                    pass

            # Check tournament mode compliance
            if self.tournament_config.is_tournament_mode():
                if not forecast.tournament_strategy:
                    violations.append(ComplianceViolation(
                        violation_type="missing_tournament_strategy",
                        severity=ComplianceStatus.VIOLATION,
                        description="Tournament strategy required in tournament mode",
                        detected_at=datetime.utcnow(),
                        question_id=forecast.question_id,
                        forecast_id=forecast.id,
                        resolution_required=True,
                        resolution_deadline=datetime.utcnow() + timedelta(hours=1)
                    ))

        # Determine overall compliance status
        if any(v.severity == ComplianceStatus.CRITICAL for v in violations):
            overall_status = ComplianceStatus.CRITICAL
        elif any(v.severity == ComplianceStatus.VIOLATION for v in violations):
            overall_status = ComplianceStatus.VIOLATION
        elif any(v.severity == ComplianceStatus.WARNING for v in violations):
            overall_status = ComplianceStatus.WARNING

        return overall_status, violations

    def _calculate_validation_score(
        self,
        forecast: Forecast,
        violations: List[ComplianceViolation]
    ) -> float:
        """Calculate overall validation score for the forecast."""
        base_score = 1.0

        # Deduct points for violations
        for violation in violations:
            if violation.severity == ComplianceStatus.CRITICAL:
                base_score -= 0.5
            elif violation.severity == ComplianceStatus.VIOLATION:
                base_score -= 0.2
            elif violation.severity == ComplianceStatus.WARNING:
                base_score -= 0.1

        # Add points for quality indicators
        if forecast.confidence_score > 0.8:
            base_score += 0.1

        if forecast.reasoning_summary and len(forecast.reasoning_summary) > 500:
            base_score += 0.05

        if forecast.research_reports and len(forecast.research_reports) > 0:
            base_score += 0.05

        # Normalize to [0, 1] range
        return max(0.0, min(1.0, base_score))

    def _optimize_log_score(
        self,
        forecast: Forecast,
        tournament_context: Optional[Dict[str, Any]]
    ) -> LogScoreOptimization:
        """Generate log score optimization strategy."""
        # Determine optimization strategy based on tournament context
        strategy = self._determine_optimization_strategy(forecast, tournament_context)

        # Calculate target log score
        target_log_score = self.log_score_targets[strategy]

        # Calculate confidence adjustment needed
        current_prediction = forecast.prediction
        confidence_adjustment = self._calculate_confidence_adjustment(
            current_prediction, target_log_score, forecast.confidence_score
        )

        # Calculate probability bounds for optimization
        probability_bounds = self._calculate_optimal_probability_bounds(
            current_prediction, strategy
        )

        # Estimate expected improvement
        expected_improvement = self._estimate_log_score_improvement(
            current_prediction, target_log_score, confidence_adjustment
        )

        # Assess risk tolerance
        risk_tolerance = self._assess_risk_tolerance(strategy, tournament_context)

        return LogScoreOptimization(
            strategy=strategy,
            target_log_score=target_log_score,
            confidence_adjustment=confidence_adjustment,
            probability_bounds=probability_bounds,
            expected_improvement=expected_improvement,
            risk_tolerance=risk_tolerance,
            metadata={
                "current_prediction": current_prediction,
                "current_confidence": forecast.confidence_score,
                "optimization_timestamp": datetime.utcnow().isoformat()
            }
        )

    def _determine_optimization_strategy(
        self,
        forecast: Forecast,
        tournament_context: Optional[Dict[str, Any]]
    ) -> OptimizationStrategy:
        """Determine the best optimization strategy for the current context."""
        if not tournament_context:
            return OptimizationStrategy.BALANCED

        # Tournament-specific strategy selection
        if self.tournament_config.is_tournament_mode():
            # Check competitive pressure
            competitive_pressure = tournament_context.get("competitive_pressure", 0.5)
            time_pressure = tournament_context.get("time_pressure", 0.5)

            if competitive_pressure > 0.8 or time_pressure > 0.8:
                return OptimizationStrategy.AGGRESSIVE
            elif competitive_pressure < 0.3 and time_pressure < 0.3:
                return OptimizationStrategy.CONSERVATIVE
            else:
                return OptimizationStrategy.TOURNAMENT_SPECIFIC

        # Adaptive strategy based on recent performance
        recent_performance = self._get_recent_performance_metrics()
        if recent_performance.get("average_log_score", 0.4) > 0.35:
            return OptimizationStrategy.CONSERVATIVE
        else:
            return OptimizationStrategy.ADAPTIVE

    def _calculate_confidence_adjustment(
        self,
        current_prediction: float,
        target_log_score: float,
        current_confidence: float
    ) -> float:
        """Calculate confidence adjustment needed to achieve target log score."""
        # Simplified confidence adjustment calculation
        # In practice, this would use more sophisticated optimization

        # Calculate current expected log score
        epsilon = 1e-15
        prob_clamped = max(epsilon, min(1 - epsilon, current_prediction))

        # Estimate adjustment needed
        if target_log_score < 0.3:  # Aggressive target
            return min(0.2, current_confidence * 0.1)
        elif target_log_score > 0.4:  # Conservative target
            return max(-0.2, -current_confidence * 0.1)
        else:  # Balanced target
            return 0.0

    def _calculate_optimal_probability_bounds(
        self,
        current_prediction: float,
        strategy: OptimizationStrategy
    ) -> Tuple[float, float]:
        """Calculate optimal probability bounds for the strategy."""
        bounds_config = {
            OptimizationStrategy.CONSERVATIVE: (0.1, 0.9),
            OptimizationStrategy.BALANCED: (0.05, 0.95),
            OptimizationStrategy.AGGRESSIVE: (0.02, 0.98),
            OptimizationStrategy.ADAPTIVE: (0.05, 0.95),
            OptimizationStrategy.TOURNAMENT_SPECIFIC: (0.03, 0.97)
        }

        base_bounds = bounds_config[strategy]

        # Adjust bounds based on current prediction
        if current_prediction < 0.3:
            # Lower prediction - tighten lower bound
            return (max(base_bounds[0], 0.05), base_bounds[1])
        elif current_prediction > 0.7:
            # Higher prediction - tighten upper bound
            return (base_bounds[0], min(base_bounds[1], 0.95))
        else:
            return base_bounds

    def _estimate_log_score_improvement(
        self,
        current_prediction: float,
        target_log_score: float,
        confidence_adjustment: float
    ) -> float:
        """Estimate expected log score improvement from optimization."""
        # Simplified improvement estimation
        # In practice, this would use historical data and more sophisticated modeling

        base_improvement = max(0.0, 0.4 - target_log_score) * 0.1
        confidence_improvement = abs(confidence_adjustment) * 0.05

        return base_improvement + confidence_improvement

    def _assess_risk_tolerance(
        self,
        strategy: OptimizationStrategy,
        tournament_context: Optional[Dict[str, Any]]
    ) -> float:
        """Assess risk tolerance for the optimization strategy."""
        base_tolerance = {
            OptimizationStrategy.CONSERVATIVE: 0.2,
            OptimizationStrategy.BALANCED: 0.5,
            OptimizationStrategy.AGGRESSIVE: 0.8,
            OptimizationStrategy.ADAPTIVE: 0.6,
            OptimizationStrategy.TOURNAMENT_SPECIFIC: 0.7
        }[strategy]

        # Adjust based on tournament context
        if tournament_context:
            competitive_pressure = tournament_context.get("competitive_pressure", 0.5)
            base_tolerance += competitive_pressure * 0.2

        return min(1.0, max(0.0, base_tolerance))

    def _get_recent_performance_metrics(self) -> Dict[str, float]:
        """Get recent performance metrics for strategy selection."""
        # This would integrate with the performance tracking service
        # For now, return default values
        return {
            "average_log_score": 0.35,
            "average_brier_score": 0.25,
            "calibration_error": 0.08,
            "prediction_count": 10
        }
    def _calculate_tournament_calibration(
        self,
        forecast: Forecast,
        tournament_context: Optional[Dict[str, Any]]
    ) -> TournamentCalibration:
        """Calculate tournament-specific calibration adjustments."""
        # Get base calibration from calibration tracker
        try:
            # This would use actual calibration data in practice
            base_calibration_error = 0.08  # Default value

            # Calculate tournament-specific adjustments
            tournament_adjustment = self._calculate_tournament_adjustment(tournament_context)
            confidence_multiplier = self._calculate_confidence_multiplier(forecast, tournament_context)
            probability_shift = self._calculate_probability_shift(forecast, tournament_context)

            # Calculate pressure factors
            competitive_pressure_factor = tournament_context.get("competitive_pressure", 0.5) if tournament_context else 0.5
            time_pressure_factor = tournament_context.get("time_pressure", 0.5) if tournament_context else 0.5

            # Calculate calibration confidence
            calibration_confidence = self._calculate_calibration_confidence(
                base_calibration_error, tournament_adjustment
            )

            return TournamentCalibration(
                base_calibration_error=base_calibration_error,
                tournament_adjustment=tournament_adjustment,
                confidence_multiplier=confidence_multiplier,
                probability_shift=probability_shift,
                competitive_pressure_factor=competitive_pressure_factor,
                time_pressure_factor=time_pressure_factor,
                calibration_confidence=calibration_confidence,
                last_updated=datetime.utcnow()
            )

        except Exception as e:
            self.logger.warning(f"Error calculating tournament calibration: {e}")
            # Return default calibration
            return TournamentCalibration(
                base_calibration_error=0.1,
                tournament_adjustment=0.0,
                confidence_multiplier=1.0,
                probability_shift=0.0,
                competitive_pressure_factor=0.5,
                time_pressure_factor=0.5,
                calibration_confidence=0.5,
                last_updated=datetime.utcnow()
            )

    def _calculate_tournament_adjustment(self, tournament_context: Optional[Dict[str, Any]]) -> float:
        """Calculate tournament-specific calibration adjustment."""
        if not tournament_context:
            return 0.0

        adjustment = 0.0

        # Adjust for competitive pressure
        competitive_pressure = tournament_context.get("competitive_pressure", 0.5)
        if competitive_pressure > 0.7:
            adjustment += 0.02  # Slight overconfidence adjustment
        elif competitive_pressure < 0.3:
            adjustment -= 0.01  # Slight underconfidence adjustment

        # Adjust for time pressure
        time_pressure = tournament_context.get("time_pressure", 0.5)
        if time_pressure > 0.8:
            adjustment += 0.03  # Higher adjustment for time pressure

        return adjustment

    def _calculate_confidence_multiplier(
        self,
        forecast: Forecast,
        tournament_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence multiplier for tournament conditions."""
        base_multiplier = 1.0

        # Adjust based on prediction variance
        variance = forecast.calculate_prediction_variance()
        if variance > 0.15:
            base_multiplier *= 0.9  # Reduce confidence for high variance
        elif variance < 0.05:
            base_multiplier *= 1.1  # Increase confidence for low variance

        # Adjust based on tournament context
        if tournament_context:
            competitive_pressure = tournament_context.get("competitive_pressure", 0.5)
            if competitive_pressure > 0.8:
                base_multiplier *= 0.95  # Slightly reduce confidence under high pressure

        return max(0.5, min(1.5, base_multiplier))

    def _calculate_probability_shift(
        self,
        forecast: Forecast,
        tournament_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate probability shift for tournament optimization."""
        shift = 0.0

        # Shift based on competitive intelligence
        if forecast.competitive_intelligence:
            market_position = forecast.competitive_intelligence.market_position_percentile
            if market_position and market_position < 0.3:
                # We're behind - consider more aggressive positioning
                if forecast.prediction < 0.5:
                    shift = -0.02  # Shift slightly lower
                else:
                    shift = 0.02   # Shift slightly higher

        return shift

    def _calculate_calibration_confidence(
        self,
        base_calibration_error: float,
        tournament_adjustment: float
    ) -> float:
        """Calculate confidence in calibration adjustments."""
        # Higher confidence for smaller adjustments and better base calibration
        base_confidence = 1.0 - base_calibration_error * 2  # Scale error to confidence
        adjustment_penalty = abs(tournament_adjustment) * 5  # Penalty for large adjustments

        confidence = base_confidence - adjustment_penalty
        return max(0.1, min(1.0, confidence))

    def _analyze_competitive_position(
        self,
        forecast: Forecast,
        tournament_context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze competitive position for the forecast."""
        position = {
            "market_position_percentile": 0.5,
            "prediction_uniqueness": 0.5,
            "timing_advantage": 0.5,
            "confidence_advantage": 0.5
        }

        if forecast.competitive_intelligence:
            ci = forecast.competitive_intelligence
            position.update({
                "market_position_percentile": ci.market_position_percentile or 0.5,
                "prediction_uniqueness": ci.prediction_uniqueness_score or 0.5,
                "timing_advantage": ci.timing_advantage_score or 0.5,
                "confidence_advantage": ci.confidence_advantage_score or 0.5
            })

        return position

    def _generate_optimization_recommendations(
        self,
        forecast: Forecast,
        violations: List[ComplianceViolation],
        log_score_optimization: Optional[LogScoreOptimization],
        calibration_adjustments: Optional[TournamentCalibration]
    ) -> List[str]:
        """Generate optimization recommendations based on validation results."""
        recommendations = []

        # Address compliance violations
        for violation in violations:
            if violation.severity in [ComplianceStatus.CRITICAL, ComplianceStatus.VIOLATION]:
                if violation.violation_type == "low_confidence":
                    recommendations.append("Increase confidence through additional research or reduce prediction extremity")
                elif violation.violation_type == "prediction_bounds":
                    recommendations.append("Adjust prediction away from extreme bounds (0.01-0.99 range)")
                elif violation.violation_type == "missing_tournament_strategy":
                    recommendations.append("Implement tournament strategy for competitive optimization")

        # Log score optimization recommendations
        if log_score_optimization:
            if log_score_optimization.expected_improvement > 0.05:
                recommendations.append(
                    f"Apply {log_score_optimization.strategy.value} optimization strategy "
                    f"for {log_score_optimization.expected_improvement:.3f} expected log score improvement"
                )

            if abs(log_score_optimization.confidence_adjustment) > 0.1:
                direction = "increase" if log_score_optimization.confidence_adjustment > 0 else "decrease"
                recommendations.append(
                    f"Consider {direction} confidence by {abs(log_score_optimization.confidence_adjustment):.2f}"
                )

        # Calibration recommendations
        if calibration_adjustments:
            if abs(calibration_adjustments.tournament_adjustment) > 0.02:
                recommendations.append(
                    f"Apply tournament calibration adjustment of {calibration_adjustments.tournament_adjustment:.3f}"
                )

            if calibration_adjustments.confidence_multiplier != 1.0:
                recommendations.append(
                    f"Apply confidence multiplier of {calibration_adjustments.confidence_multiplier:.2f}"
                )

        # General recommendations
        if forecast.calculate_prediction_variance() > 0.2:
            recommendations.append("High ensemble disagreement - consider additional research or agent tuning")

        if not recommendations:
            recommendations.append("Performance validation passed - no specific optimizations needed")

        return recommendations

    def _log_validation_result(
        self,
        forecast: Forecast,
        result: PerformanceValidationResult
    ) -> None:
        """Log detailed validation result for transparency."""
        try:
            validation_data = {
                "validation_score": result.validation_score,
                "compliance_status": result.compliance_status.value,
                "violations_count": len(result.violations),
                "recommendations_count": len(result.optimization_recommendations),
                "competitive_position": result.competitive_position,
                "optimization_strategy": result.log_score_optimization.strategy.value if result.log_score_optimization else None,
                "calibration_adjustment": result.calibration_adjustments.tournament_adjustment if result.calibration_adjustments else None,
                "violations": [
                    {
                        "type": v.violation_type,
                        "severity": v.severity.value,
                        "description": v.description
                    }
                    for v in result.violations
                ],
                "recommendations": result.optimization_recommendations
            }

            prediction_result = {
                "probability": forecast.prediction,
                "confidence": forecast.confidence_score,
                "validation_score": result.validation_score,
                "method": "tournament_validation"
            }

            self.reasoning_logger.log_reasoning_trace(
                question_id=forecast.question_id,
                agent_name="tournament_validator",
                reasoning_data=validation_data,
                prediction_result=prediction_result
            )

        except Exception as e:
            self.logger.warning(f"Error logging validation result: {e}")

    def get_compliance_monitoring_report(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate compliance monitoring report."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

            # Filter recent violations
            recent_violations = [
                v for v in self.compliance_violations
                if v.detected_at >= cutoff_time
            ]

            # Count violations by type and severity
            violation_counts = {}
            severity_counts = {}

            for violation in recent_violations:
                violation_counts[violation.violation_type] = violation_counts.get(violation.violation_type, 0) + 1
                severity_counts[violation.severity.value] = severity_counts.get(violation.severity.value, 0) + 1

            # Calculate compliance rate
            total_validations = len(recent_violations) + 10  # Assume some successful validations
            compliance_rate = max(0.0, 1.0 - len(recent_violations) / total_validations)

            return {
                "report_timestamp": datetime.utcnow().isoformat(),
                "time_window_hours": time_window_hours,
                "compliance_rate": compliance_rate,
                "total_violations": len(recent_violations),
                "violation_counts_by_type": violation_counts,
                "violation_counts_by_severity": severity_counts,
                "critical_violations": [
                    {
                        "type": v.violation_type,
                        "description": v.description,
                        "detected_at": v.detected_at.isoformat(),
                        "resolution_required": v.resolution_required
                    }
                    for v in recent_violations
                    if v.severity == ComplianceStatus.CRITICAL
                ],
                "recommendations": self._generate_compliance_recommendations(recent_violations)
            }

        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            return {
                "error": str(e),
                "report_timestamp": datetime.utcnow().isoformat()
            }

    def _generate_compliance_recommendations(
        self,
        violations: List[ComplianceViolation]
    ) -> List[str]:
        """Generate recommendations based on compliance violations."""
        recommendations = []

        # Count violation types
        violation_types = {}
        for violation in violations:
            violation_types[violation.violation_type] = violation_types.get(violation.violation_type, 0) + 1

        # Generate recommendations based on patterns
        if violation_types.get("low_confidence", 0) > 2:
            recommendations.append("Frequent low confidence violations - review confidence calibration settings")

        if violation_types.get("prediction_bounds", 0) > 1:
            recommendations.append("Multiple extreme prediction violations - implement prediction bounds checking")

        if violation_types.get("high_variance", 0) > 2:
            recommendations.append("High ensemble disagreement pattern - review agent diversity and tuning")

        if violation_types.get("deadline_pressure", 0) > 1:
            recommendations.append("Multiple deadline pressure incidents - improve scheduling and time management")

        if not recommendations:
            recommendations.append("No significant compliance patterns detected - maintain current practices")

        return recommendations

    def apply_optimization_recommendations(
        self,
        forecast: Forecast,
        validation_result: PerformanceValidationResult
    ) -> Forecast:
        """Apply optimization recommendations to improve forecast performance."""
        try:
            # Create optimized forecast copy
            optimized_forecast = forecast

            # Apply log score optimization
            if validation_result.log_score_optimization:
                optimized_forecast = self._apply_log_score_optimization(
                    optimized_forecast, validation_result.log_score_optimization
                )

            # Apply calibration adjustments
            if validation_result.calibration_adjustments:
                optimized_forecast = self._apply_calibration_adjustments(
                    optimized_forecast, validation_result.calibration_adjustments
                )

            # Log optimization application
            self.logger.info(
                f"Applied tournament optimizations to forecast {forecast.id}: "
                f"Original prediction={forecast.prediction:.3f}, "
                f"Optimized prediction={optimized_forecast.prediction:.3f}"
            )

            return optimized_forecast

        except Exception as e:
            self.logger.error(f"Error applying optimization recommendations: {e}")
            return forecast  # Return original forecast on error

    def _apply_log_score_optimization(
        self,
        forecast: Forecast,
        optimization: LogScoreOptimization
    ) -> Forecast:
        """Apply log score optimization to forecast."""
        # Adjust prediction within bounds
        current_prediction = forecast.prediction
        lower_bound, upper_bound = optimization.probability_bounds

        # Apply optimization strategy
        if optimization.strategy == OptimizationStrategy.CONSERVATIVE:
            # Move prediction toward center
            optimized_prediction = current_prediction * 0.9 + 0.5 * 0.1
        elif optimization.strategy == OptimizationStrategy.AGGRESSIVE:
            # Maintain or enhance prediction extremity within bounds
            if current_prediction < 0.5:
                optimized_prediction = max(lower_bound, current_prediction * 0.95)
            else:
                optimized_prediction = min(upper_bound, current_prediction * 1.05)
        else:
            # Balanced approach - minor adjustment toward optimal range
            optimized_prediction = current_prediction

        # Ensure bounds compliance
        optimized_prediction = max(lower_bound, min(upper_bound, optimized_prediction))

        # Update forecast prediction
        forecast.prediction = optimized_prediction

        return forecast

    def _apply_calibration_adjustments(
        self,
        forecast: Forecast,
        calibration: TournamentCalibration
    ) -> Forecast:
        """Apply tournament calibration adjustments to forecast."""
        # Apply confidence multiplier
        forecast.confidence_score *= calibration.confidence_multiplier
        forecast.confidence_score = max(0.1, min(0.99, forecast.confidence_score))

        # Apply probability shift
        forecast.prediction += calibration.probability_shift
        forecast.prediction = max(0.01, min(0.99, forecast.prediction))

        return forecast
