"""
Performance and accuracy tracking system for tournament forecasting.
Monitors forecast accuracy, calibration metrics, and API performance.
"""
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ForecastRecord:
    """Record for tracking individual forecast performance."""
    timestamp: datetime
    question_id: str
    forecast_value: float
    confidence: float
    actual_outcome: Optional[float] = None
    brier_score: Optional[float] = None
    log_score: Optional[float] = None
    calibration_bin: Optional[int] = None
    resolution_date: Optional[datetime] = None
    agent_type: str = "ensemble"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if data['resolution_date']:
            data['resolution_date'] = self.resolution_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ForecastRecord':
        """Create from dictionary for JSON deserialization."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('resolution_date'):
            data['resolution_date'] = datetime.fromisoformat(data['resolution_date'])
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: datetime
    total_forecasts: int
    resolved_forecasts: int
    overall_brier_score: float
    overall_log_score: float
    calibration_error: float
    resolution_rate: float
    accuracy_by_confidence: Dict[str, float]
    performance_trend: str  # "improving", "stable", "declining"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
class PerformanceTracker:
    """Tracks and analyzes forecast performance and accuracy metrics."""

    def __init__(self):
        """Initialize performance tracker."""
        self.forecast_records: List[ForecastRecord] = []
        self.api_performance_records = deque(maxlen=1000)  # Last 1000 API calls

        # Data persistence
        self.data_file = Path("logs/performance_tracking.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        # Performance thresholds
        self.brier_score_threshold = 0.25  # Good performance below 0.25
        self.calibration_threshold = 0.1   # Good calibration below 0.1
        self.min_forecasts_for_analysis = 10

        self._load_existing_data()
        logger.info(f"Performance tracker initialized with {len(self.forecast_records)} records")

    def record_forecast(self, question_id: str, forecast_value: float,
                       confidence: float, agent_type: str = "ensemble") -> ForecastRecord:
        """Record a new forecast for performance tracking."""
        record = ForecastRecord(
            timestamp=datetime.now(),
            question_id=question_id,
            forecast_value=forecast_value,
            confidence=confidence,
            agent_type=agent_type
        )

        self.forecast_records.append(record)

        # Save data periodically
        if len(self.forecast_records) % 10 == 0:
            self._save_data()

        logger.debug(f"Recorded forecast for {question_id}: {forecast_value:.3f} (confidence: {confidence:.3f})")
        return record

    def update_forecast_outcome(self, question_id: str, actual_outcome: float) -> bool:
        """Update forecast with actual outcome and calculate scores."""
        # Find the most recent forecast for this question
        forecast_record = None
        for record in reversed(self.forecast_records):
            if record.question_id == question_id and record.actual_outcome is None:
                forecast_record = record
                break

        if not forecast_record:
            logger.warning(f"No unresolved forecast found for question {question_id}")
            return False

        # Update with outcome
        forecast_record.actual_outcome = actual_outcome
        forecast_record.resolution_date = datetime.now()

        # Calculate performance scores
        forecast_record.brier_score = self._calculate_brier_score(
            forecast_record.forecast_value, actual_outcome
        )
        forecast_record.log_score = self._calculate_log_score(
            forecast_record.forecast_value, actual_outcome
        )
        forecast_record.calibration_bin = self._get_calibration_bin(forecast_record.confidence)

        self._save_data()

        logger.info(f"Updated forecast outcome for {question_id}: "
                   f"Brier={forecast_record.brier_score:.4f}, Log={forecast_record.log_score:.4f}")
        return True
    def record_api_performance(self, question_id: str, api_type: str,
                              success: bool, response_time: float,
                              fallback_used: bool = False) -> Dict[str, Any]:
        """Record API performance metrics."""
        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "question_id": question_id,
            "api_type": api_type,  # "research", "forecast", "community"
            "success": success,
            "response_time": response_time,
            "fallback_used": fallback_used
        }

        self.api_performance_records.append(performance_record)

        return performance_record

    def get_performance_metrics(self, days: int = 30) -> PerformanceMetrics:
        """Get comprehensive performance metrics for specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Filter records for the specified period
        recent_records = [
            record for record in self.forecast_records
            if record.timestamp >= cutoff_date and record.actual_outcome is not None
        ]

        if len(recent_records) < self.min_forecasts_for_analysis:
            logger.warning(f"Insufficient resolved forecasts ({len(recent_records)}) for analysis")
            return self._get_empty_metrics()

        # Calculate overall metrics
        brier_scores = [r.brier_score for r in recent_records if r.brier_score is not None]
        log_scores = [r.log_score for r in recent_records if r.log_score is not None]

        overall_brier = statistics.mean(brier_scores) if brier_scores else 1.0
        overall_log = statistics.mean(log_scores) if log_scores else -1.0

        # Calculate calibration error
        calibration_error = self._calculate_calibration_error(recent_records)

        # Calculate accuracy by confidence bins
        accuracy_by_confidence = self._calculate_accuracy_by_confidence(recent_records)

        # Determine performance trend
        performance_trend = self._calculate_performance_trend(recent_records)

        # Calculate resolution rate
        total_forecasts = len([r for r in self.forecast_records if r.timestamp >= cutoff_date])
        resolution_rate = len(recent_records) / max(total_forecasts, 1)

        return PerformanceMetrics(
            timestamp=datetime.now(),
            total_forecasts=total_forecasts,
            resolved_forecasts=len(recent_records),
            overall_brier_score=overall_brier,
            overall_log_score=overall_log,
            calibration_error=calibration_error,
            resolution_rate=resolution_rate,
            accuracy_by_confidence=accuracy_by_confidence,
            performance_trend=performance_trend
        )
    def get_api_success_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get API success rate and fallback usage metrics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_api_records = [
            record for record in self.api_performance_records
            if datetime.fromisoformat(record["timestamp"]) >= cutoff_time
        ]

        if not recent_api_records:
            return {
                "total_calls": 0,
                "success_rate": 0.0,
                "fallback_rate": 0.0,
                "avg_response_time": 0.0,
                "by_api_type": {}
            }

        total_calls = len(recent_api_records)
        successful_calls = sum(1 for r in recent_api_records if r["success"])
        fallback_calls = sum(1 for r in recent_api_records if r["fallback_used"])

        # Calculate by API type
        by_api_type = defaultdict(lambda: {"calls": 0, "success": 0, "fallback": 0, "response_times": []})

        for record in recent_api_records:
            api_type = record["api_type"]
            by_api_type[api_type]["calls"] += 1
            if record["success"]:
                by_api_type[api_type]["success"] += 1
            if record["fallback_used"]:
                by_api_type[api_type]["fallback"] += 1
            by_api_type[api_type]["response_times"].append(record["response_time"])

        # Calculate metrics by API type
        api_metrics = {}
        for api_type, data in by_api_type.items():
            api_metrics[api_type] = {
                "calls": data["calls"],
                "success_rate": data["success"] / data["calls"] if data["calls"] > 0 else 0.0,
                "fallback_rate": data["fallback"] / data["calls"] if data["calls"] > 0 else 0.0,
                "avg_response_time": statistics.mean(data["response_times"]) if data["response_times"] else 0.0
            }

        return {
            "total_calls": total_calls,
            "success_rate": successful_calls / total_calls,
            "fallback_rate": fallback_calls / total_calls,
            "avg_response_time": statistics.mean([r["response_time"] for r in recent_api_records]),
            "by_api_type": api_metrics
        }

    def detect_performance_degradation(self) -> List[Dict[str, Any]]:
        """Detect performance degradation and return alerts."""
        from .alert_system import alert_system

        alerts = []

        # Get recent performance
        recent_metrics = self.get_performance_metrics(days=7)
        historical_metrics = self.get_performance_metrics(days=30)
        api_metrics = self.get_api_success_metrics(hours=24)

        # Check accuracy degradation
        if recent_metrics.resolved_forecasts >= 10:  # Need sufficient data
            accuracy_alert = alert_system.check_accuracy_degradation(
                recent_metrics.overall_brier_score,
                historical_metrics.overall_brier_score,
                recent_metrics.resolved_forecasts
            )
            if accuracy_alert:
                alerts.append(accuracy_alert.to_dict())

        # Check calibration drift
        calibration_alert = alert_system.check_calibration_drift(
            recent_metrics.calibration_error,
            recent_metrics.accuracy_by_confidence
        )
        if calibration_alert:
            alerts.append(calibration_alert.to_dict())

        # Check API performance
        historical_api = self.get_api_success_metrics(hours=168)  # 7 days
        api_alerts = alert_system.check_api_performance(
            api_metrics["success_rate"],
            api_metrics["fallback_rate"],
            api_metrics["avg_response_time"],
            historical_api["avg_response_time"]
        )
        alerts.extend([alert.to_dict() for alert in api_alerts])

        return alerts
    def _calculate_brier_score(self, forecast: float, outcome: float) -> float:
        """Calculate Brier score for a forecast."""
        return (forecast - outcome) ** 2

    def _calculate_log_score(self, forecast: float, outcome: float) -> float:
        """Calculate logarithmic score for a forecast."""
        # Avoid log(0) by clamping forecast to [0.001, 0.999]
        clamped_forecast = max(0.001, min(0.999, forecast))

        if outcome == 1.0:
            return -1 * (outcome * (1 - clamped_forecast) + (1 - outcome) * clamped_forecast)
        else:
            return -1 * (outcome * clamped_forecast + (1 - outcome) * (1 - clamped_forecast))

    def _get_calibration_bin(self, confidence: float) -> int:
        """Get calibration bin (0-9) for a confidence level."""
        return min(9, int(confidence * 10))

    def _calculate_calibration_error(self, records: List[ForecastRecord]) -> float:
        """Calculate overall calibration error."""
        if not records:
            return 0.0

        # Group by calibration bins
        bins = defaultdict(list)
        for record in records:
            if record.calibration_bin is not None and record.actual_outcome is not None:
                bins[record.calibration_bin].append(record)

        total_error = 0.0
        total_weight = 0

        for bin_idx, bin_records in bins.items():
            if len(bin_records) < 2:  # Need at least 2 records for meaningful calibration
                continue

            # Calculate average confidence and accuracy for this bin
            avg_confidence = statistics.mean([r.confidence for r in bin_records])
            avg_accuracy = statistics.mean([r.actual_outcome for r in bin_records])

            # Weighted calibration error
            weight = len(bin_records)
            error = abs(avg_confidence - avg_accuracy) * weight

            total_error += error
            total_weight += weight

        return total_error / max(total_weight, 1)

    def _calculate_accuracy_by_confidence(self, records: List[ForecastRecord]) -> Dict[str, float]:
        """Calculate accuracy by confidence bins."""
        bins = defaultdict(list)

        for record in records:
            if record.actual_outcome is not None:
                bin_name = f"{int(record.confidence * 10) * 10}-{int(record.confidence * 10) * 10 + 10}%"
                bins[bin_name].append(record.actual_outcome)

        return {
            bin_name: statistics.mean(outcomes)
            for bin_name, outcomes in bins.items()
            if len(outcomes) >= 2
        }
    def _calculate_performance_trend(self, records: List[ForecastRecord]) -> str:
        """Calculate performance trend over time."""
        if len(records) < 20:  # Need sufficient data for trend analysis
            return "insufficient_data"

        # Sort by timestamp
        sorted_records = sorted(records, key=lambda r: r.timestamp)

        # Split into two halves
        mid_point = len(sorted_records) // 2
        first_half = sorted_records[:mid_point]
        second_half = sorted_records[mid_point:]

        # Calculate average Brier scores for each half
        first_half_brier = statistics.mean([r.brier_score for r in first_half if r.brier_score is not None])
        second_half_brier = statistics.mean([r.brier_score for r in second_half if r.brier_score is not None])

        # Determine trend (lower Brier score is better)
        improvement_threshold = 0.02  # 2% improvement threshold

        if first_half_brier - second_half_brier > improvement_threshold:
            return "improving"
        elif second_half_brier - first_half_brier > improvement_threshold:
            return "declining"
        else:
            return "stable"

    def _get_empty_metrics(self) -> PerformanceMetrics:
        """Get empty metrics when insufficient data is available."""
        return PerformanceMetrics(
            timestamp=datetime.now(),
            total_forecasts=0,
            resolved_forecasts=0,
            overall_brier_score=1.0,
            overall_log_score=-1.0,
            calibration_error=0.0,
            resolution_rate=0.0,
            accuracy_by_confidence={},
            performance_trend="insufficient_data"
        )

    def _save_data(self):
        """Save performance tracking data to file."""
        try:
            data = {
                "forecast_records": [record.to_dict() for record in self.forecast_records],
                "api_performance_records": list(self.api_performance_records),
                "last_updated": datetime.now().isoformat()
            }

            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")

    def _load_existing_data(self):
        """Load existing performance data if available."""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r') as f:
                    data = json.load(f)

                # Load forecast records
                records_data = data.get("forecast_records", [])
                self.forecast_records = [ForecastRecord.from_dict(record) for record in records_data]

                # Load API performance records
                api_records = data.get("api_performance_records", [])
                self.api_performance_records.extend(api_records)

                logger.info(f"Loaded {len(self.forecast_records)} forecast records and "
                           f"{len(self.api_performance_records)} API performance records")

        except Exception as e:
            logger.warning(f"Failed to load existing performance data: {e}")

    def log_performance_summary(self):
        """Log comprehensive performance summary."""
        metrics = self.get_performance_metrics()
        api_metrics = self.get_api_success_metrics()

        logger.info("=== Performance Summary ===")
        logger.info(f"Forecasts: {metrics.resolved_forecasts}/{metrics.total_forecasts} resolved "
                   f"({metrics.resolution_rate:.1%})")
        logger.info(f"Brier Score: {metrics.overall_brier_score:.4f}")
        logger.info(f"Log Score: {metrics.overall_log_score:.4f}")
        logger.info(f"Calibration Error: {metrics.calibration_error:.4f}")
        logger.info(f"Performance Trend: {metrics.performance_trend}")

        logger.info("--- API Performance ---")
        logger.info(f"Success Rate: {api_metrics['success_rate']:.1%}")
        logger.info(f"Fallback Rate: {api_metrics['fallback_rate']:.1%}")
        logger.info(f"Avg Response Time: {api_metrics['avg_response_time']:.2f}s")


# Global performance tracker instance
performance_tracker = PerformanceTracker()
