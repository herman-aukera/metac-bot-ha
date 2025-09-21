"""Tests for CalibrationTracker service."""

from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest

from src.domain.entities.prediction import (
    Prediction,
    PredictionConfidence,
    PredictionMethod,
)
from src.domain.services.calibration_service import (
    CalibrationBin,
    CalibrationDriftAlert,
    CalibrationDriftSeverity,
    CalibrationMetrics,
    CalibrationTracker,
)


class TestCalibrationBin:
    """Test calibration bin functionality."""

    def test_calibration_bin_creation(self):
        """Test creating calibration bin."""
        bin = CalibrationBin((0.0, 0.1))

        assert bin.confidence_range == (0.0, 0.1)
        assert bin.bin_center == 0.05
        assert bin.count == 0
        assert bin.observed_frequency == 0.0
        assert bin.average_predicted_probability == 0.05

    def test_calibration_bin_with_predictions(self):
        """Test calibration bin with predictions."""
        bin = CalibrationBin((0.6, 0.7))

        # Add some predictions
        bin.add_prediction(0.65, 1)  # Correct prediction
        bin.add_prediction(0.68, 0)  # Incorrect prediction
        bin.add_prediction(0.62, 1)  # Correct prediction

        assert bin.count == 3
        assert bin.observed_frequency == 2 / 3  # 2 out of 3 correct
        assert abs(bin.average_predicted_probability - 0.65) < 0.01


class TestCalibrationMetrics:
    """Test calibration metrics data structure."""

    def test_calibration_metrics_creation(self):
        """Test creating calibration metrics."""
        bins = [CalibrationBin((0.0, 0.1)), CalibrationBin((0.1, 0.2))]

        metrics = CalibrationMetrics(
            brier_score=0.15,
            calibration_error=0.08,
            reliability=0.08,
            resolution=0.05,
            uncertainty=0.25,
            sharpness=0.3,
            calibration_bins=bins,
            measurement_timestamp=datetime.utcnow(),
            time_window_days=30,
            drift_severity=CalibrationDriftSeverity.MILD,
            drift_score=0.06,
        )

        assert metrics.brier_score == 0.15
        assert metrics.calibration_error == 0.08
        assert metrics.drift_severity == CalibrationDriftSeverity.MILD
        assert len(metrics.calibration_bins) == 2

    def test_calibration_summary(self):
        """Test calibration summary generation."""
        bins = [CalibrationBin((0.0, 0.1))]
        bins[0].add_prediction(0.05, 1)

        metrics = CalibrationMetrics(
            brier_score=0.15,
            calibration_error=0.08,
            reliability=0.08,
            resolution=0.05,
            uncertainty=0.25,
            sharpness=0.3,
            calibration_bins=bins,
            measurement_timestamp=datetime.utcnow(),
            time_window_days=30,
            drift_severity=CalibrationDriftSeverity.MILD,
            drift_score=0.06,
        )

        summary = metrics.get_calibration_summary()

        assert summary["brier_score"] == 0.15
        assert summary["calibration_error"] == 0.08
        assert summary["drift_severity"] == "mild"
        assert summary["sample_size"] == 1


class TestCalibrationDriftAlert:
    """Test calibration drift alert."""

    def test_drift_alert_creation(self):
        """Test creating drift alert."""
        alert = CalibrationDriftAlert(
            severity=CalibrationDriftSeverity.MODERATE,
            drift_score=0.12,
            affected_categories=["binary", "numeric"],
            detection_timestamp=datetime.utcnow(),
            recommended_actions=["Recalibrate models", "Review thresholds"],
            current_calibration_error=0.15,
            baseline_calibration_error=0.08,
            error_increase=0.07,
        )

        assert alert.severity == CalibrationDriftSeverity.MODERATE
        assert alert.drift_score == 0.12
        assert len(alert.affected_categories) == 2
        assert len(alert.recommended_actions) == 2

    def test_alert_summary(self):
        """Test alert summary generation."""
        alert = CalibrationDriftAlert(
            severity=CalibrationDriftSeverity.SEVERE,
            drift_score=0.18,
            affected_categories=["binary"],
            detection_timestamp=datetime.utcnow(),
            recommended_actions=["Immediate recalibration"],
            current_calibration_error=0.20,
            baseline_calibration_error=0.05,
            error_increase=0.15,
        )

        summary = alert.get_alert_summary()

        assert summary["severity"] == "severe"
        assert summary["drift_score"] == 0.18
        assert summary["error_increase"] == 0.15
        assert "Immediate recalibration" in summary["recommended_actions"]


class TestCalibrationTracker:
    """Test calibration tracker service."""

    @pytest.fixture
    def tracker(self):
        """Create calibration tracker instance."""
        return CalibrationTracker(num_bins=5, min_samples_per_bin=2)

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        predictions = []
        probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]

        for i, prob in enumerate(probabilities):
            prediction = Prediction.create_binary_prediction(
                question_id=uuid4(),
                research_report_id=uuid4(),
                probability=prob,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning=f"Prediction {i}",
                created_by="test_agent",
            )
            predictions.append(prediction)

        return predictions

    @pytest.fixture
    def sample_outcomes(self):
        """Create sample outcomes for testing."""
        # Outcomes that roughly match probabilities for good calibration
        return [0, 0, 1, 1, 1]

    def test_calculate_calibration_metrics_basic(
        self, tracker, sample_predictions, sample_outcomes
    ):
        """Test basic calibration metrics calculation."""
        metrics = tracker.calculate_calibration_metrics(
            sample_predictions, sample_outcomes
        )

        assert isinstance(metrics, CalibrationMetrics)
        assert 0.0 <= metrics.brier_score <= 1.0
        assert 0.0 <= metrics.calibration_error <= 1.0
        assert 0.0 <= metrics.resolution <= 1.0
        assert 0.0 <= metrics.uncertainty <= 1.0
        assert 0.0 <= metrics.sharpness <= 1.0
        assert len(metrics.calibration_bins) == 5

    def test_calculate_calibration_metrics_perfect_calibration(self, tracker):
        """Test calibration metrics with perfect calibration."""
        # Create perfectly calibrated predictions
        predictions = []
        outcomes = []

        # 10% confidence, 10% success rate
        for _ in range(10):
            pred = Prediction.create_binary_prediction(
                question_id=uuid4(),
                research_report_id=uuid4(),
                probability=0.1,
                confidence=PredictionConfidence.LOW,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Low confidence",
                created_by="test_agent",
            )
            predictions.append(pred)
            outcomes.append(0)  # All fail

        # Add one success to make it 10%
        outcomes[0] = 1

        # 90% confidence, 90% success rate
        for _ in range(10):
            pred = Prediction.create_binary_prediction(
                question_id=uuid4(),
                research_report_id=uuid4(),
                probability=0.9,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="High confidence",
                created_by="test_agent",
            )
            predictions.append(pred)
            outcomes.append(1)  # Most succeed

        # Make one fail to get 90%
        outcomes[-1] = 0

        metrics = tracker.calculate_calibration_metrics(predictions, outcomes)

        # Should have low calibration error for well-calibrated predictions
        assert metrics.calibration_error < 0.2

    def test_calculate_calibration_metrics_poor_calibration(self, tracker):
        """Test calibration metrics with poor calibration."""
        # Create poorly calibrated predictions (overconfident)
        predictions = []
        outcomes = []

        # High confidence but low success rate
        for _ in range(10):
            pred = Prediction.create_binary_prediction(
                question_id=uuid4(),
                research_report_id=uuid4(),
                probability=0.9,  # Very confident
                confidence=PredictionConfidence.VERY_HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Very confident",
                created_by="test_agent",
            )
            predictions.append(pred)
            outcomes.append(0)  # But all fail

        metrics = tracker.calculate_calibration_metrics(predictions, outcomes)

        # Should have high calibration error for poorly calibrated predictions
        assert metrics.calibration_error > 0.5

    def test_calculate_calibration_metrics_empty_input(self, tracker):
        """Test calibration metrics with empty input."""
        with pytest.raises(ValueError, match="No predictions provided"):
            tracker.calculate_calibration_metrics([], [])

    def test_calculate_calibration_metrics_mismatched_lengths(self, tracker):
        """Test calibration metrics with mismatched input lengths."""
        predictions = [Mock()]
        outcomes = [0, 1]  # Different length

        with pytest.raises(ValueError, match="same length"):
            tracker.calculate_calibration_metrics(predictions, outcomes)

    def test_detect_calibration_drift_no_baseline(
        self, tracker, sample_predictions, sample_outcomes
    ):
        """Test drift detection with no baseline."""
        alert = tracker.detect_calibration_drift(sample_predictions, sample_outcomes)

        # Should return None when no baseline exists
        assert alert is None

    def test_detect_calibration_drift_with_baseline(self, tracker):
        """Test drift detection with established baseline."""
        # Create baseline
        baseline_predictions = []
        baseline_outcomes = []

        for i in range(10):
            pred = Prediction.create_binary_prediction(
                question_id=uuid4(),
                research_report_id=uuid4(),
                probability=0.5,
                confidence=PredictionConfidence.MEDIUM,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Baseline",
                created_by="test_agent",
            )
            baseline_predictions.append(pred)
            baseline_outcomes.append(i % 2)  # 50% success rate

        # Establish baseline
        tracker.calculate_calibration_metrics(baseline_predictions, baseline_outcomes)

        # Create drifted predictions (overconfident)
        drifted_predictions = []
        drifted_outcomes = []

        for i in range(10):
            pred = Prediction.create_binary_prediction(
                question_id=uuid4(),
                research_report_id=uuid4(),
                probability=0.9,  # Much more confident
                confidence=PredictionConfidence.VERY_HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Overconfident",
                created_by="test_agent",
            )
            drifted_predictions.append(pred)
            drifted_outcomes.append(i % 2)  # Still 50% success rate

        alert = tracker.detect_calibration_drift(drifted_predictions, drifted_outcomes)

        # Should detect drift
        assert alert is not None
        assert alert.severity != CalibrationDriftSeverity.NONE
        assert alert.drift_score > 0.0
        assert len(alert.recommended_actions) > 0

    def test_apply_calibration_correction(
        self, tracker, sample_predictions, sample_outcomes
    ):
        """Test applying calibration correction to predictions."""
        # Establish calibration baseline
        tracker.calculate_calibration_metrics(sample_predictions, sample_outcomes)

        # Create prediction to correct
        prediction = Prediction.create_binary_prediction(
            question_id=uuid4(),
            research_report_id=uuid4(),
            probability=0.8,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Original prediction",
            created_by="test_agent",
        )

        corrected = tracker.apply_calibration_correction(prediction)

        assert corrected.id != prediction.id  # New prediction
        assert "calibrated" in corrected.created_by
        assert "Calibration-corrected" in corrected.reasoning
        assert corrected.method_metadata.get("calibration_correction_applied") is True
        assert corrected.method_metadata.get("original_probability") == 0.8

    def test_apply_calibration_correction_non_binary(self, tracker):
        """Test calibration correction with non-binary prediction."""
        # Create non-binary prediction
        prediction = Prediction.create_numeric_prediction(
            question_id=uuid4(),
            research_report_id=uuid4(),
            value=42.0,
            confidence=PredictionConfidence.HIGH,
            method=PredictionMethod.CHAIN_OF_THOUGHT,
            reasoning="Numeric prediction",
            created_by="test_agent",
        )

        corrected = tracker.apply_calibration_correction(prediction)

        # Should return original prediction unchanged
        assert corrected == prediction

    def test_get_calibration_report_no_data(self, tracker):
        """Test calibration report with no data."""
        report = tracker.get_calibration_report()

        assert "error" in report
        assert "No calibration data available" in report["error"]

    def test_get_calibration_report_with_data(
        self, tracker, sample_predictions, sample_outcomes
    ):
        """Test calibration report generation with data."""
        # Generate some calibration data
        tracker.calculate_calibration_metrics(sample_predictions, sample_outcomes)

        # Add more data with different timestamp
        tracker.calculate_calibration_metrics(sample_predictions, sample_outcomes)

        report = tracker.get_calibration_report()

        assert "summary" in report
        assert "trends" in report
        assert "category_analysis" in report
        assert "drift_analysis" in report
        assert "recommendations" in report
        assert "report_timestamp" in report

        # Check summary content
        summary = report["summary"]
        assert "average_brier_score" in summary
        assert "average_calibration_error" in summary
        assert "total_predictions" in summary

    def test_update_calibration_thresholds(self, tracker):
        """Test updating calibration thresholds based on performance."""
        original_mild_threshold = tracker.drift_thresholds[
            CalibrationDriftSeverity.MILD
        ]

        # High accuracy should make thresholds more sensitive
        performance_data = {"accuracy": 0.85}
        tracker.update_calibration_thresholds(performance_data)

        new_mild_threshold = tracker.drift_thresholds[CalibrationDriftSeverity.MILD]
        assert new_mild_threshold < original_mild_threshold

    def test_brier_score_calculation(self, tracker):
        """Test Brier score calculation."""
        predicted_probs = [0.1, 0.5, 0.9]
        actual_outcomes = [0, 1, 1]

        brier_score = tracker._calculate_brier_score(predicted_probs, actual_outcomes)

        # Manual calculation: ((0.1-0)^2 + (0.5-1)^2 + (0.9-1)^2) / 3
        expected = (0.01 + 0.25 + 0.01) / 3
        assert abs(brier_score - expected) < 0.001

    def test_create_calibration_bins(self, tracker):
        """Test calibration bin creation."""
        predicted_probs = [0.05, 0.15, 0.25, 0.85, 0.95]
        actual_outcomes = [0, 0, 1, 1, 1]

        bins = tracker._create_calibration_bins(predicted_probs, actual_outcomes)

        assert len(bins) == 5  # num_bins = 5

        # Check that predictions are assigned to correct bins
        # With 5 bins, bin width is 0.2: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
        assert bins[0].count == 2  # 0.05 and 0.15 go to first bin [0-0.2)
        assert bins[1].count == 1  # 0.25 goes to second bin [0.2-0.4)
        assert bins[4].count == 2  # 0.85 and 0.95 go to last bin [0.8-1.0]

    def test_calibration_error_calculation(self, tracker):
        """Test Expected Calibration Error calculation."""
        # Create bins with known calibration error
        bins = []

        # Perfect calibration bin
        perfect_bin = CalibrationBin((0.4, 0.6))
        perfect_bin.add_prediction(0.5, 1)
        perfect_bin.add_prediction(0.5, 0)  # 50% success rate, 50% predicted
        bins.append(perfect_bin)

        # Poor calibration bin
        poor_bin = CalibrationBin((0.8, 1.0))
        poor_bin.add_prediction(0.9, 0)
        poor_bin.add_prediction(0.9, 0)  # 0% success rate, 90% predicted
        bins.append(poor_bin)

        calibration_error = tracker._calculate_calibration_error(bins)

        # Should be weighted average of bin errors
        # Perfect bin: |0.5 - 0.5| = 0.0, weight = 2/4 = 0.5
        # Poor bin: |0.9 - 0.0| = 0.9, weight = 2/4 = 0.5
        # Expected: 0.5 * 0.0 + 0.5 * 0.9 = 0.45
        assert abs(calibration_error - 0.45) < 0.01

    def test_resolution_calculation(self, tracker):
        """Test resolution calculation."""
        # Create test data
        actual_outcomes = [0, 0, 1, 1]  # 50% base rate

        bins = []

        # Bin with different base rate
        bin1 = CalibrationBin((0.0, 0.5))
        bin1.add_prediction(0.2, 0)
        bin1.add_prediction(0.3, 0)  # 0% success rate
        bins.append(bin1)

        bin2 = CalibrationBin((0.5, 1.0))
        bin2.add_prediction(0.8, 1)
        bin2.add_prediction(0.9, 1)  # 100% success rate
        bins.append(bin2)

        resolution = tracker._calculate_resolution(bins, actual_outcomes)

        # Should be positive since bins have different base rates than overall
        assert resolution > 0.0

    def test_uncertainty_calculation(self, tracker):
        """Test uncertainty calculation."""
        # 50% base rate should give maximum uncertainty
        outcomes_50 = [0, 1, 0, 1]
        uncertainty_50 = tracker._calculate_uncertainty(outcomes_50)
        assert abs(uncertainty_50 - 0.25) < 0.01  # 0.5 * (1 - 0.5) = 0.25

        # 100% base rate should give zero uncertainty
        outcomes_100 = [1, 1, 1, 1]
        uncertainty_100 = tracker._calculate_uncertainty(outcomes_100)
        assert uncertainty_100 == 0.0

    def test_sharpness_calculation(self, tracker):
        """Test sharpness calculation."""
        # Predictions close to 0.5 should have low sharpness
        low_sharpness_probs = [0.45, 0.5, 0.55]
        low_sharpness = tracker._calculate_sharpness(low_sharpness_probs)

        # Predictions far from 0.5 should have high sharpness
        high_sharpness_probs = [0.1, 0.9, 0.05]
        high_sharpness = tracker._calculate_sharpness(high_sharpness_probs)

        assert high_sharpness > low_sharpness

    def test_drift_severity_classification(self, tracker):
        """Test drift severity classification."""
        # Test different drift scores
        assert tracker._classify_drift_severity(0.01) == CalibrationDriftSeverity.NONE
        assert tracker._classify_drift_severity(0.06) == CalibrationDriftSeverity.MILD
        assert (
            tracker._classify_drift_severity(0.12) == CalibrationDriftSeverity.MODERATE
        )
        assert tracker._classify_drift_severity(0.18) == CalibrationDriftSeverity.SEVERE
        assert (
            tracker._classify_drift_severity(0.30) == CalibrationDriftSeverity.CRITICAL
        )

    def test_drift_recommendations_generation(self, tracker):
        """Test drift recommendation generation."""
        # Create mock metrics
        current_metrics = Mock()
        current_metrics.calibration_error = 0.15
        current_metrics.sharpness = 0.2

        baseline_metrics = Mock()
        baseline_metrics.calibration_error = 0.08
        baseline_metrics.sharpness = 0.3

        recommendations = tracker._generate_drift_recommendations(
            current_metrics, baseline_metrics, CalibrationDriftSeverity.SEVERE
        )

        assert len(recommendations) > 0
        assert any("recalibration" in rec.lower() for rec in recommendations)

    def test_calibration_correction_factor(
        self, tracker, sample_predictions, sample_outcomes
    ):
        """Test calibration correction factor calculation."""
        # Establish calibration data
        tracker.calculate_calibration_metrics(sample_predictions, sample_outcomes)

        # Test correction factor for different probabilities
        factor_low = tracker._get_calibration_correction_factor(0.1, None)
        factor_high = tracker._get_calibration_correction_factor(0.9, None)

        # Should return valid correction factors
        assert 0.0 <= factor_low <= 2.0
        assert 0.0 <= factor_high <= 2.0

    def test_apply_correction_factor(self, tracker):
        """Test applying correction factor to probability."""
        # Test various correction scenarios
        corrected_up = tracker._apply_correction_factor(0.5, 1.5)
        assert corrected_up == 0.75

        corrected_down = tracker._apply_correction_factor(0.8, 0.5)
        assert corrected_down == 0.4

        # Test clamping
        corrected_clamp_low = tracker._apply_correction_factor(0.1, 0.1)
        assert abs(corrected_clamp_low - 0.01) < 1e-10  # Clamped to minimum

        corrected_clamp_high = tracker._apply_correction_factor(0.9, 1.5)
        assert corrected_clamp_high == 0.99  # Clamped to maximum

    def test_trend_calculation(self, tracker):
        """Test trend calculation for value series."""
        # Increasing trend
        increasing_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert tracker._calculate_trend(increasing_values) == "increasing"

        # Decreasing trend
        decreasing_values = [0.5, 0.4, 0.3, 0.2, 0.1]
        assert tracker._calculate_trend(decreasing_values) == "decreasing"

        # Stable trend
        stable_values = [0.3, 0.31, 0.29, 0.3, 0.3]
        assert tracker._calculate_trend(stable_values) == "stable"

        # Insufficient data
        insufficient_values = [0.5]
        assert tracker._calculate_trend(insufficient_values) == "insufficient_data"

    def test_category_specific_calibration(self, tracker):
        """Test category-specific calibration tracking."""
        predictions = []
        outcomes = []

        # Create predictions for different categories
        for i in range(5):
            pred = Prediction.create_binary_prediction(
                question_id=uuid4(),
                research_report_id=uuid4(),
                probability=0.7,
                confidence=PredictionConfidence.HIGH,
                method=PredictionMethod.CHAIN_OF_THOUGHT,
                reasoning="Category A prediction",
                created_by="test_agent",
            )
            predictions.append(pred)
            outcomes.append(1)

        # Calculate metrics for specific category
        metrics = tracker.calculate_calibration_metrics(
            predictions, outcomes, category="category_a"
        )

        assert "category_a" in tracker.category_calibration
        assert len(tracker.category_calibration["category_a"]) == 1
        assert tracker.category_calibration["category_a"][0] == metrics

    def test_calibration_report_with_categories(self, tracker):
        """Test calibration report with category analysis."""
        # Create predictions for multiple categories
        for category in ["cat_a", "cat_b"]:
            predictions = []
            outcomes = []

            for i in range(3):
                pred = Prediction.create_binary_prediction(
                    question_id=uuid4(),
                    research_report_id=uuid4(),
                    probability=0.6,
                    confidence=PredictionConfidence.MEDIUM,
                    method=PredictionMethod.CHAIN_OF_THOUGHT,
                    reasoning=f"{category} prediction",
                    created_by="test_agent",
                )
                predictions.append(pred)
                outcomes.append(i % 2)

            tracker.calculate_calibration_metrics(
                predictions, outcomes, category=category
            )

        report = tracker.get_calibration_report(include_categories=True)

        assert "category_analysis" in report
        category_analysis = report["category_analysis"]
        assert "cat_a" in category_analysis
        assert "cat_b" in category_analysis

        for category in ["cat_a", "cat_b"]:
            assert "average_calibration_error" in category_analysis[category]
            assert "measurements_count" in category_analysis[category]
