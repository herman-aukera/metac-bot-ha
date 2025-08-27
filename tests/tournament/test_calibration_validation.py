"""Calibration accuracy validation and bias detection testing."""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass
import statistics
import math
import random

from src.domain.entities.question import Question, QuestionType
from src.domain.entities.forecast import Forecast
from src.domain.value_objects.probability import Probability
from src.domain.value_objects.confidence import Confidence
from src.agents.base_agent import BaseAgent


@dataclass
class BiasTestScenario:
    """Defines a bias testing scenario."""
    name: str
    bias_type: str  # "overconfidence", "underconfidence", "anchoring", "availability", "confirmation"
    expected_bias_strength: float  # 0.0 to 1.0
    test_questions: List[Dict[str, Any]]
    detection_threshold: float


class CalibrationValidator:
    """Validates agent calibration accuracy and detects biases."""

    def __init__(self):
        self.calibration_history = {}
        self.bias_detection_results = {}

    def validate_calibration_accuracy(
        self,
        forecasts: List[Forecast],
        ground_truth: Dict[int, bool],
        validation_method: str = "reliability_diagram"
    ) -> Dict[str, Any]:
        """Validate calibration accuracy using various methods."""

        if not forecasts:
            return {"error": "no_forecasts"}

        # Extract predictions and outcomes
        predictions = []
        outcomes = []
        confidences = []

        for forecast in forecasts:
            if forecast.question_id in ground_truth:
                predictions.append(forecast.prediction.value)
                outcomes.append(1.0 if ground_truth[forecast.question_id] else 0.0)
                confidences.append(forecast.confidence.value)

        if not predictions:
            return {"error": "no_matching_outcomes"}

        validation_results = {
            "method": validation_method,
            "total_forecasts": len(predictions),
            "calibration_metrics": {},
            "reliability_analysis": {},
            "sharpness_analysis": {},
            "resolution_analysis": {}
        }

        # Calculate calibration metrics
        validation_results["calibration_metrics"] = self._calculate_calibration_metrics(
            predictions, outcomes, confidences
        )

        # Perform reliability analysis
        if validation_method in ["reliability_diagram", "all"]:
            validation_results["reliability_analysis"] = self._analyze_reliability(
                predictions, outcomes
            )

        # Perform sharpness analysis
        if validation_method in ["sharpness", "all"]:
            validation_results["sharpness_analysis"] = self._analyze_sharpness(
                predictions, confidences
            )

        # Perform resolution analysis
        if validation_method in ["resolution", "all"]:
            validation_results["resolution_analysis"] = self._analyze_resolution(
                predictions, outcomes
            )

        return validation_results

    def detect_prediction_biases(
        self,
        agent: BaseAgent,
        bias_scenarios: List[BiasTestScenario]
    ) -> Dict[str, Any]:
        """Detect various prediction biases in agent behavior."""

        bias_results = {
            "agent_type": agent.__class__.__name__,
            "bias_tests": {},
            "overall_bias_score": 0.0,
            "detected_biases": [],
            "bias_mitigation_recommendations": []
        }

        for scenario in bias_scenarios:
            test_result = asyncio.run(self._test_specific_bias(agent, scenario))
            bias_results["bias_tests"][scenario.name] = test_result

            if test_result["bias_detected"]:
                bias_results["detected_biases"].append({
                    "bias_type": scenario.bias_type,
                    "strength": test_result["bias_strength"],
                    "confidence": test_result["detection_confidence"]
                })

        # Calculate overall bias score
        bias_results["overall_bias_score"] = self._calculate_overall_bias_score(
            bias_results["bias_tests"]
        )

        # Generate mitigation recommendations
        bias_results["bias_mitigation_recommendations"] = self._generate_bias_mitigation_recommendations(
            bias_results["detected_biases"]
        )

        return bias_results

    def _calculate_calibration_metrics(
        self,
        predictions: List[float],
        outcomes: List[float],
        confidences: List[float]
    ) -> Dict[str, float]:
        """Calculate comprehensive calibration metrics."""

        # Brier Score
        brier_score = statistics.mean([(p - o) ** 2 for p, o in zip(predictions, outcomes)])

        # Brier Score Decomposition
        base_rate = statistics.mean(outcomes)
        reliability = self._calculate_reliability(predictions, outcomes)
        resolution = self._calculate_resolution(predictions, outcomes, base_rate)

        # Expected Calibration Error (ECE)
        ece = self._calculate_expected_calibration_error(predictions, outcomes)

        # Maximum Calibration Error (MCE)
        mce = self._calculate_maximum_calibration_error(predictions, outcomes)

        # Overconfidence/Underconfidence
        confidence_bias = self._calculate_confidence_bias(predictions, outcomes, confidences)

        return {
            "brier_score": brier_score,
            "reliability": reliability,
            "resolution": resolution,
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "overconfidence": max(0, confidence_bias),
            "underconfidence": max(0, -confidence_bias),
            "calibration_score": 1.0 - ece  # Higher is better
        }

    def _analyze_reliability(self, predictions: List[float], outcomes: List[float]) -> Dict[str, Any]:
        """Analyze reliability using reliability diagrams."""

        # Create bins for reliability diagram
        num_bins = 10
        bins = []

        for i in range(num_bins):
            bin_start = i / num_bins
            bin_end = (i + 1) / num_bins

            bin_predictions = []
            bin_outcomes = []

            for pred, outcome in zip(predictions, outcomes):
                if bin_start <= pred < bin_end or (i == num_bins - 1 and pred == 1.0):
                    bin_predictions.append(pred)
                    bin_outcomes.append(outcome)

            if bin_predictions:
                avg_prediction = statistics.mean(bin_predictions)
                avg_outcome = statistics.mean(bin_outcomes)
                bin_size = len(bin_predictions)

                bins.append({
                    "bin_range": (bin_start, bin_end),
                    "count": bin_size,
                    "avg_prediction": avg_prediction,
                    "avg_outcome": avg_outcome,
                    "calibration_error": abs(avg_prediction - avg_outcome),
                    "proportion": bin_size / len(predictions)
                })

        # Calculate reliability metrics
        total_predictions = len(predictions)
        weighted_calibration_error = sum(
            bin_data["count"] * bin_data["calibration_error"]
            for bin_data in bins
        ) / total_predictions

        return {
            "bins": bins,
            "weighted_calibration_error": weighted_calibration_error,
            "perfect_calibration": weighted_calibration_error < 0.05,
            "reliability_score": 1.0 - weighted_calibration_error
        }

    def _analyze_sharpness(self, predictions: List[float], confidences: List[float]) -> Dict[str, Any]:
        """Analyze prediction sharpness (discrimination ability)."""

        # Calculate prediction variance (higher = more sharp)
        prediction_variance = statistics.variance(predictions) if len(predictions) > 1 else 0

        # Calculate confidence variance
        confidence_variance = statistics.variance(confidences) if len(confidences) > 1 else 0

        # Calculate sharpness score
        sharpness_score = min(1.0, prediction_variance * 4)  # Normalize to 0-1

        # Analyze prediction distribution
        prediction_bins = [0] * 10
        for pred in predictions:
            bin_idx = min(9, int(pred * 10))
            prediction_bins[bin_idx] += 1

        # Calculate entropy (lower entropy = more sharp)
        total_preds = len(predictions)
        entropy = -sum(
            (count / total_preds) * math.log(count / total_preds)
            for count in prediction_bins if count > 0
        )
        max_entropy = math.log(10)  # Maximum possible entropy for 10 bins
        normalized_entropy = entropy / max_entropy

        return {
            "prediction_variance": prediction_variance,
            "confidence_variance": confidence_variance,
            "sharpness_score": sharpness_score,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "discrimination_ability": 1.0 - normalized_entropy,
            "prediction_distribution": prediction_bins
        }

    def _analyze_resolution(self, predictions: List[float], outcomes: List[float]) -> Dict[str, Any]:
        """Analyze resolution (ability to discriminate between different outcomes)."""

        base_rate = statistics.mean(outcomes)

        # Group predictions by outcome
        positive_predictions = [p for p, o in zip(predictions, outcomes) if o == 1.0]
        negative_predictions = [p for p, o in zip(predictions, outcomes) if o == 0.0]

        # Calculate resolution components
        if positive_predictions and negative_predictions:
            pos_mean = statistics.mean(positive_predictions)
            neg_mean = statistics.mean(negative_predictions)

            # Resolution is the ability to separate positive and negative cases
            resolution_score = abs(pos_mean - neg_mean)

            # Calculate discrimination slope (ROC-like measure)
            discrimination_slope = pos_mean - neg_mean

        else:
            pos_mean = statistics.mean(positive_predictions) if positive_predictions else base_rate
            neg_mean = statistics.mean(negative_predictions) if negative_predictions else base_rate
            resolution_score = 0.0
            discrimination_slope = 0.0

        return {
            "base_rate": base_rate,
            "positive_case_avg_prediction": pos_mean,
            "negative_case_avg_prediction": neg_mean,
            "resolution_score": resolution_score,
            "discrimination_slope": discrimination_slope,
            "good_discrimination": resolution_score > 0.1
        }

    async def _test_specific_bias(self, agent: BaseAgent, scenario: BiasTestScenario) -> Dict[str, Any]:
        """Test for a specific type of bias."""

        bias_indicators = []
        forecasts = []

        # Generate forecasts for bias test questions
        for question_data in scenario.test_questions:
            question = self._create_question(question_data)

            try:
                forecast = await agent.generate_forecast(question)
                forecasts.append(forecast)
            except Exception as e:
                continue  # Skip failed forecasts

        if not forecasts:
            return {"error": "no_forecasts_generated"}

        # Analyze for specific bias type
        if scenario.bias_type == "overconfidence":
            bias_indicators = self._detect_overconfidence_bias(forecasts, scenario.test_questions)
        elif scenario.bias_type == "underconfidence":
            bias_indicators = self._detect_underconfidence_bias(forecasts, scenario.test_questions)
        elif scenario.bias_type == "anchoring":
            bias_indicators = self._detect_anchoring_bias(forecasts, scenario.test_questions)
        elif scenario.bias_type == "availability":
            bias_indicators = self._detect_availability_bias(forecasts, scenario.test_questions)
        elif scenario.bias_type == "confirmation":
            bias_indicators = self._detect_confirmation_bias(forecasts, scenario.test_questions)

        # Calculate bias strength and detection confidence
        bias_strength = statistics.mean(bias_indicators) if bias_indicators else 0.0
        detection_confidence = min(1.0, len(bias_indicators) / 10)  # More indicators = higher confidence

        return {
            "bias_type": scenario.bias_type,
            "bias_strength": bias_strength,
            "detection_confidence": detection_confidence,
            "bias_detected": bias_strength > scenario.detection_threshold,
            "bias_indicators": bias_indicators,
            "forecast_count": len(forecasts)
        }

    def _detect_overconfidence_bias(self, forecasts: List[Forecast], questions: List[Dict[str, Any]]) -> List[float]:
        """Detect overconfidence bias indicators."""
        indicators = []

        for forecast in forecasts:
            # High confidence with extreme predictions indicates overconfidence
            extreme_prediction = forecast.prediction.value < 0.1 or forecast.prediction.value > 0.9
            high_confidence = forecast.confidence.value > 0.8

            if extreme_prediction and high_confidence:
                indicators.append(forecast.confidence.value)

            # Very high confidence in general
            if forecast.confidence.value > 0.95:
                indicators.append(forecast.confidence.value - 0.8)  # Excess confidence

        return indicators

    def _detect_underconfidence_bias(self, forecasts: List[Forecast], questions: List[Dict[str, Any]]) -> List[float]:
        """Detect underconfidence bias indicators."""
        indicators = []

        for forecast in forecasts:
            # Low confidence with moderate predictions
            moderate_prediction = 0.3 <= forecast.prediction.value <= 0.7
            low_confidence = forecast.confidence.value < 0.5

            if moderate_prediction and low_confidence:
                indicators.append(0.5 - forecast.confidence.value)  # Confidence deficit

            # Generally low confidence
            if forecast.confidence.value < 0.3:
                indicators.append(0.5 - forecast.confidence.value)

        return indicators

    def _detect_anchoring_bias(self, forecasts: List[Forecast], questions: List[Dict[str, Any]]) -> List[float]:
        """Detect anchoring bias indicators."""
        indicators = []

        # Look for predictions clustered around common anchor points
        predictions = [f.prediction.value for f in forecasts]

        # Common anchors: 0.5, 0.25, 0.75, round numbers
        anchors = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]

        for anchor in anchors:
            nearby_predictions = [p for p in predictions if abs(p - anchor) < 0.05]
            if len(nearby_predictions) > len(predictions) * 0.3:  # 30% clustered around anchor
                indicators.extend([0.5] * len(nearby_predictions))  # Moderate bias indicator

        return indicators

    def _detect_availability_bias(self, forecasts: List[Forecast], questions: List[Dict[str, Any]]) -> List[float]:
        """Detect availability bias indicators."""
        indicators = []

        # Availability bias: overweighting recent or memorable events
        # Look for patterns in reasoning that suggest recent event focus

        for forecast in forecasts:
            reasoning = forecast.reasoning.lower()

            # Check for recent event keywords
            recent_keywords = ["recent", "recently", "latest", "just", "yesterday", "last week", "breaking"]
            memorable_keywords = ["dramatic", "shocking", "unprecedented", "historic", "major"]

            recent_mentions = sum(1 for keyword in recent_keywords if keyword in reasoning)
            memorable_mentions = sum(1 for keyword in memorable_keywords if keyword in reasoning)

            if recent_mentions > 2 or memorable_mentions > 1:
                indicators.append(0.6)  # Moderate availability bias indicator

        return indicators

    def _detect_confirmation_bias(self, forecasts: List[Forecast], questions: List[Dict[str, Any]]) -> List[float]:
        """Detect confirmation bias indicators."""
        indicators = []

        # Confirmation bias: seeking information that confirms prior beliefs
        # Look for one-sided reasoning or lack of counterarguments

        for forecast in forecasts:
            reasoning = forecast.reasoning.lower()

            # Check for balanced reasoning
            positive_words = ["support", "confirm", "evidence for", "indicates", "suggests"]
            negative_words = ["however", "but", "against", "contradicts", "challenges", "despite"]

            positive_count = sum(1 for word in positive_words if word in reasoning)
            negative_count = sum(1 for word in negative_words if word in reasoning)

            # Strong confirmation bias if only positive evidence mentioned
            if positive_count > 2 and negative_count == 0:
                indicators.append(0.7)
            elif positive_count > negative_count * 3:  # Heavily skewed toward confirmation
                indicators.append(0.5)

        return indicators

    def _create_question(self, question_data: Dict[str, Any]) -> Question:
        """Create Question entity from data."""
        return Question(
            id=question_data["id"],
            title=question_data["title"],
            description=question_data["description"],
            question_type=QuestionType(question_data["type"]),
            close_time=datetime.fromisoformat(question_data["close_time"]),
            resolve_time=datetime.fromisoformat(question_data["resolve_time"]),
            categories=question_data.get("categories", []),
            tags=question_data.get("tags", [])
        )

    def _calculate_reliability(self, predictions: List[float], outcomes: List[float]) -> float:
        """Calculate reliability component of Brier score."""
        # Implementation of reliability calculation
        return 0.0  # Placeholder

    def _calculate_resolution(self, predictions: List[float], outcomes: List[float], base_rate: float) -> float:
        """Calculate resolution component of Brier score."""
        # Implementation of resolution calculation
        return 0.0  # Placeholder

    def _calculate_expected_calibration_error(self, predictions: List[float], outcomes: List[float]) -> float:
        """Calculate Expected Calibration Error."""
        num_bins = 10
        total_error = 0.0
        total_count = len(predictions)

        for i in range(num_bins):
            bin_start = i / num_bins
            bin_end = (i + 1) / num_bins

            bin_predictions = []
            bin_outcomes = []

            for pred, outcome in zip(predictions, outcomes):
                if bin_start <= pred < bin_end or (i == num_bins - 1 and pred == 1.0):
                    bin_predictions.append(pred)
                    bin_outcomes.append(outcome)

            if bin_predictions:
                avg_prediction = statistics.mean(bin_predictions)
                avg_outcome = statistics.mean(bin_outcomes)
                bin_error = abs(avg_prediction - avg_outcome)
                bin_weight = len(bin_predictions) / total_count
                total_error += bin_weight * bin_error

        return total_error

    def _calculate_maximum_calibration_error(self, predictions: List[float], outcomes: List[float]) -> float:
        """Calculate Maximum Calibration Error."""
        num_bins = 10
        max_error = 0.0

        for i in range(num_bins):
            bin_start = i / num_bins
            bin_end = (i + 1) / num_bins

            bin_predictions = []
            bin_outcomes = []

            for pred, outcome in zip(predictions, outcomes):
                if bin_start <= pred < bin_end or (i == num_bins - 1 and pred == 1.0):
                    bin_predictions.append(pred)
                    bin_outcomes.append(outcome)

            if bin_predictions:
                avg_prediction = statistics.mean(bin_predictions)
                avg_outcome = statistics.mean(bin_outcomes)
                bin_error = abs(avg_prediction - avg_outcome)
                max_error = max(max_error, bin_error)

        return max_error

    def _calculate_confidence_bias(self, predictions: List[float], outcomes: List[float], confidences: List[float]) -> float:
        """Calculate confidence bias (positive = overconfident, negative = underconfident)."""
        if not confidences:
            return 0.0

        # Calculate actual accuracy for each prediction
        accuracies = [1.0 - abs(pred - outcome) for pred, outcome in zip(predictions, outcomes)]

        # Compare confidence to actual accuracy
        confidence_errors = [conf - acc for conf, acc in zip(confidences, accuracies)]

        return statistics.mean(confidence_errors)

    def _calculate_overall_bias_score(self, bias_tests: Dict[str, Any]) -> float:
        """Calculate overall bias score from individual bias tests."""
        if not bias_tests:
            return 0.0

        bias_strengths = []
        for test_result in bias_tests.values():
            if "bias_strength" in test_result:
                bias_strengths.append(test_result["bias_strength"])

        return statistics.mean(bias_strengths) if bias_strengths else 0.0

    def _generate_bias_mitigation_recommendations(self, detected_biases: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for mitigating detected biases."""
        recommendations = []

        for bias in detected_biases:
            bias_type = bias["bias_type"]
            strength = bias["strength"]

            if bias_type == "overconfidence" and strength > 0.5:
                recommendations.append("Implement confidence calibration training and encourage consideration of alternative scenarios")
            elif bias_type == "underconfidence" and strength > 0.5:
                recommendations.append("Provide feedback on prediction accuracy to build appropriate confidence levels")
            elif bias_type == "anchoring" and strength > 0.5:
                recommendations.append("Use structured decision-making processes and consider multiple reference points")
            elif bias_type == "availability" and strength > 0.5:
                recommendations.append("Implement systematic information gathering and base rate consideration")
            elif bias_type == "confirmation" and strength > 0.5:
                recommendations.append("Encourage devil's advocate reasoning and systematic consideration of counterevidence")

        if not recommendations:
            recommendations.append("Continue monitoring for bias patterns and maintain current practices")

        return recommendations


class TestCalibrationValidation:
    """Test calibration validation and bias detection."""

    @pytest.fixture
    def calibration_validator(self):
        """Create calibration validator."""
        return CalibrationValidator()

    @pytest.fixture
    def mock_calibrated_agent(self):
        """Create mock well-calibrated agent."""
        agent = Mock(spec=BaseAgent)
        agent.__class__.__name__ = "CalibratedAgent"

        async def calibrated_forecast(question):
            # Well-calibrated: confidence matches accuracy
            base_accuracy = 0.7
            noise = (question.id % 10) * 0.02 - 0.1  # ±0.1 noise
            prediction = max(0.1, min(0.9, base_accuracy + noise))
            confidence = base_accuracy + abs(noise) * 0.5  # Confidence reflects uncertainty

            return Forecast(
                question_id=question.id,
                prediction=Probability(prediction),
                confidence=Confidence(confidence),
                reasoning="Well-calibrated analysis with appropriate uncertainty",
                method="calibrated",
                sources=["calibrated_source"],
                metadata={"calibration_target": base_accuracy}
            )

        agent.generate_forecast = AsyncMock(side_effect=calibrated_forecast)
        return agent

    @pytest.fixture
    def mock_overconfident_agent(self):
        """Create mock overconfident agent."""
        agent = Mock(spec=BaseAgent)
        agent.__class__.__name__ = "OverconfidentAgent"

        async def overconfident_forecast(question):
            # Overconfident: high confidence, extreme predictions
            extreme_prediction = 0.1 if question.id % 2 == 0 else 0.9
            high_confidence = 0.95

            return Forecast(
                question_id=question.id,
                prediction=Probability(extreme_prediction),
                confidence=Confidence(high_confidence),
                reasoning="Very confident analysis with extreme prediction and recent dramatic events suggest high certainty",
                method="overconfident",
                sources=["biased_source"],
                metadata={"bias_type": "overconfidence"}
            )

        agent.generate_forecast = AsyncMock(side_effect=overconfident_forecast)
        return agent

    @pytest.fixture
    def test_ground_truth(self):
        """Create ground truth outcomes for testing."""
        # Generate deterministic outcomes for consistent testing
        ground_truth = {}
        for i in range(100):
            question_id = 7000 + i
            # Deterministic outcome based on question ID
            outcome = (question_id % 7) < 3  # ~43% positive rate
            ground_truth[question_id] = outcome
        return ground_truth

    @pytest.fixture
    def bias_test_scenarios(self):
        """Create bias test scenarios."""
        scenarios = []

        # Overconfidence bias scenario
        overconfidence_questions = []
        for i in range(10):
            overconfidence_questions.append({
                "id": 8000 + i,
                "title": f"Overconfidence test question {i+1}",
                "description": "Question designed to test overconfidence bias",
                "type": "binary",
                "close_time": "2025-12-01T00:00:00Z",
                "resolve_time": "2026-01-01T00:00:00Z",
                "categories": ["Test"],
                "tags": ["overconfidence"]
            })

        scenarios.append(BiasTestScenario(
            name="Overconfidence Test",
            bias_type="overconfidence",
            expected_bias_strength=0.6,
            test_questions=overconfidence_questions,
            detection_threshold=0.4
        ))

        # Anchoring bias scenario
        anchoring_questions = []
        for i in range(10):
            anchoring_questions.append({
                "id": 8100 + i,
                "title": f"Anchoring test question {i+1}",
                "description": "Question with potential anchoring points",
                "type": "binary",
                "close_time": "2025-12-01T00:00:00Z",
                "resolve_time": "2026-01-01T00:00:00Z",
                "categories": ["Test"],
                "tags": ["anchoring"]
            })

        scenarios.append(BiasTestScenario(
            name="Anchoring Test",
            bias_type="anchoring",
            expected_bias_strength=0.4,
            test_questions=anchoring_questions,
            detection_threshold=0.3
        ))

        return scenarios

    @pytest.mark.asyncio
    async def test_calibration_accuracy_validation(self, calibration_validator, mock_calibrated_agent, test_ground_truth):
        """Test calibration accuracy validation."""
        # Generate forecasts from well-calibrated agent
        forecasts = []
        for question_id in list(test_ground_truth.keys())[:20]:
            question_data = {
                "id": question_id,
                "title": f"Calibration test question {question_id}",
                "description": "Testing calibration accuracy",
                "type": "binary",
                "close_time": "2025-12-01T00:00:00Z",
                "resolve_time": "2026-01-01T00:00:00Z",
                "categories": ["Test"],
                "tags": ["calibration"]
            }

            question = calibration_validator._create_question(question_data)
            forecast = await mock_calibrated_agent.generate_forecast(question)
            forecasts.append(forecast)

        # Validate calibration
        validation_result = calibration_validator.validate_calibration_accuracy(
            forecasts, test_ground_truth, "all"
        )

        # Verify validation results
        assert "calibration_metrics" in validation_result
        assert "reliability_analysis" in validation_result
        assert "sharpness_analysis" in validation_result
        assert "resolution_analysis" in validation_result

        # Check calibration metrics
        metrics = validation_result["calibration_metrics"]
        assert "brier_score" in metrics
        assert "expected_calibration_error" in metrics
        assert "calibration_score" in metrics
        assert 0 <= metrics["brier_score"] <= 1
        assert 0 <= metrics["expected_calibration_error"] <= 1
        assert 0 <= metrics["calibration_score"] <= 1

        # Well-calibrated agent should have good calibration
        assert metrics["calibration_score"] >= 0.6  # Reasonable calibration

    @pytest.mark.asyncio
    async def test_overconfidence_bias_detection(self, calibration_validator, mock_overconfident_agent, bias_test_scenarios):
        """Test overconfidence bias detection."""
        overconfidence_scenario = next(s for s in bias_test_scenarios if s.bias_type == "overconfidence")

        bias_results = calibration_validator.detect_prediction_biases(
            mock_overconfident_agent, [overconfidence_scenario]
        )

        # Verify bias detection results
        assert "bias_tests" in bias_results
        assert "detected_biases" in bias_results
        assert "overall_bias_score" in bias_results

        # Check overconfidence detection
        overconfidence_test = bias_results["bias_tests"]["Overconfidence Test"]
        assert overconfidence_test["bias_type"] == "overconfidence"
        assert overconfidence_test["bias_detected"]  # Should detect overconfidence
        assert overconfidence_test["bias_strength"] > 0.3  # Significant bias

        # Verify detected biases
        detected_biases = bias_results["detected_biases"]
        assert len(detected_biases) > 0
        assert any(bias["bias_type"] == "overconfidence" for bias in detected_biases)

    @pytest.mark.asyncio
    async def test_reliability_diagram_analysis(self, calibration_validator, mock_calibrated_agent, test_ground_truth):
        """Test reliability diagram analysis."""
        # Generate forecasts with varied predictions
        forecasts = []
        for i, question_id in enumerate(list(test_ground_truth.keys())[:25]):
            question_data = {
                "id": question_id,
                "title": f"Reliability test question {question_id}",
                "description": "Testing reliability analysis",
                "type": "binary",
                "close_time": "2025-12-01T00:00:00Z",
                "resolve_time": "2026-01-01T00:00:00Z",
                "categories": ["Test"],
                "tags": ["reliability"]
            }

            question = calibration_validator._create_question(question_data)
            forecast = await mock_calibrated_agent.generate_forecast(question)
            forecasts.append(forecast)

        # Validate with reliability diagram
        validation_result = calibration_validator.validate_calibration_accuracy(
            forecasts, test_ground_truth, "reliability_diagram"
        )

        # Verify reliability analysis
        reliability = validation_result["reliability_analysis"]
        assert "bins" in reliability
        assert "weighted_calibration_error" in reliability
        assert "reliability_score" in reliability

        # Check bins
        bins = reliability["bins"]
        assert len(bins) > 0

        for bin_data in bins:
            assert "bin_range" in bin_data
            assert "count" in bin_data
            assert "avg_prediction" in bin_data
            assert "avg_outcome" in bin_data
            assert "calibration_error" in bin_data
            assert bin_data["count"] > 0
            assert 0 <= bin_data["avg_prediction"] <= 1
            assert 0 <= bin_data["avg_outcome"] <= 1

    @pytest.mark.asyncio
    async def test_sharpness_analysis(self, calibration_validator, mock_calibrated_agent, test_ground_truth):
        """Test sharpness analysis."""
        # Generate forecasts
        forecasts = []
        for question_id in list(test_ground_truth.keys())[:20]:
            question_data = {
                "id": question_id,
                "title": f"Sharpness test question {question_id}",
                "description": "Testing sharpness analysis",
                "type": "binary",
                "close_time": "2025-12-01T00:00:00Z",
                "resolve_time": "2026-01-01T00:00:00Z",
                "categories": ["Test"],
                "tags": ["sharpness"]
            }

            question = calibration_validator._create_question(question_data)
            forecast = await mock_calibrated_agent.generate_forecast(question)
            forecasts.append(forecast)

        # Validate with sharpness analysis
        validation_result = calibration_validator.validate_calibration_accuracy(
            forecasts, test_ground_truth, "sharpness"
        )

        # Verify sharpness analysis
        sharpness = validation_result["sharpness_analysis"]
        assert "prediction_variance" in sharpness
        assert "sharpness_score" in sharpness
        assert "discrimination_ability" in sharpness
        assert "prediction_distribution" in sharpness

        # Check metrics
        assert sharpness["prediction_variance"] >= 0
        assert 0 <= sharpness["sharpness_score"] <= 1
        assert 0 <= sharpness["discrimination_ability"] <= 1
        assert len(sharpness["prediction_distribution"]) == 10

    @pytest.mark.asyncio
    async def test_bias_mitigation_recommendations(self, calibration_validator, mock_overconfident_agent, bias_test_scenarios):
        """Test bias mitigation recommendations."""
        bias_results = calibration_validator.detect_prediction_biases(
            mock_overconfident_agent, bias_test_scenarios
        )

        # Verify recommendations are generated
        assert "bias_mitigation_recommendations" in bias_results
        recommendations = bias_results["bias_mitigation_recommendations"]
        assert len(recommendations) > 0

        # Check for relevant recommendations
        recommendation_text = " ".join(recommendations).lower()
        if any(bias["bias_type"] == "overconfidence" for bias in bias_results["detected_biases"]):
            assert "confidence" in recommendation_text or "calibration" in recommendation_text

        # Recommendations should be actionable
        for recommendation in recommendations:
            assert len(recommendation) > 20  # Substantial recommendations
            assert any(word in recommendation.lower() for word in ["implement", "encourage", "provide", "use", "consider"])
