"""
Forecasting Stage Service with GPT-5 and Calibration.
Implements task 4.3 requirements with uncertainty quantification and tournament compliance.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result from forecasting stage analysis."""

    forecast_type: str  # "binary", "multiple_choice", "numeric"
    prediction: Union[float, Dict[str, float], Dict[str, Any]]
    confidence_score: float
    uncertainty_bounds: Optional[Dict[str, float]]
    calibration_score: float
    overconfidence_detected: bool
    quality_validation_passed: bool
    tournament_compliant: bool
    reasoning: str
    execution_time: float
    cost_estimate: float
    model_used: str


@dataclass
class CalibrationMetrics:
    """Calibration metrics for forecast quality assessment."""

    base_rate_consideration: float
    scenario_analysis_score: float
    uncertainty_acknowledgment: float
    overconfidence_indicators: List[str]
    calibration_adjustments: List[str]
    final_calibration_score: float


@dataclass
class UncertaintyQuantification:
    """Uncertainty quantification for forecasts."""

    confidence_intervals: Dict[str, float]
    scenario_probabilities: Dict[str, float]
    key_uncertainty_factors: List[str]
    information_gaps: List[str]
    sensitivity_analysis: Dict[str, float]


class ForecastingStageService:
    """
    Advanced forecasting stage service using GPT-5 with calibration and uncertainty quantification.

    Features:
    - GPT-5 optimized forecasting prompts with maximum reasoning capability
    - Calibration checks and overconfidence reduction techniques
    - Uncertainty quantification and confidence scoring
    - Forecast quality validation and tournament compliance checks
    - Support for binary, multiple choice, and numeric forecasts
    """

    def __init__(self, tri_model_router=None):
        """Initialize the forecasting stage service."""
        self.tri_model_router = tri_model_router
        self.logger = logging.getLogger(__name__)

        # Calibration thresholds
        self.overconfidence_threshold = 0.85
        self.minimum_uncertainty_acknowledgment = 0.3
        self.base_rate_consideration_threshold = 0.6

        # Tournament compliance requirements
        self.tournament_requirements = {
            "min_reasoning_length": 100,
            "required_uncertainty_acknowledgment": True,
            "required_base_rate_consideration": True,
            "max_confidence_without_strong_evidence": 0.8,
        }

    async def generate_forecast(
        self,
        question: str,
        question_type: str,
        research_data: str,
        context: Dict[str, Any] = None,
    ) -> ForecastResult:
        """
        Generate calibrated forecast using GPT-5 with uncertainty quantification.

        Args:
            question: Forecasting question
            question_type: Type of forecast ("binary", "multiple_choice", "numeric")
            research_data: Research findings to base forecast on
            context: Additional context including options, bounds, etc.

        Returns:
            ForecastResult with calibrated prediction and quality metrics
        """
        context = context or {}
        forecast_start = datetime.now()

        self.logger.info(f"Starting GPT-5 forecasting for {question_type} question...")

        try:
            # Step 1: Create GPT-5 optimized forecasting prompt
            forecast_prompt = await self._create_gpt5_forecasting_prompt(
                question, question_type, research_data, context
            )

            # Step 2: Execute forecast with GPT-5 full model
            raw_forecast = await self._execute_gpt5_forecast(forecast_prompt)

            # Step 3: Parse and extract forecast components
            parsed_forecast = await self._parse_forecast_response(
                raw_forecast, question_type, context
            )

            # Step 4: Apply calibration checks and adjustments
            calibration_metrics = await self._perform_calibration_analysis(
                parsed_forecast, raw_forecast, question_type
            )

            # Step 5: Quantify uncertainty and confidence
            uncertainty_metrics = await self._quantify_uncertainty(
                parsed_forecast, raw_forecast, question_type, context
            )

            # Step 6: Validate forecast quality and tournament compliance
            quality_passed = await self._validate_forecast_quality(
                parsed_forecast, raw_forecast, calibration_metrics
            )

            tournament_compliant = await self._check_tournament_compliance(
                parsed_forecast, raw_forecast, calibration_metrics
            )

            execution_time = (datetime.now() - forecast_start).total_seconds()

            return ForecastResult(
                forecast_type=question_type,
                prediction=parsed_forecast["prediction"],
                confidence_score=parsed_forecast.get("confidence", 0.5),
                uncertainty_bounds=uncertainty_metrics.confidence_intervals,
                calibration_score=calibration_metrics.final_calibration_score,
                overconfidence_detected=calibration_metrics.final_calibration_score
                < 0.5,
                quality_validation_passed=quality_passed,
                tournament_compliant=tournament_compliant,
                reasoning=parsed_forecast.get("reasoning", raw_forecast),
                execution_time=execution_time,
                cost_estimate=self._estimate_forecast_cost(
                    forecast_prompt, raw_forecast
                ),
                model_used="openai/gpt-5",
            )

        except Exception as e:
            execution_time = (datetime.now() - forecast_start).total_seconds()
            self.logger.error(f"Forecasting failed: {e}")

            return ForecastResult(
                forecast_type=question_type,
                prediction=0.5 if question_type == "binary" else {},
                confidence_score=0.0,
                uncertainty_bounds=None,
                calibration_score=0.0,
                overconfidence_detected=True,
                quality_validation_passed=False,
                tournament_compliant=False,
                reasoning=f"Forecasting error: {str(e)}",
                execution_time=execution_time,
                cost_estimate=0.0,
                model_used="none",
            )

    async def _create_gpt5_forecasting_prompt(
        self,
        question: str,
        question_type: str,
        research_data: str,
        context: Dict[str, Any],
    ) -> str:
        """Create GPT-5 optimized forecasting prompt with maximum reasoning capability."""

        # Import anti-slop prompts for GPT-5 optimization
        from ...prompts.anti_slop_prompts import anti_slop_prompts

        # Get base prompt based on question type
        if question_type == "binary":
            base_prompt = anti_slop_prompts.get_binary_forecast_prompt(
                question_text=question,
                background_info=context.get("background_info", ""),
                resolution_criteria=context.get("resolution_criteria", ""),
                fine_print=context.get("fine_print", ""),
                research=research_data,
                model_tier="full",
            )
        elif question_type == "multiple_choice":
            base_prompt = anti_slop_prompts.get_multiple_choice_prompt(
                question_text=question,
                options=context.get("options", []),
                background_info=context.get("background_info", ""),
                resolution_criteria=context.get("resolution_criteria", ""),
                fine_print=context.get("fine_print", ""),
                research=research_data,
                model_tier="full",
            )
        elif question_type == "numeric":
            base_prompt = anti_slop_prompts.get_numeric_forecast_prompt(
                question_text=question,
                background_info=context.get("background_info", ""),
                resolution_criteria=context.get("resolution_criteria", ""),
                fine_print=context.get("fine_print", ""),
                research=research_data,
                unit_of_measure=context.get("unit_of_measure"),
                lower_bound=context.get("lower_bound"),
                upper_bound=context.get("upper_bound"),
                model_tier="full",
            )
        else:
            raise ValueError(f"Unsupported question type: {question_type}")

        # Enhance with GPT-5 specific calibration instructions
        gpt5_enhancements = """

## GPT-5 MAXIMUM REASONING CALIBRATION:

### ADVANCED CALIBRATION PROTOCOL:
• Apply meta-cognitive reasoning about your reasoning quality
• Consider multiple competing hypotheses and their likelihood
• Use reference class forecasting with historical base rates
• Apply inside-view vs outside-view analysis
• Consider regression to the mean effects

### OVERCONFIDENCE REDUCTION TECHNIQUES:
• Pre-mortem analysis: How could this forecast be wrong?
• Consider the planning fallacy and optimism bias
• Apply the "consider the opposite" technique
• Use confidence intervals rather than point estimates
• Account for unknown unknowns with wider uncertainty bounds

### UNCERTAINTY QUANTIFICATION REQUIREMENTS:
• Provide explicit confidence intervals for all estimates
• Identify key factors that could shift probabilities significantly
• Acknowledge information gaps and their impact on confidence
• Consider tail risks and black swan scenarios
• Use scenario analysis with probability weights

### TOURNAMENT COMPLIANCE CHECKS:
• Ensure reasoning is substantive (minimum 100 words)
• Explicitly acknowledge uncertainty and limitations
• Consider base rates and historical precedents
• Avoid extreme confidence without overwhelming evidence
• Provide clear rationale for final probability/estimate

### FINAL VERIFICATION PROTOCOL:
• Double-check: Does this forecast pass the "outside view" test?
• Calibration check: Am I being appropriately humble about uncertainty?
• Base rate check: How does this compare to similar historical cases?
• Evidence check: Is my confidence level justified by the evidence quality?
"""

        # Combine base prompt with GPT-5 enhancements
        enhanced_prompt = f"{base_prompt}\n{gpt5_enhancements}"

        return enhanced_prompt

    async def _execute_gpt5_forecast(self, prompt: str) -> str:
        """Execute forecast using GPT-5 full model with maximum reasoning capability."""

        if not self.tri_model_router:
            raise Exception("Tri-model router not available for GPT-5 forecasting")

        # Get GPT-5 full model for maximum reasoning capability
        gpt5_model = self.tri_model_router.models.get("full")
        if not gpt5_model:
            raise Exception("GPT-5 full model not available")

        try:
            # Execute with GPT-5 full model
            forecast_response = await gpt5_model.invoke(prompt)

            if not forecast_response or len(forecast_response.strip()) < 50:
                raise Exception("GPT-5 forecast response too short or empty")

            return forecast_response

        except Exception as e:
            self.logger.error(f"GPT-5 forecast execution failed: {e}")
            raise

    async def _parse_forecast_response(
        self, response: str, question_type: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse GPT-5 forecast response and extract structured components."""

        parsed = {
            "raw_response": response,
            "reasoning": "",
            "prediction": None,
            "confidence": 0.5,
            "uncertainty_factors": [],
            "base_rate_mentioned": False,
            "scenarios_considered": [],
        }

        try:
            if question_type == "binary":
                # Extract binary probability
                probability_match = re.search(
                    r"Probability:\s*(\d+(?:\.\d+)?)%", response, re.IGNORECASE
                )
                if probability_match:
                    parsed["prediction"] = float(probability_match.group(1)) / 100.0
                else:
                    # Fallback: look for percentage anywhere in response
                    percent_matches = re.findall(r"(\d+(?:\.\d+)?)%", response)
                    if percent_matches:
                        # Use the last percentage found (likely the final answer)
                        parsed["prediction"] = float(percent_matches[-1]) / 100.0
                    else:
                        parsed["prediction"] = 0.5  # Default neutral

            elif question_type == "multiple_choice":
                # Extract multiple choice probabilities
                options = context.get("options", [])
                probabilities = {}

                for option in options:
                    # Look for option with percentage
                    pattern = rf'"{re.escape(option)}":\s*(\d+(?:\.\d+)?)%'
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        probabilities[option] = float(match.group(1)) / 100.0

                # Normalize probabilities to sum to 1.0
                total = sum(probabilities.values())
                if total > 0:
                    probabilities = {k: v / total for k, v in probabilities.items()}
                    parsed["prediction"] = probabilities
                else:
                    # Default equal probabilities
                    equal_prob = 1.0 / len(options) if options else 0.5
                    parsed["prediction"] = {option: equal_prob for option in options}

            elif question_type == "numeric":
                # Extract percentile estimates
                percentiles = {}
                percentile_pattern = r"Percentile\s+(\d+):\s*([0-9,]+(?:\.\d+)?)"
                matches = re.findall(percentile_pattern, response, re.IGNORECASE)

                for percentile, value in matches:
                    # Remove commas and convert to float
                    clean_value = value.replace(",", "")
                    try:
                        percentiles[int(percentile)] = float(clean_value)
                    except ValueError:
                        continue

                if percentiles:
                    parsed["prediction"] = percentiles
                else:
                    # Fallback: look for any numbers that might be estimates
                    numbers = re.findall(r"(\d+(?:,\d{3})*(?:\.\d+)?)", response)
                    if numbers:
                        # Use median of found numbers as rough estimate
                        clean_numbers = [float(n.replace(",", "")) for n in numbers]
                        median_val = sorted(clean_numbers)[len(clean_numbers) // 2]
                        parsed["prediction"] = {50: median_val}
                    else:
                        parsed["prediction"] = {}

            # Extract confidence level
            confidence_match = re.search(
                r"Confidence:\s*(Low|Medium|High)", response, re.IGNORECASE
            )
            if confidence_match:
                confidence_level = confidence_match.group(1).lower()
                confidence_mapping = {"low": 0.3, "medium": 0.6, "high": 0.8}
                parsed["confidence"] = confidence_mapping.get(confidence_level, 0.5)

            # Extract reasoning (everything before final answer)
            reasoning_parts = []
            lines = response.split("\n")
            for line in lines:
                if not re.search(
                    r"(Probability:|Percentile|Confidence:)", line, re.IGNORECASE
                ):
                    reasoning_parts.append(line.strip())
                else:
                    break

            parsed["reasoning"] = "\n".join(reasoning_parts).strip()

            # Check for base rate consideration
            base_rate_indicators = [
                "base rate",
                "historical",
                "precedent",
                "similar cases",
                "reference class",
            ]
            parsed["base_rate_mentioned"] = any(
                indicator in response.lower() for indicator in base_rate_indicators
            )

            # Extract uncertainty factors
            uncertainty_indicators = [
                "uncertain",
                "gap",
                "unknown",
                "unclear",
                "missing",
                "limitation",
            ]
            parsed["uncertainty_factors"] = [
                line.strip()
                for line in lines
                if any(
                    indicator in line.lower() for indicator in uncertainty_indicators
                )
            ]

            return parsed

        except Exception as e:
            self.logger.error(f"Failed to parse forecast response: {e}")
            # Return minimal parsed structure
            return {
                "raw_response": response,
                "reasoning": response,
                "prediction": 0.5 if question_type == "binary" else {},
                "confidence": 0.3,
                "uncertainty_factors": [],
                "base_rate_mentioned": False,
                "scenarios_considered": [],
            }

    async def _perform_calibration_analysis(
        self, parsed_forecast: Dict[str, Any], raw_response: str, question_type: str
    ) -> CalibrationMetrics:
        """Perform calibration analysis and overconfidence detection."""

        # Analyze base rate consideration
        base_rate_score = 0.8 if parsed_forecast["base_rate_mentioned"] else 0.2

        # Analyze scenario consideration
        scenario_indicators = [
            "scenario",
            "case",
            "situation",
            "possibility",
            "alternative",
        ]
        scenario_mentions = sum(
            1 for indicator in scenario_indicators if indicator in raw_response.lower()
        )
        scenario_score = min(1.0, scenario_mentions / 3.0)  # Normalize to 0-1

        # Analyze uncertainty acknowledgment
        uncertainty_score = min(1.0, len(parsed_forecast["uncertainty_factors"]) / 3.0)

        # Detect overconfidence indicators
        overconfidence_indicators = []

        # Check for extreme confidence without strong evidence
        if question_type == "binary":
            prediction = parsed_forecast.get("prediction", 0.5)
            if isinstance(prediction, (int, float)):
                if (prediction > 0.9 or prediction < 0.1) and parsed_forecast[
                    "confidence"
                ] < 0.7:
                    overconfidence_indicators.append(
                        "Extreme probability with low confidence in evidence"
                    )

                if prediction > 0.85 and not parsed_forecast["base_rate_mentioned"]:
                    overconfidence_indicators.append(
                        "High confidence without base rate consideration"
                    )

        # Check for insufficient uncertainty acknowledgment
        if len(parsed_forecast["uncertainty_factors"]) < 2:
            overconfidence_indicators.append("Insufficient uncertainty acknowledgment")

        # Check reasoning length (short reasoning often indicates overconfidence)
        reasoning_length = len(parsed_forecast.get("reasoning", "").split())
        if reasoning_length < 50:
            overconfidence_indicators.append("Insufficient reasoning depth")

        # Generate calibration adjustments
        calibration_adjustments = []

        if base_rate_score < 0.5:
            calibration_adjustments.append(
                "Consider historical base rates and precedents"
            )

        if scenario_score < 0.5:
            calibration_adjustments.append(
                "Analyze multiple scenarios and their probabilities"
            )

        if uncertainty_score < 0.5:
            calibration_adjustments.append(
                "Acknowledge more uncertainty factors and information gaps"
            )

        if overconfidence_indicators:
            calibration_adjustments.append(
                "Reduce confidence to account for potential overconfidence"
            )

        # Calculate final calibration score
        final_calibration_score = (
            base_rate_score + scenario_score + uncertainty_score
        ) / 3.0

        # Penalize for overconfidence indicators
        overconfidence_penalty = len(overconfidence_indicators) * 0.1
        final_calibration_score = max(
            0.0, final_calibration_score - overconfidence_penalty
        )

        return CalibrationMetrics(
            base_rate_consideration=base_rate_score,
            scenario_analysis_score=scenario_score,
            uncertainty_acknowledgment=uncertainty_score,
            overconfidence_indicators=overconfidence_indicators,
            calibration_adjustments=calibration_adjustments,
            final_calibration_score=final_calibration_score,
        )

    async def _quantify_uncertainty(
        self,
        parsed_forecast: Dict[str, Any],
        raw_response: str,
        question_type: str,
        context: Dict[str, Any],
    ) -> UncertaintyQuantification:
        """Quantify uncertainty and generate confidence intervals."""

        # Extract confidence intervals based on question type
        confidence_intervals = {}

        if question_type == "binary":
            prediction = parsed_forecast.get("prediction", 0.5)
            confidence = parsed_forecast.get("confidence", 0.5)

            # Generate confidence intervals based on confidence level
            uncertainty_range = (1 - confidence) * 0.3  # Max 30% uncertainty range
            confidence_intervals = {
                "lower_bound": max(0.0, prediction - uncertainty_range),
                "upper_bound": min(1.0, prediction + uncertainty_range),
                "point_estimate": prediction,
            }

        elif question_type == "multiple_choice":
            prediction = parsed_forecast.get("prediction", {})
            if isinstance(prediction, dict):
                # For multiple choice, confidence intervals are the probability ranges
                confidence_intervals = {
                    option: {
                        "point_estimate": prob,
                        "uncertainty": (1 - parsed_forecast.get("confidence", 0.5))
                        * 0.2,
                    }
                    for option, prob in prediction.items()
                }

        elif question_type == "numeric":
            prediction = parsed_forecast.get("prediction", {})
            if isinstance(prediction, dict) and prediction:
                # Use percentile spread as confidence interval
                percentiles = sorted(prediction.keys())
                if len(percentiles) >= 3:
                    confidence_intervals = {
                        "p10": prediction.get(10, prediction[percentiles[0]]),
                        "p50": prediction.get(
                            50, prediction[percentiles[len(percentiles) // 2]]
                        ),
                        "p90": prediction.get(90, prediction[percentiles[-1]]),
                        "range": prediction[percentiles[-1]]
                        - prediction[percentiles[0]],
                    }

        # Extract scenario probabilities
        scenario_probabilities = {}
        scenario_patterns = [
            r"status quo.*?(\d+(?:\.\d+)?)%",
            r"moderate.*?(\d+(?:\.\d+)?)%",
            r"disruption.*?(\d+(?:\.\d+)?)%",
        ]

        for i, pattern in enumerate(scenario_patterns):
            match = re.search(pattern, raw_response, re.IGNORECASE)
            if match:
                scenario_names = ["status_quo", "moderate_change", "disruption"]
                scenario_probabilities[scenario_names[i]] = (
                    float(match.group(1)) / 100.0
                )

        # Extract key uncertainty factors
        key_uncertainty_factors = parsed_forecast.get("uncertainty_factors", [])

        # Extract information gaps
        gap_indicators = ["gap", "missing", "unknown", "unclear", "unavailable"]
        information_gaps = [
            line.strip()
            for line in raw_response.split("\n")
            if any(indicator in line.lower() for indicator in gap_indicators)
        ][
            :5
        ]  # Limit to top 5 gaps

        # Simple sensitivity analysis based on confidence
        confidence = parsed_forecast.get("confidence", 0.5)
        sensitivity_analysis = {
            "evidence_quality": confidence,
            "information_completeness": min(1.0, len(key_uncertainty_factors) / 5.0),
            "base_rate_reliability": (
                0.8 if parsed_forecast.get("base_rate_mentioned") else 0.3
            ),
        }

        return UncertaintyQuantification(
            confidence_intervals=confidence_intervals,
            scenario_probabilities=scenario_probabilities,
            key_uncertainty_factors=key_uncertainty_factors,
            information_gaps=information_gaps,
            sensitivity_analysis=sensitivity_analysis,
        )

    async def _validate_forecast_quality(
        self,
        parsed_forecast: Dict[str, Any],
        raw_response: str,
        calibration_metrics: CalibrationMetrics,
    ) -> bool:
        """Validate forecast quality against internal standards."""

        quality_checks = []

        # Check 1: Prediction is valid and reasonable
        prediction = parsed_forecast.get("prediction")
        if prediction is None:
            quality_checks.append(False)
        elif isinstance(prediction, (int, float)):
            # Binary forecast checks
            quality_checks.append(0.0 <= prediction <= 1.0)
        elif isinstance(prediction, dict):
            # Multiple choice or numeric forecast checks
            if all(isinstance(v, (int, float)) for v in prediction.values()):
                quality_checks.append(True)
            else:
                quality_checks.append(False)
        else:
            quality_checks.append(False)

        # Check 2: Reasoning is substantive
        reasoning_length = len(parsed_forecast.get("reasoning", "").split())
        quality_checks.append(reasoning_length >= 50)

        # Check 3: Calibration score meets threshold
        quality_checks.append(calibration_metrics.final_calibration_score >= 0.4)

        # Check 4: Not too many overconfidence indicators
        quality_checks.append(len(calibration_metrics.overconfidence_indicators) <= 2)

        # Check 5: Some uncertainty acknowledgment present
        quality_checks.append(len(parsed_forecast.get("uncertainty_factors", [])) >= 1)

        # Check 6: Response is coherent and complete
        quality_checks.append(len(raw_response.strip()) >= 100)

        # Pass if at least 4 out of 6 checks pass
        return sum(quality_checks) >= 4

    async def _check_tournament_compliance(
        self,
        parsed_forecast: Dict[str, Any],
        raw_response: str,
        calibration_metrics: CalibrationMetrics,
    ) -> bool:
        """Check tournament compliance requirements."""

        compliance_checks = []

        # Check 1: Minimum reasoning length
        reasoning_length = len(parsed_forecast.get("reasoning", "").split())
        min_length = self.tournament_requirements["min_reasoning_length"]
        compliance_checks.append(reasoning_length >= min_length)

        # Check 2: Uncertainty acknowledgment required
        if self.tournament_requirements["required_uncertainty_acknowledgment"]:
            uncertainty_present = (
                len(parsed_forecast.get("uncertainty_factors", [])) >= 1
            )
            compliance_checks.append(uncertainty_present)
        else:
            compliance_checks.append(True)

        # Check 3: Base rate consideration required
        if self.tournament_requirements["required_base_rate_consideration"]:
            base_rate_present = parsed_forecast.get("base_rate_mentioned", False)
            compliance_checks.append(base_rate_present)
        else:
            compliance_checks.append(True)

        # Check 4: Maximum confidence without strong evidence
        prediction = parsed_forecast.get("prediction")
        max_confidence = self.tournament_requirements[
            "max_confidence_without_strong_evidence"
        ]

        if isinstance(prediction, (int, float)):
            # Binary forecast
            if prediction > max_confidence or prediction < (1 - max_confidence):
                # High confidence - check if evidence is strong
                evidence_strength = calibration_metrics.final_calibration_score
                compliance_checks.append(evidence_strength >= 0.7)
            else:
                compliance_checks.append(True)
        else:
            # For non-binary, assume compliant if other checks pass
            compliance_checks.append(True)

        # Check 5: No critical overconfidence indicators
        critical_overconfidence = any(
            "extreme" in indicator.lower() or "insufficient" in indicator.lower()
            for indicator in calibration_metrics.overconfidence_indicators
        )
        compliance_checks.append(not critical_overconfidence)

        # All compliance checks must pass
        return all(compliance_checks)

    def _estimate_forecast_cost(self, prompt: str, response: str) -> float:
        """Estimate cost of GPT-5 forecast generation."""

        # Estimate tokens (rough approximation: 1 token ≈ 0.75 words)
        prompt_tokens = len(prompt.split()) / 0.75
        response_tokens = len(response.split()) / 0.75

        # GPT-5 pricing: $1.50 per million tokens (both input and output)
        input_cost = (prompt_tokens / 1_000_000) * 1.50
        output_cost = (response_tokens / 1_000_000) * 1.50

        return input_cost + output_cost

    def get_service_status(self) -> Dict[str, Any]:
        """Get current forecasting service configuration and status."""
        return {
            "service": "ForecastingStageService",
            "model_used": "openai/gpt-5",
            "supported_forecast_types": ["binary", "multiple_choice", "numeric"],
            "calibration_thresholds": {
                "overconfidence_threshold": self.overconfidence_threshold,
                "minimum_uncertainty_acknowledgment": self.minimum_uncertainty_acknowledgment,
                "base_rate_consideration_threshold": self.base_rate_consideration_threshold,
            },
            "tournament_requirements": self.tournament_requirements,
            "tri_model_router_available": bool(self.tri_model_router),
            "capabilities": [
                "gpt5_optimized_forecasting",
                "calibration_analysis",
                "overconfidence_detection",
                "uncertainty_quantification",
                "tournament_compliance_checking",
                "forecast_quality_validation",
            ],
        }
