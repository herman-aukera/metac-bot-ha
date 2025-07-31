"""Reasoning orchestrator service."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Protocol, Union
from uuid import UUID, uuid4
import structlog

from ..value_objects.reasoning_trace import ReasoningTrace, ReasoningStep, ReasoningStepType
from ..entities.question import Question


logger = structlog.get_logger(__name__)


class BiasType(Enum):
    """Types of cognitive biases to detect."""
    CONFIRMATION_BIAS = "confirmation_bias"
    ANCHORING_BIAS = "anchoring_bias"
    AVAILABILITY_HEURISTIC = "availability_heuristic"
    OVERCONFIDENCE_BIAS = "overconfidence_bias"
    REPRESENTATIVENESS_HEURISTIC = "representativeness_heuristic"
    BASE_RATE_NEGLECT = "base_rate_neglect"
    CONJUNCTION_FALLACY = "conjunction_fallacy"


@dataclass(frozen=True)
class BiasDetectionResult:
    """Result of bias detection analysis."""
    bias_type: BiasType
    detected: bool
    confidence: float
    evidence: str
    mitigation_suggestion: str


@dataclass(frozen=True)
class ReasoningValidationResult:
    """Result of reasoning step validation."""
    is_valid: bool
    confidence_adjustment: float
    validation_notes: str
    suggested_improvements: List[str]


class ReasoningAgent(Protocol):
    """Protocol for reasoning agents."""
    
    async def reason(self, question: Question, context: Dict[str, Any]) -> ReasoningTrace:
        """Generate reasoning trace for a question."""
        ...


class ReasoningOrchestrator:
    """
    Orchestrates multi-step reasoning processes with bias detection and validation.
    
    Manages the overall reasoning workflow, ensuring transparency, quality,
    and bias mitigation throughout the reasoning process.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_reasoning_depth: int = 10,
        bias_detection_enabled: bool = True,
        validation_enabled: bool = True
    ):
        """
        Initialize the reasoning orchestrator.
        
        Args:
            confidence_threshold: Minimum confidence required for conclusions
            max_reasoning_depth: Maximum number of reasoning steps allowed
            bias_detection_enabled: Whether to perform bias detection
            validation_enabled: Whether to validate reasoning steps
        """
        self.confidence_threshold = confidence_threshold
        self.max_reasoning_depth = max_reasoning_depth
        self.bias_detection_enabled = bias_detection_enabled
        self.validation_enabled = validation_enabled
        self.logger = logger.bind(component="reasoning_orchestrator")
        
        # Bias detection patterns
        self.bias_patterns = self._initialize_bias_patterns()
    
    def _initialize_bias_patterns(self) -> Dict[BiasType, Dict[str, Any]]:
        """Initialize bias detection patterns."""
        return {
            BiasType.CONFIRMATION_BIAS: {
                "keywords": ["confirms", "supports", "validates", "proves"],
                "pattern": "selective_evidence_focus",
                "threshold": 0.6
            },
            BiasType.ANCHORING_BIAS: {
                "keywords": ["first", "initial", "starting", "baseline"],
                "pattern": "excessive_initial_value_reliance",
                "threshold": 0.7
            },
            BiasType.AVAILABILITY_HEURISTIC: {
                "keywords": ["recent", "memorable", "vivid", "comes to mind"],
                "pattern": "recent_event_overweighting",
                "threshold": 0.6
            },
            BiasType.OVERCONFIDENCE_BIAS: {
                "keywords": ["certain", "definitely", "obviously", "clearly"],
                "pattern": "excessive_certainty_claims",
                "threshold": 0.8
            }
        }
    
    async def orchestrate_reasoning(
        self,
        question: Question,
        reasoning_agent: ReasoningAgent,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """
        Orchestrate a complete reasoning process with validation and bias detection.
        
        Args:
            question: The question to reason about
            reasoning_agent: The agent to perform reasoning
            context: Additional context for reasoning
            
        Returns:
            Complete reasoning trace with validation and bias detection
        """
        self.logger.info("Starting reasoning orchestration", question_id=str(question.id))
        
        context = context or {}
        
        try:
            # Generate initial reasoning trace
            initial_trace = await reasoning_agent.reason(question, context)
            
            # Validate reasoning steps
            validated_trace = await self._validate_reasoning_trace(initial_trace)
            
            # Detect and mitigate biases
            if self.bias_detection_enabled:
                bias_checked_trace = await self._detect_and_mitigate_biases(validated_trace)
            else:
                bias_checked_trace = validated_trace
            
            # Final confidence calibration
            final_trace = await self._calibrate_final_confidence(bias_checked_trace)
            
            self.logger.info(
                "Reasoning orchestration completed",
                question_id=str(question.id),
                steps=len(final_trace.steps),
                final_confidence=final_trace.overall_confidence
            )
            
            return final_trace
            
        except Exception as e:
            self.logger.error("Reasoning orchestration failed", error=str(e))
            raise
    
    async def _validate_reasoning_trace(self, trace: ReasoningTrace) -> ReasoningTrace:
        """Validate each step in the reasoning trace."""
        if not self.validation_enabled:
            return trace
        
        validated_steps = []
        validation_notes = []
        
        for step in trace.steps:
            validation_result = await self._validate_reasoning_step(step)
            
            # Adjust confidence based on validation
            adjusted_confidence = max(
                0.0,
                min(1.0, step.confidence + validation_result.confidence_adjustment)
            )
            
            validated_step = ReasoningStep.create(
                step_type=step.step_type,
                content=step.content,
                confidence=adjusted_confidence,
                metadata={
                    **step.metadata,
                    "validation_result": validation_result,
                    "original_confidence": step.confidence
                }
            )
            
            validated_steps.append(validated_step)
            validation_notes.extend(validation_result.suggested_improvements)
        
        # Create new trace with validated steps
        return ReasoningTrace.create(
            question_id=trace.question_id,
            agent_id=trace.agent_id,
            reasoning_method=trace.reasoning_method,
            steps=validated_steps,
            final_conclusion=trace.final_conclusion,
            overall_confidence=self._calculate_overall_confidence(validated_steps),
            bias_checks=trace.bias_checks,
            uncertainty_sources=trace.uncertainty_sources + validation_notes
        )
    
    async def _validate_reasoning_step(self, step: ReasoningStep) -> ReasoningValidationResult:
        """Validate a single reasoning step."""
        is_valid = True
        confidence_adjustment = 0.0
        validation_notes = ""
        suggested_improvements = []
        
        # Check step content quality
        if len(step.content.strip()) < 10:
            is_valid = False
            confidence_adjustment -= 0.2
            suggested_improvements.append("Reasoning step content is too brief")
        
        # Check confidence calibration
        if step.confidence > 0.9 and step.step_type not in [
            ReasoningStepType.OBSERVATION, ReasoningStepType.CONCLUSION
        ]:
            confidence_adjustment -= 0.1
            suggested_improvements.append("High confidence may indicate overconfidence")
        
        # Check for logical consistency
        if step.step_type == ReasoningStepType.CONCLUSION and step.confidence < 0.3:
            suggested_improvements.append("Low confidence conclusion may need more analysis")
        
        validation_notes = f"Step validation completed. Adjustments: {confidence_adjustment}"
        
        return ReasoningValidationResult(
            is_valid=is_valid,
            confidence_adjustment=confidence_adjustment,
            validation_notes=validation_notes,
            suggested_improvements=suggested_improvements
        )
    
    async def _detect_and_mitigate_biases(self, trace: ReasoningTrace) -> ReasoningTrace:
        """Detect and mitigate cognitive biases in reasoning trace."""
        bias_detection_results = []
        mitigation_steps = []
        
        for bias_type in BiasType:
            detection_result = await self._detect_bias(trace, bias_type)
            bias_detection_results.append(detection_result)
            
            if detection_result.detected:
                mitigation_step = await self._create_bias_mitigation_step(
                    detection_result, trace
                )
                mitigation_steps.append(mitigation_step)
        
        # Add bias detection steps to trace
        enhanced_steps = list(trace.steps)
        enhanced_steps.extend(mitigation_steps)
        
        # Update bias checks
        bias_checks = list(trace.bias_checks)
        bias_checks.extend([
            f"{result.bias_type.value}: {'detected' if result.detected else 'not detected'}"
            for result in bias_detection_results
        ])
        
        return ReasoningTrace.create(
            question_id=trace.question_id,
            agent_id=trace.agent_id,
            reasoning_method=trace.reasoning_method,
            steps=enhanced_steps,
            final_conclusion=trace.final_conclusion,
            overall_confidence=self._calculate_overall_confidence(enhanced_steps),
            bias_checks=bias_checks,
            uncertainty_sources=trace.uncertainty_sources
        )
    
    async def _detect_bias(self, trace: ReasoningTrace, bias_type: BiasType) -> BiasDetectionResult:
        """Detect a specific type of bias in the reasoning trace."""
        pattern_config = self.bias_patterns.get(bias_type, {})
        keywords = pattern_config.get("keywords", [])
        threshold = pattern_config.get("threshold", 0.5)
        
        # Simple keyword-based detection (can be enhanced with ML models)
        total_content = " ".join([step.content.lower() for step in trace.steps])
        keyword_matches = sum(1 for keyword in keywords if keyword in total_content)
        
        detection_confidence = min(1.0, keyword_matches / len(keywords) if keywords else 0.0)
        detected = detection_confidence >= threshold
        
        evidence = f"Found {keyword_matches} bias indicators out of {len(keywords)} patterns"
        mitigation_suggestion = self._get_bias_mitigation_suggestion(bias_type)
        
        return BiasDetectionResult(
            bias_type=bias_type,
            detected=detected,
            confidence=detection_confidence,
            evidence=evidence,
            mitigation_suggestion=mitigation_suggestion
        )
    
    def _get_bias_mitigation_suggestion(self, bias_type: BiasType) -> str:
        """Get mitigation suggestion for a specific bias type."""
        suggestions = {
            BiasType.CONFIRMATION_BIAS: "Consider alternative explanations and contradictory evidence",
            BiasType.ANCHORING_BIAS: "Evaluate multiple reference points and starting assumptions",
            BiasType.AVAILABILITY_HEURISTIC: "Seek broader data beyond recent or memorable examples",
            BiasType.OVERCONFIDENCE_BIAS: "Quantify uncertainty and consider what could go wrong",
            BiasType.REPRESENTATIVENESS_HEURISTIC: "Consider base rates and statistical reasoning",
            BiasType.BASE_RATE_NEGLECT: "Incorporate prior probabilities and base rate information",
            BiasType.CONJUNCTION_FALLACY: "Evaluate individual probabilities separately"
        }
        return suggestions.get(bias_type, "Apply general debiasing techniques")
    
    async def _create_bias_mitigation_step(
        self,
        detection_result: BiasDetectionResult,
        trace: ReasoningTrace
    ) -> ReasoningStep:
        """Create a reasoning step to mitigate detected bias."""
        content = f"Bias mitigation for {detection_result.bias_type.value}: {detection_result.mitigation_suggestion}"
        
        return ReasoningStep.create(
            step_type=ReasoningStepType.BIAS_CHECK,
            content=content,
            confidence=0.8,  # High confidence in bias mitigation process
            metadata={
                "bias_type": detection_result.bias_type.value,
                "detection_confidence": detection_result.confidence,
                "evidence": detection_result.evidence
            }
        )
    
    async def _calibrate_final_confidence(self, trace: ReasoningTrace) -> ReasoningTrace:
        """Calibrate the final confidence based on reasoning quality."""
        # Calculate base confidence from steps
        step_confidences = [step.confidence for step in trace.steps]
        base_confidence = sum(step_confidences) / len(step_confidences) if step_confidences else 0.5
        
        # Apply quality adjustments
        quality_score = trace.get_reasoning_quality_score()
        quality_adjustment = (quality_score - 0.5) * 0.2  # Scale quality impact
        
        # Apply bias detection penalty
        bias_penalty = 0.0
        if trace.has_bias_checks():
            detected_biases = len([check for check in trace.bias_checks if "detected" in check])
            bias_penalty = detected_biases * 0.05  # Small penalty per detected bias
        
        # Apply uncertainty bonus (acknowledging uncertainty is good)
        uncertainty_bonus = 0.05 if trace.has_uncertainty_assessment() else 0.0
        
        # Calculate final confidence
        final_confidence = max(
            0.1,  # Minimum confidence
            min(
                0.95,  # Maximum confidence (leave room for uncertainty)
                base_confidence + quality_adjustment - bias_penalty + uncertainty_bonus
            )
        )
        
        # Create final trace with calibrated confidence
        return ReasoningTrace.create(
            question_id=trace.question_id,
            agent_id=trace.agent_id,
            reasoning_method=trace.reasoning_method,
            steps=trace.steps,
            final_conclusion=trace.final_conclusion,
            overall_confidence=final_confidence,
            bias_checks=trace.bias_checks,
            uncertainty_sources=trace.uncertainty_sources
        )
    
    def _calculate_overall_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence from reasoning steps."""
        if not steps:
            return 0.5
        
        # Weight different step types differently
        type_weights = {
            ReasoningStepType.OBSERVATION: 0.8,
            ReasoningStepType.HYPOTHESIS: 0.6,
            ReasoningStepType.ANALYSIS: 1.0,
            ReasoningStepType.SYNTHESIS: 1.2,
            ReasoningStepType.CONCLUSION: 1.5,
            ReasoningStepType.BIAS_CHECK: 0.5,
            ReasoningStepType.UNCERTAINTY_ASSESSMENT: 0.5
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for step in steps:
            weight = type_weights.get(step.step_type, 1.0)
            weighted_sum += step.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def get_orchestrator_config(self) -> Dict[str, Any]:
        """Get current orchestrator configuration."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "max_reasoning_depth": self.max_reasoning_depth,
            "bias_detection_enabled": self.bias_detection_enabled,
            "validation_enabled": self.validation_enabled,
            "supported_bias_types": [bias_type.value for bias_type in BiasType]
        }
    
    def update_config(
        self,
        confidence_threshold: Optional[float] = None,
        max_reasoning_depth: Optional[int] = None,
        bias_detection_enabled: Optional[bool] = None,
        validation_enabled: Optional[bool] = None
    ) -> None:
        """Update orchestrator configuration."""
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError("Confidence threshold must be between 0 and 1")
            self.confidence_threshold = confidence_threshold
        
        if max_reasoning_depth is not None:
            if max_reasoning_depth < 1:
                raise ValueError("Max reasoning depth must be at least 1")
            self.max_reasoning_depth = max_reasoning_depth
        
        if bias_detection_enabled is not None:
            self.bias_detection_enabled = bias_detection_enabled
        
        if validation_enabled is not None:
            self.validation_enabled = validation_enabled
        
        self.logger.info("Orchestrator configuration updated", config=self.get_orchestrator_config())

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Protocol, Union
from uuid import UUID, uuid4
import structlog

from ..value_objects.reasoning_trace import ReasoningTrace, ReasoningStep, ReasoningStepType
from ..entities.question import Question


logger = structlog.get_logger(__name__)


class BiasType(Enum):
    """Types of cognitive biases to detect."""
    CONFIRMATION_BIAS = "confirmation_bias"
    ANCHORING_BIAS = "anchoring_bias"
    AVAILABILITY_HEURISTIC = "availability_heuristic"
    OVERCONFIDENCE_BIAS = "overconfidence_bias"
    REPRESENTATIVENESS_HEURISTIC = "representativeness_heuristic"
    BASE_RATE_NEGLECT = "base_rate_neglect"
    CONJUNCTION_FALLACY = "conjunction_fallacy"


@dataclass(frozen=True)
class BiasDetectionResult:
    """Result of bias detection analysis."""
    bias_type: BiasType
    detected: bool
    confidence: float
    evidence: str
    mitigation_suggestion: str


@dataclass(frozen=True)
class ReasoningValidationResult:
    """Result of reasoning step validation."""
    is_valid: bool
    confidence_adjustment: float
    validation_notes: str
    suggested_improvements: List[str]


class ReasoningAgent(Protocol):
    """Protocol for reasoning agents."""
    
    async def reason(self, question: Question, context: Dict[str, Any]) -> ReasoningTrace:
        """Generate reasoning trace for a question."""
        ...


class ReasoningOrchestrator:
    """
    Orchestrates multi-step reasoning processes with bias detection and validation.
    
    Manages the overall reasoning workflow, ensuring transparency, quality,
    and bias mitigation throughout the reasoning process.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_reasoning_depth: int = 10,
        bias_detection_enabled: bool = True,
        validation_enabled: bool = True
    ):
        """
        Initialize the reasoning orchestrator.
        
        Args:
            confidence_threshold: Minimum confidence required for conclusions
            max_reasoning_depth: Maximum number of reasoning steps allowed
            bias_detection_enabled: Whether to perform bias detection
            validation_enabled: Whether to validate reasoning steps
        """
        self.confidence_threshold = confidence_threshold
        self.max_reasoning_depth = max_reasoning_depth
        self.bias_detection_enabled = bias_detection_enabled
        self.validation_enabled = validation_enabled
        self.logger = logger.bind(component="reasoning_orchestrator")
        
        # Bias detection patterns
        self.bias_patterns = self._initialize_bias_patterns()
    
    def _initialize_bias_patterns(self) -> Dict[BiasType, Dict[str, Any]]:
        """Initialize bias detection patterns."""
        return {
            BiasType.CONFIRMATION_BIAS: {
                "keywords": ["confirms", "supports", "validates", "proves"],
                "pattern": "selective_evidence_focus",
                "threshold": 0.6
            },
            BiasType.ANCHORING_BIAS: {
                "keywords": ["first", "initial", "starting", "baseline"],
                "pattern": "excessive_initial_value_reliance",
                "threshold": 0.7
            },
            BiasType.AVAILABILITY_HEURISTIC: {
                "keywords": ["recent", "memorable", "vivid", "comes to mind"],
                "pattern": "recent_event_overweighting",
                "threshold": 0.6
            },
            BiasType.OVERCONFIDENCE_BIAS: {
                "keywords": ["certain", "definitely", "obviously", "clearly"],
                "pattern": "excessive_certainty_claims",
                "threshold": 0.8
            }
        }
    
    async def orchestrate_reasoning(
        self,
        question: Question,
        reasoning_agent: ReasoningAgent,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """
        Orchestrate a complete reasoning process with validation and bias detection.
        
        Args:
            question: The question to reason about
            reasoning_agent: The agent to perform reasoning
            context: Additional context for reasoning
            
        Returns:
            Complete reasoning trace with validation and bias detection
        """
        self.logger.info("Starting reasoning orchestration", question_id=str(question.id))
        
        context = context or {}
        
        try:
            # Generate initial reasoning trace
            initial_trace = await reasoning_agent.reason(question, context)
            
            # Validate reasoning steps
            validated_trace = await self._validate_reasoning_trace(initial_trace)
            
            # Detect and mitigate biases
            if self.bias_detection_enabled:
                bias_checked_trace = await self._detect_and_mitigate_biases(validated_trace)
            else:
                bias_checked_trace = validated_trace
            
            # Final confidence calibration
            final_trace = await self._calibrate_final_confidence(bias_checked_trace)
            
            self.logger.info(
                "Reasoning orchestration completed",
                question_id=str(question.id),
                steps=len(final_trace.steps),
                final_confidence=final_trace.overall_confidence
            )
            
            return final_trace
            
        except Exception as e:
            self.logger.error("Reasoning orchestration failed", error=str(e))
            raise
    
    async def _validate_reasoning_trace(self, trace: ReasoningTrace) -> ReasoningTrace:
        """Validate each step in the reasoning trace."""
        if not self.validation_enabled:
            return trace
        
        validated_steps = []
        validation_notes = []
        
        for step in trace.steps:
            validation_result = await self._validate_reasoning_step(step)
            
            # Adjust confidence based on validation
            adjusted_confidence = max(
                0.0,
                min(1.0, step.confidence + validation_result.confidence_adjustment)
            )
            
            validated_step = ReasoningStep.create(
                step_type=step.step_type,
                content=step.content,
                confidence=adjusted_confidence,
                metadata={
                    **step.metadata,
                    "validation_result": validation_result,
                    "original_confidence": step.confidence
                }
            )
            
            validated_steps.append(validated_step)
            validation_notes.extend(validation_result.suggested_improvements)
        
        # Create new trace with validated steps
        return ReasoningTrace.create(
            question_id=trace.question_id,
            agent_id=trace.agent_id,
            reasoning_method=trace.reasoning_method,
            steps=validated_steps,
            final_conclusion=trace.final_conclusion,
            overall_confidence=self._calculate_overall_confidence(validated_steps),
            bias_checks=trace.bias_checks,
            uncertainty_sources=trace.uncertainty_sources + validation_notes
        )
    
    async def _validate_reasoning_step(self, step: ReasoningStep) -> ReasoningValidationResult:
        """Validate a single reasoning step."""
        is_valid = True
        confidence_adjustment = 0.0
        validation_notes = ""
        suggested_improvements = []
        
        # Check step content quality
        if len(step.content.strip()) < 10:
            is_valid = False
            confidence_adjustment -= 0.2
            suggested_improvements.append("Reasoning step content is too brief")
        
        # Check confidence calibration
        if step.confidence > 0.9 and step.step_type not in [
            ReasoningStepType.OBSERVATION, ReasoningStepType.CONCLUSION
        ]:
            confidence_adjustment -= 0.1
            suggested_improvements.append("High confidence may indicate overconfidence")
        
        # Check for logical consistency
        if step.step_type == ReasoningStepType.CONCLUSION and step.confidence < 0.3:
            suggested_improvements.append("Low confidence conclusion may need more analysis")
        
        validation_notes = f"Step validation completed. Adjustments: {confidence_adjustment}"
        
        return ReasoningValidationResult(
            is_valid=is_valid,
            confidence_adjustment=confidence_adjustment,
            validation_notes=validation_notes,
            suggested_improvements=suggested_improvements
        )
    
    async def _detect_and_mitigate_biases(self, trace: ReasoningTrace) -> ReasoningTrace:
        """Detect and mitigate cognitive biases in reasoning trace."""
        bias_detection_results = []
        mitigation_steps = []
        
        for bias_type in BiasType:
            detection_result = await self._detect_bias(trace, bias_type)
            bias_detection_results.append(detection_result)
            
            if detection_result.detected:
                mitigation_step = await self._create_bias_mitigation_step(
                    detection_result, trace
                )
                mitigation_steps.append(mitigation_step)
        
        # Add bias detection steps to trace
        enhanced_steps = list(trace.steps)
        enhanced_steps.extend(mitigation_steps)
        
        # Update bias checks
        bias_checks = list(trace.bias_checks)
        bias_checks.extend([
            f"{result.bias_type.value}: {'detected' if result.detected else 'not detected'}"
            for result in bias_detection_results
        ])
        
        return ReasoningTrace.create(
            question_id=trace.question_id,
            agent_id=trace.agent_id,
            reasoning_method=trace.reasoning_method,
            steps=enhanced_steps,
            final_conclusion=trace.final_conclusion,
            overall_confidence=self._calculate_overall_confidence(enhanced_steps),
            bias_checks=bias_checks,
            uncertainty_sources=trace.uncertainty_sources
        )
    
    async def _detect_bias(self, trace: ReasoningTrace, bias_type: BiasType) -> BiasDetectionResult:
        """Detect a specific type of bias in the reasoning trace."""
        pattern_config = self.bias_patterns.get(bias_type, {})
        keywords = pattern_config.get("keywords", [])
        threshold = pattern_config.get("threshold", 0.5)
        
        # Simple keyword-based detection (can be enhanced with ML models)
        total_content = " ".join([step.content.lower() for step in trace.steps])
        keyword_matches = sum(1 for keyword in keywords if keyword in total_content)
        
        detection_confidence = min(1.0, keyword_matches / len(keywords) if keywords else 0.0)
        detected = detection_confidence >= threshold
        
        evidence = f"Found {keyword_matches} bias indicators out of {len(keywords)} patterns"
        mitigation_suggestion = self._get_bias_mitigation_suggestion(bias_type)
        
        return BiasDetectionResult(
            bias_type=bias_type,
            detected=detected,
            confidence=detection_confidence,
            evidence=evidence,
            mitigation_suggestion=mitigation_suggestion
        )
    
    def _get_bias_mitigation_suggestion(self, bias_type: BiasType) -> str:
        """Get mitigation suggestion for a specific bias type."""
        suggestions = {
            BiasType.CONFIRMATION_BIAS: "Consider alternative explanations and contradictory evidence",
            BiasType.ANCHORING_BIAS: "Evaluate multiple reference points and starting assumptions",
            BiasType.AVAILABILITY_HEURISTIC: "Seek broader data beyond recent or memorable examples",
            BiasType.OVERCONFIDENCE_BIAS: "Quantify uncertainty and consider what could go wrong",
            BiasType.REPRESENTATIVENESS_HEURISTIC: "Consider base rates and statistical reasoning",
            BiasType.BASE_RATE_NEGLECT: "Incorporate prior probabilities and base rate information",
            BiasType.CONJUNCTION_FALLACY: "Evaluate individual probabilities separately"
        }
        return suggestions.get(bias_type, "Apply general debiasing techniques")
    
    async def _create_bias_mitigation_step(
        self,
        detection_result: BiasDetectionResult,
        trace: ReasoningTrace
    ) -> ReasoningStep:
        """Create a reasoning step to mitigate detected bias."""
        content = f"Bias mitigation for {detection_result.bias_type.value}: {detection_result.mitigation_suggestion}"
        
        return ReasoningStep.create(
            step_type=ReasoningStepType.BIAS_CHECK,
            content=content,
            confidence=0.8,  # High confidence in bias mitigation process
            metadata={
                "bias_type": detection_result.bias_type.value,
                "detection_confidence": detection_result.confidence,
                "evidence": detection_result.evidence
            }
        )
    
    async def _calibrate_final_confidence(self, trace: ReasoningTrace) -> ReasoningTrace:
        """Calibrate the final confidence based on reasoning quality."""
        # Calculate base confidence from steps
        step_confidences = [step.confidence for step in trace.steps]
        base_confidence = sum(step_confidences) / len(step_confidences) if step_confidences else 0.5
        
        # Apply quality adjustments
        quality_score = trace.get_reasoning_quality_score()
        quality_adjustment = (quality_score - 0.5) * 0.2  # Scale quality impact
        
        # Apply bias detection penalty
        bias_penalty = 0.0
        if trace.has_bias_checks():
            detected_biases = len([check for check in trace.bias_checks if "detected" in check])
            bias_penalty = detected_biases * 0.05  # Small penalty per detected bias
        
        # Apply uncertainty bonus (acknowledging uncertainty is good)
        uncertainty_bonus = 0.05 if trace.has_uncertainty_assessment() else 0.0
        
        # Calculate final confidence
        final_confidence = max(
            0.1,  # Minimum confidence
            min(
                0.95,  # Maximum confidence (leave room for uncertainty)
                base_confidence + quality_adjustment - bias_penalty + uncertainty_bonus
            )
        )
        
        # Create final trace with calibrated confidence
        return ReasoningTrace.create(
            question_id=trace.question_id,
            agent_id=trace.agent_id,
            reasoning_method=trace.reasoning_method,
            steps=trace.steps,
            final_conclusion=trace.final_conclusion,
            overall_confidence=final_confidence,
            bias_checks=trace.bias_checks,
            uncertainty_sources=trace.uncertainty_sources
        )
    
    def _calculate_overall_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence from reasoning steps."""
        if not steps:
            return 0.5
        
        # Weight different step types differently
        type_weights = {
            ReasoningStepType.OBSERVATION: 0.8,
            ReasoningStepType.HYPOTHESIS: 0.6,
            ReasoningStepType.ANALYSIS: 1.0,
            ReasoningStepType.SYNTHESIS: 1.2,
            ReasoningStepType.CONCLUSION: 1.5,
            ReasoningStepType.BIAS_CHECK: 0.5,
            ReasoningStepType.UNCERTAINTY_ASSESSMENT: 0.5
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for step in steps:
            weight = type_weights.get(step.step_type, 1.0)
            weighted_sum += step.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def get_orchestrator_config(self) -> Dict[str, Any]:
        """Get current orchestrator configuration."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "max_reasoning_depth": self.max_reasoning_depth,
            "bias_detection_enabled": self.bias_detection_enabled,
            "validation_enabled": self.validation_enabled,
            "supported_bias_types": [bias_type.value for bias_type in BiasType]
        }
    
    def update_config(
        self,
        confidence_threshold: Optional[float] = None,
        max_reasoning_depth: Optional[int] = None,
        bias_detection_enabled: Optional[bool] = None,
        validation_enabled: Optional[bool] = None
    ) -> None:
        """Update orchestrator configuration."""
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError("Confidence threshold must be between 0 and 1")
            self.confidence_threshold = confidence_threshold
        
        if max_reasoning_depth is not None:
            if max_reasoning_depth < 1:
                raise ValueError("Max reasoning depth must be at least 1")
            self.max_reasoning_depth = max_reasoning_depth
        
        if bias_detection_enabled is not None:
            self.bias_detection_enabled = bias_detection_enabled
        
        if validation_enabled is not None:
            self.validation_enabled = validation_enabled
        
        self.logger.info("Orchestrator configuration updated", config=self.get_orchestrator_config())
