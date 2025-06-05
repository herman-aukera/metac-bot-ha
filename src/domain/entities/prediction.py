"""Prediction domain entity."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PredictionMethod(Enum):
    """Methods used to generate predictions."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REACT = "react"
    AUTO_COT = "auto_cot"
    SELF_CONSISTENCY = "self_consistency"
    ENSEMBLE = "ensemble"


@dataclass
class PredictionResult:
    """Represents the actual prediction value(s)."""
    binary_probability: Optional[float] = None
    numeric_value: Optional[float] = None
    numeric_distribution: Optional[Dict[str, float]] = None
    multiple_choice_probabilities: Optional[Dict[str, float]] = None


@dataclass
class Prediction:
    """
    Domain entity representing a prediction for a question.
    
    Contains the prediction value, confidence, reasoning,
    and metadata about how the prediction was generated.
    """
    id: UUID
    question_id: UUID
    research_report_id: UUID
    result: PredictionResult
    confidence: PredictionConfidence
    method: PredictionMethod
    reasoning: str
    reasoning_steps: List[str]
    created_at: datetime
    created_by: str  # Agent identifier
    
    # Uncertainty quantification
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    confidence_interval: Optional[float] = None  # e.g., 0.95 for 95% CI
    
    # Method-specific metadata
    method_metadata: Dict[str, Any] = None
    
    # Validation and quality metrics
    internal_consistency_score: Optional[float] = None
    evidence_strength: Optional[float] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.method_metadata is None:
            self.method_metadata = {}
    
    @classmethod
    def create(
        cls,
        question_id: UUID,
        research_report_id: UUID,
        result: PredictionResult,
        confidence: PredictionConfidence,
        method: PredictionMethod,
        reasoning: str,
        created_by: str,
        **kwargs
    ) -> "Prediction":
        """Generic factory method for predictions."""
        return cls(
            id=uuid4(),
            question_id=question_id,
            research_report_id=research_report_id,
            result=result,
            confidence=confidence,
            method=method,
            reasoning=reasoning,
            reasoning_steps=kwargs.get("reasoning_steps", []),
            created_at=datetime.utcnow(),
            created_by=created_by,
            lower_bound=kwargs.get("lower_bound"),
            upper_bound=kwargs.get("upper_bound"),
            confidence_interval=kwargs.get("confidence_interval"),
            method_metadata=kwargs.get("method_metadata", {}),
            internal_consistency_score=kwargs.get("internal_consistency_score"),
            evidence_strength=kwargs.get("evidence_strength"),
        )
    
    @classmethod
    def create_binary_prediction(
        cls,
        question_id: UUID,
        research_report_id: UUID,
        probability: float,
        confidence: PredictionConfidence,
        method: PredictionMethod,
        reasoning: str,
        created_by: str,
        **kwargs
    ) -> "Prediction":
        """Factory method for binary predictions."""
        if not 0 <= probability <= 1:
            raise ValueError("Binary probability must be between 0 and 1")
        
        result = PredictionResult(binary_probability=probability)
        
        return cls(
            id=uuid4(),
            question_id=question_id,
            research_report_id=research_report_id,
            result=result,
            confidence=confidence,
            method=method,
            reasoning=reasoning,
            reasoning_steps=kwargs.get("reasoning_steps", []),
            created_at=datetime.utcnow(),
            created_by=created_by,
            lower_bound=kwargs.get("lower_bound"),
            upper_bound=kwargs.get("upper_bound"),
            confidence_interval=kwargs.get("confidence_interval"),
            method_metadata=kwargs.get("method_metadata", {}),
            internal_consistency_score=kwargs.get("internal_consistency_score"),
            evidence_strength=kwargs.get("evidence_strength"),
        )
    
    @classmethod
    def create_numeric_prediction(
        cls,
        question_id: UUID,
        research_report_id: UUID,
        value: float,
        confidence: PredictionConfidence,
        method: PredictionMethod,
        reasoning: str,
        created_by: str,
        **kwargs
    ) -> "Prediction":
        """Factory method for numeric predictions."""
        result = PredictionResult(numeric_value=value)
        
        return cls(
            id=uuid4(),
            question_id=question_id,
            research_report_id=research_report_id,
            result=result,
            confidence=confidence,
            method=method,
            reasoning=reasoning,
            reasoning_steps=kwargs.get("reasoning_steps", []),
            created_at=datetime.utcnow(),
            created_by=created_by,
            lower_bound=kwargs.get("lower_bound"),
            upper_bound=kwargs.get("upper_bound"),
            confidence_interval=kwargs.get("confidence_interval"),
            method_metadata=kwargs.get("method_metadata", {}),
            internal_consistency_score=kwargs.get("internal_consistency_score"),
            evidence_strength=kwargs.get("evidence_strength"),
        )
    
    @classmethod
    def create_multiple_choice_prediction(
        cls,
        question_id: UUID,
        research_report_id: UUID,
        choice_probabilities: Dict[str, float],
        confidence: PredictionConfidence,
        method: PredictionMethod,
        reasoning: str,
        created_by: str,
        **kwargs
    ) -> "Prediction":
        """Factory method for multiple choice predictions."""
        # Validate probabilities sum to 1
        total_prob = sum(choice_probabilities.values())
        if abs(total_prob - 1.0) > 0.01:
            raise ValueError(f"Choice probabilities must sum to 1, got {total_prob}")
        
        result = PredictionResult(multiple_choice_probabilities=choice_probabilities)
        
        return cls(
            id=uuid4(),
            question_id=question_id,
            research_report_id=research_report_id,
            result=result,
            confidence=confidence,
            method=method,
            reasoning=reasoning,
            reasoning_steps=kwargs.get("reasoning_steps", []),
            created_at=datetime.utcnow(),
            created_by=created_by,
            method_metadata=kwargs.get("method_metadata", {}),
            internal_consistency_score=kwargs.get("internal_consistency_score"),
            evidence_strength=kwargs.get("evidence_strength"),
        )
    
    def get_confidence_score(self) -> float:
        """Convert confidence enum to numeric score."""
        confidence_mapping = {
            PredictionConfidence.VERY_LOW: 0.2,
            PredictionConfidence.LOW: 0.4,
            PredictionConfidence.MEDIUM: 0.6,
            PredictionConfidence.HIGH: 0.8,
            PredictionConfidence.VERY_HIGH: 1.0,
        }
        return confidence_mapping[self.confidence]
