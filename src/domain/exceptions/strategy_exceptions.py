"""
Strategy-specific exceptions for the tournament optimization system.
"""

from typing import Optional, Dict, Any, List
from .base_exceptions import DomainError


class StrategyError(DomainError):
    """Base class for strategy-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.context.component = "strategy"


class StrategySelectionError(StrategyError):
    """
    Raised when strategy selection fails.

    Includes information about available strategies and
    the criteria that could not be satisfied.
    """

    def __init__(
        self,
        message: str,
        available_strategies: Optional[List[str]] = None,
        selection_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.available_strategies = available_strategies or []
        self.selection_criteria = selection_criteria or {}
        self.context.metadata.update({
            "available_strategies": available_strategies,
            "selection_criteria": selection_criteria,
        })
        self.context.operation = "strategy_selection"


class StrategyExecutionError(StrategyError):
    """
    Raised when strategy execution fails.

    Includes information about the strategy being executed and
    the specific execution step that failed.
    """

    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        execution_step: Optional[str] = None,
        partial_results: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.strategy_name = strategy_name
        self.execution_step = execution_step
        self.partial_results = partial_results or {}
        self.context.metadata.update({
            "strategy_name": strategy_name,
            "execution_step": execution_step,
            "partial_results": partial_results,
        })
        self.context.operation = "strategy_execution"
        self.recoverable = bool(partial_results)


class TournamentAnalysisError(StrategyError):
    """
    Raised when tournament analysis fails.

    Includes information about the analysis type and
    the tournament data that could not be analyzed.
    """

    def __init__(
        self,
        message: str,
        analysis_type: Optional[str] = None,
        tournament_id: Optional[str] = None,
        missing_data: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.analysis_type = analysis_type
        self.tournament_id = tournament_id
        self.missing_data = missing_data or []
        self.context.metadata.update({
            "analysis_type": analysis_type,
            "tournament_id": tournament_id,
            "missing_data": missing_data,
        })
        self.context.operation = "tournament_analysis"
        self.recoverable = True


class PrioritizationError(StrategyError):
    """
    Raised when question prioritization fails.

    Includes information about prioritization criteria and
    the questions that could not be prioritized.
    """

    def __init__(
        self,
        message: str,
        prioritization_criteria: Optional[Dict[str, Any]] = None,
        failed_questions: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.prioritization_criteria = prioritization_criteria or {}
        self.failed_questions = failed_questions or []
        self.context.metadata.update({
            "prioritization_criteria": prioritization_criteria,
            "failed_questions": failed_questions,
        })
        self.context.operation = "question_prioritization"
        self.recoverable = True


class StrategyOptimizationError(StrategyError):
    """
    Raised when strategy optimization fails.

    Includes information about optimization parameters and
    the current optimization state.
    """

    def __init__(
        self,
        message: str,
        optimization_target: Optional[str] = None,
        current_performance: Optional[float] = None,
        target_performance: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.optimization_target = optimization_target
        self.current_performance = current_performance
        self.target_performance = target_performance
        self.context.metadata.update({
            "optimization_target": optimization_target,
            "current_performance": current_performance,
            "target_performance": target_performance,
        })
        self.context.operation = "strategy_optimization"
        self.recoverable = True


class StrategyValidationError(StrategyError):
    """
    Raised when strategy validation fails.

    Includes information about validation rules and
    the specific validation failures.
    """

    def __init__(
        self,
        message: str,
        validation_rules: Optional[List[str]] = None,
        validation_failures: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.validation_rules = validation_rules or []
        self.validation_failures = validation_failures or {}
        self.context.metadata.update({
            "validation_rules": validation_rules,
            "validation_failures": validation_failures,
        })
        self.context.operation = "strategy_validation"
