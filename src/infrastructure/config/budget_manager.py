"""
Budget management and cost tracking for tournament API usage.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CostTrackingRecord:
    """Record for tracking individual API call costs."""

    timestamp: datetime
    question_id: str
    model_used: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    task_type: str  # "research" or "forecast"
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostTrackingRecord":
        """Create from dictionary for JSON deserialization."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class BudgetStatus:
    """Current budget status and utilization metrics."""

    total_budget: float
    spent: float
    remaining: float
    utilization_percentage: float
    questions_processed: int
    average_cost_per_question: float
    estimated_questions_remaining: int
    status_level: str  # "normal", "conservative", "emergency"
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["last_updated"] = self.last_updated.isoformat()
        return data


class BudgetManager:
    """Manages API budget tracking and cost estimation for tournament usage."""

    def __init__(self, budget_limit: Optional[float] = None):
        """Initialize budget manager with configurable budget limit."""
        self.budget_limit = budget_limit or float(os.getenv("BUDGET_LIMIT", "100.0"))
        self.current_spend = 0.0
        self.questions_processed = 0
        self.cost_records: List[CostTrackingRecord] = []

        # OpenRouter pricing (per 1K tokens) as of 2024 (verify: https://openrouter.ai/models)
        self.cost_per_token = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            # GPT-5 tiers routed via OpenRouter; keep placeholders conservative until verified
            "gpt-5": {"input": 0.0015, "output": 0.006},
            "gpt-5-mini": {"input": 0.00025, "output": 0.001},
            "gpt-5-nano": {"input": 0.00005, "output": 0.0002},
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "perplexity/sonar-reasoning": {"input": 0.005, "output": 0.005},
            "perplexity/sonar-pro": {"input": 0.001, "output": 0.001},
        }

        # Load existing data if available
        self.data_file = Path("logs/budget_tracking.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_existing_data()

        logger.info(f"BudgetManager initialized with ${self.budget_limit} limit")
        logger.info(f"Current spend: ${self.current_spend:.4f}")

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a model call based on token usage."""
        # Normalize model name for cost lookup
        model_key = self._normalize_model_name(model)

        # Treat explicit free-tier models as zero-cost
        if ":free" in model:
            return 0.0

        if model_key not in self.cost_per_token:
            # Default to zero-cost for unknown/unpriced models to avoid noisy warnings for free tiers
            logger.warning(f"Unknown model {model}, treating as zero-cost for safety")
            return 0.0

        rates = self.cost_per_token[model_key]
        cost = (input_tokens * rates["input"] / 1000) + (
            output_tokens * rates["output"] / 1000
        )

        return cost

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for cost lookup."""
        # Remove provider prefixes
        if "/" in model:
            model = model.split("/")[-1]

        # Handle common variations
        model_mappings = {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "claude-3-5-sonnet": "claude-3-5-sonnet",
            "claude-3-haiku": "claude-3-haiku",
            "sonar-reasoning": "perplexity/sonar-reasoning",
            "sonar-pro": "perplexity/sonar-pro",
        }

        return model_mappings.get(model, model)

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if we can afford the estimated cost within budget limits."""
        # Use 95% of budget as safety margin
        safety_limit = self.budget_limit * 0.95
        return (self.current_spend + estimated_cost) <= safety_limit

    def record_cost(
        self,
        question_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str,
        success: bool = True,
    ) -> float:
        """Record actual cost for an API call."""
        estimated_cost = self.estimate_cost(model, input_tokens, output_tokens)

        record = CostTrackingRecord(
            timestamp=datetime.now(),
            question_id=question_id,
            model_used=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            task_type=task_type,
            success=success,
        )

        self.cost_records.append(record)
        self.current_spend += estimated_cost

        # Update questions processed count for successful forecasts
        if task_type == "forecast" and success:
            self.questions_processed += 1

        # Save data periodically
        if len(self.cost_records) % 10 == 0:
            self._save_data()

        logger.debug(
            f"Recorded cost: ${estimated_cost:.4f} for {task_type} on {question_id}"
        )

        return estimated_cost

    def get_budget_status(self) -> BudgetStatus:
        """Get current budget utilization status."""
        remaining = self.budget_limit - self.current_spend
        utilization = (self.current_spend / self.budget_limit) * 100

        # Calculate average cost per question
        avg_cost = self.current_spend / max(self.questions_processed, 1)

        # Estimate remaining questions based on average cost
        estimated_remaining = int(remaining / max(avg_cost, 0.01))

        # Determine status level
        status_level = self._get_status_level(utilization / 100)

        return BudgetStatus(
            total_budget=self.budget_limit,
            spent=self.current_spend,
            remaining=remaining,
            utilization_percentage=utilization,
            questions_processed=self.questions_processed,
            average_cost_per_question=avg_cost,
            estimated_questions_remaining=estimated_remaining,
            status_level=status_level,
            last_updated=datetime.now(),
        )

    def get_remaining_budget(self) -> float:
        """Return remaining budget in dollars (helper expected by tests)."""
        try:
            status = self.get_budget_status()
            return max(0.0, float(status.remaining))
        except Exception:
            # Conservative fallback
            return max(0.0, float(self.budget_limit - self.current_spend))

    def _get_status_level(self, utilization: float) -> str:
        """Determine budget status level based on utilization."""
        if utilization < 0.8:
            return "normal"
        elif utilization < 0.95:
            return "conservative"
        else:
            return "emergency"

    def should_alert_budget_usage(self) -> bool:
        """Check if budget usage warrants an alert."""
        utilization = self.current_spend / self.budget_limit
        return utilization >= 0.8

    def get_budget_alert_level(self) -> str:
        """Get budget alert level for logging."""
        utilization = self.current_spend / self.budget_limit
        if utilization >= 0.95:
            return "CRITICAL"
        elif utilization >= 0.9:
            return "HIGH"
        elif utilization >= 0.8:
            return "WARNING"
        else:
            return "NORMAL"

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown by model and task type."""
        breakdown = {
            "by_model": {},
            "by_task_type": {},
            "by_day": {},
            "total_tokens": {"input": 0, "output": 0},
        }

        for record in self.cost_records:
            # By model
            model = record.model_used
            if model not in breakdown["by_model"]:
                breakdown["by_model"][model] = {
                    "cost": 0,
                    "calls": 0,
                    "tokens": {"input": 0, "output": 0},
                }
            breakdown["by_model"][model]["cost"] += record.estimated_cost
            breakdown["by_model"][model]["calls"] += 1
            breakdown["by_model"][model]["tokens"]["input"] += record.input_tokens
            breakdown["by_model"][model]["tokens"]["output"] += record.output_tokens

            # By task type
            task = record.task_type
            if task not in breakdown["by_task_type"]:
                breakdown["by_task_type"][task] = {"cost": 0, "calls": 0}
            breakdown["by_task_type"][task]["cost"] += record.estimated_cost
            breakdown["by_task_type"][task]["calls"] += 1

            # By day
            day = record.timestamp.date().isoformat()
            if day not in breakdown["by_day"]:
                breakdown["by_day"][day] = {"cost": 0, "calls": 0}
            breakdown["by_day"][day]["cost"] += record.estimated_cost
            breakdown["by_day"][day]["calls"] += 1

            # Total tokens
            breakdown["total_tokens"]["input"] += record.input_tokens
            breakdown["total_tokens"]["output"] += record.output_tokens

        return breakdown

    def _save_data(self):
        """Save budget tracking data to file."""
        try:
            data = {
                "budget_limit": self.budget_limit,
                "current_spend": self.current_spend,
                "questions_processed": self.questions_processed,
                "cost_records": [record.to_dict() for record in self.cost_records],
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save budget data: {e}")

    def _load_existing_data(self):
        """Load existing budget tracking data if available."""
        try:
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    data = json.load(f)

                self.current_spend = data.get("current_spend", 0.0)
                self.questions_processed = data.get("questions_processed", 0)

                # Load cost records
                records_data = data.get("cost_records", [])
                self.cost_records = [
                    CostTrackingRecord.from_dict(record) for record in records_data
                ]

                logger.info(
                    f"Loaded existing budget data: ${self.current_spend:.4f} spent, "
                    f"{self.questions_processed} questions processed"
                )

        except Exception as e:
            logger.warning(f"Failed to load existing budget data: {e}")

    def reset_budget(self, new_limit: Optional[float] = None):
        """Reset budget tracking (use with caution)."""
        if new_limit:
            self.budget_limit = new_limit

        self.current_spend = 0.0
        self.questions_processed = 0
        self.cost_records = []

        # Remove existing data file
        if self.data_file.exists():
            self.data_file.unlink()

        logger.warning(f"Budget tracking reset. New limit: ${self.budget_limit}")

    def log_budget_status(self):
        """Log current budget status."""
        status = self.get_budget_status()

        logger.info("=== Budget Status ===")
        logger.info(f"Total Budget: ${status.total_budget:.2f}")
        logger.info(
            f"Spent: ${status.spent:.4f} ({status.utilization_percentage:.1f}%)"
        )
        logger.info(f"Remaining: ${status.remaining:.4f}")
        logger.info(f"Questions Processed: {status.questions_processed}")
        logger.info(f"Average Cost/Question: ${status.average_cost_per_question:.4f}")
        logger.info(
            f"Estimated Questions Remaining: {status.estimated_questions_remaining}"
        )
        logger.info(f"Status Level: {status.status_level.upper()}")

        # Alert if budget usage is high
        if self.should_alert_budget_usage():
            alert_level = self.get_budget_alert_level()
            logger.warning(
                "%s BUDGET USAGE: %.1f%% of budget used!",
                alert_level,
                status.utilization_percentage,
            )


# Global instance
budget_manager = BudgetManager()
