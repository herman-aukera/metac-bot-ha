"""
Token counting and tracking utilities for accurate cost estimation.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping

import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class TokenUsageRecord:
    """Record for tracking token usage per API call."""

    timestamp: datetime
    question_id: str
    model_used: str
    task_type: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    actual_cost: Optional[float] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenUsageRecord":
        """Create from dictionary for JSON deserialization."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class TokenTracker:
    """Tracks token usage for accurate cost estimation and real-time cost calculation."""

    def __init__(self) -> None:
        """Initialize token tracker with model encodings and cost tracking."""
        self.encodings: Dict[str, Any] = {}
        self.usage_records: List[TokenUsageRecord] = []
        self.total_tokens_used = {"input": 0, "output": 0, "total": 0}
        self.total_estimated_cost = 0.0

        # Pricing (per 1K tokens). Include GPT-4o family for legacy/test compatibility,
        # plus GPT-5 tiers. Unknown models default to gpt-4o pricing for cost estimation.
        self.cost_per_token = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-5": {"input": 0.0015, "output": 0.0015},  # assumed balanced pricing
            "gpt-5-mini": {"input": 0.0024, "output": 0.0096},
            "gpt-5-nano": {"input": 0.00012, "output": 0.0005},
            "claude-3-5-sonnet": {"input": 0.0030, "output": 0.0150},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        }

        # Data persistence
        self.data_file = Path("logs/token_usage.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_encodings()
        self._load_existing_data()

    def _initialize_encodings(self) -> None:
        """Initialize tiktoken encodings for different models."""
        try:
            # Use cl100k_base encoding as approximation across supported models
            self.encodings["gpt-5-mini"] = tiktoken.get_encoding("cl100k_base")
            self.encodings["gpt-5-nano"] = tiktoken.get_encoding("cl100k_base")
            self.encodings["gpt-4o"] = tiktoken.get_encoding("cl100k_base")
            self.encodings["gpt-4o-mini"] = tiktoken.get_encoding("cl100k_base")

            # Claude models - approximate using cl100k_base
            self.encodings["claude-3-5-sonnet"] = tiktoken.get_encoding("cl100k_base")
            self.encodings["claude-3-haiku"] = tiktoken.get_encoding("cl100k_base")

            logger.debug("Token encodings initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize some token encodings: {e}")

    def count_tokens(self, text: str, model: str = "gpt-5-mini") -> int:
        """Count tokens in text for a specific model."""
        if not text:
            return 0

        # Normalize model name
        model_key = self._normalize_model_name(model)
        # Get appropriate encoding (fallback to gpt-5-mini if missing)
        encoding = self.encodings.get(model_key) or self.encodings.get("gpt-5-mini")

        if not encoding:
            # Fallback: rough estimation (1 token ≈ 4 characters for English)
            logger.warning(
                f"No encoding available for {model}, using character-based estimation"
            )
            return max(1, len(text) // 4)

        try:
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Failed to count tokens for {model}: {e}")
            # Fallback estimation
            return max(1, len(text) // 4)

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for encoding lookup."""
        # Remove provider prefixes
        if "/" in model:
            model = model.split("/")[-1]

        # Handle common variations
        model_mappings: Mapping[str, str] = {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-5": "gpt-5",
            "gpt-5-mini": "gpt-5-mini",
            "gpt-5-nano": "gpt-5-nano",
            "claude-3-5-sonnet": "claude-3-5-sonnet",
            "claude-3-haiku": "claude-3-haiku",
        }

        return model_mappings.get(
            model, "gpt-5-mini"
        )  # Default to gpt-5-mini for encoding

    def estimate_tokens_for_prompt(
        self, prompt: str, model: str = "gpt-5-mini"
    ) -> Dict[str, int]:
        """Estimate input and output tokens for a prompt."""
        input_tokens = self.count_tokens(prompt, model)

        # Estimate output tokens based on prompt complexity and model
        output_tokens = self._estimate_output_tokens(prompt, model)

        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "total_estimated_tokens": input_tokens + output_tokens,
        }

    def _estimate_output_tokens(self, prompt: str, model: str) -> int:
        """Estimate output tokens based on prompt characteristics."""
        # Base estimation factors
        prompt_length = len(prompt)

        # Different models have different typical response lengths
        model_factors = {
            "gpt-5-mini": 0.3,
            "gpt-5-nano": 0.25,
            "claude-3-5-sonnet": 0.4,
            "claude-3-haiku": 0.2,
        }

        model_key = self._normalize_model_name(model)
        factor = model_factors.get(model_key, 0.3)

        # Adjust based on prompt type
        if "research" in prompt.lower():
            factor *= 1.5  # Research responses tend to be longer
        elif "forecast" in prompt.lower():
            factor *= 1.2  # Forecast responses are moderately long
        elif "summary" in prompt.lower():
            factor *= 0.8  # Summaries are shorter

        # Base estimation: output is typically 20-40% of input length
        base_estimate = int(
            prompt_length * 0.05 * factor
        )  # Convert chars to rough token estimate

        # Set reasonable bounds
        min_tokens = 50  # Minimum reasonable response
        max_tokens = 2000  # Maximum expected response

        return max(min_tokens, min(base_estimate, max_tokens))

    def calculate_real_time_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate real-time cost for token usage."""
        model_key = self._normalize_model_name(model)

        # Suppress costs for explicit free-tier models
        if ":free" in model:
            return 0.0

        if model_key not in self.cost_per_token:
            logger.debug(
                f"Unknown model {model}, defaulting to gpt-4o pricing for estimation"
            )
            model_key = "gpt-4o"

        rates = self.cost_per_token[model_key]
        cost = (input_tokens * rates["input"] / 1000) + (
            output_tokens * rates["output"] / 1000
        )

        return cost

    def track_api_call(
        self,
        question_id: str,
        model: str,
        task_type: str,
        input_tokens: int,
        output_tokens: int,
        success: bool = True,
        actual_cost: Optional[float] = None,
    ) -> TokenUsageRecord:
        """Track token usage and cost for an API call."""
        total_tokens = input_tokens + output_tokens
        estimated_cost = self.calculate_real_time_cost(
            model, input_tokens, output_tokens
        )

        record = TokenUsageRecord(
            timestamp=datetime.now(),
            question_id=question_id,
            model_used=model,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            success=success,
        )

        self.usage_records.append(record)

        # Update totals
        if success:
            self.total_tokens_used["input"] += input_tokens
            self.total_tokens_used["output"] += output_tokens
            self.total_tokens_used["total"] += total_tokens
            self.total_estimated_cost += estimated_cost

        # Save data periodically
        if len(self.usage_records) % 5 == 0:
            self._save_data()

        logger.debug(
            f"Tracked API call: {task_type} for {question_id}, "
            f"tokens: {input_tokens}+{output_tokens}={total_tokens}, "
            f"cost: ${estimated_cost:.4f}"
        )

        return record

    def track_actual_usage(
        self,
        prompt: str,
        response: str,
        model: str,
        question_id: str = "unknown",
        task_type: str = "general",
    ) -> Dict[str, Any]:
        """Track actual token usage for a completed API call."""
        input_tokens = self.count_tokens(prompt, model)
        output_tokens = self.count_tokens(response, model)

        # Track the usage
        record = self.track_api_call(
            question_id, model, task_type, input_tokens, output_tokens
        )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost": record.estimated_cost,
            "record": record,
        }

    def extract_token_usage_from_response(
        self, response_data: Any
    ) -> Optional[Dict[str, int]]:
        """Extract token usage from API response if available."""
        try:
            # Handle different response formats
            if hasattr(response_data, "usage"):
                usage = response_data.usage
                return {
                    "input_tokens": getattr(usage, "prompt_tokens", 0),
                    "output_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
            elif isinstance(response_data, dict) and "usage" in response_data:
                usage = response_data["usage"]
                return {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
        except Exception as e:
            logger.debug(f"Could not extract token usage from response: {e}")

        return None

    def get_model_context_limits(self, model: str) -> Dict[str, int]:
        """Get context limits for different models."""
        model_key = self._normalize_model_name(model)

        limits = {
            "gpt-5": {"context": 200000, "output": 8192},
            "gpt-5-mini": {"context": 128000, "output": 6144},
            "gpt-5-nano": {"context": 64000, "output": 4096},
            "claude-3-5-sonnet": {"context": 200000, "output": 4096},
            "claude-3-haiku": {"context": 200000, "output": 4096},
        }

        return limits.get(model_key, {"context": 8192, "output": 4096})

    def validate_prompt_length(
        self, prompt: str, model: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate that prompt fits within model context limits."""
        token_count = self.count_tokens(prompt, model)
        limits = self.get_model_context_limits(model)

        # Reserve space for output
        available_input_tokens = limits["context"] - limits["output"]

        is_valid = token_count <= available_input_tokens

        return is_valid, {
            "prompt_tokens": token_count,
            "context_limit": limits["context"],
            "output_limit": limits["output"],
            "available_input_tokens": available_input_tokens,
            "tokens_over_limit": max(0, token_count - available_input_tokens),
        }

    def truncate_prompt_if_needed(
        self,
        prompt: str,
        model: str,
        preserve_start: int = 1000,
        preserve_end: int = 1000,
    ) -> Tuple[str, bool]:
        """Truncate prompt if it exceeds model limits, preserving start and end."""
        is_valid, validation_info = self.validate_prompt_length(prompt, model)

        if is_valid:
            return prompt, False

        # Calculate how much we need to remove (implicit via context limits below)

        # Convert token counts to approximate character counts for truncation
        chars_per_token = 4  # Rough approximation
        preserve_start_chars = preserve_start * chars_per_token
        preserve_end_chars = preserve_end * chars_per_token
        # chars_to_remove not needed; we truncate structurally below

        if len(prompt) <= preserve_start_chars + preserve_end_chars:
            # Prompt is too short to truncate safely, just take the beginning
            max_chars = validation_info["available_input_tokens"] * chars_per_token
            truncated = prompt[:max_chars]
        else:
            # Truncate from the middle
            start_part = prompt[:preserve_start_chars]
            end_part = prompt[-preserve_end_chars:]

            truncated = (
                start_part + "\n\n[... content truncated for length ...]\n\n" + end_part
            )

        logger.warning(
            f"Prompt truncated for {model}: {len(prompt)} -> {len(truncated)} characters"
        )

        return truncated, True

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        if not self.usage_records:
            return {
                "total_calls": 0,
                "total_tokens": {"input": 0, "output": 0, "total": 0},
                "total_cost": 0.0,
                "by_model": {},
                "by_task_type": {},
                "success_rate": 0.0,
            }
        summary: Dict[str, Any] = {
            "total_calls": len(self.usage_records),
            "total_tokens": self.total_tokens_used.copy(),
            "total_cost": self.total_estimated_cost,
            "by_model": {},
            "by_task_type": {},
            "by_day": {},
            "success_rate": 0.0,
        }
        successful_calls = 0

        for record in self.usage_records:
            if record.success:
                successful_calls += 1

            # By model
            model = record.model_used
            if model not in summary["by_model"]:
                summary["by_model"][model] = {
                    "calls": 0,
                    "tokens": {"input": 0, "output": 0, "total": 0},
                    "cost": 0.0,
                    "success_rate": 0.0,
                }

            model_stats = summary["by_model"][model]
            model_stats["calls"] += 1
            if record.success:
                model_stats["tokens"]["input"] += record.input_tokens
                model_stats["tokens"]["output"] += record.output_tokens
                model_stats["tokens"]["total"] += record.total_tokens
                model_stats["cost"] += record.estimated_cost

            # By task type
            task = record.task_type
            if task not in summary["by_task_type"]:
                summary["by_task_type"][task] = {
                    "calls": 0,
                    "tokens": {"input": 0, "output": 0, "total": 0},
                    "cost": 0.0,
                    "success_rate": 0.0,
                }

            task_stats = summary["by_task_type"][task]
            task_stats["calls"] += 1
            if record.success:
                task_stats["tokens"]["input"] += record.input_tokens
                task_stats["tokens"]["output"] += record.output_tokens
                task_stats["tokens"]["total"] += record.total_tokens
                task_stats["cost"] += record.estimated_cost

            # By day
            day = record.timestamp.date().isoformat()
            if day not in summary["by_day"]:
                summary["by_day"][day] = {"calls": 0, "tokens": 0, "cost": 0.0}
            summary["by_day"][day]["calls"] += 1
            if record.success:
                summary["by_day"][day]["tokens"] += record.total_tokens
                summary["by_day"][day]["cost"] += record.estimated_cost

        # Calculate success rates
        summary["success_rate"] = (
            successful_calls / len(self.usage_records) if self.usage_records else 0.0
        )

        for model_name, model_stats in summary["by_model"].items():
            successes = sum(
                1
                for r in self.usage_records
                if r.model_used == model_name and r.success
            )
            total_calls = sum(
                1 for r in self.usage_records if r.model_used == model_name
            )
            model_stats["success_rate"] = successes / max(total_calls, 1)

        for task_name, task_stats in summary["by_task_type"].items():
            successes = sum(
                1 for r in self.usage_records if r.task_type == task_name and r.success
            )
            total_calls = sum(1 for r in self.usage_records if r.task_type == task_name)
            task_stats["success_rate"] = successes / max(total_calls, 1)

        return summary

    def get_cost_efficiency_metrics(self) -> Dict[str, Any]:
        """Get cost efficiency metrics for optimization."""
        summary = self.get_usage_summary()

        if not self.usage_records:
            return {"error": "No usage data available"}

        metrics = {
            "average_cost_per_call": summary["total_cost"] / summary["total_calls"],
            "average_tokens_per_call": summary["total_tokens"]["total"]
            / summary["total_calls"],
            "cost_per_token": summary["total_cost"]
            / max(summary["total_tokens"]["total"], 1),
            "model_efficiency": {},
            "task_efficiency": {},
        }

        # Model efficiency (cost per successful token)
        for model, stats in summary["by_model"].items():
            if stats["tokens"]["total"] > 0:
                metrics["model_efficiency"][model] = {
                    "cost_per_token": stats["cost"] / stats["tokens"]["total"],
                    "tokens_per_call": stats["tokens"]["total"]
                    / max(stats["calls"], 1),
                    "success_rate": stats["success_rate"],
                }

        # Task efficiency
        for task, stats in summary["by_task_type"].items():
            if stats["tokens"]["total"] > 0:
                metrics["task_efficiency"][task] = {
                    "cost_per_token": stats["cost"] / stats["tokens"]["total"],
                    "tokens_per_call": stats["tokens"]["total"]
                    / max(stats["calls"], 1),
                    "success_rate": stats["success_rate"],
                }

        return metrics

    def log_usage_summary(self) -> None:
        """Log comprehensive usage summary."""
        summary = self.get_usage_summary()

        logger.info("=== Token Usage Summary ===")
        logger.info(f"Total API Calls: {summary['total_calls']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(
            f"Total Tokens: {summary['total_tokens']['total']:,} "
            f"(Input: {summary['total_tokens']['input']:,}, "
            f"Output: {summary['total_tokens']['output']:,})"
        )
        logger.info(f"Total Estimated Cost: ${summary['total_cost']:.4f}")

        if summary["total_calls"] > 0:
            logger.info(
                f"Average Cost/Call: ${summary['total_cost'] / summary['total_calls']:.4f}"
            )
            logger.info(
                f"Average Tokens/Call: {summary['total_tokens']['total'] / summary['total_calls']:.0f}"
            )

        # Model breakdown
        logger.info("--- By Model ---")
        for model, stats in summary["by_model"].items():
            logger.info(
                f"{model}: {stats['calls']} calls, "
                f"{stats['tokens']['total']:,} tokens, "
                f"${stats['cost']:.4f}, "
                f"{stats['success_rate']:.1%} success"
            )

        # Task breakdown
        logger.info("--- By Task Type ---")
        for task, stats in summary["by_task_type"].items():
            logger.info(
                f"{task}: {stats['calls']} calls, "
                f"{stats['tokens']['total']:,} tokens, "
                f"${stats['cost']:.4f}, "
                f"{stats['success_rate']:.1%} success"
            )

    def _save_data(self) -> None:
        """Save token usage data to file."""
        try:
            data = {
                "total_tokens_used": self.total_tokens_used,
                "total_estimated_cost": self.total_estimated_cost,
                "usage_records": [record.to_dict() for record in self.usage_records],
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save token usage data: {e}")

    def _load_existing_data(self) -> None:
        """Load existing token usage data if available."""
        try:
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    data = json.load(f)

                self.total_tokens_used = data.get(
                    "total_tokens_used", {"input": 0, "output": 0, "total": 0}
                )
                self.total_estimated_cost = data.get("total_estimated_cost", 0.0)

                # Load usage records
                records_data = data.get("usage_records", [])
                self.usage_records = [
                    TokenUsageRecord.from_dict(record) for record in records_data
                ]

                logger.info(
                    f"Loaded existing token usage data: {len(self.usage_records)} records, "
                    f"{self.total_tokens_used['total']:,} tokens, "
                    f"${self.total_estimated_cost:.4f} estimated cost"
                )

        except Exception as e:
            logger.warning(f"Failed to load existing token usage data: {e}")

    def reset_tracking(self) -> None:
        """Reset token usage tracking (use with caution)."""
        self.usage_records = []
        self.total_tokens_used = {"input": 0, "output": 0, "total": 0}
        self.total_estimated_cost = 0.0

        # Remove existing data file
        if self.data_file.exists():
            self.data_file.unlink()

        logger.warning("Token usage tracking reset")


# Global instance
token_tracker = TokenTracker()
