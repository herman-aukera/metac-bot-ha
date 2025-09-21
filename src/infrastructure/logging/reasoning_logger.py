"""
Reasoning trace logging system for forecasting agents.

This module handles saving detailed reasoning traces from forecasting agents
to markdown files in the logs/reasoning/ directory structure.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

logger = logging.getLogger(__name__)


class ReasoningLogger:
    """
    Logger for capturing and saving detailed reasoning traces from forecasting agents.

    Creates markdown files in format: logs/reasoning/question-{id}_agent-{name}.md
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the reasoning logger.

        Args:
            base_dir: Base directory for logs (defaults to logs/reasoning/)
        """
        if base_dir is None:
            # Default to logs/reasoning/ relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            base_dir = project_root / "logs" / "reasoning"

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Reasoning logger initialized with base directory: {self.base_dir}"
        )

    def log_reasoning_trace(
        self,
        question_id: Union[str, int, UUID],
        agent_name: str,
        reasoning_data: Dict[str, Any],
        prediction_result: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Log detailed reasoning trace to a markdown file.

        Args:
            question_id: ID of the question being forecasted
            agent_name: Name of the forecasting agent
            reasoning_data: Detailed reasoning data from the agent
            prediction_result: Final prediction result (optional)

        Returns:
            Path to the created log file
        """
        # Sanitize filenames
        safe_question_id = str(question_id).replace("/", "_").replace("\\", "_")
        safe_agent_name = agent_name.replace("/", "_").replace("\\", "_")

        filename = f"question-{safe_question_id}_agent-{safe_agent_name}.md"
        file_path = self.base_dir / filename

        # Generate markdown content
        markdown_content = self._generate_markdown_content(
            question_id=question_id,
            agent_name=agent_name,
            reasoning_data=reasoning_data,
            prediction_result=prediction_result,
        )

        # Write to file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            logger.info(f"Reasoning trace logged to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to write reasoning trace to {file_path}: {e}")
            raise

    def _generate_markdown_content(
        self,
        question_id: Union[str, int, UUID],
        agent_name: str,
        reasoning_data: Dict[str, Any],
        prediction_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate markdown content for the reasoning trace.

        Args:
            question_id: ID of the question
            agent_name: Name of the agent
            reasoning_data: Reasoning data from the agent
            prediction_result: Final prediction result

        Returns:
            Formatted markdown content
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        lines = [
            f"# Reasoning Trace: {agent_name}",
            "",
            f"**Question ID:** {question_id}",
            f"**Agent:** {agent_name}",
            f"**Timestamp:** {timestamp}",
            "",
            "---",
            "",
        ]

        # Add prediction result if available
        if prediction_result:
            lines.extend(
                [
                    "## Prediction Result",
                    "",
                    f"**Probability:** {prediction_result.get('probability', 'N/A')}",
                    f"**Confidence:** {prediction_result.get('confidence', 'N/A')}",
                    f"**Method:** {prediction_result.get('method', 'N/A')}",
                    "",
                ]
            )

        # Add main reasoning
        if "reasoning" in reasoning_data:
            lines.extend(["## Main Reasoning", "", reasoning_data["reasoning"], ""])

        # Add reasoning steps if available
        if "reasoning_steps" in reasoning_data:
            lines.extend(["## Reasoning Steps", ""])
            steps = reasoning_data["reasoning_steps"]
            if isinstance(steps, list):
                for i, step in enumerate(steps, 1):
                    lines.append(f"{i}. {step}")
            else:
                lines.append(str(steps))
            lines.append("")

        # Add research findings if available
        if "research_findings" in reasoning_data:
            lines.extend(
                ["## Research Findings", "", reasoning_data["research_findings"], ""]
            )

        # Add sources if available
        if "sources" in reasoning_data:
            lines.extend(["## Sources", ""])
            sources = reasoning_data["sources"]
            if isinstance(sources, list):
                for source in sources:
                    if isinstance(source, dict):
                        title = source.get("title", "Unknown Title")
                        url = source.get("url", "#")
                        lines.append(f"- [{title}]({url})")
                    else:
                        lines.append(f"- {source}")
            else:
                lines.append(str(sources))
            lines.append("")

        # Add confidence analysis if available
        if "confidence_analysis" in reasoning_data:
            lines.extend(
                [
                    "## Confidence Analysis",
                    "",
                    reasoning_data["confidence_analysis"],
                    "",
                ]
            )

        # Add key factors if available
        if "key_factors" in reasoning_data:
            lines.extend(["## Key Factors", ""])
            factors = reasoning_data["key_factors"]
            if isinstance(factors, list):
                for factor in factors:
                    lines.append(f"- {factor}")
            else:
                lines.append(str(factors))
            lines.append("")

        # Add evidence for/against if available
        if "evidence_for" in reasoning_data:
            lines.extend(["## Evidence For", ""])
            evidence = reasoning_data["evidence_for"]
            if isinstance(evidence, list):
                for item in evidence:
                    lines.append(f"- {item}")
            else:
                lines.append(str(evidence))
            lines.append("")

        if "evidence_against" in reasoning_data:
            lines.extend(["## Evidence Against", ""])
            evidence = reasoning_data["evidence_against"]
            if isinstance(evidence, list):
                for item in evidence:
                    lines.append(f"- {item}")
            else:
                lines.append(str(evidence))
            lines.append("")

        # Add uncertainties if available
        if "uncertainties" in reasoning_data:
            lines.extend(["## Uncertainties", ""])
            uncertainties = reasoning_data["uncertainties"]
            if isinstance(uncertainties, list):
                for uncertainty in uncertainties:
                    lines.append(f"- {uncertainty}")
            else:
                lines.append(str(uncertainties))
            lines.append("")

        # Add method-specific data
        method_specific_keys = [
            "thoughts",
            "evaluations",
            "actions",
            "observations",
            "tree_structure",
            "thought_evaluations",
            "chain_of_thought",
            "action_sequences",
            "self_consistency_checks",
        ]

        for key in method_specific_keys:
            if key in reasoning_data:
                section_title = key.replace("_", " ").title()
                lines.extend(
                    [
                        f"## {section_title}",
                        "",
                        self._format_data_structure(reasoning_data[key]),
                        "",
                    ]
                )

        # Add raw metadata if available
        if "metadata" in reasoning_data:
            lines.extend(
                [
                    "## Metadata",
                    "",
                    "```json",
                    self._format_data_structure(reasoning_data["metadata"]),
                    "```",
                    "",
                ]
            )

        # Add any remaining data
        processed_keys = {
            "reasoning",
            "reasoning_steps",
            "research_findings",
            "sources",
            "confidence_analysis",
            "key_factors",
            "evidence_for",
            "evidence_against",
            "uncertainties",
            "metadata",
        } | set(method_specific_keys)

        remaining_data = {
            k: v for k, v in reasoning_data.items() if k not in processed_keys
        }
        if remaining_data:
            lines.extend(["## Additional Data", ""])
            for key, value in remaining_data.items():
                section_title = key.replace("_", " ").title()
                lines.extend(
                    [f"### {section_title}", "", self._format_data_structure(value), ""]
                )

        return "\n".join(lines)

    def _format_data_structure(self, data: Any) -> str:
        """Format a data structure for markdown display."""
        if isinstance(data, str):
            return data
        elif isinstance(data, (list, tuple)):
            if not data:
                return "_(empty)_"
            return "\n".join(f"- {item}" for item in data)
        elif isinstance(data, dict):
            if not data:
                return "_(empty)_"
            import json

            try:
                return json.dumps(data, indent=2)
            except:
                return str(data)
        else:
            return str(data)

    def log_ensemble_reasoning(
        self,
        question_id: Union[str, int, UUID],
        individual_agents: List[Dict[str, Any]],
        ensemble_result: Dict[str, Any],
        aggregation_method: str = "ensemble",
    ) -> Path:
        """
        Log ensemble reasoning trace combining multiple agents.

        Args:
            question_id: ID of the question
            individual_agents: List of individual agent reasoning data
            ensemble_result: Final ensemble result
            aggregation_method: Method used for aggregation

        Returns:
            Path to the created log file
        """
        # Create ensemble reasoning data
        reasoning_data = {
            "reasoning": f"Ensemble forecast combining {len(individual_agents)} agents using {aggregation_method} aggregation",
            "aggregation_method": aggregation_method,
            "individual_agents": individual_agents,
            "ensemble_statistics": self._calculate_ensemble_statistics(
                individual_agents
            ),
            "final_reasoning": ensemble_result.get("reasoning", ""),
            "metadata": {
                "agent_count": len(individual_agents),
                "aggregation_method": aggregation_method,
                "consensus_strength": self._calculate_consensus_strength(
                    individual_agents
                ),
            },
        }

        prediction_result = {
            "probability": ensemble_result.get("prediction"),
            "confidence": ensemble_result.get("confidence"),
            "method": "ensemble",
        }

        return self.log_reasoning_trace(
            question_id=question_id,
            agent_name="ensemble",
            reasoning_data=reasoning_data,
            prediction_result=prediction_result,
        )

    def _calculate_ensemble_statistics(
        self, individual_agents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics for ensemble reasoning."""
        if not individual_agents:
            return {}

        def _to_float_list(items: List[Dict[str, Any]], key: str) -> List[float]:
            values: List[float] = []
            for it in items:
                raw = it.get(key)
                try:
                    if raw is not None:
                        values.append(float(raw))
                except (TypeError, ValueError):
                    continue
            return values

        predictions = _to_float_list(individual_agents, "prediction")
        confidences = _to_float_list(individual_agents, "confidence")

        if not predictions:
            return {}

        import statistics

        stats = {
            "prediction_stats": {
                "mean": statistics.mean(predictions),
                "median": statistics.median(predictions),
                "min": min(predictions),
                "max": max(predictions),
                "std": statistics.stdev(predictions) if len(predictions) > 1 else 0.0,
            }
        }

        if confidences:
            stats["confidence_stats"] = {
                "mean": statistics.mean(confidences),
                "median": statistics.median(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "std": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            }

        return stats

    def _calculate_consensus_strength(
        self, individual_agents: List[Dict[str, Any]]
    ) -> float:
        """Calculate consensus strength based on prediction variance."""
        predictions: List[float] = []
        for agent in individual_agents:
            raw = agent.get("prediction")
            try:
                if raw is not None:
                    predictions.append(float(raw))
            except (TypeError, ValueError):
                continue

        if len(predictions) < 2:
            return 1.0

        import statistics

        variance = statistics.variance(predictions)
        # Convert variance to consensus strength (1.0 = perfect consensus, 0.0 = no consensus)
        # Max variance for binary predictions is 0.25 (0.5^2)
        max_variance = 0.25
        consensus_strength = max(0.0, 1.0 - (variance / max_variance))

        return consensus_strength

    def clear_logs(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear old reasoning log files.

        Args:
            older_than_days: Delete files older than this many days (default: all files)

        Returns:
            Number of files deleted
        """
        if not self.base_dir.exists():
            return 0

        deleted_count = 0

        for file_path in self.base_dir.glob("*.md"):
            should_delete = True

            if older_than_days is not None:
                # Check file modification time
                import time

                file_age_days = (time.time() - file_path.stat().st_mtime) / (24 * 3600)
                should_delete = file_age_days > older_than_days

            if should_delete:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted reasoning log: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")

        logger.info(f"Cleared {deleted_count} reasoning log files")
        return deleted_count


# Global reasoning logger instance
_global_reasoning_logger = None


def get_reasoning_logger(base_dir: Optional[Path] = None) -> ReasoningLogger:
    """Get the global reasoning logger instance."""
    global _global_reasoning_logger

    if _global_reasoning_logger is None:
        _global_reasoning_logger = ReasoningLogger(base_dir=base_dir)

    return _global_reasoning_logger


def log_agent_reasoning(
    question_id: Union[str, int, UUID],
    agent_name: str,
    reasoning_data: Dict[str, Any],
    prediction_result: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Convenience function to log agent reasoning using the global logger.

    Args:
        question_id: ID of the question being forecasted
        agent_name: Name of the forecasting agent
        reasoning_data: Detailed reasoning data from the agent
        prediction_result: Final prediction result (optional)

    Returns:
        Path to the created log file
    """
    logger_instance = get_reasoning_logger()
    return logger_instance.log_reasoning_trace(
        question_id=question_id,
        agent_name=agent_name,
        reasoning_data=reasoning_data,
        prediction_result=prediction_result,
    )


def log_ensemble_reasoning(
    question_id: Union[str, int, UUID],
    individual_agents: List[Dict[str, Any]],
    ensemble_result: Dict[str, Any],
    aggregation_method: str = "ensemble",
) -> Path:
    """
    Convenience function to log ensemble reasoning using the global logger.

    Args:
        question_id: ID of the question
        individual_agents: List of individual agent reasoning data
        ensemble_result: Final ensemble result
        aggregation_method: Method used for aggregation

    Returns:
        Path to the created log file
    """
    logger_instance = get_reasoning_logger()
    return logger_instance.log_ensemble_reasoning(
        question_id=question_id,
        individual_agents=individual_agents,
        ensemble_result=ensemble_result,
        aggregation_method=aggregation_method,
    )
