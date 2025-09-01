import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Literal
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOption,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

logger = logging.getLogger(__name__)
SAFE_REASONING_FALLBACK = "Forecast generated without detailed reasoning due to fallback."

# Tournament components - import after forecasting_tools to avoid conflicts
try:
    # Add src to path for tournament components
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from infrastructure.external_apis.tournament_asknews_client import TournamentAskNewsClient
    from infrastructure.config.tournament_config import get_tournament_config
    from infrastructure.config.api_keys import api_key_manager
    TOURNAMENT_COMPONENTS_AVAILABLE = True
    logger.info("Tournament components loaded successfully")
except ImportError as e:
    logger.warning(f"Tournament components not available: {e}")
    TOURNAMENT_COMPONENTS_AVAILABLE = False


class TemplateForecaster(ForecastBot):
    """
    Enhanced template bot for Q2 2025 Metaculus AI Tournament with tournament optimizations.

    Features:
    - Tournament-optimized AskNews client with quota management
    - Metaculus proxy client for free credits with fallback to OpenRouter
    - Robust fallback system for all API providers
    - Usage monitoring and alerting
    - Tournament-specific configurations
    - Budget management and cost-aware model selection
    - Token tracking and real-time cost monitoring

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with tournament optimizations and budget management."""
        super().__init__(*args, **kwargs)

        # Initialize budget management components
        try:
            from src.infrastructure.config.budget_manager import budget_manager
            from src.infrastructure.config.budget_alerts import budget_alert_system
            from src.infrastructure.config.enhanced_llm_config import enhanced_llm_config
            from src.infrastructure.config.token_tracker import token_tracker

            self.budget_manager = budget_manager
            self.budget_alert_system = budget_alert_system
            self.enhanced_llm_config = enhanced_llm_config
            self.token_tracker = token_tracker

            # Log initial budget status
            self.budget_manager.log_budget_status()
            self.enhanced_llm_config.log_configuration_status()

            logger.info("Budget management system initialized")

        except ImportError as e:
            logger.warning(f"Budget management components not available: {e}")
            self.budget_manager = None
            self.budget_alert_system = None
            self.enhanced_llm_config = None
            self.token_tracker = None

        # Initialize enhanced tri-model router for GPT-5 variants with anti-slop directives
        try:
            from src.infrastructure.config.tri_model_router import tri_model_router
            from src.prompts.anti_slop_prompts import anti_slop_prompts
            from src.domain.services.multi_stage_validation_pipeline import MultiStageValidationPipeline
            from src.infrastructure.config.budget_aware_operation_manager import budget_aware_operation_manager
            from src.infrastructure.reliability.comprehensive_error_recovery import ComprehensiveErrorRecoveryManager

            self.tri_model_router = tri_model_router
            self.anti_slop_prompts = anti_slop_prompts

            # Initialize multi-stage validation pipeline (Task 8.1)
            self.multi_stage_pipeline = MultiStageValidationPipeline(
                tri_model_router=self.tri_model_router,
                tournament_asknews=getattr(self, 'tournament_asknews', None)
            )

            # Initialize budget-aware operation manager (Task 8.2)
            self.budget_aware_manager = budget_aware_operation_manager

            # Initialize comprehensive error recovery (Task 8.1)
            self.error_recovery_manager = ComprehensiveErrorRecoveryManager(
                tri_model_router=self.tri_model_router,
                budget_manager=self.budget_manager
            )

            # Integrate tri-model router with budget management systems (Task 8.2)
            self.tri_model_router.integrate_with_budget_manager(
                budget_manager=self.budget_manager,
                budget_aware_manager=self.budget_aware_manager
            )

            # Log tri-model status
            model_status = self.tri_model_router.get_model_status()
            logger.info("Enhanced tri-model router initialized:")
            for tier, status in model_status.items():
                logger.info(f"  {tier}: {status}")

            logger.info("Budget manager integration with tri-model router completed")

            # Log multi-stage pipeline status
            pipeline_config = self.multi_stage_pipeline.get_pipeline_configuration()
            logger.info(f"Multi-stage validation pipeline initialized with {len(pipeline_config['stages'])} stages")

            # Log comprehensive system status (Task 8.2)
            self.log_enhanced_system_status()

        except ImportError as e:
            logger.warning(f"Enhanced tri-model components not available: {e}")
            self.tri_model_router = None
            self.anti_slop_prompts = None
            self.multi_stage_pipeline = None
            self.budget_aware_manager = None
            self.error_recovery_manager = None

        # Initialize OpenRouter API key configuration
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key or self.openrouter_api_key.startswith("dummy_"):
            logger.error("OpenRouter API key not configured! This is required for tournament operation.")
        else:
            logger.info("OpenRouter API key configured successfully")

        # Initialize performance monitoring integration (Task 8.2)
        try:
            from src.infrastructure.monitoring.integrated_monitoring_service import integrated_monitoring_service
            self.performance_monitor = integrated_monitoring_service
            logger.info("Performance monitoring integration initialized")
        except ImportError as e:
            logger.warning(f"Performance monitoring not available: {e}")
            self.performance_monitor = None

        # Initialize error handling and fallback state
        self.emergency_mode_active = False
        self.api_failure_count = 0
        self.max_api_failures = int(os.getenv("MAX_API_FAILURES", "5"))
        self.fallback_models = {
            "emergency": os.getenv("EMERGENCY_FALLBACK_MODEL", "openai/gpt-4o-mini"),
            "proxy": "metaculus/gpt-4o-mini",
            "last_resort": "openai/gpt-3.5-turbo"
        }

        # Initialize tournament components if available
        if TOURNAMENT_COMPONENTS_AVAILABLE:
            try:
                self.tournament_config = get_tournament_config()
                self.tournament_asknews = TournamentAskNewsClient()

                # Initialize multi-stage research pipeline (Task 4.1)
                from domain.services.multi_stage_research_pipeline import MultiStageResearchPipeline
                self._multi_stage_pipeline = MultiStageResearchPipeline(
                    tri_model_router=self.tri_model_router,
                    tournament_asknews=self.tournament_asknews
                )
                logger.info("Multi-stage research pipeline initialized successfully")

                # Update concurrency based on tournament config
                self._max_concurrent_questions = self.tournament_config.max_concurrent_questions
                self._concurrency_limiter = asyncio.Semaphore(self._max_concurrent_questions)

                # Log tournament initialization
                logger.info(f"Tournament mode initialized: {self.tournament_config.tournament_name}")
                logger.info(f"Max concurrent questions: {self._max_concurrent_questions}")

                # Validate API keys
                api_key_manager.log_key_status()

            except Exception as e:
                logger.warning(f"Failed to initialize tournament components: {e}")
                self.tournament_config = None
                self.tournament_asknews = None
                self._multi_stage_pipeline = None
        else:
            self.tournament_config = None
            self.tournament_asknews = None
            self._multi_stage_pipeline = None

        # Set default concurrency if not set by tournament config
        if not hasattr(self, '_max_concurrent_questions'):
            self._max_concurrent_questions = 2
            self._concurrency_limiter = asyncio.Semaphore(self._max_concurrent_questions)

        # Initialize a search client for tests and research integrations
        try:
            from src.infrastructure.config.settings import get_settings, Settings
            from src.infrastructure.external_apis.search_client import create_search_client, SearchClient

            settings = get_settings()
            self.search_client = create_search_client(settings)
            logger.info("Search client initialized for TemplateForecaster")
        except Exception as e:
            logger.warning(f"Search client not available, using no-op stub: {e}")

            class _NoOpSearchClient:
                async def search(self, query: str, max_results: int = 10):  # pragma: no cover
                    _ = (query, max_results)
                    return []

                async def health_check(self) -> bool:  # pragma: no cover
                    await asyncio.sleep(0)
                    return True

            self.search_client = _NoOpSearchClient()

    def _has_openrouter_key(self) -> bool:
        """Check if OpenRouter API key is available."""
        return bool(self.openrouter_api_key and not self.openrouter_api_key.startswith("dummy_"))

    def _has_perplexity_key(self) -> bool:
        """Check if Perplexity API key is available."""
        key = os.getenv("PERPLEXITY_API_KEY")
        return bool(key and not key.startswith("dummy_"))

    def _has_exa_key(self) -> bool:
        """Check if Exa API key is available."""
        key = os.getenv("EXA_API_KEY")
        return bool(key and not key.startswith("dummy_"))

    def _has_metaculus_proxy(self) -> bool:
        """Check if Metaculus proxy is enabled."""
        return os.getenv("ENABLE_PROXY_CREDITS", "true").lower() == "true"

    async def _call_llm_based_research(self, question: str) -> str:
        """Fallback research method using LLM when no external APIs are available."""
        if self.tri_model_router and self.anti_slop_prompts:
            try:
                # Use the enhanced research prompt with mini model
                research_prompt = self.anti_slop_prompts.get_research_prompt(
                    question_text=question,
                    model_tier="mini"
                )

                # Add context that this is LLM-based research
                enhanced_prompt = f"""
{research_prompt}

**IMPORTANT NOTE**: External research APIs are not available. Please provide research based on your training data knowledge, clearly indicating the limitations and knowledge cutoff date. Focus on:
- General background information about the topic
- Historical context and patterns
- Known factors that typically influence such questions
- Explicit acknowledgment of information limitations

Be very clear about what information may be outdated or incomplete.
"""

                # Get budget-aware context for routing
                budget_context = self.tri_model_router.get_budget_aware_routing_context()
                budget_remaining = budget_context.remaining_percentage if budget_context else 100.0

                research = await self.tri_model_router.route_query(
                    task_type="research",
                    content=enhanced_prompt,
                    complexity="medium",
                    budget_remaining=budget_remaining
                )

                return research

            except Exception as e:
                logger.warning(f"LLM-based research failed: {e}")
                return ""

        return ""

    def _integrate_budget_manager_with_operation_modes(self):
        """Integrate budget manager with operation mode transitions and alerts (Task 8.2)."""
        if not (self.budget_manager and self.budget_aware_manager):
            return

        try:
            # Monitor budget utilization and trigger operation mode changes
            monitoring_result = self.budget_aware_manager.monitor_budget_utilization()

            # Check for threshold alerts and log them
            if monitoring_result.get("threshold_alerts"):
                for alert in monitoring_result["threshold_alerts"]:
                    logger.warning(f"Budget threshold alert: {alert['threshold_name']} "
                                 f"({alert['current_utilization']:.1f}% utilization)")

                    # Send alert through budget alert system if available
                    if self.budget_alert_system:
                        self.budget_alert_system.check_and_alert()

            # Detect and execute operation mode transitions
            mode_switched, transition_log = self.budget_aware_manager.detect_and_switch_operation_mode()

            if mode_switched and transition_log:
                logger.info(f"Operation mode transition executed: "
                          f"{transition_log.from_mode.value} → {transition_log.to_mode.value}, "
                          f"estimated savings: ${transition_log.cost_savings_estimate:.4f}")

                # Update performance monitoring with mode transition
                if self.performance_monitor:
                    self.performance_monitor.record_model_usage(
                        question_id="mode_transition",
                        task_type="operation_mode_change",
                        selected_model=f"mode_{transition_log.to_mode.value}",
                        selected_tier="system",
                        routing_rationale=transition_log.trigger_reason,
                        estimated_cost=transition_log.cost_savings_estimate,
                        operation_mode=transition_log.to_mode.value,
                        budget_remaining=transition_log.remaining_budget
                    )

                # Update tri-model router with new operation mode for budget-aware routing
                if self.tri_model_router:
                    # Apply operation mode adjustments to model selection
                    current_mode = transition_log.to_mode.value
                    logger.info(f"Updating tri-model router operation mode to: {current_mode}")

                    # The router will automatically use the budget-aware operation manager
                    # for future routing decisions based on the new mode

        except Exception as e:
            logger.error(f"Budget manager integration error: {e}")

    def _track_question_processing_cost(self, question_id: str, task_type: str,
                                      cost: float, model_used: str, success: bool):
        """Track cost and performance metrics for question processing (Task 8.2)."""
        try:
            # Estimate token usage for budget manager (approximate)
            estimated_input_tokens = 1000  # Default estimate
            estimated_output_tokens = 500  # Default estimate

            # Update budget manager with actual cost
            if self.budget_manager:
                self.budget_manager.record_cost(
                    question_id=question_id,
                    model=model_used,
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    task_type=task_type,
                    success=success
                )

            # Update budget-aware operation manager performance metrics
            if self.budget_aware_manager:
                current_mode = self.budget_aware_manager.operation_mode_manager.get_current_mode()

                # Update question processing count by mode
                mode_key = current_mode.value
                if mode_key in self.budget_aware_manager.performance_metrics["questions_processed_by_mode"]:
                    self.budget_aware_manager.performance_metrics["questions_processed_by_mode"][mode_key] += 1

                # Update average cost by mode
                if mode_key in self.budget_aware_manager.performance_metrics["average_cost_by_mode"]:
                    current_avg = self.budget_aware_manager.performance_metrics["average_cost_by_mode"][mode_key]
                    question_count = self.budget_aware_manager.performance_metrics["questions_processed_by_mode"][mode_key]

                    # Calculate new average
                    new_avg = ((current_avg * (question_count - 1)) + cost) / question_count
                    self.budget_aware_manager.performance_metrics["average_cost_by_mode"][mode_key] = new_avg

            # Update performance monitoring with execution outcome
            if self.performance_monitor:
                self.performance_monitor.record_execution_outcome(
                    question_id=question_id,
                    actual_cost=cost,
                    execution_time=1.0,  # Default execution time
                    quality_score=0.8 if success else 0.3,  # Estimated quality based on success
                    success=success,
                    fallback_used=False  # Would need to be passed from caller
                )

        except Exception as e:
            logger.error(f"Cost tracking error for question {question_id}: {e}")

    def _check_tournament_compliance_integration(self) -> Dict[str, Any]:
        """Check tournament compliance with integrated systems (Task 8.2)."""
        compliance_status = {
            "budget_compliant": True,
            "operation_mode_compliant": True,
            "performance_compliant": True,
            "error_recovery_compliant": True,
            "tri_model_integration_compliant": True,
            "cost_tracking_compliant": True,
            "issues": [],
            "recommendations": []
        }

        try:
            # Check budget compliance
            if self.budget_manager:
                budget_status = self.budget_manager.get_budget_status()
                if budget_status.utilization_percentage > 100:
                    compliance_status["budget_compliant"] = False
                    compliance_status["issues"].append("Budget exceeded 100% utilization")
                    compliance_status["recommendations"].append("Activate emergency mode immediately")
                elif budget_status.utilization_percentage > 95:
                    compliance_status["recommendations"].append("Consider switching to critical operation mode")

            # Check operation mode compliance
            if self.budget_aware_manager:
                current_mode = self.budget_aware_manager.operation_mode_manager.get_current_mode()
                emergency_protocol = self.budget_aware_manager.current_emergency_protocol

                if emergency_protocol.value != "none":
                    compliance_status["operation_mode_compliant"] = False
                    compliance_status["issues"].append(f"Emergency protocol active: {emergency_protocol.value}")
                    compliance_status["recommendations"].append("Monitor system closely during emergency protocol")

                # Check if operation mode matches budget utilization
                budget_util = self.budget_manager.get_budget_status().utilization_percentage if self.budget_manager else 0
                expected_mode = self.budget_aware_manager.get_operation_mode_for_budget(budget_util)
                if current_mode.value != expected_mode:
                    compliance_status["operation_mode_compliant"] = False
                    compliance_status["issues"].append(f"Operation mode mismatch: current={current_mode.value}, expected={expected_mode}")

            # Check tri-model router integration compliance
            if self.tri_model_router and self.budget_aware_manager:
                try:
                    # Verify router can access budget-aware operation manager
                    router_status = self.tri_model_router.get_model_status()
                    if isinstance(router_status, dict):
                        # Check if status objects have is_available attribute
                        unavailable_tiers = []
                        for tier, status in router_status.items():
                            if hasattr(status, 'is_available') and not status.is_available:
                                unavailable_tiers.append(tier)

                        if unavailable_tiers:
                            compliance_status["tri_model_integration_compliant"] = False
                            compliance_status["issues"].append(f"Tri-model router tiers unavailable: {', '.join(unavailable_tiers)}")
                            compliance_status["recommendations"].append("Check model availability and fallback chains")
                except Exception as e:
                    compliance_status["tri_model_integration_compliant"] = False
                    compliance_status["issues"].append(f"Tri-model router integration error: {str(e)}")

            # Check cost tracking integration compliance
            if self.budget_manager and self.performance_monitor:
                try:
                    # Verify cost tracking is working (allow empty records for fresh start)
                    recent_records = len(self.budget_manager.cost_records[-10:]) if self.budget_manager.cost_records else 0
                    if recent_records == 0 and self.budget_manager.questions_processed > 0:
                        # Only flag as issue if we've processed questions but have no records
                        compliance_status["cost_tracking_compliant"] = False
                        compliance_status["issues"].append("No recent cost tracking records found despite processing questions")
                        compliance_status["recommendations"].append("Verify cost tracking integration is functioning")
                    # If no questions processed yet, cost tracking is still compliant
                except Exception as e:
                    compliance_status["cost_tracking_compliant"] = False
                    compliance_status["issues"].append(f"Cost tracking integration error: {str(e)}")

            # Check performance monitoring compliance
            if self.performance_monitor:
                try:
                    comprehensive_status = self.performance_monitor.get_comprehensive_status()
                    overall_health = comprehensive_status.overall_health
                    if overall_health in ["concerning", "critical"]:
                        compliance_status["performance_compliant"] = False
                        compliance_status["issues"].append(f"System health: {overall_health}")
                        compliance_status["recommendations"].extend(comprehensive_status.optimization_recommendations[:3])
                except Exception as e:
                    compliance_status["performance_compliant"] = False
                    compliance_status["issues"].append(f"Performance monitoring error: {str(e)}")

            # Check error recovery compliance
            if self.error_recovery_manager:
                try:
                    recovery_status = self.error_recovery_manager.get_recovery_status()
                    if recovery_status.get("system_health", {}).get("status") == "critical":
                        compliance_status["error_recovery_compliant"] = False
                        compliance_status["issues"].append("Error recovery system in critical state")
                        compliance_status["recommendations"].append("Review error recovery logs and reset if necessary")
                except Exception as e:
                    compliance_status["error_recovery_compliant"] = False
                    compliance_status["issues"].append(f"Error recovery check failed: {str(e)}")

            # Overall compliance assessment
            compliance_status["overall_compliant"] = all([
                compliance_status["budget_compliant"],
                compliance_status["operation_mode_compliant"],
                compliance_status["performance_compliant"],
                compliance_status["error_recovery_compliant"],
                compliance_status["tri_model_integration_compliant"],
                compliance_status["cost_tracking_compliant"]
            ])

            # Add timestamp for monitoring
            compliance_status["last_checked"] = datetime.now().isoformat()

        except Exception as e:
            compliance_status["issues"].append(f"Compliance check error: {str(e)}")
            compliance_status["overall_compliant"] = False
            logger.error(f"Tournament compliance check error: {e}")

        return compliance_status

    def _handle_budget_exhaustion(self, question_id: str = "unknown") -> bool:
        """Handle budget exhaustion scenarios with graceful degradation."""
        if not self.budget_manager:
            return False

        budget_status = self.budget_manager.get_budget_status()

        if budget_status.status_level == "emergency":
            if not self.emergency_mode_active:
                logger.critical(f"EMERGENCY MODE ACTIVATED: Budget utilization at {budget_status.utilization_percentage:.1f}%")
                logger.critical(f"Remaining budget: ${budget_status.remaining:.4f}")
                logger.critical(f"Estimated questions remaining: {budget_status.estimated_questions_remaining}")
                self.emergency_mode_active = True

                # Alert system if available
                if self.budget_alert_system:
                    self.budget_alert_system.send_critical_alert(
                        f"Emergency mode activated for question {question_id}",
                        budget_status
                    )

            # In emergency mode, only process high-priority questions
            return True

        elif budget_status.utilization_percentage >= 100:
            logger.critical("BUDGET EXHAUSTED: Cannot process any more questions")
            if self.budget_alert_system:
                self.budget_alert_system.send_critical_alert(
                    "Budget completely exhausted",
                    budget_status
                )
            return True

        return False

    def _handle_api_failure(self, error: Exception, model: str, task_type: str) -> str:
        """Handle API failures with intelligent fallbacks."""
        self.api_failure_count += 1
        logger.warning(f"API failure #{self.api_failure_count} for {model} ({task_type}): {error}")

        # If too many failures, activate emergency mode
        if self.api_failure_count >= self.max_api_failures:
            logger.error(f"Too many API failures ({self.api_failure_count}), activating emergency protocols")
            self.emergency_mode_active = True

        # Determine fallback strategy
        if "openrouter" in model.lower():
            # OpenRouter failed, try Metaculus proxy
            if os.getenv("ENABLE_PROXY_CREDITS", "true").lower() == "true":
                fallback_model = self.fallback_models["proxy"]
                logger.info(f"Falling back to Metaculus proxy: {fallback_model}")
                return fallback_model
            else:
                # No proxy available, use emergency model
                fallback_model = self.fallback_models["emergency"]
                logger.info(f"Using emergency fallback model: {fallback_model}")
                return fallback_model

        elif "metaculus" in model.lower():
            # Proxy failed, try OpenRouter
            if self.openrouter_api_key and not self.openrouter_api_key.startswith("dummy_"):
                fallback_model = self.fallback_models["emergency"]
                logger.info(f"Proxy failed, falling back to OpenRouter: {fallback_model}")
                return fallback_model
            else:
                # No OpenRouter key, use last resort
                fallback_model = self.fallback_models["last_resort"]
                logger.warning(f"Using last resort model: {fallback_model}")
                return fallback_model

        else:
            # Unknown provider failed, use emergency model
            fallback_model = self.fallback_models["emergency"]
            logger.info(f"Unknown provider failed, using emergency model: {fallback_model}")
            return fallback_model

    def _create_emergency_response(self, task_type: str, question_text: str = "") -> str:
        """Create emergency response when all APIs fail."""
        if task_type == "research":
            return (
                "Research unavailable due to API failures. "
                "Proceeding with forecast based on question information only."
            )
        elif task_type == "forecast":
            return (
                f"Unable to generate detailed forecast due to API failures. "
                f"Question: {question_text[:200]}... "
                f"Based on limited analysis, assigning neutral probability due to uncertainty."
            )
        else:
            return "Task unavailable due to system limitations."

    async def _safe_llm_invoke(self, llm, prompt: str, task_type: str, question_id: str = "unknown",
                              max_retries: int = 3) -> str:
        """Safely invoke LLM with error handling and fallbacks."""
        last_error = None

        for attempt in range(max_retries):
            try:
                # Check budget before each attempt
                if self._handle_budget_exhaustion(question_id):
                    if task_type == "research":
                        return "Research skipped due to budget constraints."
                    else:
                        return self._create_emergency_response(task_type, prompt[:200])

                # Attempt LLM call
                response = await llm.invoke(prompt)

                # Reset failure count on success
                if self.api_failure_count > 0:
                    logger.info(f"API call successful after {self.api_failure_count} previous failures")
                    self.api_failure_count = max(0, self.api_failure_count - 1)

                return response

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    # Try fallback model
                    fallback_model_name = self._handle_api_failure(e, llm.model, task_type)

                    try:
                        # Create fallback LLM
                        fallback_llm = GeneralLlm(
                            model=fallback_model_name,
                            api_key=self.openrouter_api_key if "openrouter" in fallback_model_name else None,
                            temperature=llm.temperature if hasattr(llm, 'temperature') else 0.1,
                            timeout=30,  # Shorter timeout for fallbacks
                            allowed_tries=1
                        )
                        llm = fallback_llm
                        logger.info(f"Retrying with fallback model: {fallback_model_name}")

                    except Exception as fallback_error:
                        logger.error(f"Failed to create fallback LLM: {fallback_error}")
                        if attempt == max_retries - 1:
                            break
                else:
                    break

                # Brief delay before retry
                await asyncio.sleep(min(2 ** attempt, 10))

        # All attempts failed
        logger.error(f"All LLM attempts failed for {task_type}. Last error: {last_error}")
        return self._create_emergency_response(task_type, prompt[:200])

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Enhanced research with multi-stage validation pipeline, tri-model routing, and comprehensive error handling."""
        async with self._concurrency_limiter:
            # Get budget status and operation mode for intelligent routing
            budget_remaining = 100.0
            operation_mode = "normal"

            if self.budget_manager:
                budget_status = self.budget_manager.get_budget_status()
                budget_remaining = 100.0 - budget_status.utilization_percentage

                # Check and alert on budget status
                if self.budget_alert_system:
                    alert = self.budget_alert_system.check_and_alert()

                # Get current operation mode from budget-aware manager
                if self.budget_aware_manager:
                    mode_switched, transition_log = self.budget_aware_manager.detect_and_switch_operation_mode()
                    if mode_switched:
                        logger.info(f"Operation mode switched: {transition_log.from_mode.value} → {transition_log.to_mode.value}")

                    operation_mode = self.budget_aware_manager.operation_mode_manager.get_current_mode().value

                # Check if question should be skipped based on operation mode
                if self.budget_aware_manager:
                    should_skip, skip_reason = self.budget_aware_manager.should_skip_question(
                        question_priority="normal",  # Could be extracted from question metadata
                        question_complexity="medium"
                    )
                    if should_skip:
                        logger.warning(f"Skipping research for {question.page_url}: {skip_reason}")
                        return f"Research skipped: {skip_reason}"

            # PRIORITY: Try enhanced multi-stage validation pipeline first (Task 8.1)
            if self.multi_stage_pipeline:
                try:
                    # Use the complete multi-stage validation pipeline for research
                    pipeline_result = await self.multi_stage_pipeline.process_question(
                        question=question.question_text,
                        question_type="research",  # Special type for research-only processing
                        context={
                            "question_url": question.page_url,
                            "budget_remaining": budget_remaining,
                            "operation_mode": operation_mode,
                            "background_info": getattr(question, 'background_info', ''),
                            "resolution_criteria": getattr(question, 'resolution_criteria', ''),
                            "fine_print": getattr(question, 'fine_print', '')
                        }
                    )

                    if pipeline_result.pipeline_success and pipeline_result.research_result.content:
                        logger.info(f"Multi-stage validation pipeline successful for URL {question.page_url}, "
                                  f"cost: ${pipeline_result.total_cost:.4f}, quality: {pipeline_result.quality_score:.2f}")

                        # Integrate budget manager with cost tracking and operation modes (Task 8.2)
                        self._track_question_processing_cost(
                            question_id=str(getattr(question, 'id', 'unknown')),
                            task_type="research_pipeline",
                            cost=pipeline_result.total_cost,
                            model_used="multi_stage_pipeline",
                            success=True
                        )

                        # Check and update operation modes based on budget utilization
                        self._integrate_budget_manager_with_operation_modes()

                        return pipeline_result.research_result.content

                    else:
                        logger.warning(f"Multi-stage validation pipeline failed for {question.page_url}: "
                                     f"Success={pipeline_result.pipeline_success}, "
                                     f"Quality={pipeline_result.quality_score:.2f}")
                        # Continue to fallback methods

                except Exception as e:
                    logger.warning(f"Multi-stage validation pipeline failed: {e}")

                    # Use comprehensive error recovery (Task 8.1)
                    if self.error_recovery_manager:
                        try:
                            from src.infrastructure.reliability.error_classification import ErrorContext
                            error_context = ErrorContext(
                                task_type="research",
                                model_tier="mini",
                                operation_mode=operation_mode,
                                budget_remaining=budget_remaining,
                                attempt_number=1,
                                question_id=str(getattr(question, 'id', 'unknown')),
                                provider="multi_stage_pipeline"
                            )

                            recovery_result = await self.error_recovery_manager.recover_from_error(e, error_context)
                            if recovery_result.success:
                                logger.info(f"Error recovery successful: {recovery_result.message}")
                                # Could retry with recovered configuration, but for now continue to fallback
                            else:
                                logger.warning(f"Error recovery failed: {recovery_result.message}")
                        except Exception as recovery_error:
                            logger.error(f"Error recovery system failed: {recovery_error}")

                    # Continue to fallback methods

            # FALLBACK: Try tri-model router for intelligent research
            if self.tri_model_router and self.anti_slop_prompts:
                try:
                    # Create anti-slop research prompt
                    research_prompt = self.anti_slop_prompts.get_research_prompt(
                        question_text=question.question_text,
                        model_tier="mini"  # Use mini model for research by default
                    )

                    # Get budget-aware context for routing
                    budget_context = self.tri_model_router.get_budget_aware_routing_context()
                    budget_remaining = budget_context.remaining_percentage if budget_context else 100.0

                    # Route to optimal model based on budget and complexity
                    research = await self.tri_model_router.route_query(
                        task_type="research",
                        content=research_prompt,
                        complexity="medium",
                        budget_remaining=budget_remaining
                    )

                    if research and len(research.strip()) > 50:
                        logger.info(f"Tri-model research successful for URL {question.page_url}")
                        return research

                except Exception as e:
                    logger.warning(f"Tri-model research failed: {e}")
                    # Continue to fallback methods

            research = ""

            # Try tournament-optimized AskNews client first
            if self.tournament_asknews:
                try:
                    research = await self.tournament_asknews.get_news_research(question.question_text)

                    if research and len(research.strip()) > 0:
                        # Log usage stats periodically
                        stats = self.tournament_asknews.get_usage_stats()
                        if stats["total_requests"] % 10 == 0:  # Log every 10 requests
                            logger.info(f"AskNews usage: {stats['estimated_quota_used']}/{stats['quota_limit']} "
                                      f"({stats['quota_usage_percentage']:.1f}%), "
                                      f"Success rate: {stats['success_rate']:.1f}%")

                        # Alert on high quota usage
                        if self.tournament_asknews.should_alert_quota_usage():
                            alert_level = self.tournament_asknews.get_quota_alert_level()
                            logger.warning(f"AskNews quota usage {alert_level}: "
                                         f"{stats['quota_usage_percentage']:.1f}% used")

                        logger.info(f"Tournament AskNews research successful for URL {question.page_url}")
                        return research

                except Exception as e:
                    logger.warning(f"Tournament AskNews client failed: {e}")
                    # Continue to other research methods

            # Fallback to original AskNews if available
            if not research and os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                try:
                    research = await AskNewsSearcher().get_formatted_news_async(question.question_text)
                    if research and len(research.strip()) > 0:
                        logger.info(f"Original AskNews research successful for URL {question.page_url}")
                        return research
                except Exception as e:
                    logger.warning(f"Original AskNews failed: {e}")

            # Fallback to OpenRouter Perplexity (if available)
            if not research and self._has_openrouter_key():
                try:
                    research = await self._call_perplexity(question.question_text, use_open_router=True)
                    if research and len(research.strip()) > 0:
                        logger.info(f"OpenRouter Perplexity research successful for URL {question.page_url}")
                        return research
                except Exception as e:
                    logger.warning(f"OpenRouter Perplexity search failed: {e}")

            # Fallback to Perplexity direct (if available)
            if not research and self._has_perplexity_key():
                try:
                    research = await self._call_perplexity(question.question_text)
                    if research and len(research.strip()) > 0:
                        logger.info(f"Perplexity research successful for URL {question.page_url}")
                        return research
                except Exception as e:
                    logger.warning(f"Perplexity search failed: {e}")

            # Fallback to Exa (if available)
            if not research and self._has_exa_key():
                try:
                    research = await self._call_exa_smart_searcher(question.question_text)
                    if research and len(research.strip()) > 0:
                        logger.info(f"Exa research successful for URL {question.page_url}")
                        return research
                except Exception as e:
                    logger.warning(f"Exa search failed: {e}")

            # Final fallback: Use LLM-based research if no external APIs available
            if not research and (self._has_openrouter_key() or self._has_metaculus_proxy()):
                try:
                    research = await self._call_llm_based_research(question.question_text)
                    if research and len(research.strip()) > 0:
                        logger.info(f"LLM-based research successful for URL {question.page_url}")
                        return research
                except Exception as e:
                    logger.warning(f"LLM-based research failed: {e}")

            # If all research methods fail, check if we're in emergency mode
            if not research:
                if self.emergency_mode_active or self._handle_budget_exhaustion(str(getattr(question, 'id', 'unknown'))):
                    logger.warning(f"Emergency mode: Skipping research for question URL {question.page_url}")
                    research = "Research unavailable due to system constraints."
                else:
                    logger.warning(f"All research providers failed for question URL {question.page_url}. "
                                 f"Proceeding with empty research.")
                    research = ""

            logger.info(f"Research completed for URL {question.page_url} (length: {len(research)})")
            return research

    async def _call_perplexity(
        self, question: str, use_open_router: bool = False
    ) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question}
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        # Use safe invoke for Perplexity calls
        response = await self._safe_llm_invoke(model, prompt, "research")
        return response

    def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including all integrated components (Task 8.2)."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_components": {},
            "budget_status": {},
            "operation_mode": {},
            "performance_metrics": {},
            "tournament_compliance": {},
            "error_recovery": {},
            "overall_health": "unknown"
        }

        try:
            # Budget manager status
            if self.budget_manager:
                budget_status = self.budget_manager.get_budget_status()
                status["budget_status"] = {
                    "utilization_percentage": budget_status.utilization_percentage,
                    "remaining": budget_status.remaining,
                    "status_level": budget_status.status_level,
                    "estimated_questions_remaining": budget_status.estimated_questions_remaining
                }

            # Budget-aware operation manager status
            if self.budget_aware_manager:
                operation_details = self.budget_aware_manager.get_operation_mode_details(
                    status["budget_status"].get("utilization_percentage", 0)
                )
                status["operation_mode"] = operation_details

                # Get performance metrics
                status["performance_metrics"] = self.budget_aware_manager.get_performance_metrics()

            # Tri-model router status
            if self.tri_model_router:
                status["system_components"]["tri_model_router"] = {
                    "available": True,
                    "model_status": self.tri_model_router.get_model_status(),
                    "provider_routing": self.tri_model_router.get_openrouter_provider_routing_info()
                }

            # Multi-stage pipeline status
            if self.multi_stage_pipeline:
                status["system_components"]["multi_stage_pipeline"] = {
                    "available": True,
                    "configuration": self.multi_stage_pipeline.get_pipeline_configuration()
                }

            # Error recovery status
            if self.error_recovery_manager:
                recovery_status = self.error_recovery_manager.get_recovery_status()
                status["error_recovery"] = recovery_status

            # Performance monitoring status
            if self.performance_monitor:
                status["system_components"]["performance_monitor"] = {
                    "available": True,
                    "system_health": self.performance_monitor.get_system_health_status()
                }

            # Tournament compliance check
            status["tournament_compliance"] = self._check_tournament_compliance_integration()

            # Determine overall health
            health_indicators = []

            if status["budget_status"].get("utilization_percentage", 0) < 95:
                health_indicators.append("budget_healthy")

            if status["tournament_compliance"].get("budget_compliant", False):
                health_indicators.append("compliance_healthy")

            if status["error_recovery"].get("system_health", {}).get("status") in ["healthy", "degraded"]:
                health_indicators.append("recovery_healthy")

            if len(health_indicators) >= 2:
                status["overall_health"] = "healthy"
            elif len(health_indicators) >= 1:
                status["overall_health"] = "degraded"
            else:
                status["overall_health"] = "unhealthy"

        except Exception as e:
            status["error"] = f"Status collection error: {str(e)}"
            status["overall_health"] = "error"
            logger.error(f"Enhanced system status collection error: {e}")

        return status

    def log_enhanced_system_status(self):
        """Log comprehensive system status for monitoring (Task 8.2)."""
        try:
            status = self.get_enhanced_system_status()

            logger.info("=== ENHANCED SYSTEM STATUS ===")
            logger.info(f"Overall Health: {status['overall_health'].upper()}")

            # Budget status
            if status.get("budget_status"):
                budget = status["budget_status"]
                logger.info(f"Budget: {budget.get('utilization_percentage', 0):.1f}% used, "
                          f"${budget.get('remaining', 0):.4f} remaining, "
                          f"~{budget.get('estimated_questions_remaining', 0)} questions left")

            # Operation mode
            if status.get("operation_mode"):
                mode = status["operation_mode"]
                logger.info(f"Operation Mode: {mode.get('operation_mode', 'unknown').upper()}")
                logger.info(f"Mode Description: {mode.get('mode_description', 'N/A')}")

            # Performance metrics
            if status.get("performance_metrics"):
                perf = status["performance_metrics"]
                logger.info(f"Mode Switches: {perf.get('mode_switches_count', 0)}")
                logger.info(f"Emergency Activations: {perf.get('emergency_activations', 0)}")
                logger.info(f"Cost Savings: ${perf.get('cost_savings_achieved', 0):.4f}")

            # Tournament compliance
            if status.get("tournament_compliance"):
                compliance = status["tournament_compliance"]
                compliant_count = sum(1 for v in compliance.values() if isinstance(v, bool) and v)
                total_checks = sum(1 for v in compliance.values() if isinstance(v, bool))
                logger.info(f"Tournament Compliance: {compliant_count}/{total_checks} checks passed")

                if compliance.get("issues"):
                    logger.warning("Compliance Issues:")
                    for issue in compliance["issues"]:
                        logger.warning(f"  - {issue}")

            # System components
            if status.get("system_components"):
                components = status["system_components"]
                available_components = [name for name, info in components.items()
                                      if info.get("available", False)]
                logger.info(f"Available Components: {', '.join(available_components)}")

        except Exception as e:
            logger.error(f"Enhanced system status logging error: {e}")

    async def _call_exa_smart_searcher(self, question: str) -> str:
        """
        SmartSearcher is a custom class that is a wrapper around an search on Exa.ai
        """
        searcher = SmartSearcher(
            model=self.get_llm("default", "llm"),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )  # You can ask the searcher to filter by date, exclude/include a domain, and run specific searches for finding sources vs finding highlights within a source
        response = await searcher.invoke(prompt)
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:

        question_id = str(getattr(question, 'id', 'unknown'))

        # Get budget status and operation mode for intelligent routing
        budget_remaining = 100.0
        operation_mode = "normal"

        if self.budget_manager:
            budget_status = self.budget_manager.get_budget_status()
            budget_remaining = 100.0 - budget_status.utilization_percentage

        if self.budget_aware_manager:
            operation_mode = self.budget_aware_manager.operation_mode_manager.get_current_mode().value

        # PRIORITY: Try enhanced multi-stage validation pipeline for complete forecasting (Task 8.1)
        if self.multi_stage_pipeline:
            try:
                # Use the complete multi-stage validation pipeline for forecasting
                pipeline_result = await self.multi_stage_pipeline.process_question(
                    question=question.question_text,
                    question_type="binary",
                    context={
                        "question_url": question.page_url,
                        "budget_remaining": budget_remaining,
                        "operation_mode": operation_mode,
                        "background_info": question.background_info,
                        "resolution_criteria": getattr(question, 'resolution_criteria', ''),
                        "fine_print": getattr(question, 'fine_print', ''),
                        "research_data": research
                    }
                )

                if (pipeline_result.pipeline_success and
                    pipeline_result.forecast_result.quality_validation_passed and
                    pipeline_result.forecast_result.tournament_compliant):

                    logger.info(f"Multi-stage binary forecast successful for question {question_id}, "
                              f"cost: ${pipeline_result.total_cost:.4f}, "
                              f"quality: {pipeline_result.quality_score:.2f}, "
                              f"calibration: {pipeline_result.forecast_result.calibration_score:.2f}")

                    # Integrate budget manager with cost tracking and operation modes (Task 8.2)
                    self._track_question_processing_cost(
                        question_id=question_id,
                        task_type="binary_forecast_pipeline",
                        cost=pipeline_result.total_cost,
                        model_used="multi_stage_pipeline",
                        success=True
                    )

                    # Check and update operation modes based on budget utilization
                    self._integrate_budget_manager_with_operation_modes()

                    # Extract prediction and reasoning from pipeline result
                    prediction = pipeline_result.final_forecast
                    reasoning = pipeline_result.reasoning

                    return ReasonedPrediction(
                        prediction_value=prediction, reasoning=reasoning
                    )

                else:
                    logger.warning(f"Multi-stage binary forecast quality issues for {question_id}: "
                                 f"Success={pipeline_result.pipeline_success}, "
                                 f"Quality={pipeline_result.forecast_result.quality_validation_passed}, "
                                 f"Compliant={pipeline_result.forecast_result.tournament_compliant}")
                    # Continue to fallback methods

            except Exception as e:
                logger.warning(f"Multi-stage binary forecast failed: {e}")

                # Use comprehensive error recovery (Task 8.1)
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import ErrorContext
                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier="full",
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=1,
                            question_id=question_id,
                            provider="multi_stage_pipeline"
                        )

                        recovery_result = await self.error_recovery_manager.recover_from_error(e, error_context)
                        if recovery_result.success:
                            logger.info(f"Binary forecast error recovery successful: {recovery_result.message}")
                        else:
                            logger.warning(f"Binary forecast error recovery failed: {recovery_result.message}")
                    except Exception as recovery_error:
                        logger.error(f"Binary forecast error recovery system failed: {recovery_error}")

                # Continue to fallback methods

        # FALLBACK: Try enhanced tri-model router with tier-optimized prompts
        if self.tri_model_router and self.anti_slop_prompts:
            try:
                # Apply budget-aware model selection adjustments (Task 8.2)
                selected_model_tier = "full"  # Default for forecasting
                if self.budget_aware_manager:
                    # Get model selection adjustments based on current operation mode
                    adjusted_model = self.budget_aware_manager.apply_model_selection_adjustments("forecast")

                    # Map model to tier for prompt optimization
                    if "gpt-5-nano" in adjusted_model or "gpt-4o-mini" in adjusted_model:
                        selected_model_tier = "nano"
                    elif "gpt-5-mini" in adjusted_model:
                        selected_model_tier = "mini"
                    else:
                        selected_model_tier = "full"

                # Create tier-optimized anti-slop binary forecast prompt (Task 8.1)
                prompt = self.anti_slop_prompts.get_binary_forecast_prompt(
                    question_text=question.question_text,
                    background_info=question.background_info,
                    resolution_criteria=getattr(question, 'resolution_criteria', ''),
                    fine_print=getattr(question, 'fine_print', ''),
                    research=research,
                    model_tier=selected_model_tier
                )

                # Get budget-aware context for routing
                budget_context = self.tri_model_router.get_budget_aware_routing_context()
                budget_remaining = budget_context.remaining_percentage if budget_context else 100.0

                # Route to optimal model with budget-aware selection
                reasoning = await self.tri_model_router.route_query(
                    task_type="forecast",
                    content=prompt,
                    complexity="high",
                    budget_remaining=budget_remaining
                )

                logger.info(f"Enhanced tri-model binary forecast successful for question {question_id} "
                          f"using {selected_model_tier} tier")

            except Exception as e:
                logger.warning(f"Enhanced tri-model binary forecast failed: {e}")

                # Use comprehensive error recovery for tri-model failures
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import ErrorContext
                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier=selected_model_tier,
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=2,
                            question_id=question_id,
                            provider="tri_model_router"
                        )

                        recovery_result = await self.error_recovery_manager.recover_from_error(e, error_context)
                        if recovery_result.success and recovery_result.fallback_result:
                            logger.info(f"Tri-model error recovery successful: {recovery_result.message}")
                            # Could use recovered model, but for now fallback to legacy
                        else:
                            logger.warning(f"Tri-model error recovery failed: {recovery_result.message}")
                    except Exception as recovery_error:
                        logger.error(f"Tri-model error recovery system failed: {recovery_error}")

                # Fallback to legacy method
                reasoning = await self._legacy_binary_forecast(question, research)
        else:
            # Fallback to legacy method if enhanced components not available
            reasoning = await self._legacy_binary_forecast(question, research)

        # Extract prediction from reasoning with enhanced error handling
        try:
            prediction: float = PredictionExtractor.extract_last_percentage_value(
                reasoning, max_prediction=1, min_prediction=0
            )
        except Exception as e:
            logger.warning(f"Prediction extraction failed for {question_id}: {e}")
            prediction = 0.5  # Default neutral prediction

        logger.info(
            f"Binary forecast completed for URL {question.page_url} as {prediction:.3f} "
            f"(mode: {operation_mode}, budget: {budget_remaining:.1f}%)"
        )

        safe_reasoning = reasoning or SAFE_REASONING_FALLBACK

        return ReasonedPrediction(
            prediction_value=prediction, reasoning=safe_reasoning
        )

    async def _legacy_binary_forecast(self, question: BinaryQuestion, research: str) -> str:
        """Legacy binary forecasting method as fallback."""
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {getattr(question, 'resolution_criteria', '')}

            {getattr(question, 'fine_print', '')}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        # Get appropriate LLM based on complexity analysis and budget status
        if self.enhanced_llm_config:
            complexity_assessment = self.enhanced_llm_config.assess_question_complexity(
                question.question_text,
                question.background_info,
                getattr(question, 'resolution_criteria', ''),
                getattr(question, 'fine_print', '')
            )
            llm = self.enhanced_llm_config.get_llm_for_task("forecast", complexity_assessment=complexity_assessment)
        else:
            llm = self.get_llm("default", "llm")

        # Use safe LLM invoke with error handling and fallbacks
        return await self._safe_llm_invoke(
            llm, prompt, "forecast",
            question_id=str(getattr(question, 'id', 'unknown'))
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:

        question_id = str(getattr(question, 'id', 'unknown'))

        # Get budget status and operation mode for intelligent routing
        budget_remaining = 100.0
        operation_mode = "normal"

        if self.budget_manager:
            budget_status = self.budget_manager.get_budget_status()
            budget_remaining = 100.0 - budget_status.utilization_percentage

        if self.budget_aware_manager:
            operation_mode = self.budget_aware_manager.operation_mode_manager.get_current_mode().value

        # PRIORITY: Try enhanced multi-stage validation pipeline for complete forecasting (Task 8.1)
        if self.multi_stage_pipeline:
            try:
                # Use the complete multi-stage validation pipeline for forecasting
                pipeline_result = await self.multi_stage_pipeline.process_question(
                    question=question.question_text,
                    question_type="multiple_choice",
                    context={
                        "question_url": question.page_url,
                        "budget_remaining": budget_remaining,
                        "operation_mode": operation_mode,
                        "background_info": question.background_info,
                        "resolution_criteria": getattr(question, 'resolution_criteria', ''),
                        "fine_print": getattr(question, 'fine_print', ''),
                        "options": question.options,
                        "research_data": research
                    }
                )

                if (pipeline_result.pipeline_success and
                    pipeline_result.forecast_result.quality_validation_passed and
                    pipeline_result.forecast_result.tournament_compliant):

                    logger.info(f"Multi-stage multiple choice forecast successful for question {question_id}, "
                              f"cost: ${pipeline_result.total_cost:.4f}, "
                              f"quality: {pipeline_result.quality_score:.2f}")

                    # Integrate budget manager with cost tracking and operation modes (Task 8.2)
                    self._track_question_processing_cost(
                        question_id=question_id,
                        task_type="multiple_choice_forecast_pipeline",
                        cost=pipeline_result.total_cost,
                        model_used="multi_stage_pipeline",
                        success=True
                    )

                    # Check and update operation modes based on budget utilization
                    self._integrate_budget_manager_with_operation_modes()

                    # Convert pipeline result to expected format
                    if isinstance(pipeline_result.final_forecast, dict):
                        # Convert dict to PredictedOptionList format
                        option_predictions = []
                        for option, probability in pipeline_result.final_forecast.items():
                            option_predictions.append(f"{option}: {probability:.1%}")

                        # Create PredictedOptionList from the predictions
                        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
                            "\n".join(option_predictions), question.options
                        )
                    else:
                        # Fallback extraction from reasoning
                        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
                            pipeline_result.reasoning, question.options
                        )

                    return ReasonedPrediction(
                        prediction_value=prediction, reasoning=pipeline_result.reasoning
                    )

                else:
                    logger.warning(f"Multi-stage multiple choice forecast quality issues for {question_id}")
                    # Continue to fallback methods

            except Exception as e:
                logger.warning(f"Multi-stage multiple choice forecast failed: {e}")

                # Use comprehensive error recovery (Task 8.1)
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import ErrorContext
                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier="full",
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=1,
                            question_id=question_id,
                            provider="multi_stage_pipeline"
                        )

                        recovery_result = await self.error_recovery_manager.recover_from_error(e, error_context)
                        if recovery_result.success:
                            logger.info(f"Multiple choice forecast error recovery successful: {recovery_result.message}")
                        else:
                            logger.warning(f"Multiple choice forecast error recovery failed: {recovery_result.message}")
                    except Exception as recovery_error:
                        logger.error(f"Multiple choice forecast error recovery system failed: {recovery_error}")

        # FALLBACK: Try enhanced tri-model router with tier-optimized prompts
        if self.tri_model_router and self.anti_slop_prompts:
            try:
                # Apply budget-aware model selection adjustments (Task 8.2)
                selected_model_tier = "full"  # Default for forecasting
                if self.budget_aware_manager:
                    adjusted_model = self.budget_aware_manager.apply_model_selection_adjustments("forecast")

                    # Map model to tier for prompt optimization
                    if "gpt-5-nano" in adjusted_model or "gpt-4o-mini" in adjusted_model:
                        selected_model_tier = "nano"
                    elif "gpt-5-mini" in adjusted_model:
                        selected_model_tier = "mini"
                    else:
                        selected_model_tier = "full"

                # Create tier-optimized anti-slop multiple choice forecast prompt (Task 8.1)
                prompt = self.anti_slop_prompts.get_multiple_choice_prompt(
                    question_text=question.question_text,
                    options=question.options,
                    background_info=question.background_info,
                    resolution_criteria=getattr(question, 'resolution_criteria', ''),
                    fine_print=getattr(question, 'fine_print', ''),
                    research=research,
                    model_tier=selected_model_tier
                )

                # Get budget-aware context for routing
                budget_context = self.tri_model_router.get_budget_aware_routing_context()
                budget_remaining = budget_context.remaining_percentage if budget_context else 100.0

                # Route to optimal model with budget-aware selection
                reasoning = await self.tri_model_router.route_query(
                    task_type="forecast",
                    content=prompt,
                    complexity="high",
                    budget_remaining=budget_remaining
                )

                logger.info(f"Enhanced tri-model multiple choice forecast successful for question {question_id} "
                          f"using {selected_model_tier} tier")

            except Exception as e:
                logger.warning(f"Enhanced tri-model multiple choice forecast failed: {e}")

                # Use comprehensive error recovery for tri-model failures
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import ErrorContext
                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier=selected_model_tier,
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=2,
                            question_id=question_id,
                            provider="tri_model_router"
                        )

                        recovery_result = await self.error_recovery_manager.recover_from_error(e, error_context)
                        if recovery_result.success:
                            logger.info(f"Tri-model multiple choice error recovery successful: {recovery_result.message}")
                        else:
                            logger.warning(f"Tri-model multiple choice error recovery failed: {recovery_result.message}")
                    except Exception as recovery_error:
                        logger.error(f"Tri-model multiple choice error recovery system failed: {recovery_error}")

                # Fallback to legacy method
                reasoning = await self._legacy_multiple_choice_forecast(question, research)
        else:
            # Fallback to legacy method if enhanced components not available
            reasoning = await self._legacy_multiple_choice_forecast(question, research)

        # Extract prediction from reasoning with enhanced error handling
        try:
            prediction: PredictedOptionList = (
                PredictionExtractor.extract_option_list_with_percentage_afterwards(
                    reasoning, question.options
                )
            )
        except Exception as e:
            logger.warning(f"Multiple choice prediction extraction failed for {question_id}: {e}")
            # Create default equal probability distribution
            from forecasting_tools import PredictedOption
            equal_prob = 1.0 / len(question.options)
            predicted_options = [
                PredictedOption(option_name=option, probability=equal_prob)
                for option in question.options
            ]
            prediction = PredictedOptionList(predicted_options=predicted_options)

        logger.info(
            f"Multiple choice forecast completed for URL {question.page_url} "
            f"(mode: {operation_mode}, budget: {budget_remaining:.1f}%)"
        )

        safe_reasoning = reasoning or SAFE_REASONING_FALLBACK

        return ReasonedPrediction(
            prediction_value=prediction, reasoning=safe_reasoning
        )

    async def _legacy_multiple_choice_forecast(self, question: MultipleChoiceQuestion, research: str) -> str:
        """Legacy multiple choice forecasting method as fallback."""
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {getattr(question, 'resolution_criteria', '')}

            {getattr(question, 'fine_print', '')}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )

        # Get appropriate LLM and use safe invoke
        if self.enhanced_llm_config:
            complexity_assessment = self.enhanced_llm_config.assess_question_complexity(
                question.question_text, question.background_info
            )
            llm = self.enhanced_llm_config.get_llm_for_task("forecast", complexity_assessment=complexity_assessment)
        else:
            llm = self.get_llm("default", "llm")

        return await self._safe_llm_invoke(
            llm, prompt, "forecast",
            question_id=str(getattr(question, 'id', 'unknown'))
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:

        question_id = str(getattr(question, 'id', 'unknown'))

        # Get budget status and operation mode for intelligent routing
        budget_remaining = 100.0
        operation_mode = "normal"

        if self.budget_manager:
            budget_status = self.budget_manager.get_budget_status()
            budget_remaining = 100.0 - budget_status.utilization_percentage

        if self.budget_aware_manager:
            operation_mode = self.budget_aware_manager.operation_mode_manager.get_current_mode().value

        # PRIORITY: Try enhanced multi-stage validation pipeline for complete forecasting (Task 8.1)
        if self.multi_stage_pipeline:
            try:
                # Use the complete multi-stage validation pipeline for forecasting
                pipeline_result = await self.multi_stage_pipeline.process_question(
                    question=question.question_text,
                    question_type="numeric",
                    context={
                        "question_url": question.page_url,
                        "budget_remaining": budget_remaining,
                        "operation_mode": operation_mode,
                        "background_info": question.background_info,
                        "resolution_criteria": getattr(question, 'resolution_criteria', ''),
                        "fine_print": getattr(question, 'fine_print', ''),
                        "unit_of_measure": question.unit_of_measure,
                        "lower_bound": question.lower_bound if not question.open_lower_bound else None,
                        "upper_bound": question.upper_bound if not question.open_upper_bound else None,
                        "open_lower_bound": question.open_lower_bound,
                        "open_upper_bound": question.open_upper_bound,
                        "research_data": research
                    }
                )

                if (pipeline_result.pipeline_success and
                    pipeline_result.forecast_result.quality_validation_passed and
                    pipeline_result.forecast_result.tournament_compliant):

                    logger.info(f"Multi-stage numeric forecast successful for question {question_id}, "
                              f"cost: ${pipeline_result.total_cost:.4f}, "
                              f"quality: {pipeline_result.quality_score:.2f}")

                    # Integrate budget manager with cost tracking and operation modes (Task 8.2)
                    self._track_question_processing_cost(
                        question_id=question_id,
                        task_type="numeric_forecast_pipeline",
                        cost=pipeline_result.total_cost,
                        model_used="multi_stage_pipeline",
                        success=True
                    )

                    # Check and update operation modes based on budget utilization
                    self._integrate_budget_manager_with_operation_modes()

                    # Convert pipeline result to expected format
                    if isinstance(pipeline_result.final_forecast, dict):
                        # Convert percentile dict to NumericDistribution
                        try:
                            percentiles = {}
                            for key, value in pipeline_result.final_forecast.items():
                                if isinstance(key, (int, str)) and str(key).isdigit():
                                    percentiles[int(key)] = float(value)

                            if percentiles:
                                from forecasting_tools.data_models.numeric_report import Percentile
                                percentile_list = [
                                    Percentile(percentile=float(p)/100.0, value=float(v))
                                    for p, v in percentiles.items()
                                ]
                                prediction = NumericDistribution(
                                    declared_percentiles=percentile_list,
                                    open_upper_bound=question.open_upper_bound,
                                    open_lower_bound=question.open_lower_bound,
                                    upper_bound=question.upper_bound,
                                    lower_bound=question.lower_bound,
                                    zero_point=None
                                )
                            else:
                                # Fallback extraction from reasoning
                                prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                                    pipeline_result.reasoning, question
                                )
                        except Exception as conversion_error:
                            logger.warning(f"Numeric prediction conversion failed: {conversion_error}")
                            # Fallback extraction from reasoning
                            prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                                pipeline_result.reasoning, question
                            )
                    else:
                        # Fallback extraction from reasoning
                        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                            pipeline_result.reasoning, question
                        )

                    return ReasonedPrediction(
                        prediction_value=prediction, reasoning=pipeline_result.reasoning
                    )

                else:
                    logger.warning(f"Multi-stage numeric forecast quality issues for {question_id}")
                    # Continue to fallback methods

            except Exception as e:
                logger.warning(f"Multi-stage numeric forecast failed: {e}")

                # Use comprehensive error recovery (Task 8.1)
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import ErrorContext
                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier="full",
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=1,
                            question_id=question_id,
                            provider="multi_stage_pipeline"
                        )

                        recovery_result = await self.error_recovery_manager.recover_from_error(e, error_context)
                        if recovery_result.success:
                            logger.info(f"Numeric forecast error recovery successful: {recovery_result.message}")
                        else:
                            logger.warning(f"Numeric forecast error recovery failed: {recovery_result.message}")
                    except Exception as recovery_error:
                        logger.error(f"Numeric forecast error recovery system failed: {recovery_error}")

        # FALLBACK: Try enhanced tri-model router with tier-optimized prompts
        if self.tri_model_router and self.anti_slop_prompts:
            try:
                # Apply budget-aware model selection adjustments (Task 8.2)
                selected_model_tier = "full"  # Default for forecasting
                if self.budget_aware_manager:
                    adjusted_model = self.budget_aware_manager.apply_model_selection_adjustments("forecast")

                    # Map model to tier for prompt optimization
                    if "gpt-5-nano" in adjusted_model or "gpt-4o-mini" in adjusted_model:
                        selected_model_tier = "nano"
                    elif "gpt-5-mini" in adjusted_model:
                        selected_model_tier = "mini"
                    else:
                        selected_model_tier = "full"

                # Create tier-optimized anti-slop numeric forecast prompt (Task 8.1)
                prompt = self.anti_slop_prompts.get_numeric_forecast_prompt(
                    question_text=question.question_text,
                    background_info=question.background_info,
                    resolution_criteria=getattr(question, 'resolution_criteria', ''),
                    fine_print=getattr(question, 'fine_print', ''),
                    research=research,
                    unit_of_measure=question.unit_of_measure,
                    lower_bound=question.lower_bound if not question.open_lower_bound else None,
                    upper_bound=question.upper_bound if not question.open_upper_bound else None,
                    model_tier=selected_model_tier
                )

                # Get budget-aware context for routing
                budget_context = self.tri_model_router.get_budget_aware_routing_context()
                budget_remaining = budget_context.remaining_percentage if budget_context else 100.0

                # Route to optimal model with budget-aware selection
                reasoning = await self.tri_model_router.route_query(
                    task_type="forecast",
                    content=prompt,
                    complexity="high",
                    budget_remaining=budget_remaining
                )

                logger.info(f"Enhanced tri-model numeric forecast successful for question {question_id} "
                          f"using {selected_model_tier} tier")

            except Exception as e:
                logger.warning(f"Enhanced tri-model numeric forecast failed: {e}")

                # Use comprehensive error recovery for tri-model failures
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import ErrorContext
                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier=selected_model_tier,
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=2,
                            question_id=question_id,
                            provider="tri_model_router"
                        )

                        recovery_result = await self.error_recovery_manager.recover_from_error(e, error_context)
                        if recovery_result.success:
                            logger.info(f"Tri-model numeric error recovery successful: {recovery_result.message}")
                        else:
                            logger.warning(f"Tri-model numeric error recovery failed: {recovery_result.message}")
                    except Exception as recovery_error:
                        logger.error(f"Tri-model numeric error recovery system failed: {recovery_error}")

                # Fallback to legacy method
                reasoning = await self._legacy_numeric_forecast(question, research)
        else:
            # Fallback to legacy method if enhanced components not available
            reasoning = await self._legacy_numeric_forecast(question, research)

        # Extract prediction from reasoning with enhanced error handling
        try:
            prediction: NumericDistribution = (
                PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                    reasoning, question
                )
            )
        except Exception as e:
            logger.warning(f"Numeric prediction extraction failed for {question_id}: {e}")
            # Create default distribution with median estimate
            median_estimate = (question.lower_bound + question.upper_bound) / 2 if (
                not question.open_lower_bound and not question.open_upper_bound
            ) else 1.0

            from forecasting_tools.data_models.numeric_report import Percentile
            percentiles = [
                Percentile(percentile=0.1, value=median_estimate * 0.5),
                Percentile(percentile=0.5, value=median_estimate),
                Percentile(percentile=0.9, value=median_estimate * 1.5)
            ]
            prediction = NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=question.open_upper_bound,
                open_lower_bound=question.open_lower_bound,
                upper_bound=question.upper_bound,
                lower_bound=question.lower_bound,
                zero_point=None
            )

        logger.info(
            f"Numeric forecast completed for URL {question.page_url} "
            f"(mode: {operation_mode}, budget: {budget_remaining:.1f}%)"
        )

        safe_reasoning = reasoning or SAFE_REASONING_FALLBACK

        return ReasonedPrediction(
            prediction_value=prediction, reasoning=safe_reasoning
        )

    async def _legacy_numeric_forecast(self, question: NumericQuestion, research: str) -> str:
        """Legacy numeric forecasting method as fallback."""
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {getattr(question, 'resolution_criteria', '')}

            {getattr(question, 'fine_print', '')}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    # Perform OpenRouter startup validation and auto-configuration (Task 9.2)
    async def validate_openrouter_startup():
        """Validate OpenRouter configuration on startup."""
        try:
            from src.infrastructure.config.openrouter_startup_validator import OpenRouterStartupValidator
            from src.infrastructure.config.tri_model_router import OpenRouterTriModelRouter

            logger.info("Performing OpenRouter startup validation...")

            # Run validation
            validator = OpenRouterStartupValidator()
            validation_success = await validator.run_startup_validation(exit_on_failure=False)

            if validation_success:
                logger.info("✅ OpenRouter configuration validated successfully")

                # Create router with auto-configuration
                router = await OpenRouterTriModelRouter.create_with_auto_configuration()
                logger.info("✅ OpenRouter tri-model router configured and ready")

                return router
            else:
                logger.warning("⚠️ OpenRouter validation failed - system may have limited functionality")
                return None

        except ImportError as e:
            logger.warning(f"OpenRouter validation components not available: {e}")
            return None
        except Exception as e:
            logger.error(f"OpenRouter startup validation failed: {e}")
            return None

    # Run OpenRouter validation
    openrouter_router = asyncio.run(validate_openrouter_startup())

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    # Create enhanced LLM configuration with tri-model GPT-5 routing
    def create_enhanced_llms():
        """Create LLM configuration with tri-model GPT-5 routing and budget-aware selection."""
        llms = {}

        # Try to use tri-model router first
        try:
            from src.infrastructure.config.tri_model_router import tri_model_router

            # Get models from tri-model router
            router_models = tri_model_router.models

            # Map router models to expected LLM names
            llms["default"] = router_models["full"]      # GPT-5 full for main forecasting
            llms["summarizer"] = router_models["nano"]   # GPT-5 nano for simple tasks
            llms["researcher"] = router_models["mini"]   # GPT-5 mini for research

            logger.info("Using tri-model GPT-5 configuration:")
            for name, model in llms.items():
                logger.info(f"  {name}: {model.model}")

            return llms

        except ImportError as e:
            logger.warning(f"Tri-model router not available, falling back to legacy models: {e}")
            # Continue to legacy configuration below

        # Get OpenRouter API key
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        if not openrouter_key or openrouter_key.startswith("dummy_"):
            logger.error("OpenRouter API key not configured! Using fallback configuration.")
            openrouter_key = None

        # Try to use tournament-optimized models with proxy support
        if TOURNAMENT_COMPONENTS_AVAILABLE:
            try:
                tournament_config = get_tournament_config()

                # Default model with OpenRouter primary, proxy fallback
                try:
                    if openrouter_key:
                        default_model = os.getenv("PRIMARY_FORECAST_MODEL", "openai/gpt-4o")
                        llms["default"] = GeneralLlm(
                            model=default_model,
                            api_key=openrouter_key,
                            temperature=0.3,
                            timeout=60,
                            allowed_tries=3,
                        )
                        logger.info(f"Using OpenRouter model for default: {default_model}")
                    else:
                        default_model = os.getenv("METACULUS_DEFAULT_MODEL", "metaculus/claude-3-5-sonnet")
                        llms["default"] = GeneralLlm(
                            model=default_model,
                            temperature=0.3,
                            timeout=60,
                            allowed_tries=3,
                        )
                        logger.info(f"Using proxy model for default: {default_model}")
                except Exception as e:
                    logger.warning(f"Failed to create default model: {e}")
                    llms["default"] = GeneralLlm(
                        model="openrouter/anthropic/claude-3-5-sonnet",
                        api_key=openrouter_key,
                        temperature=0.3,
                        timeout=60,
                        allowed_tries=3,
                    )

                # Summarizer model with OpenRouter primary
                try:
                    if openrouter_key:
                        summarizer_model = os.getenv("SIMPLE_TASK_MODEL", "openai/gpt-4o-mini")
                        llms["summarizer"] = GeneralLlm(
                            model=summarizer_model,
                            api_key=openrouter_key,
                            temperature=0.0,
                            timeout=45,
                            allowed_tries=3,
                        )
                        logger.info(f"Using OpenRouter model for summarizer: {summarizer_model}")
                    else:
                        summarizer_model = os.getenv("METACULUS_SUMMARIZER_MODEL", "metaculus/gpt-4o-mini")
                        llms["summarizer"] = GeneralLlm(
                            model=summarizer_model,
                            temperature=0.0,
                            timeout=45,
                            allowed_tries=3,
                        )
                        logger.info(f"Using proxy model for summarizer: {summarizer_model}")
                except Exception as e:
                    logger.warning(f"Failed to create summarizer model: {e}")
                    llms["summarizer"] = GeneralLlm(
                        model="openai/gpt-4o-mini",
                        api_key=openrouter_key,
                        temperature=0.0,
                        timeout=45,
                        allowed_tries=3,
                    )

                # Research model with OpenRouter primary
                try:
                    if openrouter_key:
                        research_model = os.getenv("PRIMARY_RESEARCH_MODEL", "openai/gpt-4o-mini")
                        llms["researcher"] = GeneralLlm(
                            model=research_model,
                            api_key=openrouter_key,
                            temperature=0.1,
                            timeout=90,
                            allowed_tries=2,
                        )
                        logger.info(f"Using OpenRouter model for researcher: {research_model}")
                    else:
                        research_model = os.getenv("METACULUS_RESEARCH_MODEL", "metaculus/gpt-4o")
                        llms["researcher"] = GeneralLlm(
                            model=research_model,
                            temperature=0.1,
                            timeout=90,
                            allowed_tries=2,
                        )
                        logger.info(f"Using proxy model for researcher: {research_model}")
                except Exception as e:
                    logger.warning(f"Failed to create research model: {e}")
                    llms["researcher"] = GeneralLlm(
                        model="openrouter/openai/gpt-4o",
                        api_key=openrouter_key,
                        temperature=0.1,
                        timeout=90,
                        allowed_tries=2,
                    )

            except Exception as e:
                logger.warning(f"Failed to initialize tournament LLMs: {e}")

        # Fallback to OpenRouter-based models if tournament components failed
        if not llms:
            logger.info("Using OpenRouter-based fallback LLM configuration")
            llms = {
                "default": GeneralLlm(
                    model=os.getenv("PRIMARY_FORECAST_MODEL", "openai/gpt-4o"),
                    api_key=openrouter_key,
                    temperature=0.3,
                    timeout=60,
                    allowed_tries=3,
                ),
                "summarizer": GeneralLlm(
                    model=os.getenv("SIMPLE_TASK_MODEL", "openai/gpt-4o-mini"),
                    api_key=openrouter_key,
                    temperature=0.0,
                    timeout=45,
                    allowed_tries=3,
                ),
                "researcher": GeneralLlm(
                    model=os.getenv("PRIMARY_RESEARCH_MODEL", "openai/gpt-4o-mini"),
                    api_key=openrouter_key,
                    temperature=0.1,
                    timeout=90,
                    allowed_tries=2,
                ),
            }

        return llms

    # Initialize bot with enhanced configuration
    enhanced_llms = create_enhanced_llms()

    # Get tournament configuration for bot parameters
    if TOURNAMENT_COMPONENTS_AVAILABLE:
        try:
            tournament_config = get_tournament_config()
            research_reports = tournament_config.max_research_reports_per_question
            predictions_per_report = tournament_config.max_predictions_per_report
            publish_reports = tournament_config.publish_reports and not tournament_config.dry_run
            skip_previously_forecasted = tournament_config.skip_previously_forecasted
        except Exception as e:
            logger.warning(f"Failed to get tournament config: {e}")
            # Fallback to environment variables
            research_reports = int(os.getenv("MAX_RESEARCH_REPORTS_PER_QUESTION", "1"))
            predictions_per_report = int(os.getenv("MAX_PREDICTIONS_PER_REPORT", "5"))
            publish_reports = os.getenv("PUBLISH_REPORTS", "true").lower() == "true" and not os.getenv("DRY_RUN", "false").lower() == "true"
            skip_previously_forecasted = os.getenv("SKIP_PREVIOUSLY_FORECASTED", "true").lower() == "true"
    else:
        # Use environment variables for configuration
        research_reports = int(os.getenv("MAX_RESEARCH_REPORTS_PER_QUESTION", "1"))
        predictions_per_report = int(os.getenv("MAX_PREDICTIONS_PER_REPORT", "5"))
        publish_reports = os.getenv("PUBLISH_REPORTS", "true").lower() == "true" and not os.getenv("DRY_RUN", "false").lower() == "true"
        skip_previously_forecasted = os.getenv("SKIP_PREVIOUSLY_FORECASTED", "true").lower() == "true"

    # Log configuration
    logger.info(f"Bot Configuration:")
    logger.info(f"  Research reports per question: {research_reports}")
    logger.info(f"  Predictions per research report: {predictions_per_report}")
    logger.info(f"  Publish reports to Metaculus: {publish_reports}")
    logger.info(f"  Skip previously forecasted: {skip_previously_forecasted}")
    logger.info(f"  Tournament mode: {os.getenv('TOURNAMENT_MODE', 'false')}")
    logger.info(f"  Tournament ID: {os.getenv('AIB_TOURNAMENT_ID', '32813')}")
    logger.info(f"  Budget limit: ${os.getenv('BUDGET_LIMIT', '100.0')}")
    logger.info(f"  Scheduling frequency: {os.getenv('SCHEDULING_FREQUENCY_HOURS', '4')} hours")

    template_bot = TemplateForecaster(
        research_reports_per_question=research_reports,
        predictions_per_research_report=predictions_per_report,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_reports,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_previously_forecasted,
        llms=enhanced_llms,
    )

    forecast_reports = []  # ensure defined for summary even if failures occur
    try:
        if run_mode == "tournament":
            # Use specific tournament ID from environment variable (Fall 2025 tournament)
            tournament_id = int(os.getenv("AIB_TOURNAMENT_ID", "32813"))
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    tournament_id, return_exceptions=True
                )
            )
        elif run_mode == "quarterly_cup":
            # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
            # The new quarterly cup may not be initialized near the beginning of a quarter
            template_bot.skip_previously_forecasted_questions = False
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
                )
            )
        elif run_mode == "test_questions":
            # Example questions are a good way to test the bot's performance on a single question
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
                "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
                "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            ]
            template_bot.skip_previously_forecasted_questions = False
            questions = [
                MetaculusApi.get_question_by_url(question_url)
                for question_url in EXAMPLE_QUESTIONS
            ]
            forecast_reports = asyncio.run(
                template_bot.forecast_questions(questions, return_exceptions=True)
            )
    except Exception as e:
        logger.error("Forecasting run failed: %s", e)
        # Preserve the exception in the report list so downstream summary still works
        forecast_reports = [e]
    # Log comprehensive report summary (tolerant to missing/exception entries)
    try:
        TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore
    except Exception as e:
        logger.warning("Failed to log report summary: %s", e)

    # Log budget usage statistics if available
    if hasattr(template_bot, 'budget_manager') and template_bot.budget_manager:
        try:
            logger.info("=== Budget Usage Statistics ===")
            template_bot.budget_manager.log_budget_status()

            # Generate and log budget report
            if hasattr(template_bot, 'budget_alert_system') and template_bot.budget_alert_system:
                template_bot.budget_alert_system.log_budget_summary()

                # Get cost optimization suggestions
                suggestions = template_bot.budget_alert_system.get_cost_optimization_suggestions()
                if suggestions:
                    logger.info("Cost Optimization Suggestions:")
                    for i, suggestion in enumerate(suggestions[:3], 1):
                        logger.info("  %d. %s", i, suggestion)

        except Exception as e:
            logger.warning("Failed to log budget statistics: %s", e)

    # Log tournament usage statistics if available
    if TOURNAMENT_COMPONENTS_AVAILABLE and hasattr(template_bot, 'tournament_asknews') and template_bot.tournament_asknews:
        try:
            stats = template_bot.tournament_asknews.get_usage_stats()
            logger.info("=== Tournament Usage Statistics ===")
            logger.info(f"AskNews Total Requests: {stats['total_requests']}")
            logger.info(f"AskNews Success Rate: {stats['success_rate']:.1f}%")
            logger.info(f"AskNews Fallback Rate: {stats['fallback_rate']:.1f}%")
            logger.info(f"AskNews Quota Used: {stats['estimated_quota_used']}/{stats['quota_limit']} "
                       f"({stats['quota_usage_percentage']:.1f}%)")
            logger.info(f"AskNews Daily Requests: {stats['daily_request_count']}/{stats.get('daily_limit', 'N/A')}")

            # Alert if quota usage is high
            if stats['quota_usage_percentage'] > 80:
                logger.warning(
                    "HIGH QUOTA USAGE: %.1f%% of AskNews quota used!",
                    stats['quota_usage_percentage']
                )

            # Log fallback provider status
            fallback_status = template_bot.tournament_asknews.get_fallback_providers_status()
            logger.info("Fallback Providers Status:")
            for provider, available in fallback_status.items():
                status = "✓ Available" if available else "✗ Not configured"
                logger.info(f"  {provider}: {status}")

        except Exception as e:
            logger.warning(f"Failed to log tournament statistics: {e}")

    # Final status summary
    try:
        successful_forecasts = len([r for r in forecast_reports if not isinstance(r, Exception)])
        failed_forecasts = len([r for r in forecast_reports if isinstance(r, Exception)])

        logger.info("=== Final Summary ===")
        logger.info("Successful forecasts: %d", successful_forecasts)
        logger.info("Failed forecasts: %d", failed_forecasts)
        logger.info("Total questions processed: %d", len(forecast_reports))

        if failed_forecasts > 0:
            logger.warning("Some forecasts failed. Check logs above for details.")
            # Log first few exceptions for debugging
            exceptions = [r for r in forecast_reports if isinstance(r, Exception)][:3]
            for i, exc in enumerate(exceptions, 1):
                logger.error("Exception %d: %s: %s", i, type(exc).__name__, exc)
    except Exception as e:
        logger.warning("Failed to log final summary: %s", e)

    logger.info("Bot execution completed.")
