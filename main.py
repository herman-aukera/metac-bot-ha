import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Literal
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
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

logger = logging.getLogger(__name__)

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

        # Initialize tri-model router for GPT-5 variants with anti-slop directives
        try:
            from src.infrastructure.config.tri_model_router import tri_model_router
            from src.prompts.anti_slop_prompts import anti_slop_prompts

            self.tri_model_router = tri_model_router
            self.anti_slop_prompts = anti_slop_prompts

            # Log tri-model status
            model_status = self.tri_model_router.get_model_status()
            logger.info("Tri-model router initialized:")
            for tier, status in model_status.items():
                logger.info(f"  {tier}: {status}")

        except ImportError as e:
            logger.warning(f"Tri-model router not available: {e}")
            self.tri_model_router = None
            self.anti_slop_prompts = None

        # Initialize OpenRouter API key configuration
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "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")
        if not self.openrouter_api_key or self.openrouter_api_key.startswith("dummy_"):
            logger.error("OpenRouter API key not configured! This is required for tournament operation.")
        else:
            logger.info("OpenRouter API key configured successfully")

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
        else:
            self.tournament_config = None
            self.tournament_asknews = None

        # Set default concurrency if not set by tournament config
        if not hasattr(self, '_max_concurrent_questions'):
            self._max_concurrent_questions = 2
            self._concurrency_limiter = asyncio.Semaphore(self._max_concurrent_questions)

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
        """Enhanced research with tri-model routing, anti-slop directives, and budget management."""
        async with self._concurrency_limiter:
            # Get budget status for model selection
            budget_remaining = 100.0
            if self.budget_manager:
                budget_status = self.budget_manager.get_budget_status()
                budget_remaining = 100.0 - budget_status.utilization_percentage

                # Check and alert on budget status
                if self.budget_alert_system:
                    alert = self.budget_alert_system.check_and_alert()

                # If in emergency mode, skip research to save budget
                if budget_status.status_level == "emergency":
                    logger.warning(f"Emergency budget mode: skipping research for {question.page_url}")
                    return "Research skipped due to emergency budget mode."

            # Try tri-model router first for intelligent research
            if self.tri_model_router and self.anti_slop_prompts:
                try:
                    # Create anti-slop research prompt
                    research_prompt = self.anti_slop_prompts.get_research_prompt(
                        question_text=question.question_text,
                        model_tier="mini"  # Use mini model for research by default
                    )

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

            # Fallback to Exa
            if not research and os.getenv("EXA_API_KEY") and not os.getenv("EXA_API_KEY").startswith("dummy_"):
                try:
                    research = await self._call_exa_smart_searcher(question.question_text)
                    if research and len(research.strip()) > 0:
                        logger.info(f"Exa research successful for URL {question.page_url}")
                        return research
                except Exception as e:
                    logger.warning(f"Exa search failed: {e}")

            # Fallback to Perplexity
            if not research and os.getenv("PERPLEXITY_API_KEY") and not os.getenv("PERPLEXITY_API_KEY").startswith("dummy_"):
                try:
                    research = await self._call_perplexity(question.question_text)
                    if research and len(research.strip()) > 0:
                        logger.info(f"Perplexity research successful for URL {question.page_url}")
                        return research
                except Exception as e:
                    logger.warning(f"Perplexity search failed: {e}")

            # Fallback to OpenRouter Perplexity
            if not research and os.getenv("OPENROUTER_API_KEY"):
                try:
                    research = await self._call_perplexity(question.question_text, use_open_router=True)
                    if research and len(research.strip()) > 0:
                        logger.info(f"OpenRouter Perplexity research successful for URL {question.page_url}")
                        return research
                except Exception as e:
                    logger.warning(f"OpenRouter Perplexity search failed: {e}")

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

        # Get budget status for model selection
        budget_remaining = 100.0
        if self.budget_manager:
            budget_status = self.budget_manager.get_budget_status()
            budget_remaining = 100.0 - budget_status.utilization_percentage

        # Try tri-model router with anti-slop prompts first
        if self.tri_model_router and self.anti_slop_prompts:
            try:
                # Create anti-slop binary forecast prompt
                prompt = self.anti_slop_prompts.get_binary_forecast_prompt(
                    question_text=question.question_text,
                    background_info=question.background_info,
                    resolution_criteria=getattr(question, 'resolution_criteria', ''),
                    fine_print=getattr(question, 'fine_print', ''),
                    research=research,
                    model_tier="full"  # Use full model for final forecasting
                )

                # Route to optimal model (likely GPT-5 full for forecasting)
                reasoning = await self.tri_model_router.route_query(
                    task_type="forecast",
                    content=prompt,
                    complexity="high",
                    budget_remaining=budget_remaining
                )

                logger.info(f"Tri-model binary forecast successful for question {question_id}")

            except Exception as e:
                logger.warning(f"Tri-model binary forecast failed: {e}")
                # Fallback to legacy method
                reasoning = await self._legacy_binary_forecast(question, research)
        else:
            # Fallback to legacy method if tri-model not available
            reasoning = await self._legacy_binary_forecast(question, research)

        # Extract prediction from reasoning
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )

        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )

        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
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

        # Get budget status for model selection
        budget_remaining = 100.0
        if self.budget_manager:
            budget_status = self.budget_manager.get_budget_status()
            budget_remaining = 100.0 - budget_status.utilization_percentage

        # Try tri-model router with anti-slop prompts first
        if self.tri_model_router and self.anti_slop_prompts:
            try:
                # Create anti-slop multiple choice forecast prompt
                prompt = self.anti_slop_prompts.get_multiple_choice_prompt(
                    question_text=question.question_text,
                    options=question.options,
                    background_info=question.background_info,
                    resolution_criteria=getattr(question, 'resolution_criteria', ''),
                    fine_print=getattr(question, 'fine_print', ''),
                    research=research,
                    model_tier="full"  # Use full model for final forecasting
                )

                # Route to optimal model
                reasoning = await self.tri_model_router.route_query(
                    task_type="forecast",
                    content=prompt,
                    complexity="high",
                    budget_remaining=budget_remaining
                )

                logger.info(f"Tri-model multiple choice forecast successful for question {question_id}")

            except Exception as e:
                logger.warning(f"Tri-model multiple choice forecast failed: {e}")
                # Fallback to legacy method
                reasoning = await self._legacy_multiple_choice_forecast(question, research)
        else:
            # Fallback to legacy method if tri-model not available
            reasoning = await self._legacy_multiple_choice_forecast(question, research)
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
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

        # Get budget status for model selection
        budget_remaining = 100.0
        if self.budget_manager:
            budget_status = self.budget_manager.get_budget_status()
            budget_remaining = 100.0 - budget_status.utilization_percentage

        # Try tri-model router with anti-slop prompts first
        if self.tri_model_router and self.anti_slop_prompts:
            try:
                # Create anti-slop numeric forecast prompt
                prompt = self.anti_slop_prompts.get_numeric_forecast_prompt(
                    question_text=question.question_text,
                    background_info=question.background_info,
                    resolution_criteria=getattr(question, 'resolution_criteria', ''),
                    fine_print=getattr(question, 'fine_print', ''),
                    research=research,
                    unit_of_measure=question.unit_of_measure,
                    lower_bound=question.lower_bound if not question.open_lower_bound else None,
                    upper_bound=question.upper_bound if not question.open_upper_bound else None,
                    model_tier="full"  # Use full model for final forecasting
                )

                # Route to optimal model
                reasoning = await self.tri_model_router.route_query(
                    task_type="forecast",
                    content=prompt,
                    complexity="high",
                    budget_remaining=budget_remaining
                )

                logger.info(f"Tri-model numeric forecast successful for question {question_id}")

            except Exception as e:
                logger.warning(f"Tri-model numeric forecast failed: {e}")
                # Fallback to legacy method
                reasoning = await self._legacy_numeric_forecast(question, research)
        else:
            # Fallback to legacy method if tri-model not available
            reasoning = await self._legacy_numeric_forecast(question, research)
        # Extract prediction from reasoning
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
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
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "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")

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
    # Log comprehensive report summary
    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore

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
                        logger.info(f"  {i}. {suggestion}")

        except Exception as e:
            logger.warning(f"Failed to log budget statistics: {e}")

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
                logger.warning(f"HIGH QUOTA USAGE: {stats['quota_usage_percentage']:.1f}% of AskNews quota used!")

            # Log fallback provider status
            fallback_status = template_bot.tournament_asknews.get_fallback_providers_status()
            logger.info("Fallback Providers Status:")
            for provider, available in fallback_status.items():
                status = "✓ Available" if available else "✗ Not configured"
                logger.info(f"  {provider}: {status}")

        except Exception as e:
            logger.warning(f"Failed to log tournament statistics: {e}")

    # Final status summary
    successful_forecasts = len([r for r in forecast_reports if not isinstance(r, Exception)])
    failed_forecasts = len([r for r in forecast_reports if isinstance(r, Exception)])

    logger.info("=== Final Summary ===")
    logger.info(f"Successful forecasts: {successful_forecasts}")
    logger.info(f"Failed forecasts: {failed_forecasts}")
    logger.info(f"Total questions processed: {len(forecast_reports)}")

    if failed_forecasts > 0:
        logger.warning(f"Some forecasts failed. Check logs above for details.")
        # Log first few exceptions for debugging
        exceptions = [r for r in forecast_reports if isinstance(r, Exception)][:3]
        for i, exc in enumerate(exceptions, 1):
            logger.error(f"Exception {i}: {type(exc).__name__}: {exc}")

    logger.info("Bot execution completed.")
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
