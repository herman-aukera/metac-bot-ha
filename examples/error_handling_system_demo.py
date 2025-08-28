"""
Demonstration of the comprehensive error handling and recovery system
for OpenRouter tri-model optimization.

This example shows how the error handling system works with various
error scenarios and recovery strategies.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.infrastructure.reliability.error_classification import (
    ErrorContext, ModelError, BudgetError, APIError, QualityError
)
from src.infrastructure.reliability.comprehensive_error_recovery import (
    ComprehensiveErrorRecoveryManager, RecoveryConfiguration
)
from src.infrastructure.config.tri_model_router import OpenRouterTriModelRouter
from src.infrastructure.config.budget_aware_operation_manager import BudgetAwareOperationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorHandlingSystemDemo:
    """
    Demonstration of the comprehensive error handling and recovery system.
    """

    def __init__(self):
        """Initialize the demo with mock components."""
        self.tri_model_router = None
        self.budget_manager = None
        self.recovery_manager = None

    async def initialize_system(self):
        """Initialize the error handling system components."""
        logger.info("Initializing error handling system components...")

        try:
            # Initialize tri-model router (if API keys available)
            if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENROUTER_API_KEY", "").startswith("dummy_"):
                logger.info("Initializing OpenRouter tri-model router...")
                self.tri_model_router = OpenRouterTriModelRouter()
            else:
                logger.warning("OpenRouter API key not available - using mock router")
                self.tri_model_router = self._create_mock_router()

            # Initialize budget manager
            logger.info("Initializing budget manager...")
            self.budget_manager = self._create_mock_budget_manager()

            # Initialize comprehensive error recovery manager
            config = RecoveryConfiguration(
                max_recovery_attempts=3,
                max_recovery_time=120.0,
                enable_circuit_breakers=True,
                enable_emergency_mode=True,
                enable_quality_recovery=True,
                budget_threshold_for_emergency=5.0
            )

            self.recovery_manager = ComprehensiveErrorRecoveryManager(
                self.tri_model_router,
                self.budget_manager,
                config
            )

            logger.info("Error handling system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize error handling system: {e}")
            raise

    def _create_mock_router(self):
        """Create a mock tri-model router for demonstration."""
        class MockRouter:
            def __init__(self):
                self.models = {
                    "full": "openai/gpt-5",
                    "mini": "openai/gpt-5-mini",
                    "nano": "openai/gpt-5-nano"
                }
                self.model_configs = {
                    "full": type('Config', (), {"model_name": "openai/gpt-5"})(),
                    "mini": type('Config', (), {"model_name": "openai/gpt-5-mini"})(),
                    "nano": type('Config', (), {"model_name": "openai/gpt-5-nano"})()
                }

            async def detect_model_availability(self):
                return {
                    "openai/gpt-5": True,
                    "openai/gpt-5-mini": True,
                    "openai/gpt-5-nano": True,
                    "openai/gpt-oss-20b:free": True,
                    "moonshotai/kimi-k2:free": True
                }

            async def check_model_health(self, tier):
                from src.infrastructure.config.tri_model_router import ModelStatus
                return ModelStatus(
                    tier=tier,
                    model_name=self.models.get(tier, "unknown"),
                    is_available=True,
                    last_check=0
                )

        return MockRouter()

    def _create_mock_budget_manager(self):
        """Create a mock budget manager for demonstration."""
        class MockBudgetManager:
            def __init__(self):
                self.budget_remaining = 50.0

            async def get_budget_status(self):
                return {
                    "remaining_percentage": self.budget_remaining,
                    "total_budget": 100.0,
                    "used_budget": 100.0 - self.budget_remaining
                }

            def set_budget_remaining(self, percentage):
                self.budget_remaining = percentage

        return MockBudgetManager()

    async def demonstrate_error_classification(self):
        """Demonstrate error classification capabilities."""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING ERROR CLASSIFICATION")
        logger.info("="*60)

        test_context = ErrorContext(
            task_type="forecast",
            model_tier="mini",
            operation_mode="normal",
            budget_remaining=50.0,
            attempt_number=1,
            model_name="openai/gpt-5-mini",
            provider="openrouter"
        )

        # Test different error types
        test_errors = [
            ("Rate Limit Error", Exception("Rate limit exceeded for requests")),
            ("Model Unavailable", Exception("Model not found or unavailable")),
            ("Context Too Long", Exception("Context length exceeded maximum")),
            ("Network Timeout", Exception("Request timed out after 30 seconds")),
            ("Authentication Failed", Exception("Invalid API key or unauthorized")),
            ("Quality Validation", Exception("Response failed quality validation")),
            ("Missing Citations", Exception("Required citations not found")),
            ("Budget Warning", Exception("Budget utilization approaching limit"))
        ]

        classifier = self.recovery_manager.error_classifier

        for error_name, error in test_errors:
            logger.info(f"\nClassifying: {error_name}")
            logger.info(f"Error message: {error}")

            classification = classifier.classify_error(error, test_context)

            logger.info(f"  Category: {classification.category.value}")
            logger.info(f"  Severity: {classification.severity.value}")
            logger.info(f"  Error Code: {classification.error_code}")
            logger.info(f"  Recovery Strategies: {[s.value for s in classification.recovery_strategies]}")
            logger.info(f"  Max Retries: {classification.max_retries}")
            logger.info(f"  Retry Delay: {classification.retry_delay}s")

        # Show error statistics
        stats = classifier.get_error_statistics()
        logger.info(f"\nError Statistics:")
        logger.info(f"  Total Errors: {stats['total_errors']}")
        logger.info(f"  Error Categories: {stats['error_categories']}")

    async def demonstrate_model_fallback(self):
        """Demonstrate model tier fallback strategies."""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING MODEL TIER FALLBACK")
        logger.info("="*60)

        fallback_manager = self.recovery_manager.fallback_orchestrator.model_fallback_manager

        # Test fallback scenarios
        scenarios = [
            ("Normal Budget - Full Tier", "full", 70.0),
            ("Low Budget - Full Tier", "full", 15.0),
            ("Critical Budget - Mini Tier", "mini", 5.0),
            ("Emergency - Nano Tier", "nano", 2.0)
        ]

        for scenario_name, tier, budget_remaining in scenarios:
            logger.info(f"\nScenario: {scenario_name}")
            logger.info(f"Original Tier: {tier}, Budget Remaining: {budget_remaining}%")

            context = ErrorContext(
                task_type="forecast",
                model_tier=tier,
                operation_mode="normal",
                budget_remaining=budget_remaining,
                attempt_number=1
            )

            try:
                result = await fallback_manager.execute_fallback(tier, context, budget_remaining)

                if result.success:
                    logger.info(f"  ✓ Fallback Successful")
                    logger.info(f"  Fallback Model: {result.fallback_used.name}")
                    logger.info(f"  Performance Impact: {result.performance_impact:.2f}")
                    logger.info(f"  Cost Impact: {result.cost_impact:.2f}")
                    logger.info(f"  Recovery Time: {result.recovery_time:.2f}s")
                else:
                    logger.info(f"  ✗ Fallback Failed: {result.message}")

            except Exception as e:
                logger.error(f"  ✗ Fallback Error: {e}")

        # Show fallback statistics
        stats = fallback_manager.get_fallback_statistics()
        logger.info(f"\nFallback Statistics:")
        logger.info(f"  Total Fallbacks: {stats['total_fallbacks']}")
        logger.info(f"  Recent Fallbacks (24h): {stats.get('recent_fallbacks_24h', 0)}")

    async def demonstrate_provider_fallback(self):
        """Demonstrate cross-provider fallback strategies."""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING PROVIDER FALLBACK")
        logger.info("="*60)

        provider_manager = self.recovery_manager.fallback_orchestrator.provider_fallback_manager

        # Test provider fallback scenarios
        scenarios = [
            ("OpenRouter to Metaculus", "openrouter"),
            ("Metaculus to Free Models", "metaculus_proxy"),
            ("Unknown Provider", "unknown_provider")
        ]

        for scenario_name, original_provider in scenarios:
            logger.info(f"\nScenario: {scenario_name}")
            logger.info(f"Original Provider: {original_provider}")

            context = ErrorContext(
                task_type="forecast",
                model_tier="mini",
                operation_mode="normal",
                budget_remaining=50.0,
                attempt_number=1,
                provider=original_provider
            )

            try:
                result = await provider_manager.execute_provider_fallback(original_provider, context)

                if result.success:
                    logger.info(f"  ✓ Provider Fallback Successful")
                    logger.info(f"  Fallback Provider: {result.fallback_used.name}")
                    logger.info(f"  Recovery Time: {result.recovery_time:.2f}s")
                else:
                    logger.info(f"  ✗ Provider Fallback Failed: {result.message}")

            except Exception as e:
                logger.error(f"  ✗ Provider Fallback Error: {e}")

    async def demonstrate_emergency_mode(self):
        """Demonstrate emergency mode activation."""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING EMERGENCY MODE")
        logger.info("="*60)

        emergency_manager = self.recovery_manager.fallback_orchestrator.emergency_manager

        # Test emergency mode scenarios
        scenarios = [
            ("Budget Exhaustion", BudgetError("Budget exhausted", 0.0, 10.0)),
            ("Critical System Failure", Exception("Multiple system failures detected"))
        ]

        for scenario_name, trigger_error in scenarios:
            logger.info(f"\nScenario: {scenario_name}")
            logger.info(f"Trigger Error: {trigger_error}")

            context = ErrorContext(
                task_type="forecast",
                model_tier="mini",
                operation_mode="critical",
                budget_remaining=2.0,
                attempt_number=1
            )

            try:
                # Activate emergency mode
                recovery_action = await emergency_manager.activate_emergency_mode(trigger_error, context)

                logger.info(f"  ✓ Emergency Mode Activated")
                logger.info(f"  Strategy: {recovery_action.strategy.value}")
                logger.info(f"  Free Models Only: {recovery_action.parameters.get('free_models_only', False)}")
                logger.info(f"  Minimal Functionality: {recovery_action.parameters.get('minimal_functionality', False)}")

                # Show emergency status
                status = emergency_manager.get_emergency_status()
                logger.info(f"  Emergency Active: {status['active']}")
                if status['active']:
                    logger.info(f"  Duration: {status['duration_seconds']:.1f}s")

                # Test deactivation (with mock conditions)
                # Note: In real scenario, this would check actual system conditions
                logger.info(f"  Attempting to deactivate emergency mode...")
                deactivated = await emergency_manager.deactivate_emergency_mode()
                logger.info(f"  Deactivation {'successful' if deactivated else 'failed'}")

            except Exception as e:
                logger.error(f"  ✗ Emergency Mode Error: {e}")

    async def demonstrate_comprehensive_recovery(self):
        """Demonstrate comprehensive error recovery scenarios."""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING COMPREHENSIVE ERROR RECOVERY")
        logger.info("="*60)

        # Test comprehensive recovery scenarios
        scenarios = [
            ("Model Error Recovery", ModelError("GPT-5 model failed", "openai/gpt-5", "full")),
            ("API Error Recovery", APIError("OpenRouter API error", "openrouter", 500)),
            ("Budget Error Recovery", BudgetError("Budget limit reached", 1.0, 5.0)),
            ("Quality Error Recovery", QualityError("Quality validation failed", ["missing_citations"], 0.3))
        ]

        for scenario_name, error in scenarios:
            logger.info(f"\nScenario: {scenario_name}")
            logger.info(f"Error Type: {type(error).__name__}")
            logger.info(f"Error Message: {error}")

            # Adjust context based on error type
            if isinstance(error, BudgetError):
                budget_remaining = 2.0
                operation_mode = "critical"
            else:
                budget_remaining = 50.0
                operation_mode = "normal"

            context = ErrorContext(
                task_type="forecast",
                model_tier="mini",
                operation_mode=operation_mode,
                budget_remaining=budget_remaining,
                attempt_number=1,
                model_name="openai/gpt-5-mini",
                provider="openrouter"
            )

            try:
                # Execute comprehensive recovery
                result = await self.recovery_manager.recover_from_error(error, context)

                logger.info(f"  Recovery Result:")
                logger.info(f"    Success: {result.success}")
                logger.info(f"    Strategy: {result.recovery_strategy.value}")
                logger.info(f"    Recovery Time: {result.recovery_time:.2f}s")
                logger.info(f"    Attempts Made: {result.attempts_made}")
                logger.info(f"    Performance Impact: {result.performance_impact:.2f}")
                logger.info(f"    Cost Impact: {result.cost_impact:.2f}")
                logger.info(f"    Message: {result.message}")

                if not result.success and result.final_error:
                    logger.info(f"    Final Error: {result.final_error}")

            except Exception as e:
                logger.error(f"  ✗ Recovery Error: {e}")

    async def demonstrate_system_monitoring(self):
        """Demonstrate system monitoring and health assessment."""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING SYSTEM MONITORING")
        logger.info("="*60)

        # Get recovery system status
        status = self.recovery_manager.get_recovery_status()

        logger.info("Recovery System Status:")
        logger.info(f"  Active Recoveries: {status['active_recoveries']}")
        logger.info(f"  Emergency Mode Active: {status['emergency_mode_active']}")
        logger.info(f"  Recent Recovery Count: {status['recent_recovery_count']}")

        # Show recovery statistics
        stats = status['recovery_statistics']
        logger.info(f"\nRecovery Statistics:")
        logger.info(f"  Total Recoveries: {stats['total_recoveries']}")
        logger.info(f"  Successful Recoveries: {stats['successful_recoveries']}")
        logger.info(f"  Failed Recoveries: {stats['failed_recoveries']}")
        logger.info(f"  Average Recovery Time: {stats['average_recovery_time']:.2f}s")

        if stats['strategy_effectiveness']:
            logger.info(f"\nStrategy Effectiveness:")
            for strategy, effectiveness in stats['strategy_effectiveness'].items():
                logger.info(f"  {strategy}:")
                logger.info(f"    Attempts: {effectiveness['attempts']}")
                logger.info(f"    Success Rate: {effectiveness['success_rate']:.2f}")
                logger.info(f"    Avg Recovery Time: {effectiveness['avg_recovery_time']:.2f}s")

        # Show system health
        health = status['system_health']
        logger.info(f"\nSystem Health:")
        logger.info(f"  Status: {health['status']}")
        logger.info(f"  Health Score: {health['score']:.2f}")
        if health['issues']:
            logger.info(f"  Issues: {health['issues']}")

        # Test recovery system
        logger.info(f"\nTesting Recovery System...")
        test_results = await self.recovery_manager.test_recovery_system()

        logger.info(f"Test Results:")
        for test_name, result in test_results['test_results'].items():
            if result['success']:
                logger.info(f"  ✓ {test_name}: Passed")
            else:
                logger.info(f"  ✗ {test_name}: Failed - {result.get('error', 'Unknown error')}")

    async def run_comprehensive_demo(self):
        """Run the complete error handling system demonstration."""
        logger.info("Starting Comprehensive Error Handling System Demo")
        logger.info("="*80)

        try:
            # Initialize system
            await self.initialize_system()

            # Run demonstrations
            await self.demonstrate_error_classification()
            await self.demonstrate_model_fallback()
            await self.demonstrate_provider_fallback()
            await self.demonstrate_emergency_mode()
            await self.demonstrate_comprehensive_recovery()
            await self.demonstrate_system_monitoring()

            logger.info("\n" + "="*80)
            logger.info("Error Handling System Demo Completed Successfully")
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise


async def main():
    """Main function to run the error handling system demo."""
    demo = ErrorHandlingSystemDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
