"""
Demo of ForecastingStageService with GPT-5 and calibration.
Demonstrates advanced forecasting capabilities with uncertainty quantification.
"""

import asyncio
import logging
from src.domain.services.forecasting_stage_service import ForecastingStageService
from src.infrastructure.config.tri_model_router import OpenRouterTriModelRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_binary_forecast():
    """Demonstrate binary forecasting with calibration."""

    print("\n" + "="*60)
    print("BINARY FORECAST DEMO - GPT-5 with Calibration")
    print("="*60)

    # Initialize services
    router = OpenRouterTriModelRouter()
    forecasting_service = ForecastingStageService(router)

    # Sample binary question
    question = "Will AI achieve AGI (Artificial General Intelligence) by 2030?"

    research_data = """
    Recent research findings:
    - Current AI models show rapid capability improvements
    - Major tech companies investing billions in AGI research
    - Expert surveys show mixed predictions (20-80% by 2030)
    - Technical challenges remain in reasoning and generalization
    - Regulatory frameworks still developing
    - Compute requirements may be limiting factor
    """

    context = {
        "background_info": "AGI defined as AI matching human performance across all cognitive tasks",
        "resolution_criteria": "Resolves YES if consensus among AI researchers that AGI achieved",
        "fine_print": "Must be general intelligence, not narrow AI capabilities"
    }

    try:
        print(f"Question: {question}")
        print(f"Research Data: {research_data[:100]}...")
        print("\nGenerating GPT-5 forecast with calibration...")

        result = await forecasting_service.generate_forecast(
            question=question,
            question_type="binary",
            research_data=research_data,
            context=context
        )

        print(f"\n📊 FORECAST RESULTS:")
        print(f"Prediction: {result.prediction:.1%}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Calibration Score: {result.calibration_score:.2f}")
        print(f"Overconfidence Detected: {result.overconfidence_detected}")
        print(f"Quality Validation: {'✅ PASSED' if result.quality_validation_passed else '❌ FAILED'}")
        print(f"Tournament Compliant: {'✅ YES' if result.tournament_compliant else '❌ NO'}")

        if result.uncertainty_bounds:
            print(f"\n🎯 UNCERTAINTY BOUNDS:")
            print(f"Lower Bound: {result.uncertainty_bounds.get('lower_bound', 0):.1%}")
            print(f"Upper Bound: {result.uncertainty_bounds.get('upper_bound', 1):.1%}")

        print(f"\n💰 COST: ${result.cost_estimate:.4f}")
        print(f"⏱️ TIME: {result.execution_time:.2f}s")
        print(f"🤖 MODEL: {result.model_used}")

        print(f"\n📝 REASONING EXCERPT:")
        reasoning_preview = result.reasoning[:300] + "..." if len(result.reasoning) > 300 else result.reasoning
        print(reasoning_preview)

    except Exception as e:
        print(f"❌ Demo failed: {e}")


async def demo_multiple_choice_forecast():
    """Demonstrate multiple choice forecasting."""

    print("\n" + "="*60)
    print("MULTIPLE CHOICE FORECAST DEMO")
    print("="*60)

    router = OpenRouterTriModelRouter()
    forecasting_service = ForecastingStageService(router)

    question = "Which company will achieve the highest market cap by end of 2025?"
    options = ["Apple", "Microsoft", "Google/Alphabet", "Amazon", "Other"]

    research_data = """
    Current market analysis:
    - Apple: Strong iPhone sales, services growth, $3T market cap
    - Microsoft: Cloud dominance, AI integration, enterprise focus
    - Google: Search monopoly, AI leadership, regulatory challenges
    - Amazon: E-commerce recovery, AWS growth, cost optimization
    - Market volatility and economic uncertainty affecting all
    """

    context = {
        "options": options,
        "background_info": "Based on publicly traded market capitalization",
        "resolution_criteria": "Highest market cap on last trading day of 2025",
        "fine_print": "Adjusted for stock splits and other corporate actions"
    }

    try:
        print(f"Question: {question}")
        print(f"Options: {', '.join(options)}")
        print("\nGenerating forecast...")

        result = await forecasting_service.generate_forecast(
            question=question,
            question_type="multiple_choice",
            research_data=research_data,
            context=context
        )

        print(f"\n📊 PROBABILITY DISTRIBUTION:")
        if isinstance(result.prediction, dict):
            for option, probability in result.prediction.items():
                print(f"{option}: {probability:.1%}")

        print(f"\nCalibration Score: {result.calibration_score:.2f}")
        print(f"Quality Validation: {'✅ PASSED' if result.quality_validation_passed else '❌ FAILED'}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


async def demo_numeric_forecast():
    """Demonstrate numeric forecasting with percentiles."""

    print("\n" + "="*60)
    print("NUMERIC FORECAST DEMO")
    print("="*60)

    router = OpenRouterTriModelRouter()
    forecasting_service = ForecastingStageService(router)

    question = "What will be the global temperature anomaly (°C above 1951-1980 average) in 2025?"

    research_data = """
    Climate data analysis:
    - 2023 temperature anomaly: +1.17°C (record high)
    - El Niño/La Niña cycles affect annual temperatures
    - Long-term warming trend continues at ~0.18°C per decade
    - 2024 showing continued high temperatures
    - Climate models predict continued warming
    - Natural variability can cause ±0.2°C annual variation
    """

    context = {
        "background_info": "Global mean surface temperature anomaly relative to 1951-1980 baseline",
        "resolution_criteria": "NASA GISS global temperature data for 2025",
        "fine_print": "Annual average, not monthly peaks",
        "unit_of_measure": "°C",
        "lower_bound": 0.5,
        "upper_bound": 2.0
    }

    try:
        print(f"Question: {question}")
        print("\nGenerating numeric forecast...")

        result = await forecasting_service.generate_forecast(
            question=question,
            question_type="numeric",
            research_data=research_data,
            context=context
        )

        print(f"\n📊 PERCENTILE ESTIMATES:")
        if isinstance(result.prediction, dict):
            for percentile in sorted(result.prediction.keys()):
                value = result.prediction[percentile]
                print(f"P{percentile}: {value:.2f}°C")

        print(f"\nCalibration Score: {result.calibration_score:.2f}")
        print(f"Quality Validation: {'✅ PASSED' if result.quality_validation_passed else '❌ FAILED'}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


async def main():
    """Run all forecasting demos."""

    print("🚀 FORECASTING STAGE SERVICE DEMO")
    print("GPT-5 with Advanced Calibration and Uncertainty Quantification")

    try:
        await demo_binary_forecast()
        await demo_multiple_choice_forecast()
        await demo_numeric_forecast()

        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Demo suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
