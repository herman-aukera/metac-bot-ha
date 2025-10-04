#!/usr/bin/env python3
"""Test script for date question forecasting implementation."""

import sys
import asyncio
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.domain.services.date_question_forecaster import DateQuestionForecaster


def test_date_forecaster():
    """Test the date question forecaster with a sample question."""
    print("🧪 Testing Date Question Forecaster")
    print("=" * 50)

    forecaster = DateQuestionForecaster()

    # Test with a sample date question similar to the failing ones
    question_text = "When will the U.S. Solicitor General file a CVSG brief in Monsanto Company v. John L. Durnell?"
    background_info = "The U.S. Supreme Court case involves administrative law and regulatory procedures."
    resolution_criteria = "This question resolves when the U.S. Solicitor General files a brief."

    # Use realistic date bounds (e.g., next 2 years)
    lower_bound = datetime(2025, 10, 1)
    upper_bound = datetime(2027, 10, 1)

    research_data = "Recent legal filings suggest regulatory review processes typically take 6-18 months."

    print(f"Question: {question_text}")
    print(f"Date range: {lower_bound.date()} to {upper_bound.date()}")
    print()

    try:
        # Generate forecast
        result = forecaster.forecast_date_question(
            question_text=question_text,
            background_info=background_info,
            resolution_criteria=resolution_criteria,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            research_data=research_data,
        )

        print("✅ Forecast generated successfully!")
        print()
        print("📊 Results:")
        print(f"  Confidence: {result.confidence:.1%}")
        print()
        print("📅 Date Percentiles:")
        for percentile in sorted(result.percentiles.keys()):
            date_val = result.percentiles[percentile]
            print(f"  {percentile*100:2.0f}th percentile: {date_val.strftime('%Y-%m-%d')}")

        print()
        print("💭 Reasoning:")
        print(result.reasoning)

        print()
        print("🔗 Metaculus Format:")
        formatted = forecaster.format_percentiles_for_metaculus(result.percentiles)
        for percentile, date_str in formatted:
            print(f"  {percentile}: {date_str}")

        return True

    except Exception as e:
        print(f"❌ Forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_date_report_models():
    """Test the date report data models."""
    print("\n" + "=" * 50)
    print("🧪 Testing Date Report Models")
    print("=" * 50)

    try:
        from src.domain.models.date_report import DateDistribution, DatePercentile, DateForecastResult

        # Test creating date percentiles
        lower_bound = datetime(2025, 1, 1)
        upper_bound = datetime(2025, 12, 31)

        # Create a simple distribution
        distribution = DateDistribution.uniform_distribution(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            num_percentiles=5
        )

        print("✅ DateDistribution created successfully!")
        print(f"  Median date: {distribution.median_date.strftime('%Y-%m-%d')}")

        ci_low, ci_high = distribution.confidence_interval_90
        print(f"  90% CI: {ci_low.strftime('%Y-%m-%d')} to {ci_high.strftime('%Y-%m-%d')}")

        # Test Metaculus format conversion
        metaculus_format = distribution.to_metaculus_format()
        print(f"  Metaculus format: {list(metaculus_format.keys())}")

        # Test forecast result
        forecast_result = DateForecastResult(
            distribution=distribution,
            reasoning="Test reasoning",
            confidence=0.7
        )

        print("✅ DateForecastResult created successfully!")
        print(f"  Summary: {forecast_result.prediction_summary}")

        return True

    except Exception as e:
        print(f"❌ Date models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Date Question Implementation Tests\n")

    # Test the forecaster
    forecaster_test = test_date_forecaster()

    # Test the data models
    models_test = test_date_report_models()

    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    print(f"Date Forecaster: {'✅ PASS' if forecaster_test else '❌ FAIL'}")
    print(f"Date Models:     {'✅ PASS' if models_test else '❌ FAIL'}")

    if forecaster_test and models_test:
        print("\n🎉 All date question tests passed!")
        print("Ready to handle date questions in the bot.")
    else:
        print("\n⚠️  Some tests failed - check implementation.")
        sys.exit(1)
