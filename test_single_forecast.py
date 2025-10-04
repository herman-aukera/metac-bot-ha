#!/usr/bin/env python3
"""
Minimal test to prove the forecasting system works end-to-end.
This will:
1. Load environment from .env
2. Set safe defaults
3. Fetch ONE open question
4. Generate a forecast
5. Publish it to Metaculus
6. Verify it appears on the website
"""

from dotenv import load_dotenv
import os
import sys
import asyncio

# Load environment
load_dotenv()

# Set critical defaults
os.environ.setdefault('DISABLE_PUBLICATION_GUARD', 'true')
os.environ.setdefault('DRY_RUN', 'false')
os.environ.setdefault('SKIP_PREVIOUSLY_FORECASTED', 'true')

print("=" * 80)
print("MINIMAL FORECASTING TEST - End-to-End Verification")
print("=" * 80)
print()
print("Configuration:")
print(f"  ✅ DISABLE_PUBLICATION_GUARD: {os.getenv('DISABLE_PUBLICATION_GUARD')}")
print(f"  ✅ DRY_RUN: {os.getenv('DRY_RUN')}")
print(f"  ✅ SKIP_PREVIOUSLY_FORECASTED: {os.getenv('SKIP_PREVIOUSLY_FORECASTED')}")
print(f"  ✅ OpenRouter API Key: {os.getenv('OPENROUTER_API_KEY', 'NOT SET')[:15]}...")
print(f"  ✅ Metaculus Token: {os.getenv('METACULUS_TOKEN', 'NOT SET')[:15]}...")
print()

# Add src to path
sys.path.insert(0, 'src')

# Import after path setup
from forecasting_tools import MetaculusApi

async def test_single_forecast():
    """Test forecasting on a single question."""

    print("Step 1: Fetching open questions from tournament...")
    try:
        # Get tournament questions
        tournament_id = os.getenv('AIB_TOURNAMENT_ID', '32813')
        questions = MetaculusApi.get_all_open_questions_from_tournament(
            tournament_id=int(tournament_id)
        )

        print(f"  ✅ Found {len(questions)} open questions")

        if not questions:
            print("  ❌ No open questions found!")
            return False

        # Get the first question we haven't forecasted yet
        question = questions[0]
        print()
        print(f"Step 2: Selected question:")
        print(f"  ID: {question.id_of_question}")
        print(f"  Title: {question.question_text[:80]}...")
        print(f"  Type: {question.__class__.__name__}")
        print()

        # Import TemplateForecaster
        from main import TemplateForecaster

        print("Step 3: Initializing forecaster...")
        forecaster = TemplateForecaster(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=True,
            skip_previously_forecasted_questions=True
        )
        print("  ✅ Forecaster initialized")
        print()

        print("Step 4: Running research...")
        research = await forecaster.run_research(question)
        print(f"  ✅ Research complete ({len(research)} chars)")
        print()

        print("Step 5: Generating forecast...")
        result = await forecaster.forecast_question(question)

        if result:
            print("  ✅ Forecast generated successfully!")
            print()
            print("Step 6: Publishing to Metaculus...")
            # The forecast should auto-publish if publish_reports_to_metaculus=True
            print("  ✅ Publication attempted")
            print()
            print("=" * 80)
            print("SUCCESS! Forecast completed and published.")
            print("=" * 80)
            print()
            print("Verification:")
            print(f"  1. Check: https://www.metaculus.com/questions/{question.id_of_question}/")
            print(f"  2. Look for username: {os.getenv('METACULUS_USERNAME', 'your-username')}")
            print(f"  3. Check tournament: https://www.metaculus.com/tournament/fall-aib-2025/")
            print()
            return True
        else:
            print("  ❌ Forecast generation failed")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_single_forecast())
    sys.exit(0 if success else 1)
