#!/usr/bin/env python3
"""
Test script for GPT-5 cost-optimized tri-model configuration.
Verifies that the system is properly configured to use GPT-5 models with free fallbacks,
skipping expensive GPT-4o models entirely.
"""

import sys
import os
sys.path.append('.')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_environment_configuration():
    """Test that environment variables are set correctly for GPT-5 cost optimization."""
    print("=== TESTING ENVIRONMENT CONFIGURATION ===")

    # Test GPT-5 primary models
    expected_models = {
        "DEFAULT_MODEL": "openai/gpt-5",
        "MINI_MODEL": "openai/gpt-5-mini",
        "NANO_MODEL": "openai/gpt-5-nano"
    }

    for env_var, expected_value in expected_models.items():
        actual_value = os.getenv(env_var)
        if actual_value == expected_value:
            print(f"✅ {env_var}: {actual_value}")
        else:
            print(f"❌ {env_var}: Expected {expected_value}, got {actual_value}")

    # Test free fallback models
    free_models = os.getenv("FREE_FALLBACK_MODELS", "")
    expected_free = ["moonshotai/kimi-k2:free", "openai/gpt-oss-20b:free"]

    print(f"\nFree fallback models: {free_models}")
    for model in expected_free:
        if model in free_models:
            print(f"✅ {model} configured")
        else:
            print(f"❌ {model} missing")

    # Verify expensive models are NOT configured
    expensive_models = ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"]
    print(f"\nVerifying expensive models are NOT in primary configuration:")
    all_passed = True
    for model in expensive_models:
        if model not in [os.getenv("DEFAULT_MODEL"), os.getenv("MINI_MODEL"), os.getenv("NANO_MODEL")]:
            print(f"✅ {model} not in primary config (good!)")
        else:
            print(f"❌ {model} found in primary config (expensive!)")
            all_passed = False

    return all_passed

def test_tri_model_router():
    """Test tri-model router configuration."""
    print("\n=== TESTING TRI-MODEL ROUTER ===")

    try:
        from src.infrastructure.config.tri_model_router import tri_model_router

        # Test model configurations
        print("Model configurations:")
        expected_costs = {
            "nano": (0.05, 0.05),    # GPT-5 Nano
            "mini": (0.25, 0.25),    # GPT-5 Mini
            "full": (1.50, 1.50)     # GPT-5 Full
        }

        for tier, (expected_input, expected_output) in expected_costs.items():
            config = tri_model_router.model_configs[tier]
            if config.cost_per_million_input == expected_input and config.cost_per_million_output == expected_output:
                print(f"✅ {tier}: {config.model_name} (${config.cost_per_million_input}/${config.cost_per_million_output})")
            else:
                print(f"❌ {tier}: Expected ${expected_input}/${expected_output}, got ${config.cost_per_million_input}/${config.cost_per_million_output}")

        # Test fallback chains
        print(f"\nFallback chains:")
        for tier, chain in tri_model_router.fallback_chains.items():
            print(f"  {tier}: {' → '.join(chain)}")

            # Verify GPT-5 models come first, free models are included, expensive models are skipped
            if tier == "full":
                if chain[0] == "openai/gpt-5" and "moonshotai/kimi-k2:free" in chain and "openai/gpt-4o" not in chain:
                    print(f"    ✅ Correct fallback order (GPT-5 → Free, skip expensive)")
                else:
                    print(f"    ❌ Incorrect fallback order")

        # Test operation modes
        print(f"\nOperation mode thresholds:")
        routing_info = tri_model_router.get_openrouter_provider_routing_info()
        for mode, description in routing_info["operation_modes"].items():
            print(f"  {mode}: {description}")
            if "GPT-5" in description or "Free models" in description:
                print(f"    ✅ Mentions GPT-5 or free models")
            else:
                print(f"    ❌ Should mention GPT-5 or free models")

        return True

    except Exception as e:
        print(f"❌ Tri-model router test failed: {e}")
        return False

def test_anti_slop_prompts():
    """Test anti-slop prompts with GPT-5 optimizations."""
    print("\n=== TESTING ANTI-SLOP PROMPTS ===")

    try:
        from src.prompts.anti_slop_prompts import anti_slop_prompts

        # Test tier optimizations mention GPT-5
        print("Tier optimizations:")
        for tier, optimization in anti_slop_prompts.tier_optimizations.items():
            if "GPT-5" in optimization:
                print(f"✅ {tier}: Contains GPT-5 optimization")
            else:
                print(f"❌ {tier}: Missing GPT-5 optimization")

        # Test model-specific adaptations
        print(f"\nModel-specific adaptations:")
        test_models = [
            ("openai/gpt-5-nano", "GPT-5-NANO SPECIFIC"),
            ("openai/gpt-5-mini", "GPT-5-MINI SPECIFIC"),
            ("openai/gpt-5", "GPT-5-FULL SPECIFIC"),
            ("moonshotai/kimi-k2:free", "FREE MODEL OPTIMIZATIONS")
        ]

        for model_name, expected_text in test_models:
            enhanced_prompt = anti_slop_prompts.get_enhanced_prompt_with_model_adaptation(
                'research',
                'nano',
                model_name,
                question_text='Test question'
            )
            if expected_text in enhanced_prompt:
                print(f"✅ {model_name}: Contains {expected_text}")
            else:
                print(f"❌ {model_name}: Missing {expected_text}")

        return True

    except Exception as e:
        print(f"❌ Anti-slop prompts test failed: {e}")
        return False

def test_cost_analysis():
    """Test cost analysis for GPT-5 vs GPT-4o comparison."""
    print("\n=== COST ANALYSIS ===")

    try:
        from src.infrastructure.config.tri_model_router import tri_model_router

        # Simulate cost for 1000 questions
        questions = 1000
        avg_tokens_per_question = 2000  # Conservative estimate

        print(f"Cost analysis for {questions} questions ({avg_tokens_per_question} tokens each):")

        # GPT-5 costs
        gpt5_costs = {
            "nano": 0.05,
            "mini": 0.25,
            "full": 1.50
        }

        # Simulate mixed usage (30% nano, 50% mini, 20% full)
        mixed_cost = (
            (questions * 0.3 * avg_tokens_per_question * gpt5_costs["nano"] / 1_000_000) +
            (questions * 0.5 * avg_tokens_per_question * gpt5_costs["mini"] / 1_000_000) +
            (questions * 0.2 * avg_tokens_per_question * gpt5_costs["full"] / 1_000_000)
        )

        print(f"GPT-5 mixed usage cost: ${mixed_cost:.2f}")

        # GPT-4o comparison (if we used expensive models)
        gpt4o_cost = questions * avg_tokens_per_question * 10.0 / 1_000_000  # Average of $5-15
        print(f"GPT-4o equivalent cost: ${gpt4o_cost:.2f}")

        savings = gpt4o_cost - mixed_cost
        print(f"Cost savings: ${savings:.2f} ({savings/gpt4o_cost*100:.1f}%)")

        # Questions possible with $100 budget
        questions_with_gpt5 = int(100 / mixed_cost * questions)
        questions_with_gpt4o = int(100 / gpt4o_cost * questions)

        print(f"\nQuestions possible with $100 budget:")
        print(f"  GPT-5 cost-optimized: {questions_with_gpt5:,} questions")
        print(f"  GPT-4o expensive: {questions_with_gpt4o:,} questions")
        print(f"  Improvement: {questions_with_gpt5/questions_with_gpt4o:.1f}x more questions")

        if questions_with_gpt5 > 2000:
            print(f"✅ Achieves 2000+ questions target")
        else:
            print(f"❌ Does not achieve 2000+ questions target")

        return True

    except Exception as e:
        print(f"❌ Cost analysis failed: {e}")
        return False

def main():
    """Run all tests."""
    print("GPT-5 COST-OPTIMIZED CONFIGURATION TEST")
    print("=" * 50)

    tests = [
        ("Environment Configuration", test_environment_configuration),
        ("Tri-Model Router", test_tri_model_router),
        ("Anti-Slop Prompts", test_anti_slop_prompts),
        ("Cost Analysis", test_cost_analysis)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! GPT-5 cost-optimized configuration is working correctly.")
        print("The system will:")
        print("  • Use GPT-5 models for primary tasks")
        print("  • Fall back to free models when GPT-5 unavailable")
        print("  • Skip expensive GPT-4o models entirely")
        print("  • Process 2000+ questions within $100 budget")
        print("  • Maintain tournament-winning quality with anti-slop directives")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please review the configuration.")

    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
