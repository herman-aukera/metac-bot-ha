"""
OpenRouter Startup Configuration Validator.
Validates OpenRouter configuration and provides setup guidance.
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .tri_model_router import OpenRouterTriModelRouter

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    configuration_status: Dict[str, Any]


class OpenRouterStartupValidator:
    """Validates OpenRouter configuration and provides setup guidance."""

    def __init__(self):
        self.required_env_vars = ["OPENROUTER_API_KEY"]

        self.recommended_env_vars = [
            "OPENROUTER_BASE_URL",
            "OPENROUTER_HTTP_REFERER",
            "OPENROUTER_APP_TITLE",
            "DEFAULT_MODEL",
            "MINI_MODEL",
            "NANO_MODEL",
        ]

        self.default_values = {
            "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
            "DEFAULT_MODEL": "openai/gpt-5",
            "MINI_MODEL": "openai/gpt-5-mini",
            "NANO_MODEL": "openai/gpt-5-nano",
        }

    async def validate_configuration(self) -> ValidationResult:
        """Perform comprehensive OpenRouter configuration validation."""
        logger.info("Starting OpenRouter configuration validation...")

        errors = []
        warnings = []
        recommendations = []

        # Check required environment variables
        missing_required = []
        for var in self.required_env_vars:
            value = os.getenv(var)
            if not value:
                missing_required.append(var)
            elif value.startswith("dummy_"):
                warnings.append(f"{var} is set to a dummy value")
                recommendations.append(
                    f"Replace dummy {var} with real OpenRouter API key"
                )

        if missing_required:
            errors.extend(
                [
                    f"Missing required environment variable: {var}"
                    for var in missing_required
                ]
            )

        # Check recommended environment variables
        missing_recommended = []
        for var in self.recommended_env_vars:
            if not os.getenv(var):
                missing_recommended.append(var)

        if missing_recommended:
            warnings.extend(
                [
                    f"Recommended environment variable not set: {var}"
                    for var in missing_recommended
                ]
            )
            recommendations.append(
                "Set recommended environment variables for optimal configuration"
            )

        # Validate base URL
        base_url = os.getenv(
            "OPENROUTER_BASE_URL", self.default_values["OPENROUTER_BASE_URL"]
        )
        if base_url != "https://openrouter.ai/api/v1":
            warnings.append(f"Non-standard OpenRouter base URL: {base_url}")
            recommendations.append(
                "Use standard OpenRouter base URL: https://openrouter.ai/api/v1"
            )

        # Check model configurations
        model_config_status = self._validate_model_configurations()
        if model_config_status["errors"]:
            errors.extend(model_config_status["errors"])
        if model_config_status["warnings"]:
            warnings.extend(model_config_status["warnings"])
        if model_config_status["recommendations"]:
            recommendations.extend(model_config_status["recommendations"])

        # Test OpenRouter connectivity if API key is available
        connectivity_status = {"available": False, "tested_models": {}}
        if not missing_required and not any(
            os.getenv(var, "").startswith("dummy_") for var in self.required_env_vars
        ):
            try:
                connectivity_status = await self._test_openrouter_connectivity()
                if not connectivity_status["available"]:
                    errors.append("OpenRouter API connectivity test failed")
                    recommendations.append(
                        "Check OpenRouter API key validity and network connectivity"
                    )
            except Exception as e:
                warnings.append(f"OpenRouter connectivity test failed: {e}")
                recommendations.append(
                    "Verify OpenRouter API key and network connectivity"
                )

        # Determine overall validation status
        is_valid = len(errors) == 0

        configuration_status = {
            "environment_variables": self._get_env_var_status(),
            "model_configurations": model_config_status,
            "connectivity": connectivity_status,
            "validation_timestamp": asyncio.get_event_loop().time(),
        }

        logger.info(
            f"Configuration validation complete: {'VALID' if is_valid else 'INVALID'} "
            f"({len(errors)} errors, {len(warnings)} warnings)"
        )

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            configuration_status=configuration_status,
        )

    def _validate_model_configurations(self) -> Dict[str, Any]:
        """Validate model configuration settings."""
        errors = []
        warnings = []
        recommendations = []

        # Check model name configurations
        model_vars = ["DEFAULT_MODEL", "MINI_MODEL", "NANO_MODEL"]
        expected_models = {
            "DEFAULT_MODEL": "openai/gpt-5",
            "MINI_MODEL": "openai/gpt-5-mini",
            "NANO_MODEL": "openai/gpt-5-nano",
        }

        for var in model_vars:
            value = os.getenv(var)
            expected = expected_models[var]

            if not value:
                warnings.append(
                    f"Model configuration {var} not set, using default: {expected}"
                )
            elif value != expected:
                warnings.append(
                    f"Non-standard model for {var}: {value} (expected: {expected})"
                )
                recommendations.append(
                    f"Consider using standard model {expected} for {var}"
                )

        # Check operation mode thresholds
        threshold_vars = [
            "NORMAL_MODE_THRESHOLD",
            "CONSERVATIVE_MODE_THRESHOLD",
            "EMERGENCY_MODE_THRESHOLD",
            "CRITICAL_MODE_THRESHOLD",
        ]

        for var in threshold_vars:
            value = os.getenv(var)
            if value:
                try:
                    threshold = float(value)
                    if not 0 <= threshold <= 100:
                        warnings.append(
                            f"Invalid threshold value for {var}: {threshold} (should be 0-100)"
                        )
                except ValueError:
                    warnings.append(
                        f"Invalid threshold format for {var}: {value} (should be numeric)"
                    )

        return {
            "errors": errors,
            "warnings": warnings,
            "recommendations": recommendations,
            "model_settings": {
                var: os.getenv(var, expected_models.get(var, "not_set"))
                for var in model_vars
            },
        }

    async def _test_openrouter_connectivity(self) -> Dict[str, Any]:
        """Test OpenRouter API connectivity and model availability."""
        try:
            # Create a temporary router instance for testing
            router = OpenRouterTriModelRouter()

            # Test model availability
            availability = await router.detect_model_availability()

            # Check if any models are available
            available_models = [
                model for model, available in availability.items() if available
            ]

            return {
                "available": len(available_models) > 0,
                "tested_models": availability,
                "available_count": len(available_models),
                "total_tested": len(availability),
            }

        except Exception as e:
            logger.error(f"OpenRouter connectivity test failed: {e}")
            return {
                "available": False,
                "error": str(e),
                "tested_models": {},
                "available_count": 0,
                "total_tested": 0,
            }

    def _get_env_var_status(self) -> Dict[str, Any]:
        """Get status of all relevant environment variables."""
        all_vars = self.required_env_vars + self.recommended_env_vars

        status = {}
        for var in all_vars:
            value = os.getenv(var)
            if not value:
                status[var] = {"status": "missing", "value": None}
            elif value.startswith("dummy_"):
                status[var] = {"status": "dummy", "value": "dummy_*****"}
            else:
                # Mask sensitive values
                if "key" in var.lower() or "token" in var.lower():
                    masked_value = (
                        value[:8] + "*" * (len(value) - 8)
                        if len(value) > 8
                        else "*****"
                    )
                    status[var] = {"status": "configured", "value": masked_value}
                else:
                    status[var] = {"status": "configured", "value": value}

        return status

    def generate_setup_guide(self, validation_result: ValidationResult) -> str:
        """Generate a setup guide based on validation results."""
        guide_lines = [
            "# OpenRouter Configuration Setup Guide",
            "",
            "## Current Status",
            f"Configuration Valid: {'✓ YES' if validation_result.is_valid else '✗ NO'}",
            f"Errors: {len(validation_result.errors)}",
            f"Warnings: {len(validation_result.warnings)}",
            "",
        ]

        if validation_result.errors:
            guide_lines.extend(["## ❌ Critical Errors (Must Fix)", ""])
            for i, error in enumerate(validation_result.errors, 1):
                guide_lines.append(f"{i}. {error}")
            guide_lines.append("")

        if validation_result.warnings:
            guide_lines.extend(["## ⚠️ Warnings", ""])
            for i, warning in enumerate(validation_result.warnings, 1):
                guide_lines.append(f"{i}. {warning}")
            guide_lines.append("")

        if validation_result.recommendations:
            guide_lines.extend(["## 💡 Recommendations", ""])
            for i, rec in enumerate(validation_result.recommendations, 1):
                guide_lines.append(f"{i}. {rec}")
            guide_lines.append("")

        # Add environment variable setup instructions
        guide_lines.extend(
            [
                "## Environment Variable Setup",
                "",
                "Add these to your .env file:",
                "",
                "```bash",
                "# Required",
                "OPENROUTER_API_KEY=your_openrouter_api_key_here",
                "",
                "# Recommended",
                "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1",
                "OPENROUTER_HTTP_REFERER=your_app_url_here",
                "OPENROUTER_APP_TITLE=your_app_name_here",
                "",
                "# Model Configuration",
                "DEFAULT_MODEL=openai/gpt-5",
                "MINI_MODEL=openai/gpt-5-mini",
                "NANO_MODEL=openai/gpt-5-nano",
                "",
                "# Free Fallback Models",
                "FREE_FALLBACK_MODELS=openai/gpt-oss-20b:free,moonshotai/kimi-k2:free",
                "```",
                "",
                "## Next Steps",
                "",
                "1. Set the required environment variables",
                "2. Restart the application",
                "3. Run validation again to confirm setup",
                "",
            ]
        )

        return "\n".join(guide_lines)

    async def run_startup_validation(self, exit_on_failure: bool = False) -> bool:
        """Run startup validation and optionally exit on failure."""
        try:
            validation_result = await self.validate_configuration()

            # Print results
            print("\n" + "=" * 60)
            print("OpenRouter Configuration Validation")
            print("=" * 60)

            if validation_result.is_valid:
                print("✅ Configuration is VALID")
            else:
                print("❌ Configuration is INVALID")

            if validation_result.errors:
                print(f"\n❌ Errors ({len(validation_result.errors)}):")
                for error in validation_result.errors:
                    print(f"  • {error}")

            if validation_result.warnings:
                print(f"\n⚠️  Warnings ({len(validation_result.warnings)}):")
                for warning in validation_result.warnings:
                    print(f"  • {warning}")

            if validation_result.recommendations:
                print(
                    f"\n💡 Recommendations ({len(validation_result.recommendations)}):"
                )
                for rec in validation_result.recommendations:
                    print(f"  • {rec}")

            print("\n" + "=" * 60)

            # Generate and save setup guide if there are issues
            if not validation_result.is_valid or validation_result.warnings:
                setup_guide = self.generate_setup_guide(validation_result)

                # Try to save setup guide
                try:
                    with open("openrouter_setup_guide.md", "w") as f:
                        f.write(setup_guide)
                    print("📝 Setup guide saved to: openrouter_setup_guide.md")
                except Exception as e:
                    print(f"⚠️  Could not save setup guide: {e}")
                    print("\nSetup Guide:")
                    print("-" * 40)
                    print(setup_guide)

            # Exit on failure if requested
            if not validation_result.is_valid and exit_on_failure:
                print("\n❌ Exiting due to configuration errors")
                sys.exit(1)

            return validation_result.is_valid

        except Exception as e:
            print(f"\n❌ Validation failed with error: {e}")
            if exit_on_failure:
                sys.exit(1)
            return False


async def main():
    """Main function for running validation as a script."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate OpenRouter configuration")
    parser.add_argument(
        "--exit-on-failure",
        action="store_true",
        help="Exit with error code if validation fails",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)

    validator = OpenRouterStartupValidator()
    success = await validator.run_startup_validation(
        exit_on_failure=args.exit_on_failure
    )

    if not success and not args.exit_on_failure:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
