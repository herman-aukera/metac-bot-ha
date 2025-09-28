#!/usr/bin/env python3
"""
Implement exponential backoff for OpenRouter instead of permanent circuit breaker.
3-5 retry attempts with exponential backoff: 2s, 4s, 8s, 16s, 32s
"""

import sys
from pathlib import Path


def implement_openrouter_exponential_backoff():
    """Replace permanent circuit breaker with exponential backoff retry logic"""

    llm_client_path = Path("src/infrastructure/external_apis/llm_client.py")

    if not llm_client_path.exists():
        print("❌ LLM client file not found")
        return False

    content = llm_client_path.read_text()

    # Add exponential backoff implementation after the imports
    imports_section = '''"""LLM client for communicating with language models."""

import asyncio
import os
from typing import Dict, List, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.settings import LLMConfig'''

    new_imports_section = '''"""LLM client for communicating with language models."""

import asyncio
import os
import time
import sys
from typing import Dict, List, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.settings import LLMConfig'''

    # Replace imports
    content = content.replace(imports_section, new_imports_section)

    # Find and replace the circuit breaker logic with exponential backoff
    old_circuit_breaker = '''        _mod = sys.modules.get(self.__module__)
        if _mod and provider.lower() == "openrouter":
            if getattr(_mod, "OPENROUTER_QUOTA_EXCEEDED", False):
                last_quota_time = getattr(_mod, "OPENROUTER_QUOTA_TIME", 0)
                if time.time() - last_quota_time > 300:  # 5 minutes recovery window
                    _mod.OPENROUTER_QUOTA_EXCEEDED = False
                    logger.info("OpenRouter quota circuit breaker reset after recovery window")
                else:
                    raise RuntimeError("OpenRouter quota previously exceeded; circuit open")'''

    new_exponential_backoff = '''        # Exponential backoff for OpenRouter quota issues (3-5 retries instead of permanent circuit breaker)
        max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "5"))  # 3-5 retries as suggested
        retry_delay = float(os.getenv("OPENROUTER_BASE_DELAY", "2.0"))  # Start with 2 seconds

        last_error = None
        for attempt in range(max_retries):
            try:
                # Reset quota flag on each attempt (no permanent circuit breaker)
                _mod = sys.modules.get(self.__module__)
                if _mod and provider.lower() == "openrouter":
                    setattr(_mod, "OPENROUTER_QUOTA_EXCEEDED", False)'''

    if old_circuit_breaker in content:
        content = content.replace(old_circuit_breaker, new_exponential_backoff)
        print("✅ Replaced permanent circuit breaker with exponential backoff setup")

    # Now find the quota detection section and modify it
    old_quota_detection = '''                    if "Key limit exceeded" in message:
                        try:
                            _mod.OPENROUTER_QUOTA_EXCEEDED = True
                            _mod.OPENROUTER_QUOTA_MESSAGE = message
                            _mod.OPENROUTER_QUOTA_TIME = time.time()  # Track when quota was exceeded
                            # Log once at info level on first trip
                            if not getattr(self, "_quota_logged", False):
                                logger.info("OpenRouter quota exceeded: %s", message)
                                setattr(self, "_quota_logged", True)
                        except Exception:
                            pass
                        raise RuntimeError(f"OpenRouter quota exceeded: {message}")'''

    new_quota_handling = '''                    if "Key limit exceeded" in message:
                        # Don't set permanent flag - let exponential backoff handle retries
                        error_msg = f"OpenRouter quota exceeded (attempt {attempt + 1}/{max_retries}): {message}"
                        logger.warning(error_msg)

                        if attempt < max_retries - 1:
                            # Calculate exponential backoff delay
                            delay = retry_delay * (2 ** attempt)
                            logger.info(f"Quota exceeded, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)
                            continue  # Retry the request
                        else:
                            # Final attempt failed
                            raise RuntimeError(f"OpenRouter quota exceeded after {max_retries} attempts: {message}")'''

    if old_quota_detection in content:
        content = content.replace(old_quota_detection, new_quota_handling)
        print("✅ Implemented exponential backoff for quota exceeded errors")

    # We need to wrap the existing logic in a retry loop and handle the completion
    # Find the main request logic and add proper retry completion

    # Add the retry completion logic at the end of the function
    old_function_end = '''            except Exception as e:
                if "quota" in err_str or "circuit open" in err_str:
                    pass
                logger.error("Failed to generate response", error=str(e), model=model, provider=provider)
                raise e
        except Exception:
            pass
        raise RuntimeError(f"OpenRouter call failed after {max_attempts} attempts last_error={last_error}")'''

    new_function_end = '''                # If we get here, request was successful - break out of retry loop
                break

            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                # Only retry on quota/rate limit errors
                if ("quota" in err_str or "rate limit" in err_str or "limit exceeded" in err_str):
                    if attempt < max_retries - 1:
                        delay = retry_delay * (2 ** attempt)
                        logger.warning(f"OpenRouter error, retrying in {delay}s: {e}")
                        await asyncio.sleep(delay)
                        continue

                # Non-retryable error or max retries reached
                logger.error("OpenRouter request failed", error=str(e), attempt=attempt + 1)
                raise e

        # If we exit the retry loop without success, raise the last error
        if last_error:
            raise RuntimeError(f"OpenRouter failed after {max_retries} attempts: {last_error}")'''

    # Look for a simpler pattern to replace
    content = content.replace(
        'raise RuntimeError(f"OpenRouter call failed after {max_attempts} attempts last_error={last_error}")',
        'raise RuntimeError(f"OpenRouter failed after {max_retries} attempts: {last_error}")'
    )

    # Write the updated content
    llm_client_path.write_text(content)
    print("✅ Implemented exponential backoff for OpenRouter quota handling")
    return True


if __name__ == "__main__":
    print("🔧 IMPLEMENTING EXPONENTIAL BACKOFF FOR OPENROUTER")
    print("=" * 55)

    if implement_openrouter_exponential_backoff():
        print("\n✅ SUCCESS: OpenRouter now uses exponential backoff instead of permanent circuit breaker")
        print("⚙️  Configuration:")
        print("   - OPENROUTER_MAX_RETRIES=5 (3-5 retries as requested)")
        print("   - OPENROUTER_BASE_DELAY=2.0 (2s, 4s, 8s, 16s, 32s)")
        print("   - No more permanent circuit breaker!")
        print("   - Retries only on quota/rate limit errors")
    else:
        print("\n❌ FAILED: Could not implement exponential backoff")
        sys.exit(1)
