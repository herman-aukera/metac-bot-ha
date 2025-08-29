"""LLM client for communicating with language models."""

import asyncio
from typing import Any, Dict, List, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.settings import LLMConfig

logger = structlog.get_logger(__name__)


class LLMClient:
    """
    Client for interacting with language models.

    Supports multiple providers and includes retry logic,
    rate limiting, and structured logging.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logger.bind(provider=config.provider, model=config.model)
        self._rate_limiter = asyncio.Semaphore(5)  # Allow 5 concurrent requests

        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate text using the configured LLM.

        Args:
            prompt: Input prompt for the model
            model: Override the default model
            temperature: Override the default temperature
            max_tokens: Override the default max_tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response
        """
        async with self._rate_limiter:
            model = model or self.config.model
            temperature = (
                temperature if temperature is not None else self.config.temperature
            )
            max_tokens = max_tokens or self.config.max_tokens

            self.logger.info(
                "Generating response",
                model=model,
                temperature=temperature,
                prompt_length=len(prompt),
            )

            try:
                if self.config.provider == "openai":
                    response = await self._call_openai(
                        prompt, model, temperature, max_tokens, **kwargs
                    )
                elif self.config.provider == "anthropic":
                    response = await self._call_anthropic(
                        prompt, model, temperature, max_tokens, **kwargs
                    )
                elif self.config.provider == "openrouter":
                    response = await self._call_openrouter(
                        prompt, model, temperature, max_tokens, **kwargs
                    )
                else:
                    raise ValueError(
                        f"Unsupported LLM provider: {self.config.provider}"
                    )

                self.logger.info(
                    "Response generated", response_length=len(response), model=model
                )

                return response

            except Exception as e:
                self.logger.error(
                    "Failed to generate response", error=str(e), model=model
                )
                raise

    async def _call_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> str:
        """Call OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        if max_tokens:
            data["max_tokens"] = max_tokens

        # Add any additional kwargs
        data.update(kwargs)

        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=data
        )

        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]

    async def _call_anthropic(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> str:
        """Call Anthropic API."""
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens or 4000,
        }

        # Add any additional kwargs
        data.update(kwargs)

        response = await self.client.post(
            "https://api.anthropic.com/v1/messages", headers=headers, json=data
        )

        response.raise_for_status()
        result = response.json()

        return result["content"][0]["text"]

    async def _call_openrouter(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> str:
        """Call OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/metaculus-bot-ha",
            "X-Title": "Metaculus Forecasting Bot HA",
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        if max_tokens:
            data["max_tokens"] = max_tokens

        # Add any additional kwargs
        data.update(kwargs)

        response = await self.client.post(
            "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data
        )

        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]

    async def batch_generate(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of input prompts
            model: Override the default model
            temperature: Override the default temperature
            max_tokens: Override the default max_tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            List of generated responses
        """
        tasks = [
            self.generate(prompt, model, temperature, max_tokens, **kwargs)
            for prompt in prompts
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions in the batch
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(
                    "Batch generation failed for prompt",
                    prompt_index=i,
                    error=str(response),
                )
                results.append("")  # Empty string for failed generations
            else:
                results.append(response)

        return results

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate chat completion using messages format.

        This method provides compatibility with agents that expect
        OpenAI-style chat completion format.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Override the default model
            temperature: Override the default temperature
            max_tokens: Override the default max_tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response
        """
        # Convert messages format to simple prompt
        # Most LLM providers accept this format, but we simplify for our generate method
        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Extract the user message content (assuming last message is from user)
        user_messages = [
            msg["content"] for msg in messages if msg.get("role") == "user"
        ]
        if not user_messages:
            raise ValueError("No user message found in messages")

        # Use the last user message as the prompt
        prompt = user_messages[-1]

        # If there are system messages, prepend them
        system_messages = [
            msg["content"] for msg in messages if msg.get("role") == "system"
        ]
        if system_messages:
            prompt = f"{system_messages[0]}\n\n{prompt}"

        # Call the existing generate method
        return await self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response - alias for generate method to maintain test compatibility.

        Args:
            prompt: Input prompt for the model
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        return await self.generate(prompt, **kwargs)

    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.

        Returns:
            Dictionary of headers for the API request
        """
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _format_prompt(self, prompt) -> str:
        """
        Format prompt for the LLM.

        Args:
            prompt: Input prompt (can be string or dict)

        Returns:
            Formatted prompt string
        """
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, dict):
            # Handle structured prompts
            parts = []
            if "question" in prompt:
                parts.append(f"Question: {prompt['question']}")
            if "context" in prompt:
                parts.append(f"Context: {prompt['context']}")
            if "format" in prompt:
                parts.append(f"Format: {prompt['format']}")
            return "\n\n".join(parts)
        else:
            return str(prompt)

    async def health_check(self) -> bool:
        """
        Check if the LLM service is available.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Simple test request
            await self.generate("test", max_tokens=1)
            return True
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False
