"""
API key management with fallback handling for missing credentials.
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages API keys with fallback handling for missing credentials."""

    def __init__(self):
        self.required_keys = {
            'METACULUS_TOKEN': 'Required for Metaculus API access',
            'OPENROUTER_API_KEY': 'Required for LLM API access (tournament provided key)'
        }

        self.optional_keys = {
            'PERPLEXITY_API_KEY': 'Optional for enhanced search capabilities',
            'EXA_API_KEY': 'Optional for web search',
            'OPENAI_API_KEY': 'Optional for OpenAI models',
            'ANTHROPIC_API_KEY': 'Optional for Claude models',
            'ASKNEWS_CLIENT_ID': 'Optional for news search',
            'ASKNEWS_SECRET': 'Optional for news search'
        }

    def get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key with fallback handling."""
        value = os.getenv(key_name)

        if not value or value.startswith('dummy_'):
            if key_name in self.required_keys:
                logger.warning(f"Required API key {key_name} is missing or dummy")
                return None
            else:
                logger.info(f"Optional API key {key_name} not configured, using fallback")
                return self._get_fallback_key(key_name)

        return value

    def _get_fallback_key(self, key_name: str) -> str:
        """Get fallback/dummy key for optional services."""
        fallback_keys = {
            'PERPLEXITY_API_KEY': 'fallback_perplexity',
            'EXA_API_KEY': 'fallback_exa',
            'OPENAI_API_KEY': 'fallback_openai',
            'ANTHROPIC_API_KEY': 'fallback_anthropic',
            'ASKNEWS_CLIENT_ID': 'fallback_asknews_client',
            'ASKNEWS_SECRET': 'fallback_asknews_secret'
        }
        return fallback_keys.get(key_name, 'fallback_key')

    def validate_required_keys(self) -> Dict[str, Any]:
        """Validate that all required API keys are present."""
        validation_result = {
            'valid': True,
            'missing_keys': [],
            'warnings': []
        }

        for key_name, description in self.required_keys.items():
            value = self.get_api_key(key_name)
            if not value:
                validation_result['valid'] = False
                validation_result['missing_keys'].append({
                    'key': key_name,
                    'description': description
                })

        for key_name, description in self.optional_keys.items():
            value = os.getenv(key_name)
            if not value or value.startswith('dummy_'):
                validation_result['warnings'].append({
                    'key': key_name,
                    'description': description,
                    'status': 'using_fallback'
                })

        return validation_result

    def get_all_keys(self) -> Dict[str, str]:
        """Get all API keys with fallback handling."""
        keys = {}

        for key_name in {**self.required_keys, **self.optional_keys}:
            keys[key_name] = self.get_api_key(key_name)

        return keys

    def log_key_status(self):
        """Log the status of all API keys."""
        validation = self.validate_required_keys()

        if validation['valid']:
            logger.info("All required API keys are configured")
        else:
            logger.error(f"Missing required API keys: {[k['key'] for k in validation['missing_keys']]}")

        if validation['warnings']:
            logger.info(f"Using fallback for optional keys: {[w['key'] for w in validation['warnings']]}")


# Global instance
api_key_manager = APIKeyManager()
