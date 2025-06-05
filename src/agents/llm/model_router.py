# model_router.py
"""
ModelRouter: Returns a LangChain LLM instance for a given model_id string.
Supports: openai/gpt-4, anthropic/claude-3, mistral/mixtral-8x7b (OpenRouter), and fallback.
"""
from src.agents.llm import MockLLM

def get_llm(model_id: str):
    """
    Given a model_id, return a LangChain-compatible LLM instance.
    """
    if not model_id:
        return MockLLM()
    model_id = model_id.lower()
    if model_id in ["openai/gpt-4", "gpt-4", "openai-gpt-4"]:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4")
        except ImportError:
            return MockLLM()
    if model_id in ["anthropic/claude-3", "claude-3", "anthropic-claude-3"]:
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-3-opus-20240229")
        except ImportError:
            return MockLLM()
    if model_id in ["mistral/mixtral-8x7b", "mixtral-8x7b", "openrouter/mixtral-8x7b"]:
        try:
            from langchain_openrouter import ChatOpenRouter
            return ChatOpenRouter(model="mistralai/mixtral-8x7b-32768")
        except ImportError:
            return MockLLM()
    # Fallback
    return MockLLM()
