# llm.py
# Minimal mockable LLM wrapper for ForecastChain

class MockLLM:
    def invoke(self, input_dict):
        # Return a deterministic mock response for testing
        return {'forecast': 0.77, 'justification': 'Mock LLM justification.'}
