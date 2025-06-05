# search.py
# AskNews/Perplexity search tool wrapper for agent evidence gathering

class SearchTool:
    def __init__(self, client=None):
        self.client = client  # Inject AskNews/Perplexity client or mock

    def search(self, query: str) -> str:
        # TODO: Integrate real AskNews/Perplexity API
        # For now, return stubbed evidence
        return f"Stubbed evidence for: {query}"
