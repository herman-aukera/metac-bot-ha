# search.py
# AskNews/Perplexity search tool wrapper for agent evidence gathering

class SearchTool:
    def __init__(self, client=None):
        self.client = client  # Inject AskNews/Perplexity client or mock

    def search(self, query: str) -> str:
        # Integrar API real si es necesario
        # Por ahora, devolver evidencia simulada
        return f"Evidence for: {query} (stubbed)"
