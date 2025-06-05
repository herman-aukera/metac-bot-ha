# wikipedia.py
"""
WikipediaTool: LangChain-compatible tool for Wikipedia summaries.
- Returns a short, source-attributed summary for a given query.
- CI/offline safe: can be mocked in tests.
"""
import requests

class WikipediaTool:
    def __init__(self, lang="en"):
        self.lang = lang

    def run(self, query: str) -> str:
        """Return a short summary for the query from Wikipedia."""
        if not query or not isinstance(query, str):
            return "[WikipediaTool] Invalid query."
        url = f"https://{self.lang}.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                summary = data.get("extract")
                title = data.get("title")
                if summary:
                    return f"[Wikipedia: {title}] {summary}"
                else:
                    return f"[Wikipedia: {title}] No summary available."
            elif resp.status_code == 404:
                return f"[WikipediaTool] No Wikipedia page found for '{query}'."
            else:
                return f"[WikipediaTool] Error: {resp.status_code}"
        except Exception as e:
            return f"[WikipediaTool] Exception: {e}"
