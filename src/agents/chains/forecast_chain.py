"""
ForecastChain: Chain-of-Thought + Evidence pipeline for Metaculus forecasting.
- Extracts question details
- Gathers evidence via search_tool
- Builds CoT prompt
- Invokes LLM (LangChain Runnable)
- Parses and returns forecast dict
"""

from typing import Dict, Any
from src.agents.search import SearchTool

class ForecastChain:
    def __init__(self, llm, search_tool: SearchTool):
        self.llm = llm
        self.search_tool = search_tool

    def run(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Steps:
        1. Extract question details (title, background, type)
        2. Use `search_tool.search(query)` for news evidence
        3. Build CoT prompt (MCP/templated)
        4. Invoke LLM with structured input
        5. Parse and return structured forecast with justification
        """
        if not question or 'question_id' not in question or 'question_text' not in question:
            return {"error": "Missing required question fields"}
        question_id = question['question_id']
        question_text = question['question_text']
        # 2. Evidence
        evidence = self.search_tool.search(question_text)
        # 3. CoT prompt
        prompt = self._build_prompt(question_text, evidence)
        # 4. LLM inference
        llm_response = self.llm.invoke({"prompt": prompt})
        # 5. Parse
        forecast, justification = self._parse_llm_response(llm_response)
        return {
            "question_id": question_id,
            "forecast": forecast,
            "justification": justification
        }

    def _build_prompt(self, question_text, evidence):
        return f"""You are a forecasting agent.\nQuestion: {question_text}\nEvidence: {evidence}\nThink step by step and output a probability (0-1) and justification as JSON: {{'forecast': float, 'justification': str}}"""

    def _parse_llm_response(self, llm_response):
        import json
        # Accepts dict or JSON string
        if isinstance(llm_response, dict):
            forecast = llm_response.get('forecast', 0.5)
            justification = llm_response.get('justification', str(llm_response))
        else:
            try:
                parsed = json.loads(llm_response)
                forecast = parsed.get('forecast', 0.5)
                justification = parsed.get('justification', str(parsed))
            except Exception:
                forecast = 0.5
                justification = str(llm_response)
        return forecast, justification
