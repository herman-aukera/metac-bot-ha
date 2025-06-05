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
        question_type = question.get('type', 'binary')
        evidence = self.search_tool.search(question_text)
        if question_type in ('mc', 'multiple_choice'):
            options = question.get('options')
            if not options or not isinstance(options, list):
                return {"error": "Missing or invalid options for multi-choice question"}
            prompt = self._build_mc_prompt(question_text, options, evidence)
            llm_response = self.llm.invoke({"prompt": prompt})
            forecast, justification = self._parse_mc_llm_response(llm_response, options)
        elif question_type == 'numeric':
            prompt = self._build_numeric_prompt(question_text, evidence)
            llm_response = self.llm.invoke({"prompt": prompt})
            forecast, low, high, justification = self._parse_numeric_llm_response(llm_response)
            return {
                "question_id": question_id,
                "prediction": forecast,
                "low": low,
                "high": high,
                "justification": justification
            }
        else:
            prompt = self._build_prompt(question_text, evidence)
            llm_response = self.llm.invoke({"prompt": prompt})
            forecast, justification = self._parse_llm_response(llm_response)
        return {
            "question_id": question_id,
            "forecast": forecast,
            "justification": justification
        }

    def _build_prompt(self, question_text, evidence):
        return f"""You are a forecasting agent.\nQuestion: {question_text}\nEvidence: {evidence}\nThink step by step and output a probability (0-1) and justification as JSON: {{'forecast': float, 'justification': str}}"""

    def _build_mc_prompt(self, question_text, options, evidence):
        return (
            f"You are a forecasting agent.\n"
            f"Question: {question_text}\n"
            f"Options: {options}\n"
            f"Evidence: {evidence}\n"
            "Think step by step and output a probability list (summing to 1) and justification as JSON: "
            "{'forecast': [float, ...], 'justification': str}"
        )

    def _build_numeric_prompt(self, question_text, evidence):
        return (
            f"You are a forecasting agent.\n"
            f"Question: {question_text}\n"
            f"Evidence: {evidence}\n"
            "Think step by step and output a numeric prediction and 90% confidence interval as JSON: "
            "{'prediction': float, 'low': float, 'high': float, 'justification': str}"
        )

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

    def _parse_mc_llm_response(self, llm_response, options):
        import json
        # Accepts dict or JSON string
        if isinstance(llm_response, dict):
            forecast = llm_response.get('forecast', [1.0/len(options)]*len(options))
            justification = llm_response.get('justification', str(llm_response))
        else:
            try:
                parsed = json.loads(llm_response)
                forecast = parsed.get('forecast', [1.0/len(options)]*len(options))
                justification = parsed.get('justification', str(parsed))
            except Exception:
                forecast = [1.0/len(options)]*len(options)
                justification = str(llm_response)
        # Ensure forecast is a list of floats of correct length
        if not (isinstance(forecast, list) and len(forecast) == len(options)):
            forecast = [1.0/len(options)]*len(options)
        return forecast, justification

    def _parse_numeric_llm_response(self, llm_response):
        import json
        # Accepts dict or JSON string
        if isinstance(llm_response, dict):
            pred = llm_response.get('prediction')
            low = llm_response.get('low')
            high = llm_response.get('high')
            justification = llm_response.get('justification', str(llm_response))
        else:
            try:
                parsed = json.loads(llm_response)
                pred = parsed.get('prediction')
                low = parsed.get('low')
                high = parsed.get('high')
                justification = parsed.get('justification', str(parsed))
            except Exception:
                pred, low, high = None, None, None
                justification = str(llm_response)
        # Fallbacks
        if pred is None:
            pred = 0.0
        if low is None:
            low = pred
        if high is None:
            high = pred
        return pred, low, high, justification
