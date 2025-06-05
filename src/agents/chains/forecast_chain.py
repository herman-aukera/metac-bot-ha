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
from src.agents.tools import tool_list
from datetime import datetime

class ForecastChain:
    def __init__(self, llm, search_tool: Any, tools=None):
        self.llm = llm
        self.search_tool = search_tool
        self.tools = tools if tools is not None else tool_list
        self.trace = []

    def _log_trace(self, step_type, input_data, output_data):
        self.trace.append({
            "step": len(self.trace) + 1,
            "type": step_type,
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    def _call_tools(self, question_text):
        """Call all tools and collect their outputs for the question."""
        tool_outputs = {}
        for tool in self.tools:
            # WikipediaTool: only call if question is a string and not too long
            if tool.__class__.__name__ == "WikipediaTool" and isinstance(question_text, str) and len(question_text) < 100:
                wiki = tool.run(question_text)
                self._log_trace("tool", {"tool": "WikipediaTool", "query": question_text}, wiki)
                if wiki and "Wikipedia" in wiki:
                    tool_outputs["WikipediaTool"] = wiki
            # MathTool: only call if question looks like a math expression
            elif tool.__class__.__name__ == "MathTool" and any(op in question_text for op in ["+", "-", "*", "/", "%", "^"]):
                math = tool.run(question_text)
                self._log_trace("tool", {"tool": "MathTool", "expr": question_text}, math)
                if math and "Error" not in math:
                    tool_outputs["MathTool"] = math
        return tool_outputs

    def run(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Steps:
        1. Extract question details (title, background, type)
        2. Use `search_tool.search(query)` for news evidence
        3. Build CoT prompt (MCP/templated)
        4. Invoke LLM with structured input
        5. Parse and return structured forecast with justification
        """
        self.trace = []
        if not question or 'question_id' not in question or 'question_text' not in question:
            return {"error": "Missing required question fields"}
        self._log_trace("input", question, None)
        question_id = question['question_id']
        question_text = question['question_text']
        question_type = question.get('type', 'binary')
        evidence = self.search_tool.search(question_text)
        self._log_trace("evidence", question_text, evidence)
        tool_outputs = self._call_tools(question_text)
        # Add tool outputs to evidence if present
        if tool_outputs:
            evidence = f"{evidence}\nTool outputs: {tool_outputs}"
        if question_type in ('mc', 'multiple_choice'):
            options = question.get('options')
            if not options or not isinstance(options, list):
                return {"error": "Missing or invalid options for multi-choice question"}
            prompt = self._build_mc_prompt(question_text, options, evidence)
            self._log_trace("prompt", {"type": "mc", "prompt": prompt}, None)
            llm_response = self.llm.invoke({"prompt": prompt})
            self._log_trace("llm", {"prompt": prompt}, llm_response)
            forecast, justification = self._parse_mc_llm_response(llm_response, options)
            result = {
                "question_id": question_id,
                "forecast": forecast,
                "justification": justification
            }
        elif question_type == 'numeric':
            prompt = self._build_numeric_prompt(question_text, evidence)
            self._log_trace("prompt", {"type": "numeric", "prompt": prompt}, None)
            llm_response = self.llm.invoke({"prompt": prompt})
            self._log_trace("llm", {"prompt": prompt}, llm_response)
            forecast, low, high, justification = self._parse_numeric_llm_response(llm_response)
            result = {
                "question_id": question_id,
                "prediction": forecast,
                "low": low,
                "high": high,
                "justification": justification
            }
        else:
            prompt = self._build_prompt(question_text, evidence)
            self._log_trace("prompt", {"type": "binary", "prompt": prompt}, None)
            llm_response = self.llm.invoke({"prompt": prompt})
            self._log_trace("llm", {"prompt": prompt}, llm_response)
            forecast, justification = self._parse_llm_response(llm_response)
            result = {
                "question_id": question_id,
                "forecast": forecast,
                "justification": justification
            }
        result["trace"] = self.trace
        return result

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
