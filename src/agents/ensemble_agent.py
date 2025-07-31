# ensemble_agent.py
# Implements EnsembleForecastAgent for multi-agent forecasting and reasoning logging

import os
import json
from datetime import datetime
from src.agents.forecast_agent import ForecastAgent
from src.agents.llm import MockLLM
from src.agents.search import SearchTool

AGENT_CONFIGS = [
    {"name": "gpt-4", "llm": MockLLM()},
    {"name": "claude", "llm": MockLLM()},
    {"name": "gemini", "llm": MockLLM()},
]

class EnsembleForecastAgent:
    def __init__(self, agent_configs=None, search_tool=None, tools=None):
        self.agent_configs = agent_configs or AGENT_CONFIGS
        self.search_tool = search_tool or SearchTool()
        self.tools = tools
        self.log_dir = os.path.join("logs", "reasoning")
        os.makedirs(self.log_dir, exist_ok=True)

    def run_ensemble_forecast(self, question_json):
        forecasts = []
        justifications = []
        traces = []
        agent_results = []
        for agent_cfg in self.agent_configs:
            agent = ForecastAgent(llm=agent_cfg["llm"], search_tool=self.search_tool, tools=self.tools)
            result = agent.invoke(question_json)
            agent_results.append((agent_cfg["name"], result))
            # Save reasoning log
            self._log_reasoning(question_json, agent_cfg["name"], result)
            # Extract forecast/justification
            forecast = result.get("forecast")
            if forecast is None:
                forecast = result.get("prediction")
            just = result.get("justification") or result.get("reasoning") or "No justification."
            forecasts.append(forecast)
            justifications.append(f"[{agent_cfg['name']}]: {just}")
            traces.append(result.get("trace"))
        # Combine forecasts (average for binary/numeric, mean for MC)
        combined = self._combine_forecasts(forecasts, question_json)
        combined_just = "\n\n".join(justifications)
        return {
            "question_id": question_json.get("question_id"),
            "forecast": combined,
            "justification": combined_just,
            "traces": traces,
            "agent_results": agent_results
        }

    def _combine_forecasts(self, forecasts, question_json):
        # Simple average for binary/numeric, elementwise mean for MC
        if not forecasts:
            return None
        qtype = question_json.get("type", "binary")
        if qtype in ("mc", "multiple_choice") and isinstance(forecasts[0], list):
            # Elementwise mean
            n = len(forecasts[0])
            avg = [sum(f[i] for f in forecasts) / len(forecasts) for i in range(n)]
            return avg
        elif isinstance(forecasts[0], (int, float)):
            return sum(forecasts) / len(forecasts)
        elif isinstance(forecasts[0], dict):
            # Numeric dict: average each key
            keys = forecasts[0].keys()
            avg = {k: sum(f[k] for f in forecasts) / len(forecasts) for k in keys}
            return avg
        return forecasts[0]

    def _log_reasoning(self, question_json, agent_name, result):
        qid = question_json.get("question_id", "unknown")
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        fname = f"question-{qid}_agent-{agent_name}.md"
        path = os.path.join(self.log_dir, fname)
        prompt = None
        for step in (result.get("trace") or []):
            if step["type"] == "prompt":
                prompt = step["input"].get("prompt")
                break
        with open(path, "w") as f:
            f.write(f"# Reasoning Log\n")
            f.write(f"- Timestamp: {ts}\n")
            f.write(f"- Agent: {agent_name}\n")
            f.write(f"- Question ID: {qid}\n")
            if prompt:
                f.write(f"- Prompt:\n\n{prompt}\n\n")
            f.write(f"- Full Explanation/Justification:\n\n{result.get('justification') or result.get('reasoning') or 'No justification.'}\n")
            if "trace" in result:
                f.write(f"\n---\n\n## Trace\n\n{json.dumps(result['trace'], indent=2)}\n")
