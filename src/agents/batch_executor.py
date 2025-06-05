# batch_executor.py
"""
Batch executor for running forecasts on a list of questions.
- Runs ForecastChain
- Logs results
- Optionally submits via MetaculusClient
"""
import json
from src.agents.forecast_agent import ForecastAgent
from src.api.metaculus_client import MetaculusClient

def run_batch(questions, submit=False, log_file=None):
    agent = ForecastAgent()
    client = MetaculusClient() if submit else None
    logs = []
    for q in questions:
        result = agent.invoke(q)
        log_entry = {
            "question_id": q["question_id"],
            "forecast": result["forecast"],
            "justification": result["justification"]
        }
        if "trace" in result:
            log_entry["trace"] = result["trace"]
        if q.get("type") in ("mc", "multiple_choice"):
            log_entry["options"] = q.get("options")
        if submit and client:
            submit_result = client.submit_forecast(q["question_id"], result["forecast"], result["justification"])
            log_entry["submission"] = submit_result
        print(json.dumps(log_entry, indent=2))
        logs.append(log_entry)
    if log_file:
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    return logs
