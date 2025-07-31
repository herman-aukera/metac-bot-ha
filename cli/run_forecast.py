import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents.batch_executor import run_batch
from src.agents.ensemble_agent import EnsembleForecastAgent

def main():
    parser = argparse.ArgumentParser(description="Run forecasts on a batch of questions.")
    parser.add_argument("--submit", action="store_true", help="Submit forecasts to Metaculus")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble of agents for forecasting")
    parser.add_argument("--questions", type=str, default="data/questions.json", help="Path to questions JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.questions):
        print(f"Questions file not found: {args.questions}")
        return
    with open(args.questions, "r") as f:
        questions = json.load(f)

    # Normalize question_id field if needed
    for q in questions:
        if "question_id" not in q and "id" in q:
            q["question_id"] = q["id"]

    # Map 'title' to 'question_text' if needed for each question
    for q in questions:
        if 'question_text' not in q and 'title' in q:
            q['question_text'] = q['title']

    if args.ensemble:
        agent = EnsembleForecastAgent()
        for q in questions:
            result = agent.run_ensemble_forecast(q)
            print(f"\n=== Ensemble Forecast for Question {q.get('question_id')} ===")
            print(f"Forecast: {result['forecast']}")
            print(f"Justification:\n{result['justification']}")
            print(f"Reasoning logs saved to logs/reasoning/question-{q.get('question_id')}_agent-*.md\n")
    else:
        logs = run_batch(questions, submit=args.submit)
        for log in logs:
            qid = log.get("question_id")
            prob = log.get("forecast")
            if isinstance(prob, float):
                prob_pct = f"{prob*100:.1f}%"
            elif isinstance(prob, list):
                prob_pct = ", ".join(f"{p*100:.1f}%" for p in prob)
            else:
                prob_pct = str(prob)
            reasoning = log.get("justification") or log.get("reasoning")
            print(f"QID: {qid}\nForecast: {prob_pct}\nReasoning: {reasoning}\n{'-'*40}")

if __name__ == "__main__":
    main()
