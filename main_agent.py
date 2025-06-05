# main_agent.py
# CLI entry for running the Metaculus agent (dry-run or CI trigger)

import argparse
import json
from src.agents.agent_runner import run_agent
from src.api.metaculus_client import MetaculusClient

def main():
    parser = argparse.ArgumentParser(description="Run Metaculus Agentic Bot")
    parser.add_argument('--dryrun', action='store_true', help='Run in dry mode (no submission)')
    parser.add_argument('--submit', action='store_true', help='Submit forecast to Metaculus')
    parser.add_argument('--question', type=str, help='Path to question JSON file (optional)')
    args = parser.parse_args()

    # Example question (fallback)
    question = {
        'question_id': 1,
        'question_text': 'Will AGI be achieved by 2030?'
    }
    if args.question:
        with open(args.question, 'r') as f:
            question = json.load(f)

    from src.agents.forecast_agent import ForecastAgent
    agent = ForecastAgent()
    result = agent.invoke(question)
    print(json.dumps(result, indent=2))
    if args.submit:
        client = MetaculusClient()
        submit_result = client.submit_forecast(result['question_id'], result['forecast'], result['justification'])
        print("Submission result:", json.dumps(submit_result, indent=2))
    elif args.dryrun:
        print("[DRYRUN] No submission performed.")

if __name__ == "__main__":
    main()
