# main_agent.py
# CLI entry for running the Metaculus agent (dry-run or CI trigger)

import argparse
import json
import os
import importlib.util
from src.agents.agent_runner import run_agent
from src.api.metaculus_client import MetaculusClient
from src.api.question_fetcher import fetch_new_questions
from src.agents.batch_executor import run_batch

def main():
    parser = argparse.ArgumentParser(description="Run Metaculus Agentic Bot")
    parser.add_argument('--dryrun', action='store_true', help='Run in dry mode (no submission)')
    parser.add_argument('--submit', action='store_true', help='Submit forecast to Metaculus')
    parser.add_argument('--question', type=str, help='Path to question JSON file (optional)')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'], help='Run mode: single or batch')
    parser.add_argument('--limit', type=int, default=5, help='Max questions for batch mode')
    parser.add_argument('--logfile', type=str, help='Optional log file for batch results')
    parser.add_argument('--model', type=str, help='Model ID for LLM (e.g. openai/gpt-4, anthropic/claude-3, mistral/mixtral-8x7b)')
    parser.add_argument('--show-trace', action='store_true', help='Show step-by-step reasoning trace in output')
    args = parser.parse_args()

    if args.mode == 'batch':
        questions = fetch_new_questions(limit=args.limit)
        if not questions:
            print("No new questions to forecast.")
            return
        run_batch(questions, submit=args.submit, log_file=args.logfile)
        return

    # Example question (fallback)
    question = {
        'question_id': 1,
        'question_text': 'Will AGI be achieved by 2030?'
    }
    # Add MC example if no --question provided
    mc_example = {
        'question_id': 2,
        'question_text': 'Which city will win the bid?',
        'type': 'mc',
        'options': ['London', 'Paris', 'Berlin']
    }
    if args.question:
        with open(args.question, 'r') as f:
            question = json.load(f)
    else:
        # If user wants MC, they can uncomment below
        # question = mc_example
        pass

    model_id = args.model
    from src.agents.llm.model_router import get_llm
    from src.agents.forecast_agent import ForecastAgent
    llm = get_llm(model_id)
    agent = ForecastAgent(llm=llm)
    result = agent.invoke(question)
    print(json.dumps(result, indent=2))
    if args.show_trace and 'trace' in result:
        print("\n--- Reasoning Trace ---")
        print(json.dumps(result['trace'], indent=2))
    if 'options' in question:
        print('Options:', question['options'])
    if args.submit:
        client = MetaculusClient()
        submit_result = client.submit_forecast(result['question_id'], result['forecast'], result['justification'])
        print("Submission result:", json.dumps(submit_result, indent=2))
    elif args.dryrun:
        print("[DRYRUN] No submission performed.")

if __name__ == "__main__":
    if "PYTHONPATH" not in os.environ:
        print("\u26A0\uFE0F Warning: PYTHONPATH is not set. Set it to your project root to avoid import errors.")
    main()
