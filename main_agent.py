# main_agent.py
# CLI entry for running the Metaculus agent (dry-run or CI trigger)

import argparse
import json
from src.agents.agent_runner import run_agent

def main():
    parser = argparse.ArgumentParser(description="Run Metaculus Agentic Bot")
    parser.add_argument('--dryrun', action='store_true', help='Run in dry mode (no submission)')
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

    run_agent(question, dryrun=args.dryrun)

if __name__ == "__main__":
    main()
