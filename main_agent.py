# main_agent.py
# CLI entry for running the Metaculus agent (dry-run or CI trigger)

import argparse
import json
import os
from src.api.metaculus_client import MetaculusClient
from src.agents.batch_executor import run_batch

# Carga variables de entorno desde .env si está presente (solo en desarrollo)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    parser = argparse.ArgumentParser(description="Run Metaculus Agentic Bot with Tournament Optimization")
    parser.add_argument('--dryrun', action='store_true', help='Run in dry mode (no submission)')
    parser.add_argument('--submit', action='store_true', help='Submit forecast to Metaculus')
    parser.add_argument('--question', type=str, help='Path to question JSON file (optional)')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'], help='Run mode: single or batch')
    parser.add_argument('--limit', type=int, default=5, help='Max questions for batch mode')
    parser.add_argument('--logfile', type=str, help='Optional log file for batch results')
    parser.add_argument('--model', type=str, help='Model ID for LLM (e.g. openai/gpt-4, anthropic/claude-3, mistral/mixtral-8x7b)')
    parser.add_argument('--show-trace', action='store_true', help='Show step-by-step reasoning trace in output')
    parser.add_argument('--plugin-dir', type=str, help='Directory to load external PluginTools from')
    parser.add_argument('--enable-webhooks', action='store_true', help='Enable webhook plugin support')
    parser.add_argument('--version', action='store_true', help='Show version and exit')
    parser.add_argument('--use-optimization', action='store_true', default=True, help='Use tournament optimization features')
    parser.add_argument('--legacy-mode', action='store_true', help='Force legacy mode without optimization')
    parser.add_argument('--tournament-id', type=int, help='Tournament ID for context')
    args = parser.parse_args()

    if args.version:
        print("Metaculus Agentic Bot version: v2.0.0-tournament-optimization")
        return

    # Determine whether to use optimization features
    use_optimization = args.use_optimization and not args.legacy_mode

    if args.mode == 'batch':
        from src.api.question_fetcher import fetch_new_questions
        questions = fetch_new_questions(limit=args.limit)
        if not questions:
            print("No new questions to forecast.")
            return

        if use_optimization:
            print("Running batch mode with tournament optimization...")
            # In a real implementation, this would use the tournament optimization system
            print("Tournament optimization for batch mode not yet implemented, falling back to legacy")
            use_optimization = False

        if not use_optimization:
            run_batch(questions, submit=args.submit, log_file=args.logfile)
        return

    # Single question processing
    # Example question (fallback)
    question = {
        'question_id': 1,
        'question_text': 'Will AGI be achieved by 2030?'
    }
    if args.question:
        with open(args.question, 'r') as f:
            question = json.load(f)

    if use_optimization:
        print("Processing question with tournament optimization...")

        try:
            # Try to use the new integration service
            import asyncio
            from src.application.services.integration_service import IntegrationService

            # In a real implementation, this would properly initialize the integration service
            # For now, we'll simulate the integration
            print("Tournament optimization system not fully initialized, falling back to legacy mode")
            use_optimization = False

        except ImportError as e:
            print(f"Tournament optimization not available: {e}")
            print("Falling back to legacy mode")
            use_optimization = False

    if not use_optimization:
        print("Processing question in legacy mode...")

        model_id = args.model
        from src.agents.llm.model_router import get_llm
        from src.agents.forecast_agent import ForecastAgent
        llm = get_llm(model_id)
        agent = ForecastAgent(llm=llm)

        # Plugin loading
        plugins = []
        if args.plugin_dir:
            from src.agents.tools.plugin_loader import load_plugins
            plugins = load_plugins(args.plugin_dir, enable_webhooks=args.enable_webhooks)
            if plugins:
                print(f"Loaded plugins: {[p.name for p in plugins]}")
            else:
                print("No plugins loaded.")
        if plugins:
            agent.chain.add_plugins(plugins)

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

        # Add optimization status to output
        print(f"\nProcessing completed in {'legacy' if not use_optimization else 'optimized'} mode")
        if args.tournament_id:
            print(f"Tournament context: {args.tournament_id}")
    else:
        # Use tournament optimization system
        print("Advanced tournament optimization processing would be implemented here")
        # Implementation would go here when fully integrated

if __name__ == "__main__":
    if "PYTHONPATH" not in os.environ:
        print("\u26A0\uFE0F Warning: PYTHONPATH is not set. Set it to your project root to avoid import errors.")
    main()
