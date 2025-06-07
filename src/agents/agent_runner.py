# agent_runner.py
# Core LangChain agent entrypoint for Metaculus forecasting agent

from src.agents.forecast_agent import ForecastAgent
from src.agents.metaculus_client import MetaculusClient
from src.agents.tools import tool_list

# Entrypoint for running the agent in CLI or CI

def run_agent(question_json, dryrun=True):
    agent = ForecastAgent()
    result = agent.invoke(question_json)
    result['tools_used'] = [tool.__class__.__name__ for tool in tool_list]
    if 'trace' in result:
        print("--- Reasoning Trace ---")
        print(result['trace'])
    if dryrun:
        print(result)
    else:
        # Submit forecast via Metaculus client
        client = MetaculusClient()
        client.submit_forecast(result['question_id'], result['forecast'], result['justification'])
    return result
