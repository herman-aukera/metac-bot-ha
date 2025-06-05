# agent_runner.py
# Core LangChain agent entrypoint for Metaculus forecasting agent

from langchain_core.runnables import Runnable
from src.agents.forecast_agent import ForecastAgent
from src.agents.metaculus_client import MetaculusClient

# Entrypoint for running the agent in CLI or CI

def run_agent(question_json, dryrun=True):
    agent = ForecastAgent()
    result = agent.invoke(question_json)
    if dryrun:
        print(result)
    else:
        # Submit forecast via Metaculus client
        client = MetaculusClient()
        client.submit_forecast(result['question_id'], result['forecast'], result['justification'])
    return result
