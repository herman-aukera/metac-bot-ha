# forecast_agent.py
# Implements ForecastAgent using LangChain's Runnable interface

from langchain_core.runnables import Runnable
from src.agents.chains.forecast_chain import ForecastChain

class ForecastAgent(Runnable):
    def __init__(self):
        self.chain = ForecastChain()

    def invoke(self, question_json):
        """
        Given a question JSON, fetch evidence, build reasoning chain, and return forecast object.
        Returns:
            dict: {
                'question_id': ...,
                'forecast': float,
                'justification': str
            }
        """
        return self.chain.run(question_json)
