# forecast_agent.py
# Implements ForecastAgent using LangChain's Runnable interface

from langchain_core.runnables import Runnable
from src.agents.llm import MockLLM
from src.agents.search import SearchTool
from src.agents.chains.forecast_chain import ForecastChain

class ForecastAgent(Runnable):
    def __init__(self, llm=None, search_tool=None):
        llm = llm or MockLLM()
        search_tool = search_tool or SearchTool()
        self.chain = ForecastChain(llm=llm, search_tool=search_tool)

    def invoke(self, question_json):
        """
        Given a question JSON (binary or multi-choice), fetch evidence, build reasoning chain, and return forecast object.
        Returns:
            dict: {
                'question_id': ...,
                'forecast': float or list of floats,
                'justification': str
            }
        """
        return self.chain.run(question_json)
