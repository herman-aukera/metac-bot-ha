# forecast_agent.py
# Implements ForecastAgent using LangChain's Runnable interface

from src.agents.llm import MockLLM
from src.agents.search import SearchTool
from src.agents.chains.forecast_chain import ForecastChain
from src.agents.tools import tool_list

class ForecastAgent:
    def __init__(self, llm=None, search_tool=None, tools=None):
        llm = llm or MockLLM()
        search_tool = search_tool or SearchTool()
        self.chain = ForecastChain(llm=llm, search_tool=search_tool, tools=tools or tool_list)

    def invoke(self, question_json, submission_response=None):
        """
        Given a question JSON (binary or multi-choice), fetch evidence, build reasoning chain, and return forecast object.
        If submission_response is provided, call plugin post_submit hooks.
        """
        result = self.chain.run(question_json)
        if submission_response is not None:
            self.chain.post_submit_plugins(submission_response)
        return result
