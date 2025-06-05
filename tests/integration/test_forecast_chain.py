# test_forecast_chain.py
# Integration tests for ForecastChain with WikipediaTool and MathTool
import pytest
from unittest.mock import patch
from src.agents.chains.forecast_chain import ForecastChain
from src.agents.tools import WikipediaTool, MathTool

class DummyLLM:
    def invoke(self, input_dict):
        # Echo the prompt for inspection
        return {'forecast': 0.5, 'justification': input_dict['prompt']}

class DummySearch:
    def search(self, query):
        return f"[DummySearch] Evidence for: {query}"

def test_wikipedia_tool_integration():
    chain = ForecastChain(llm=DummyLLM(), search_tool=DummySearch(), tools=[WikipediaTool()])
    question = {'question_id': 1, 'question_text': 'Quantum computing'}
    result = chain.run(question)
    # WikipediaTool output should be in justification or evidence
    assert 'Wikipedia' in result['justification'] or 'Wikipedia' in result.get('evidence', '')

def test_math_tool_integration():
    chain = ForecastChain(llm=DummyLLM(), search_tool=DummySearch(), tools=[MathTool()])
    question = {'question_id': 2, 'question_text': '2 + 2 * 2'}
    result = chain.run(question)
    # MathTool output should be in justification or evidence
    assert '4' in result['justification'] or '4' in result.get('evidence', '')

def test_tool_fallback():
    class FailingWiki:
        def run(self, query):
            return ''
    chain = ForecastChain(llm=DummyLLM(), search_tool=DummySearch(), tools=[FailingWiki()])
    question = {'question_id': 3, 'question_text': 'Nonexistent'}
    result = chain.run(question)
    # Should still return a forecast even if tool fails
    assert 'forecast' in result
    assert 'justification' in result
