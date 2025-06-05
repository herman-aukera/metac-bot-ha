# test_forecast_chain.py
# Integration tests for ForecastChain with WikipediaTool and MathTool
import pytest
from unittest.mock import patch
from src.agents.chains.forecast_chain import ForecastChain
from src.agents.tools import WikipediaTool, MathTool
import types

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
    assert '6' in result['justification'] or '6' in result.get('evidence', '')

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

def test_trace_logging():
    class DummySearch:
        def search(self, query):
            return f"[DummySearch] Evidence for: {query}"
    chain = ForecastChain(llm=DummyLLM(), search_tool=DummySearch(), tools=[WikipediaTool(), MathTool()])
    question = {'question_id': 10, 'question_text': 'Quantum computing and 2 + 2 * 2'}
    result = chain.run(question)
    trace = result.get('trace', [])
    assert isinstance(trace, list)
    assert any(step['type'] == 'tool' and step['input'].get('tool') == 'WikipediaTool' for step in trace)
    assert any(step['type'] == 'tool' and step['input'].get('tool') == 'MathTool' for step in trace)
    assert any(step['type'] == 'llm' for step in trace)
    # Check that step-by-step trace is in order
    assert trace[0]['type'] == 'input'
    assert trace[-1]['type'] == 'llm' or trace[-1]['type'] == 'prompt'
