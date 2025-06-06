# test_plugin_lifecycle.py
import pytest
from src.agents.chains.forecast_chain import ForecastChain
from src.agents.tools.plugin_loader import PluginTool

class DummySearch:
    def search(self, q):
        return "dummy_evidence"
class DummyLLM:
    def invoke(self, x):
        return {"forecast": 0.5, "justification": "stub"}

class PrePlugin(PluginTool):
    name = "PrePlugin"
    def pre_forecast(self, q):
        return "pre_hook_called"
class PostPlugin(PluginTool):
    name = "PostPlugin"
    def post_submit(self, resp):
        return "post_hook_called"
class BothPlugin(PluginTool):
    name = "BothPlugin"
    def pre_forecast(self, q):
        return "both_pre"
    def post_submit(self, resp):
        return "both_post"

def test_pre_hook_trace():
    chain = ForecastChain(llm=DummyLLM(), search_tool=DummySearch(), tools=[PrePlugin()])
    q = {"question_id": 1, "question_text": "foo"}
    result = chain.run(q)
    pre_traces = [t for t in result["trace"] if t["type"] == "plugin_pre_forecast"]
    assert pre_traces and pre_traces[0]["output"] == "pre_hook_called"

def test_post_hook_trace():
    chain = ForecastChain(llm=DummyLLM(), search_tool=DummySearch(), tools=[PostPlugin()])
    chain.run({"question_id": 1, "question_text": "foo"})
    chain.post_submit_plugins({"status": "ok"})
    post_traces = [t for t in chain.trace if t["type"] == "plugin_post_submit"]
    assert post_traces and post_traces[0]["output"] == "post_hook_called"

def test_both_hooks_trace():
    chain = ForecastChain(llm=DummyLLM(), search_tool=DummySearch(), tools=[BothPlugin()])
    q = {"question_id": 1, "question_text": "foo"}
    result = chain.run(q)
    chain.post_submit_plugins({"status": "ok"})
    pre = [t for t in chain.trace if t["type"] == "plugin_pre_forecast"]
    post = [t for t in chain.trace if t["type"] == "plugin_post_submit"]
    assert pre and pre[0]["output"] == "both_pre"
    assert post and post[0]["output"] == "both_post"
