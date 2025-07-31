# test_plugin_loader.py
import os
import tempfile
import textwrap
from src.agents.tools.plugin_loader import load_plugins, PluginTool, WebhookPluginTool

def test_load_plugins_from_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a simple plugin
        plugin_code = textwrap.dedent('''
            from src.agents.tools.plugin_loader import PluginTool
            class MyTestPlugin(PluginTool):
                name = "MyTestPlugin"
                description = "Test plugin."
                def invoke(self, input_str):
                    return "plugin_output:" + input_str
        ''')
        plugin_path = os.path.join(tmpdir, "my_plugin.py")
        with open(plugin_path, "w") as f:
            f.write(plugin_code)
        plugins = load_plugins(tmpdir)
        assert any(p.name == "MyTestPlugin" for p in plugins)
        assert plugins[0].invoke("foo").startswith("plugin_output:")

def test_webhook_plugin_tool(monkeypatch):
    class DummyResp:
        def __init__(self, code, data):
            self.status_code = code
            self._data = data
        def json(self):
            return self._data
        def raise_for_status(self):
            if self.status_code != 200:
                raise Exception("HTTP error")
    def fake_post(url, json, headers=None, timeout=None):
        return DummyResp(200, {"output": "webhook_result"})
    monkeypatch.setattr("requests.post", fake_post)
    tool = WebhookPluginTool(name="WebhookTest", url="http://fake")
    out = tool.invoke("bar")
    assert "webhook_result" in out

def test_plugin_loader_handles_errors():
    # Should not raise even if plugin is broken
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "bad.py"), "w") as f:
            f.write("raise Exception('fail')\n")
        plugins = load_plugins(tmpdir)
        assert isinstance(plugins, list)

def test_load_webhook_plugin_from_json(monkeypatch):
    import tempfile
    import json as pyjson
    # Mock requests.post
    class DummyResp:
        def __init__(self, code, data):
            self.status_code = code
            self._data = data
            self.headers = {}
        def json(self):
            return self._data
        def raise_for_status(self):
            if self.status_code != 200:
                raise Exception("HTTP error")
    def fake_post(url, json, headers, timeout):
        return DummyResp(200, {"output": "webhook_json_result"})
    monkeypatch.setattr("requests.post", fake_post)
    with tempfile.TemporaryDirectory() as tmpdir:
        webhook_def = {
            "name": "WebhookFromJson",
            "url": "http://fake/hook",
            "description": "Test webhook plugin"
        }
        with open(os.path.join(tmpdir, "webhook.json"), "w") as f:
            pyjson.dump(webhook_def, f)
        plugins = load_plugins(tmpdir)
        found = [p for p in plugins if getattr(p, 'name', None) == "WebhookFromJson"]
        assert found, "Webhook plugin from JSON should be loaded"
        assert found[0].invoke("foo").find("webhook_json_result") != -1

def test_webhook_plugin_http_error(monkeypatch):
    class DummyResp:
        def __init__(self, code):
            self.status_code = code
            self.headers = {}
        def json(self):
            return {"output": "fail"}
        def raise_for_status(self):
            raise Exception("HTTP error")
    def fake_post(url, json, headers, timeout):
        return DummyResp(500)
    monkeypatch.setattr("requests.post", fake_post)
    tool = WebhookPluginTool(name="WebhookTest", url="http://fake")
    out = tool.invoke("bar")
    assert "Exception" in out or "error" in out.lower()

def test_plugin_loader_bad_json():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "bad.json"), "w") as f:
            f.write("not a json")
        plugins = load_plugins(tmpdir)
        assert isinstance(plugins, list)

class PreOnlyPlugin(PluginTool):
    name = "PreOnlyPlugin"
    def pre_forecast(self, q):
        return {**q, "tag": "pre"}

class PostOnlyPlugin(PluginTool):
    name = "PostOnlyPlugin"
    def post_submit(self, resp):
        return f"post:{resp.get('status','none')}"

class BothHooksPlugin(PluginTool):
    name = "BothHooksPlugin"
    def pre_forecast(self, q):
        return {**q, "tag": "both-pre"}
    def post_submit(self, resp):
        return f"both-post:{resp.get('status','none')}"

def test_pre_forecast_hook():
    from src.agents.chains.forecast_chain import ForecastChain
    class DummySearch:
        def search(self, q):
            return "dummy_evidence"
    class DummyLLM:
        def invoke(self, x):
            return {"forecast": 0.5, "justification": "stub"}
    chain = ForecastChain(llm=DummyLLM(), search_tool=DummySearch(), tools=[PreOnlyPlugin()])
    q = {"question_id": 1, "question_text": "foo"}
    result = chain.run(q)
    pre_traces = [t for t in result["trace"] if t["type"] == "plugin_pre_forecast"]
    assert pre_traces, "No plugin_pre_forecast trace found"
    assert pre_traces[0]["output"] and pre_traces[0]["output"].get("tag") == "pre"

def test_post_submit_hook():
    from src.agents.chains.forecast_chain import ForecastChain
    class DummySearch:
        def search(self, q):
            return "dummy_evidence"
    class DummyLLM:
        def invoke(self, x):
            return {"forecast": 0.5, "justification": "stub"}
    chain = ForecastChain(llm=DummyLLM(), search_tool=DummySearch(), tools=[PostOnlyPlugin()])
    chain.run({"question_id": 1, "question_text": "foo"})
    chain.post_submit_plugins({"status": "ok"})
    # Should log post_submit
    assert any(t["type"] == "plugin_post_submit" for t in chain.trace)

def test_both_hooks():
    from src.agents.chains.forecast_chain import ForecastChain
    class DummySearch:
        def search(self, q):
            return "dummy_evidence"
    class DummyLLM:
        def invoke(self, x):
            return {"forecast": 0.5, "justification": "stub"}
    chain = ForecastChain(llm=DummyLLM(), search_tool=DummySearch(), tools=[BothHooksPlugin()])
    q = {"question_id": 1, "question_text": "foo"}
    result = chain.run(q)
    chain.post_submit_plugins({"status": "ok"})
    types = [t["type"] for t in chain.trace]
    assert "plugin_pre_forecast" in types and "plugin_post_submit" in types
