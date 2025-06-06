# plugin_loader.py
"""
Dynamic plugin loader for ForecastChain tools.
- Loads PluginTool subclasses from plugins/ directory or specified path
- Supports both internal and external (webhook) tools
"""
import os
import importlib.util
from typing import List, Any, Optional
import sys
import json

class PluginTool:
    name: str = "PluginTool"
    description: str = ""
    def invoke(self, input_str: str) -> str:
        raise NotImplementedError
    # --- Lifecycle hooks ---
    def pre_forecast(self, question: dict) -> 'Optional[str]':
        """
        Optional lifecycle hook: called before reasoning. Can mutate, annotate, or log.
        Return value (if any) will be logged in the trace.
        """
        return None

    def post_submit(self, forecast: dict) -> 'Optional[str]':
        """
        Optional lifecycle hook: called after submission. Can log, alert, or trigger side effects.
        Return value (if any) will be logged in the trace.
        """
        return None

class WebhookPluginTool(PluginTool):
    def __init__(self, name, url, description="", headers=None, timeout=10):
        self.name = name
        self.url = url
        self.description = description
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
    def invoke(self, input_str: str) -> str:
        import requests
        try:
            resp = requests.post(self.url, json={"input": input_str}, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            try:
                return json.dumps(resp.json())
            except Exception:
                return resp.text
        except Exception as e:
            return f"[WebhookPluginTool] Exception: {e}"


def load_plugins(plugin_dir: str, enable_webhooks: bool = True) -> List[Any]:
    """
    Dynamically load PluginTool subclasses and webhook plugins from a directory.
    - Python plugins: .py files with PluginTool subclasses
    - Webhook plugins: .json files with {"name", "url", "description", ...}
    """
    plugins = []
    if not os.path.isdir(plugin_dir):
        return plugins
    sys.path.insert(0, plugin_dir)
    for fname in os.listdir(plugin_dir):
        fpath = os.path.join(plugin_dir, fname)
        if fname.endswith(".py") and not fname.startswith("_"):
            spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    if isinstance(obj, type) and issubclass(obj, PluginTool) and obj is not PluginTool:
                        plugins.append(obj())
            except Exception as e:
                print(f"[PluginLoader] Failed to load {fname}: {e}")
        elif enable_webhooks and fname.endswith(".json"):
            try:
                with open(fpath) as f:
                    data = json.load(f)
                name = data["name"]
                url = data["url"]
                desc = data.get("description", "Webhook plugin")
                headers = data.get("headers", None)
                timeout = data.get("timeout", 10)
                plugins.append(WebhookPluginTool(name, url, desc, headers, timeout))
            except Exception as e:
                print(f"[PluginLoader] Failed to load webhook {fname}: {e}")
    sys.path.pop(0)
    return plugins

# Plugin registry for ForecastChain and CLI
plugin_registry: List[PluginTool] = []

def register_plugins(plugin_dir: str, enable_webhooks: bool = True):
    """
    Loads plugins and sets the global plugin_registry.
    """
    global plugin_registry
    plugin_registry = load_plugins(plugin_dir, enable_webhooks)
    return plugin_registry

def get_registered_plugins() -> List[PluginTool]:
    return plugin_registry
