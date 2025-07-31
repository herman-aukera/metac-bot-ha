# test_math.py
# Unit tests for MathTool
from src.agents.tools.math import MathTool, safe_eval

def test_mathtool_basic():
    tool = MathTool()
    assert tool.run("2+2") == "4"
    assert tool.run("sqrt(16)") == "4.0"
    assert tool.run("abs(-5)") == "5"
    assert tool.run("min(1,2,3)") == "1"
    assert tool.run("max(1,2,3)") == "3"

def test_mathtool_invalid():
    tool = MathTool()
    assert "Invalid expression" in tool.run(123)
    assert "Error" in tool.run("__import__('os')")
    assert "Error" in tool.run("exec('1+1')")
    assert "Error" in tool.run("open('file')")
    assert "Error" in tool.run("lambda x: x+1")

def test_safe_eval_edge_cases():
    assert "Error" in str(safe_eval("import os"))
    assert "Error" in str(safe_eval("os.system('ls')"))
    assert safe_eval("sum([1,2,3])") == 6
