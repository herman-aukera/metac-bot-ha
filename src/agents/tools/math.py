# math.py
"""
MathTool: Safe math evaluator for forecast agent.
- Evaluates numeric/logical expressions for quick calculations.
- Rejects unsafe input (no __import__, exec, etc).
- CI/offline safe: can be mocked in tests.
"""
import math
import ast

SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
SAFE_NAMES.update({"abs": abs, "min": min, "max": max, "sum": sum})
SAFE_OPS = {
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Mod, ast.FloorDiv, ast.BitXor, ast.BitOr, ast.BitAnd
}

def safe_eval(expr: str):
    """Safely evaluate a math expression string."""
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if not (isinstance(node.func, ast.Name) and node.func.id in SAFE_NAMES):
                    raise ValueError("Unsafe function call")
            elif isinstance(node, ast.Name):
                if node.id not in SAFE_NAMES:
                    raise ValueError(f"Unsafe name: {node.id}")
            elif type(node) in (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda, ast.FunctionDef, ast.ClassDef, ast.With, ast.AsyncWith, ast.Try, ast.While, ast.For, ast.If, ast.Delete, ast.Assign, ast.AugAssign, ast.Raise, ast.Yield, ast.YieldFrom, ast.Await, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp):
                raise ValueError("Unsafe statement")
            elif isinstance(node, ast.Expr) and type(node.value) not in SAFE_OPS | {ast.Num, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load}:
                raise ValueError("Unsafe expression")
        return eval(compile(tree, "<string>", mode="eval"), {"__builtins__": {}}, SAFE_NAMES)
    except Exception as e:
        return f"[MathTool] Error: {e}"

class MathTool:
    def run(self, expr: str) -> str:
        """Evaluate a math expression safely."""
        if not expr or not isinstance(expr, str):
            return "[MathTool] Invalid expression."
        return str(safe_eval(expr))
