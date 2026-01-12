import numpy as np
#!/usr/bin/env python3
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# SciPy optimizers (recommended)
from scipy.optimize import minimize_scalar, minimize


# ---------------------------
# Safe 1D expression eval
# ---------------------------

_SAFE_MATH = {
    "pi": math.pi,
    "e": math.e,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "fabs": math.fabs,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
    "min": min,
    "max": max,
    "abs": abs,
}

_BLOCKLIST = [
    "__", "import", "open(", "exec", "eval", "os.", "sys.", "subprocess", "socket",
    "pickle", "marshal", "shutil", "pathlib", "glob", "inspect", "builtins",
]


def _check_expression_safe(expr: str) -> None:
    expr_l = expr.lower()
    for bad in _BLOCKLIST:
        if bad in expr_l:
            raise ValueError(f"Expression contains disallowed token: '{bad}'")


def _make_f(expr: str):
    """Return f(x)->float from safe expression in variable x."""
    _check_expression_safe(expr)
    code = compile(expr, "<user_expr>", "eval")

    safe_globals = {"__builtins__": {}}
    safe_globals.update(_SAFE_MATH)

    def f(x: float) -> float:
        val = eval(code, safe_globals, {"x": float(x)})
        val = float(val)
        if not np.isfinite(val):
            return float("inf")
        return val

    return f


# ---------------------------
# Tool: 1D optimizer
# ---------------------------

def _auto_algorithm(bounds: Optional[Tuple[float, float]], starts: Optional[List[float]]) -> str:
    """
    Heuristic:
    - If bounds provided: use bounded (golden/Brent bounded) via minimize_scalar(method='bounded')
    - Else:
        - If a start provided: use BFGS on 1D (minimize) as a local optimizer
        - Else: use brent (unbounded line search) via minimize_scalar(method='brent')
    """
    if bounds is not None:
        return "bounded"
    if starts:
        return "bfgs"
    return "brent"


@tool
def optimize_1d(
    expr: str,
    bounds: Optional[List[float]] = None,   # [lo, hi]
    goal: str = "min",                      # "min" or "max"
    starts: Optional[List[float]] = None,   # [x0, x1, ...]
    algorithm: Optional[str] = None,        # "bounded","brent","golden","bfgs"
    maxiter: int = 400
) -> Dict[str, Any]:
    """
    Optimize a 1D function f(x).

    Args:
      expr: string function of x, e.g. "(x-2)**2 + sin(x)"
      bounds: optional [lo, hi]. Strongly recommended.
      goal: "min" or "max"
      starts: optional starting points for BFGS (multi-start)
      algorithm: optional optimizer: "bounded","brent","golden","bfgs"
      maxiter: max iterations

    Returns dict with x_best, f_best, success, algorithm, message.
    """
    try:
        if not isinstance(expr, str) or not expr.strip():
            return {"error": "expr must be a non-empty string."}

        if goal not in ("min", "max"):
            return {"error": "goal must be 'min' or 'max'."}

        bounds_t: Optional[Tuple[float, float]] = None
        if bounds is not None:
            if not (isinstance(bounds, list) and len(bounds) == 2):
                return {"error": "bounds must be a list [lo, hi]."}
            lo, hi = float(bounds[0]), float(bounds[1])
            if not (lo < hi):
                return {"error": "bounds must satisfy lo < hi."}
            bounds_t = (lo, hi)

        starts_list: Optional[List[float]] = None
        if starts is not None:
            if not (isinstance(starts, list) and all(isinstance(v, (int, float)) for v in starts)):
                return {"error": "starts must be a list of numbers."}
            starts_list = [float(v) for v in starts]

        algo = algorithm.strip() if isinstance(algorithm, str) and algorithm.strip() else None
        if algo is None:
            algo = _auto_algorithm(bounds_t, starts_list)

        f_raw = _make_f(expr)

        # Convert max to min by negating
        def f(x: float) -> float:
            fx = f_raw(x)
            return -fx if goal == "max" else fx

        # 1) Bounded line search (best if bounds exist)
        if algo in ("bounded", "golden", "brent") and bounds_t is not None:
            method = "bounded" if algo == "bounded" else algo
            res = minimize_scalar(
                f,
                bounds=bounds_t,
                method=method,
                options={"maxiter": int(maxiter)},
            )
            x_best = float(res.x)
            return {
                "x_best": x_best,
                "f_best": float(f_raw(x_best)),
                "success": bool(res.success),
                "algorithm": f"{method} (minimize_scalar)",
                "message": str(res.message),
            }

        # 2) Unbounded line search
        if algo in ("brent", "golden") and bounds_t is None:
            res = minimize_scalar(f, method=algo, options={"maxiter": int(maxiter)})
            x_best = float(res.x)
            return {
                "x_best": x_best,
                "f_best": float(f_raw(x_best)),
                "success": bool(res.success),
                "algorithm": f"{algo} (minimize_scalar)",
                "message": str(res.message),
            }

        # 3) Multi-start local optimization (BFGS)
        if algo == "bfgs":
            if not starts_list:
                starts_list = [0.5 * (bounds_t[0] + bounds_t[1])] if bounds_t else [0.0]

            best_x, best_fx = None, float("inf")
            any_success = False
            msgs = []

            for x0 in starts_list:
                res = minimize(
                    lambda z: f(float(z[0])),
                    x0=np.array([x0], dtype=float),
                    method="BFGS",
                    options={"maxiter": int(maxiter)},
                )
                x = float(res.x[0])
                fx = float(f(x))
                msgs.append(str(res.message))
                any_success = any_success or bool(res.success)
                if fx < best_fx:
                    best_fx, best_x = fx, x

            return {
                "x_best": float(best_x),
                "f_best": float(f_raw(best_x)),
                "success": any_success,
                "algorithm": "BFGS (multi-start)" if len(starts_list) > 1 else "BFGS",
                "message": " | ".join(msgs[:2]) + (" | ..." if len(msgs) > 2 else ""),
            }

        return {"error": f"Unsupported algorithm '{algo}' for the given inputs."}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}



# ---------------------------
# LangGraph ReAct agent
# ---------------------------

SYSTEM = """You are a 1D optimization assistant.

When the user asks to optimize, you MUST do exactly:
1) Call optimize_1d ONCE with a JSON spec string.
2) After the tool returns, you MUST reply with a final answer in plain English and STOP.

Do NOT call optimize_1d more than once per user request.
Do NOT ask follow-up questions if the user provided bounds.
If bounds are missing, ask exactly one question: "What bounds [lo, hi] should I search over?"

Tool calling:
- Always call optimize_1d with a single argument named spec, which is a JSON string.
- The JSON must include: expr, bounds (if provided), and goal ("min" unless user asks to maximize).

Final answer format:
x_best = ...
f(x_best) = ...
(method = ...)

Formatting rules:
- Do NOT use LaTeX or math delimiters.
- Do NOT use \( \), \[ \], or $.
- Write all math in plain text.
"""

def main():
    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)

    agent = create_react_agent(
        model=llm,
        tools=[optimize_1d],
        #@debug=True,   # uncomment if you want verbose tracing
    )

    history = [SystemMessage(content=SYSTEM)]
    print("1D Optimization Agent (local). Type /exit to quit, /reset to clear.\n")

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in ("/exit", "exit", "quit"):
            break
        if user.lower() in ("/reset", "reset"):
            history = [history[0]]
            print("(history cleared)\n")
            continue

        history.append(HumanMessage(content=user))

        out = agent.invoke({"messages": history})
        msgs = out["messages"]

        # --- Find a human-readable thing to print ---
        text_to_print = None

        # 1) Prefer the most recent non-empty AIMessage content
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and getattr(m, "content", "").strip():
                text_to_print = m.content
                break

        # 2) If no assistant text, fall back to most recent ToolMessage content
        if text_to_print is None:
            for m in reversed(msgs):
                if isinstance(m, ToolMessage) and getattr(m, "content", "").strip():
                    text_to_print = f"(tool output)\n{m.content}"
                    break

        # 3) If still nothing, print message types for debugging
        if text_to_print is None:
            types = [type(m).__name__ for m in msgs]
            text_to_print = f"(no printable content)\nMessage types: {types}"

        print(f"AI: {text_to_print}\n")

        # Keep the updated messages as history for multi-turn
        history = msgs


if __name__ == "__main__":
    main()

