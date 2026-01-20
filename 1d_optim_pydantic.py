#!/usr/bin/env python3
# 1d optimizer agent (LCEL + pydantic) that does NOT hallucinate bounds
# - only triggers optimize flow when intent keywords show up (min/max/optimize/etc)
# - if bounds missing -> asks once: "What bounds [lo, hi] should I search over?"
# - once bounds provided, it runs tool exactly once
# - explains "hit bounds" deterministically (no llm hallucination)

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
from pydantic import BaseModel, ValidationError
from scipy.optimize import minimize_scalar, minimize

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda


# ---------------------------
# safe math + eval guard
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
    # return f(x) from expr string, but keep it safe
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
# intent + bounds parsing
# ---------------------------

# only these words should trigger optimization mode
_OPT_INTENT_RE = re.compile(r"\b(min|minimize|max|maximize|optimi[sz]e|argmin|argmax)\b", re.I)

# parse [lo, hi] anywhere in the user text, like "[-1,1]" or "[ -1.2 , 3 ]"
_BOUNDS_RE = re.compile(
    r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]"
)

def is_opt_request(text: str) -> bool:
    return bool(_OPT_INTENT_RE.search(text))

def extract_bounds(text: str) -> Optional[Tuple[float, float]]:
    m = _BOUNDS_RE.search(text)
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2))
    if not (lo < hi):
        return None
    return (lo, hi)


# ---------------------------
# pydantic schema for tool
# ---------------------------

class Optimize1DInput(BaseModel):
    expr: str
    bounds: Tuple[float, float]  # required. no bounds, no tool call
    goal: Literal["min", "max"] = "min"
    starts: Optional[List[float]] = None
    algorithm: Optional[Literal["bounded", "brent", "golden", "bfgs"]] = None
    maxiter: int = 400


def _auto_algorithm(bounds: Optional[Tuple[float, float]], starts: Optional[List[float]]) -> str:
    if bounds is not None:
        return "bounded"
    if starts:
        return "bfgs"
    return "brent"


# ---------------------------
# tool: 1d optimizer
# ---------------------------

@tool(args_schema=Optimize1DInput)
def optimize_1d(
    expr: str,
    bounds: Tuple[float, float],
    goal: str = "min",
    starts: Optional[List[float]] = None,
    algorithm: Optional[str] = None,
    maxiter: int = 400,
) -> Dict[str, Any]:
    
    """
    1D optimizer tool.
    
    Inputs:
      - expr: string expression in variable x
      - bounds: (lo, hi) required
      - goal: "min" or "max"
      - starts/algorithm/maxiter: optional
    
    Output:
      dict with x_best, f_best, success, algorithm, message, bounds, goal, expr
    """
    # actual math part. no llm stuff here
    try:
        if not isinstance(expr, str) or not expr.strip():
            return {"error": "expr must be a non-empty string."}

        lo, hi = float(bounds[0]), float(bounds[1])
        if not (lo < hi):
            return {"error": "bounds must satisfy lo < hi."}
        bounds_t = (lo, hi)

        starts_list = [float(v) for v in starts] if starts else None

        algo = algorithm.strip() if isinstance(algorithm, str) and algorithm.strip() else None
        if algo is None:
            algo = _auto_algorithm(bounds_t, starts_list)

        f_raw = _make_f(expr)

        # convert max to min by negating
        def f(x: float) -> float:
            fx = f_raw(x)
            return -fx if goal == "max" else fx

        # bounded line search
        if algo in ("bounded", "golden", "brent"):
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
                "bounds": [lo, hi],
                "goal": goal,
                "expr": expr,
            }

        # multi-start local opt (if you ever want it)
        if algo == "bfgs":
            if not starts_list:
                starts_list = [0.5 * (lo + hi)]

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
                "bounds": [lo, hi],
                "goal": goal,
                "expr": expr,
            }

        return {"error": f"Unsupported algorithm '{algo}' for the given inputs."}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------
# lcel agent pieces
# ---------------------------

SYSTEM = """You are a 1D optimization assistant.

Rules:
- Only do optimization if user clearly asks to min/max/optimize (intent keywords).
- If user asks to optimize but bounds are missing, ask exactly one question:
  "What bounds [lo, hi] should I search over?"
- If bounds are provided, call optimize_1d exactly once.
- After tool returns, reply with:
  x_best = ...
  f(x_best) = ...
  (method = ...)
- Do NOT use LaTeX or math delimiters like \\( \\) or $.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("messages"),
])

# local slm
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
llm_with_tools = llm.bind_tools([optimize_1d])


def _boundary_note(bounds: Tuple[float, float], x_best: float) -> str:
    # deterministic "why bounds" message. dont let llm guess
    lo, hi = bounds
    # tolerance based on interval size
    span = max(1.0, abs(hi - lo))
    tol = 1e-5 * span

    if abs(x_best - lo) <= tol:
        return f"note: solution is at lower bound {lo}. likely true optimum is < {lo}."
    if abs(x_best - hi) <= tol:
        return f"note: solution is at upper bound {hi}. likely true optimum is > {hi}."
    return "note: solution is inside the bounds (not at an endpoint)."


def intent_gate(state: Dict[str, Any]) -> Dict[str, Any]:
    # always return dict, never AIMessage (this avoids your crash)
    msgs = state["messages"]
    last_user = next(m for m in reversed(msgs) if isinstance(m, HumanMessage)).content.strip()

    # basic session-y words
    if last_user.lower() in ("done", "stop", "exit", "quit"):
        return {
            "messages": msgs + [AIMessage(content="Got it — stopping. If you want another one, send: min/max + expr + bounds [lo, hi].")],
            "skip": True,
            "need_bounds": False,
        }

    # if user is not asking to optimize, dont start asking for bounds
    if not is_opt_request(last_user):
        return {
            "messages": msgs + [AIMessage(content="If you want to optimize: say min/max, the expression in x, and bounds like [lo, hi].")],
            "skip": True,
            "need_bounds": False,
        }

    # optimization intent exists -> check if bounds exist in this message
    b = extract_bounds(last_user)
    if b is None:
        # ask EXACTLY this question (no tool calls)
        return {
            "messages": msgs + [AIMessage(content="What bounds [lo, hi] should I search over?")],
            "skip": True,         # skip model/tool flow for this turn
            "need_bounds": True,  # remember we are waiting for bounds next
        }

    # bounds present, allow tool flow
    return {"messages": msgs, "skip": False, "need_bounds": False}


def bounds_followup_gate(state: Dict[str, Any]) -> Dict[str, Any]:
    # this handles the second turn: user replies with "[-1, 1]" etc
    msgs = state["messages"]

    # if we are not waiting for bounds, do nothing
    if not state.get("need_bounds"):
        return state

    last_user = next(m for m in reversed(msgs) if isinstance(m, HumanMessage)).content.strip()
    b = extract_bounds(last_user)
    if b is None:
        # still no bounds, ask once again same question
        return {
            "messages": msgs + [AIMessage(content="What bounds [lo, hi] should I search over?")],
            "skip": True,
            "need_bounds": True,
        }

    # ok bounds provided, now we need to run optimize on the PREVIOUS expr request
    # find the last human message that had opt intent (min/max/optimize) and keep that as the query
    opt_user = None
    for m in reversed(msgs):
        if isinstance(m, HumanMessage) and is_opt_request(m.content) and extract_bounds(m.content) is None:
            opt_user = m.content.strip()
            break

    if opt_user is None:
        # edge case: cant find prior request, just ask user to restate
        return {
            "messages": msgs + [AIMessage(content="Please restate the optimization request with min/max and the expression in x.")],
            "skip": True,
            "need_bounds": False,
        }

    # inject a synthetic message that includes bounds so model can tool-call cleanly
    # (this avoids the model inventing bounds)
    synth = f"{opt_user} bounds {list(b)}"
    return {
        "messages": msgs + [HumanMessage(content=synth)],
        "skip": False,
        "need_bounds": False,
    }


def _run_model_once(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("skip"):
        return state
    msgs = state["messages"]
    rendered = prompt.invoke({"messages": msgs})
    ai = llm_with_tools.invoke(rendered.to_messages())
    return {"messages": msgs + [ai], "skip": False, "need_bounds": False}


def _maybe_run_tool_once(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("skip"):
        return state

    msgs = state["messages"]
    last = msgs[-1]
    if not isinstance(last, AIMessage):
        return state

    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return state

    # enforce exactly 1 call
    tc = tool_calls[0]
    if tc.get("name") != "optimize_1d":
        return state

    # IMPORTANT: overwrite tool bounds with user-provided bounds (prevents hallucinated bounds)
    # we will take the last bounds we can parse from ANY human message
    b = None
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            b = extract_bounds(m.content)
            if b is not None:
                break

    args = dict(tc.get("args", {}))
    if b is not None:
        args["bounds"] = b  # clamp to what user actually gave

    try:
        tool_out = optimize_1d.invoke(args)
    except ValidationError as ve:
        tool_out = {"error": f"PydanticValidationError: {ve}"}
    except Exception as e:
        tool_out = {"error": f"{type(e).__name__}: {e}"}

    tool_msg = ToolMessage(
        content=json.dumps(tool_out),
        tool_call_id=tc.get("id", "tool_call"),
    )
    return {"messages": msgs + [tool_msg], "skip": False, "need_bounds": False}


def _finalize_answer(state: Dict[str, Any]) -> AIMessage:
    # last step returns AIMessage only
    msgs = state["messages"]

    # if we skipped, we already appended assistant reply in gate
    if state.get("skip"):
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and (m.content or "").strip():
                return m
        return AIMessage(content="(no response)")

    # find tool output
    tool_json = None
    for m in reversed(msgs):
        if isinstance(m, ToolMessage) and m.content.strip():
            tool_json = m.content
            break

    if tool_json is None:
        # model didnt call tool; just print its latest message
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and (m.content or "").strip():
                return m
        return AIMessage(content="(no response)")

    # parse tool output and build final message deterministically (no llm guessing here)
    try:
        out = json.loads(tool_json)
    except Exception:
        return AIMessage(content=f"(tool output)\n{tool_json}")

    if "error" in out:
        return AIMessage(content=f"error: {out['error']}")

    x_best = float(out["x_best"])
    f_best = float(out["f_best"])
    method = str(out.get("algorithm", "unknown"))
    bounds_list = out.get("bounds", None)

    extra = ""
    if bounds_list and isinstance(bounds_list, list) and len(bounds_list) == 2:
        extra = _boundary_note((float(bounds_list[0]), float(bounds_list[1])), x_best)

    text = (
        f"x_best = {x_best}\n"
        f"f(x_best) = {f_best}\n"
        f"(method = {method})"
    )
    if extra:
        text += f"\n{extra}"

    return AIMessage(content=text)


# chain order:
# 1) intent gate: block random chat from triggering bounds/tool stuff
# 2) if we were waiting for bounds, handle that (inject synth message with bounds)
# 3) run model once (tool call or not)
# 4) run tool once (if called), with bounds clamped to user bounds
# 5) finalize deterministically
chain = (
    RunnableLambda(intent_gate)
    | RunnableLambda(bounds_followup_gate)
    | RunnableLambda(_run_model_once)
    | RunnableLambda(_maybe_run_tool_once)
    | RunnableLambda(_finalize_answer)
)


def main():
    history: List[Any] = [SystemMessage(content=SYSTEM)]
    print("1D Optimization Agent (LCEL + Pydantic, local). Type /exit to quit, /reset to clear.\n")

    while True:
        user = input("You: ").strip()
        if not user:
            continue

        # manual cmds (keep these)
        if user.lower() in ("/exit", "exit", "quit"):
            break
        if user.lower() in ("/reset", "reset"):
            history = [history[0]]
            print("(history cleared)\n")
            continue

        history.append(HumanMessage(content=user))

        out = chain.invoke({"messages": history})

        if isinstance(out, AIMessage):
            print(f"AI: {out.content}\n")
            history.append(out)
        else:
            print(f"AI: {str(out)}\n")


if __name__ == "__main__":
    main()
