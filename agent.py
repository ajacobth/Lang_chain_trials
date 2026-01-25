from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    input: str
    output: str

def node(state: State):
    return {"output": f"Echo: {state['input']}"}

builder = StateGraph(State)
builder.add_node("echo", node)
builder.set_entry_point("echo")
builder.set_finish_point("echo")

graph = builder.compile()
