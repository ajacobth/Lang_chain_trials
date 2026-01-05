import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import os, json
from typing import Literal, Optional
from pydantic import BaseModel, Field

os.environ["LANGCHAIN_TRACING_V2"] = "false"  # disable LangSmith warning

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- Tool (normal Python function) ---
def add(a: int, b: int) -> int:
    return a + b

# --- Structured decision schema ---
class ToolDecision(BaseModel):
    action: Literal["call_tool", "final"] = Field(..., description="Whether to call a tool or answer directly.")
    tool_name: Optional[Literal["add"]] = None
    a: Optional[int] = None
    b: Optional[int] = None
    final_answer: Optional[str] = None

llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)

# Step 1: Decide whether to call a tool, and with what args
decide_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a tool-using assistant.\n"
     "If a tool is needed, choose action='call_tool' and provide tool_name and args.\n"
     "If no tool is needed, choose action='final' and provide final_answer.\n"
     "Only use the tool add(a:int,b:int) for arithmetic.\n"),
    ("human", "{question}")
])

decider = decide_prompt | llm.with_structured_output(ToolDecision)

# Step 2: Final answer prompt (after tool observation)
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are concise. Use the observation to answer."),
    ("human", "Question: {question}\nObservation: {observation}\nAnswer:")
])

def run_agent(question: str) -> str:
    decision: ToolDecision = decider.invoke({"question": question})

    if decision.action == "final":
        return decision.final_answer or ""

    if decision.action == "call_tool":
        if decision.tool_name != "add" or decision.a is None or decision.b is None:
            raise ValueError(f"Bad tool decision: {decision}")

        obs = add(decision.a, decision.b)
        return (final_prompt | llm).invoke({"question": question, "observation": obs}).content

    raise ValueError(f"Unknown action: {decision.action}")

print(run_agent("What is 123 + 456? Use a tool if needed."))

