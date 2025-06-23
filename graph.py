from __future__ import annotations
"""LangGraph agent – project‑aware extractor (no memory layer yet)

API contract (unchanged):
POST /invoke { "transcript": str }
Response ⇒ { "projects": [...], "people": [...], "meeting_summary": str }
"""

from typing import TypedDict, Optional, List, Dict, Literal
from json import loads

from pydantic import BaseModel, Field, ValidationError
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph

# ────────────────────────────────────────────────────────────────────────────────
# 1. LangGraph state definition
# -----------------------------------------------------------------------------
class GraphState(TypedDict):
    transcript: str               # raw text input (required)
    parsed: Optional[dict]        # validated extraction
    summary: Optional[str]        # overall recap

# ────────────────────────────────────────────────────────────────────────────────
# 2. Pydantic schema for extraction
# -----------------------------------------------------------------------------
class ProjectBlock(BaseModel):
    name: str
    summary: str
    open_tasks: List[Dict[str, str]] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)

class Extracted(BaseModel):
    projects: List[ProjectBlock]
    people: List[str]

# ────────────────────────────────────────────────────────────────────────────────
# 3. LLM setup and prompts
# -----------------------------------------------------------------------------
LLM = ChatOpenAI(model="gpt-4o")

EXTRACT_PROMPT = """
You are a meeting‑analysis agent.

Goal: Identify each project discussed and output structured JSON.

Return ONLY valid JSON (no markdown) matching this exact schema:
{"projects":[{"name":"","summary":"","open_tasks":[],"blockers":[]}],"people":[],"meeting_summary":""}

Transcript:\n{transcript}
"""

FALLBACK_PROMPT = """
You previously returned invalid JSON. Try again. Output strictly:
{"projects":[],"people":[],"meeting_summary":""}
Transcript:\n{transcript}
"""

# ────────────────────────────────────────────────────────────────────────────────
# 4. Node lambdas
# -----------------------------------------------------------------------------

def extract(state: GraphState) -> GraphState:
    resp = LLM.invoke(EXTRACT_PROMPT.format(transcript=state["transcript"]))
    return {**state, "parsed": resp.content}


def validate(state: GraphState) -> GraphState:
    raw = state.get("parsed")
    try:
        cleaned = str(raw).replace("```json", "").replace("```", "").strip()
        data = Extracted.model_validate(loads(cleaned)).model_dump()
        return {**state, "parsed": data}
    except (ValidationError, ValueError):
        return {**state, "parsed": None}


def retry(state: GraphState) -> GraphState:
    resp = LLM.invoke(FALLBACK_PROMPT.format(transcript=state["transcript"]))
    return {**state, "parsed": resp.content}


def summarise(state: GraphState) -> GraphState:
    recap = LLM.invoke(
        f"Provide a concise one‑paragraph summary of this meeting:\n{state['transcript']}"
    ).content
    return {**state, "summary": recap}

# ────────────────────────────────────────────────────────────────────────────────
# 5. Build LangGraph
# -----------------------------------------------------------------------------

graph = StateGraph(GraphState)

graph.add_node("extract", RunnableLambda(extract))
graph.add_node("validate", RunnableLambda(validate))
graph.add_node("retry", RunnableLambda(retry))
graph.add_node("summarise", RunnableLambda(summarise))

# flow
graph.set_entry_point("extract")
graph.add_edge("extract", "validate")

graph.add_conditional_edges(
    "validate",
    lambda s: "retry" if s.get("parsed") is None else "summarise",
    {"retry": "retry", "summarise": "summarise"}
)

# loop retry
graph.add_edge("retry", "validate")

# finish
graph.set_finish_point("summarise")

# compile for serve.py
app = graph.compile()
