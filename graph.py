from typing import TypedDict, Optional
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

class GraphState(TypedDict):
    transcript: str
    result: Optional[str]

def extract_data(state: GraphState) -> GraphState:
    text = state["transcript"]
    return {"transcript": text, "result": f"Parsed from: {text[:50]}..."}

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("extractor", RunnableLambda(extract_data))
    graph.set_entry_point("extractor")
    graph.set_finish_point("extractor")
    return graph.compile()
