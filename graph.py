from typing import TypedDict, Optional
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph

# Define input/output schema
class GraphState(TypedDict):
    transcript: str
    result: Optional[str]

# Load GPT-4o model
llm = ChatOpenAI(model="gpt-4o")

# Processing logic
def parse_transcript(state: GraphState) -> GraphState:
    prompt = f"""
You are an AI assistant that summarizes meeting transcripts.

Extract these:
1. tasks: list of {{"description": "...", "owner": "...", "due": "..."}}
2. people: list of names
3. project_updates: list of progress or blockers

Return this structure:
{{
  "tasks": [...],
  "people": [...],
  "project_updates": [...]
}}

Transcript:
{state['transcript']}
"""
    response = llm.invoke(prompt)
    return {
        "transcript": state["transcript"],
        "result": response.content
    }

# Build the LangGraph flow
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("parse_transcript", RunnableLambda(parse_transcript))
    graph.set_entry_point("parse_transcript")
    graph.set_finish_point("parse_transcript")
    return graph.compile()
