from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import os
from vector_store import create_vector_store, query_vector_store, VECTOR_STORE_PATH
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

@tool
def rag_query(query: str) -> str:
    """Query the HR manual database for relevant information."""

    if not os.path.exists(VECTOR_STORE_PATH):
        create_vector_store()

    results = query_vector_store(query, top_k=3)
    
    if not results:
        return "No relevant information found in the HR manual."
    
    formatted_results = []
    for i, (chunk, score) in enumerate(results, 1):
        formatted_results.append(f"Relevant Section {i} (Score: {score:.3f}):\n{chunk}")
    
    return "\n\n".join(formatted_results)


def main():
    tools = [rag_query]
    print(tools)
    def assistant(state: MessagesState):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(state["messages"])
        print(response)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    agent = graph.compile()

    return agent
