import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from src.vectorstore import get_vectorstore


# Load .env
load_dotenv()


# -----------------------------
# State
# -----------------------------

class AgentState(TypedDict):
    question: str
    context: str
    answer: str


# -----------------------------
# LLM
# -----------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)


# -----------------------------
# Retrieve documents
# -----------------------------

def retrieve(state: AgentState):

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    documents = retriever.invoke(
        state["question"]
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return {
        "context": context
    }


# -----------------------------
# Generate answer
# -----------------------------

def generate_answer(state: AgentState):

    prompt = f"""
You are an AI company policy assistant.

Answer the user's question using ONLY the context below.

Context:
{state["context"]}

Question:
{state["question"]}

If the answer is not available in the context,
say:

"I could not find this information in the company policy."

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# -----------------------------
# Create LangGraph Agent
# -----------------------------

def create_agent():

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node(
        "retrieve",
        retrieve
    )

    graph.add_node(
        "generate",
        generate_answer
    )

    # Workflow
    graph.add_edge(
        START,
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "generate"
    )

    graph.add_edge(
        "generate",
        END
    )

    # Compile
    return graph.compile()