from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from src.vectorstore import get_vectorstore


class AgentState(TypedDict):
    question: str
    context: str
    answer: str


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def retrieve(state: AgentState):

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    documents = retriever.invoke(
        state["question"]
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    return {
        "context": context
    }


def generate_answer(state: AgentState):

    prompt = f"""
You are a document research assistant.

Use ONLY the following context.

Context:
{state["context"]}

Question:
{state["question"]}

If the answer is not available,
say:

"I could not find this information in the document."

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def create_agent():

    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()