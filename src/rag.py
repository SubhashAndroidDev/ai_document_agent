from langchain_openai import ChatOpenAI

from src.vectorstore import get_vectorstore


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def ask_question(question):

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are an AI document research assistant.

Answer the question using ONLY the provided context.

If the answer is not present in the context,
say:

"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content