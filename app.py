import streamlit as st

from src.agent import create_agent


st.set_page_config(
    page_title="AI Document Agent",
    page_icon="🤖"
)


st.title("🤖 AI Document Research Agent")

st.write(
    "Ask questions about your PDF document."
)


question = st.text_input(
    "Enter your question:"
)


if st.button("Ask AI"):

    if not question:

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            agent = create_agent()

            result = agent.invoke({
                "question": question,
                "context": "",
                "answer": ""
            })

        st.subheader("Answer")

        st.write(
            result["answer"]
        )