import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from pydantic import SecretStr

from agents import LangfuseDocsAgent

load_dotenv()

st.title("Learn Langfuse, Maybe")

# Get API key from environment or sidebar input
openai_api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input(
    "OpenAI API Key", type="password"
)

# Agent selection for future extensibility
agent_options = {
    "Langfuse Docs": "langfuse_docs",
}
selected_agent = st.sidebar.selectbox("Select Agent", list(agent_options.keys()))


def get_agent(agent_type: str, api_key: str):
    """Get the appropriate agent based on selection."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=SecretStr(api_key),
    )

    if agent_type == "langfuse_docs":
        return LangfuseDocsAgent(llm=llm)

    raise ValueError(f"Unknown agent type: {agent_type}")


def generate_response(input_text: str, agent_type: str):
    """Generate response using the selected agent."""
    agent = get_agent(agent_type, openai_api_key)
    with st.spinner(f"Querying {agent.name}..."):
        response = agent.run(input_text)
    st.markdown(response)


with st.form("my_form"):
    text = st.text_area(
        "Enter your question about Langfuse:",
        "What is Langfuse and how do I get started?",
    )
    submitted = st.form_submit_button("Submit")

    if submitted:
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key!", icon="⚠")
        else:
            generate_response(text, agent_options[selected_agent])
