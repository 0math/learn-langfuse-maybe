import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from pydantic import SecretStr

from agents import (
    LangfuseDocsAgent,
    LangfuseSupportAgent,
    clear_knowledge_base,
    get_knowledge_base_stats,
    index_discussions,
)
from agents.langfuse_support import fetch_all_support_discussions

load_dotenv()

st.title("Learn Langfuse, Maybe")

# Get API key from environment or sidebar input
openai_api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input(
    "OpenAI API Key", type="password"
)

# Agent selection for future extensibility
agent_options = {
    "Langfuse Docs": "langfuse_docs",
    "Langfuse Support": "langfuse_support",
}
selected_agent = st.sidebar.selectbox("Select Agent", list(agent_options.keys()))

# Knowledge base record counter
st.sidebar.markdown("---")
kb_counter = st.sidebar.empty()


def update_kb_counter():
    """Update the knowledge base counter display."""
    try:
        stats = get_knowledge_base_stats()
        record_count = stats.get("total_documents", 0)
        kb_counter.markdown(f"**knowledge_base** - {record_count} records")
    except Exception:
        kb_counter.markdown("**knowledge_base** - 0 records")


# Initial counter display
update_kb_counter()

# Sync button
if st.sidebar.button("Sync Knowledge Base"):
    if not os.getenv("GITHUB_TOKEN"):
        st.sidebar.error("GITHUB_TOKEN not set")
    else:
        with st.sidebar.status("Syncing...", expanded=True) as status:
            st.write("Fetching discussions from GitHub...")
            discussions = fetch_all_support_discussions(max_discussions=100)
            st.write(f"Found {len(discussions)} discussions")

            st.write("Indexing documents...")

            def on_progress(current, total):
                update_kb_counter()

            indexed = index_discussions(discussions, on_progress=on_progress)
            if indexed == 0:
                status.update(
                    label="Already up to date - no new discussions", state="complete"
                )
            else:
                status.update(
                    label=f"Sync complete! Added {indexed} documents", state="complete"
                )
            update_kb_counter()

# Clear button
if st.sidebar.button("Clear Knowledge Base"):
    from agents.knowledge_base import DISCUSSIONS_COLLECTION, get_vector_store

    try:
        vector_store = get_vector_store()
        # Delete all documents from the collection
        collection = vector_store._collection
        # Get all IDs and delete them
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        st.sidebar.success("Knowledge base cleared")
        update_kb_counter()
    except Exception as e:
        st.sidebar.error(f"Error: {e}")


def get_agent(agent_type: str, api_key: str):
    """Get the appropriate agent based on selection."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=SecretStr(api_key),
    )

    if agent_type == "langfuse_docs":
        return LangfuseDocsAgent(llm=llm)
    if agent_type == "langfuse_support":
        return LangfuseSupportAgent(llm=llm)

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
