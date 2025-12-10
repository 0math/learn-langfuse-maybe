import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from pydantic import SecretStr

from agents import (
    ControllerAgent,
    clear_knowledge_base,
    get_knowledge_base_stats,
    index_discussions,
)
from agents.langfuse_support import fetch_all_support_discussions

load_dotenv()

# Initialize Langfuse client for tracing
langfuse = get_client()

st.title("Learn Langfuse, Maybe")

# Get API key from environment or sidebar input
openai_api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input(
    "OpenAI API Key", type="password"
)

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
    try:
        if clear_knowledge_base():
            st.sidebar.success("Knowledge base cleared")
            update_kb_counter()
        else:
            st.sidebar.error("Failed to clear knowledge base")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")


def get_agent(api_key: str):
    """Get the controller agent that automatically routes to specialized agents."""
    # Create Langfuse callback handler for tracing LangChain/LangGraph calls
    langfuse_handler = LangfuseCallbackHandler()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=SecretStr(api_key),
        callbacks=[langfuse_handler],  # Trace all LLM calls
    )

    return ControllerAgent(llm=llm, langfuse_handler=langfuse_handler)


def generate_response(input_text: str):
    """Generate response using the controller agent with streaming status."""
    agent = get_agent(openai_api_key)

    # Create a status container for showing tool execution
    status_container = st.status("Processing...", expanded=True)

    tool_start_time = None
    current_tool = None
    response = None

    def update_status(tool_name: str, elapsed: float):
        """Update the status display with tool name and elapsed time."""
        status_container.update(label=f"Thinking... {tool_name}")

    with status_container:
        for item in agent.stream(input_text):
            if item.startswith("__FINAL__"):
                # Extract final response
                response = item[9:]  # Remove "__FINAL__" prefix
            else:
                # This is a tool name
                tool_name = item

                # If we were timing a previous tool, log it
                if current_tool and tool_start_time:
                    elapsed = time.time() - tool_start_time
                    st.write(f"Thought {elapsed:.0f}s {current_tool}")

                # Start timing the new tool
                current_tool = tool_name
                tool_start_time = time.time()
                update_status(tool_name, 0)

                # Update status periodically while tool is running
                # (The actual update happens on next iteration)

        # Log the last tool if there was one
        if current_tool and tool_start_time:
            elapsed = time.time() - tool_start_time
            st.write(f"Thought {elapsed:.0f}s {current_tool}")

    status_container.update(label="Complete", state="complete", expanded=False)

    # Display the final response
    if response:
        st.markdown(response)
    else:
        st.warning("No response received from the agent.")

    # Flush Langfuse events to ensure traces are sent
    langfuse.flush()


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
            generate_response(text)
