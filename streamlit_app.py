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

st.title("Learn Langfuse Maybe")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get API keys from environment or sidebar input
openai_api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input(
    "OpenAI API Key", type="password"
)
github_token = os.getenv("GITHUB_TOKEN") or st.sidebar.text_input(
    "GitHub token", type="password"
)

# Knowledge base record counter
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


def generate_response(input_text: str, history: list[dict[str, str]]) -> str:
    """Generate response using the controller agent with streaming status.

    Args:
        input_text: The current user query.
        history: Conversation history (list of {"role": "user"|"assistant", "content": "..."}).

    Returns:
        The agent's response.
    """
    agent = get_agent(openai_api_key)

    # Create a status container for showing tool execution
    status_container = st.status("Processing...", expanded=True)

    tool_start_time = None
    current_tool = None
    response = None

    def update_status(tool_name: str):
        """Update the status display with tool name."""
        status_container.update(label=f"Thinking... {tool_name}")

    with status_container:
        for item in agent.stream(input_text, history=history):
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
                update_status(tool_name)

        # Log the last tool if there was one
        if current_tool and tool_start_time:
            elapsed = time.time() - tool_start_time
            st.write(f"Thought {elapsed:.0f}s {current_tool}")

    status_container.update(label="Complete", state="complete", expanded=False)

    # Flush Langfuse events to ensure traces are sent
    langfuse.flush()

    return response or "No response received from the agent."


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Langfuse..."):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.", icon="⚠")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response with conversation history (before adding current message)
        with st.chat_message("assistant"):
            response = generate_response(prompt, history=st.session_state.messages)
            st.markdown(response)

        # Add both messages to history after generation
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Update knowledge base counter (may have changed during sync_knowledge_base tool)
        update_kb_counter()
