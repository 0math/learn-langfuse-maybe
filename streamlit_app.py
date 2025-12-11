import os
import time
import uuid
from typing import Optional

import httpx
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

# GitHub API configuration for fetching popular discussions
GITHUB_API_URL = "https://api.github.com/graphql"
LANGFUSE_REPO_OWNER = "langfuse"
LANGFUSE_REPO_NAME = "langfuse"

load_dotenv()

# Initialize Langfuse client for tracing
langfuse = get_client()


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_popular_discussions(limit: int = 3) -> list[dict]:
    """Fetch most popular GitHub discussions from the Support category.

    Sorts by comment count as a proxy for popularity/engagement.

    Args:
        limit: Number of discussions to return.

    Returns:
        List of discussion dicts with title, url, comment_count, and topic.
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        return []

    graphql_query = """
    query PopularDiscussions($owner: String!, $repo: String!, $first: Int!) {
      repository(owner: $owner, name: $repo) {
        discussions(
          first: $first,
          categoryId: null,
          orderBy: {field: UPDATED_AT, direction: DESC}
        ) {
          nodes {
            title
            url
            comments {
              totalCount
            }
            category {
              name
              slug
            }
            labels(first: 5) {
              nodes {
                name
              }
            }
          }
        }
      }
    }
    """

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                GITHUB_API_URL,
                json={
                    "query": graphql_query,
                    "variables": {
                        "owner": LANGFUSE_REPO_OWNER,
                        "repo": LANGFUSE_REPO_NAME,
                        "first": 50,  # Fetch more to filter and sort
                    },
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {github_token}",
                },
            )
            response.raise_for_status()
            data = response.json()

            discussions = (
                data.get("data", {})
                .get("repository", {})
                .get("discussions", {})
                .get("nodes", [])
            )

            # Filter to support category and sort by comment count
            support_discussions = [
                {
                    "title": d["title"],
                    "url": d["url"],
                    "comment_count": d["comments"]["totalCount"],
                    "labels": [l["name"] for l in d.get("labels", {}).get("nodes", [])],
                }
                for d in discussions
                if d and d.get("category", {}).get("slug") == "support"
            ]

            # Sort by comment count (most engaged first)
            support_discussions.sort(key=lambda x: x["comment_count"], reverse=True)

            return support_discussions[:limit]

    except Exception:
        return []


def get_popular_topics_summary() -> Optional[dict]:
    """Get a summary of popular discussion topics.

    Returns:
        Dict with total_count, topic_summary, and top_discussion, or None if unavailable.
    """
    discussions = fetch_popular_discussions(limit=3)

    if not discussions:
        return None

    total_comments = sum(d["comment_count"] for d in discussions)
    top_discussion = discussions[0] if discussions else None

    # Extract a common theme from titles (simplified)
    titles = [d["title"].lower() for d in discussions]
    common_topics = []
    keywords = [
        "tracing",
        "integration",
        "error",
        "setup",
        "sdk",
        "python",
        "langchain",
    ]
    for keyword in keywords:
        if any(keyword in title for title in titles):
            common_topics.append(keyword)

    topic_summary = common_topics[0] if common_topics else "common issues"

    return {
        "total_count": total_comments,
        "topic_summary": topic_summary,
        "top_discussion": top_discussion,
        "discussions": discussions,
    }


def render_empty_chat_state():
    """Render the empty chat state with statistics and helpful links."""
    st.markdown("### Welcome to LLM")
    st.markdown(
        "Ask me anything about Langfuse - I can search the official docs and community discussions"
    )

    st.markdown("---")

    # Popular discussions section
    popular = get_popular_topics_summary()
    if popular and popular["top_discussion"]:
        top = popular["top_discussion"]
        st.markdown("#### Popular in the Community")
        st.markdown(
            f"Users often ask about **{popular['topic_summary']}** "
            f"({popular['total_count']} comments across top discussions). "
            f"[See the most active discussion]({top['url']})"
        )

        with st.expander("View top discussions"):
            for i, d in enumerate(popular["discussions"], 1):
                st.markdown(
                    f"{i}. [{d['title']}]({d['url']}) ({d['comment_count']} comments)"
                )

    # Learn the basics section
    st.markdown("#### Learn the Basics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Langfuse Variables**")
        st.code(
            """# Environment variables
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com"
""",
            language="bash",
        )
        st.markdown("[Quick Start Guide](https://langfuse.com/docs/get-started)")

    with col2:
        st.markdown("**Prompt Management**")
        st.markdown(
            "Create, version, and deploy prompts independently from your application code."
        )
        st.markdown(
            "[Prompt Management Docs](https://langfuse.com/docs/prompts/get-started)"
        )


def get_session_id() -> str:
    """Get or create a session ID for the current Streamlit session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def get_user_id() -> str:
    """Get or create a user ID for the current Streamlit session.

    In a real application, this would come from authentication.
    For now, we generate a persistent ID per browser session.
    """
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"
    return st.session_state.user_id


st.title("Learn Langfuse Maybe (LLM)")

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

# Documentation links
st.sidebar.markdown(
    "[Why is this here?](https://github.com/0math/learn-langfuse-maybe/blob/main/architecture.md#part-2-chromadb-knowledge-base)"
)
st.sidebar.markdown(
    "[About Knowledge Bases](https://docs.trychroma.com/guides/build/intro-to-retrieval)"
)


def get_agent(api_key: str, session_id: str, user_id: str):
    """Get the controller agent that automatically routes to specialized agents.

    Args:
        api_key: OpenAI API key.
        session_id: Session ID for Langfuse tracing.
        user_id: User ID for Langfuse tracing.
    """
    # Create Langfuse callback handler for tracing LangChain/LangGraph calls
    langfuse_handler = LangfuseCallbackHandler()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=SecretStr(api_key),
        callbacks=[langfuse_handler],
    )

    return ControllerAgent(
        llm=llm,
        langfuse_handler=langfuse_handler,
        session_id=session_id,
        user_id=user_id,
    )


def generate_response(input_text: str, history: list[dict[str, str]]) -> tuple:
    """Generate response using the controller agent with streaming status.

    Args:
        input_text: The current user query.
        history: Conversation history (list of {"role": "user"|"assistant", "content": "..."}).

    Returns:
        Tuple of (agent's response, trace_id or None).
    """
    session_id = get_session_id()
    user_id = get_user_id()
    agent = get_agent(openai_api_key, session_id, user_id)

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

    return response or "No response received from the agent.", agent.last_trace_id


# Display chat history or empty state
if not st.session_state.messages:
    render_empty_chat_state()
else:
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
            response, trace_id = generate_response(
                prompt, history=st.session_state.messages
            )
            st.markdown(response)

        # Add both messages to history after generation (include trace_id in metadata)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append(
            {"role": "assistant", "content": response, "trace_id": trace_id}
        )

        # Update knowledge base counter (may have changed during sync_knowledge_base tool)
        update_kb_counter()
