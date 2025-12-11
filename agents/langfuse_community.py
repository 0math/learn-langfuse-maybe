"""Langfuse community agent for searching Reddit and StackOverflow."""

import gzip
import io
from typing import Any
from urllib.parse import quote_plus

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.base import BaseAgent

# Reddit subreddits to search for Langfuse self-hosted content
REDDIT_SUBREDDITS = ["selfhosted", "LangChain", "LocalLLaMA"]

# StackExchange API endpoint
STACKEXCHANGE_API_URL = "https://api.stackexchange.com/2.3"


def _search_reddit_subreddit(subreddit: str, query: str, limit: int = 5) -> list[dict]:
    """Search a single Reddit subreddit using the JSON API.

    Args:
        subreddit: The subreddit name (without r/).
        query: The search query.
        limit: Maximum number of results.

    Returns:
        List of post dictionaries.
    """
    # Add langfuse to query if not already present
    search_query = query if "langfuse" in query.lower() else f"langfuse {query}"
    encoded_query = quote_plus(search_query)

    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": search_query,
        "restrict_sr": "1",
        "limit": str(limit),
        "sort": "relevance",
    }

    headers = {
        "User-Agent": "LangfuseBot/1.0 (Learning Project)",
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

    posts = []
    for child in data.get("data", {}).get("children", []):
        post_data = child.get("data", {})
        posts.append(
            {
                "title": post_data.get("title", ""),
                "url": f"https://reddit.com{post_data.get('permalink', '')}",
                "selftext": post_data.get("selftext", "")[:500],
                "author": post_data.get("author", "Unknown"),
                "subreddit": subreddit,
                "score": post_data.get("score", 0),
                "num_comments": post_data.get("num_comments", 0),
                "created_utc": post_data.get("created_utc", 0),
            }
        )

    return posts


def _format_reddit_results(posts: list[dict]) -> str:
    """Format Reddit search results into readable text.

    Args:
        posts: List of post dictionaries.

    Returns:
        Formatted string with post information.
    """
    if not posts:
        return "No Reddit posts found matching your query."

    output = [f"Found {len(posts)} Reddit post(s):\n"]

    for i, post in enumerate(posts, 1):
        output.append(f"## {i}. {post['title']}")
        output.append(
            f"**Subreddit:** r/{post['subreddit']} | **Score:** {post['score']} | **Comments:** {post['num_comments']}"
        )
        output.append(f"**URL:** {post['url']}")
        output.append(f"**Author:** u/{post['author']}")
        if post["selftext"]:
            output.append(f"**Content:**\n{post['selftext']}...")
        output.append("\n---\n")

    return "\n".join(output)


def _search_stackoverflow(query: str, limit: int = 5) -> dict:
    """Search StackOverflow for Langfuse-related questions.

    Args:
        query: The search query.
        limit: Maximum number of results.

    Returns:
        Dictionary containing search results.
    """
    # Add langfuse tag to the search
    search_query = query if "langfuse" in query.lower() else f"langfuse {query}"

    url = f"{STACKEXCHANGE_API_URL}/search/advanced"
    params = {
        "order": "desc",
        "sort": "relevance",
        "q": search_query,
        "tagged": "langfuse",
        "site": "stackoverflow",
        "pagesize": str(limit),
        "filter": "withbody",
    }

    headers = {
        "Accept-Encoding": "gzip",
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params, headers=headers)
        response.raise_for_status()

        # Handle gzip-encoded response
        if response.headers.get("Content-Encoding") == "gzip":
            decompressed = gzip.decompress(response.content)
            import json

            return json.loads(decompressed)
        return response.json()


def _format_stackoverflow_results(data: dict) -> str:
    """Format StackOverflow search results into readable text.

    Args:
        data: Raw API response data.

    Returns:
        Formatted string with question information.
    """
    items = data.get("items", [])

    if not items:
        return "No StackOverflow questions found matching your query."

    output = [f"Found {len(items)} StackOverflow question(s):\n"]

    for i, item in enumerate(items, 1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        body = item.get("body", "")[:500] if item.get("body") else ""
        owner = item.get("owner", {}).get("display_name", "Unknown")
        score = item.get("score", 0)
        answer_count = item.get("answer_count", 0)
        is_answered = item.get("is_answered", False)
        tags = ", ".join(item.get("tags", []))

        answered_marker = " [ANSWERED]" if is_answered else ""

        output.append(f"## {i}. {title}{answered_marker}")
        output.append(f"**Score:** {score} | **Answers:** {answer_count}")
        output.append(f"**URL:** {link}")
        output.append(f"**Author:** {owner}")
        output.append(f"**Tags:** {tags}")
        if body:
            # Strip HTML tags for cleaner output
            import re

            clean_body = re.sub(r"<[^>]+>", "", body)
            output.append(f"**Question:**\n{clean_body}...")
        output.append("\n---\n")

    return "\n".join(output)


@tool
def search_reddit(query: str) -> str:
    """Search Reddit for Langfuse self-hosted discussions.

    Searches across r/selfhosted, r/LangChain, and r/LocalLLaMA subreddits
    for discussions related to Langfuse deployment and configuration.

    Use this for:
    - Self-hosted deployment experiences
    - Docker/Kubernetes configuration tips
    - Community workarounds and solutions
    - Real-world deployment patterns

    Args:
        query: The search query about Langfuse self-hosting.
    """
    try:
        all_posts = []
        for subreddit in REDDIT_SUBREDDITS:
            try:
                posts = _search_reddit_subreddit(subreddit, query, limit=3)
                all_posts.extend(posts)
            except Exception:
                # Continue with other subreddits if one fails
                continue

        # Sort by score and take top results
        all_posts.sort(key=lambda x: x["score"], reverse=True)
        return _format_reddit_results(all_posts[:10])
    except Exception as e:
        return f"Error searching Reddit: {str(e)}"


@tool
def search_stackoverflow(query: str) -> str:
    """Search StackOverflow for Langfuse-related questions.

    Searches for questions tagged with 'langfuse' on StackOverflow.

    Use this for:
    - Technical implementation questions
    - Error troubleshooting
    - Integration issues
    - Code examples and solutions

    Args:
        query: The search query about Langfuse issues.
    """
    try:
        data = _search_stackoverflow(query, limit=5)
        return _format_stackoverflow_results(data)
    except httpx.HTTPStatusError as e:
        return f"StackOverflow API error: {e.response.status_code}"
    except Exception as e:
        return f"Error searching StackOverflow: {str(e)}"


class LangfuseCommunityAgent(BaseAgent):
    """Agent that searches Reddit and StackOverflow for Langfuse community answers.

    This agent is specifically designed for self-hosted Langfuse questions
    and should be used when official docs and GitHub discussions don't have answers.
    """

    def __init__(self, llm: BaseChatModel, **kwargs: Any):
        """Initialize the Langfuse community agent.

        Args:
            llm: The language model to use.
            **kwargs: Additional configuration.
        """
        super().__init__(llm, **kwargs)
        self._setup_agent()

    def _setup_agent(self) -> None:
        """Set up the ReAct agent with Reddit and StackOverflow search tools."""
        self.tools = [
            search_reddit,
            search_stackoverflow,
        ]
        self.agent = create_react_agent(self.llm, self.tools)
        self._callbacks = [self.langfuse_handler] if self.langfuse_handler else []

    @property
    def name(self) -> str:
        return "Langfuse Community Agent"

    @property
    def description(self) -> str:
        return "Searches Reddit and StackOverflow for Langfuse self-hosted solutions"

    def run(self, query: str) -> str:
        """Search community platforms for Langfuse self-hosted answers.

        Args:
            query: The user's question about Langfuse self-hosting.

        Returns:
            The agent's response based on community search results.
        """
        system_prompt = """You are a helpful assistant that finds answers to Langfuse self-hosted
deployment questions by searching community platforms.

You have access to:
- search_reddit: Searches r/selfhosted, r/LangChain, r/LocalLLaMA for relevant discussions
- search_stackoverflow: Searches StackOverflow for langfuse-tagged questions

**When to use each tool:**
- Use search_reddit for deployment experiences, Docker/Kubernetes configs, and community workarounds
- Use search_stackoverflow for technical implementation questions and error troubleshooting

**Your workflow:**
1. Search Reddit first for deployment and configuration discussions
2. Search StackOverflow for technical questions and error solutions
3. Synthesize findings from both sources
4. Provide clear, actionable answers with source links

**Important:**
- Always cite the source URLs when referencing solutions
- Note if a solution is from a specific deployment context (Docker, K8s, etc.)
- Mention if information might be outdated
- If no relevant results found, clearly state that"""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]

        config = {"callbacks": self._callbacks} if self._callbacks else {}
        result = self.agent.invoke({"messages": messages}, config=config)

        if result.get("messages"):
            return result["messages"][-1].content
        return "Unable to find community discussions for this query."
