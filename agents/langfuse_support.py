"""Langfuse support agent for searching GitHub Discussions."""

import os
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.base import BaseAgent

GITHUB_API_URL = "https://api.github.com/graphql"
LANGFUSE_REPO_OWNER = "langfuse"
LANGFUSE_REPO_NAME = "langfuse"
SUPPORT_CATEGORY_SLUG = "support"


def _search_github_discussions(
    query: str, category_slug: str = SUPPORT_CATEGORY_SLUG, first: int = 10
) -> dict:
    """Search GitHub Discussions using the GraphQL API.

    Args:
        query: The search query.
        category_slug: Discussion category to filter by.
        first: Number of results to return.

    Returns:
        Dictionary containing discussion search results.
    """
    # Build the search query for GitHub Discussions
    search_query = f"repo:{LANGFUSE_REPO_OWNER}/{LANGFUSE_REPO_NAME} category:{category_slug} {query}"

    graphql_query = """
    query SearchDiscussions($query: String!, $first: Int!) {
      search(query: $query, type: DISCUSSION, first: $first) {
        discussionCount
        nodes {
          ... on Discussion {
            title
            url
            body
            createdAt
            author {
              login
            }
            category {
              name
            }
            comments(first: 5) {
              nodes {
                body
                author {
                  login
                }
                isAnswer
              }
            }
            answer {
              body
              author {
                login
              }
            }
          }
        }
      }
    }
    """

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            GITHUB_API_URL,
            json={
                "query": graphql_query,
                "variables": {"query": search_query, "first": first},
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            },
        )
        response.raise_for_status()
        return response.json()


def _format_discussion_results(data: dict) -> str:
    """Format GitHub Discussions search results into readable text.

    Args:
        data: Raw GraphQL response data.

    Returns:
        Formatted string with discussion information.
    """
    if "errors" in data:
        return f"GitHub API Error: {data['errors']}"

    search_data = data.get("data", {}).get("search", {})
    discussions = search_data.get("nodes", [])
    total_count = search_data.get("discussionCount", 0)

    if not discussions:
        return "No discussions found matching your query."

    output = [f"Found {total_count} discussion(s). Showing top results:\n"]

    for i, discussion in enumerate(discussions, 1):
        if not discussion:
            continue

        title = discussion.get("title", "Untitled")
        url = discussion.get("url", "")
        body = discussion.get("body", "")[:500]  # Truncate body
        author = discussion.get("author", {}).get("login", "Unknown")
        created_at = discussion.get("createdAt", "")[:10]  # Just date part

        output.append(f"## {i}. {title}")
        output.append(f"**URL:** {url}")
        output.append(f"**Author:** {author} | **Created:** {created_at}")
        output.append(f"**Question:**\n{body}...")

        # Include the accepted answer if available
        answer = discussion.get("answer")
        if answer:
            answer_body = answer.get("body", "")[:500]
            answer_author = answer.get("author", {}).get("login", "Unknown")
            output.append(
                f"\n**Accepted Answer** (by {answer_author}):\n{answer_body}..."
            )
        else:
            # Include top comments if no accepted answer
            comments = discussion.get("comments", {}).get("nodes", [])
            if comments:
                output.append("\n**Top Comments:**")
                for comment in comments[:2]:
                    if comment:
                        comment_body = comment.get("body", "")[:300]
                        comment_author = comment.get("author", {}).get(
                            "login", "Unknown"
                        )
                        is_answer = comment.get("isAnswer", False)
                        marker = " [ANSWER]" if is_answer else ""
                        output.append(
                            f"- **{comment_author}**{marker}: {comment_body}..."
                        )

        output.append("\n---\n")

    return "\n".join(output)


@tool
def search_langfuse_support(query: str) -> str:
    """Search Langfuse GitHub Discussions in the Support category.

    Use this to find answers to questions that other users have asked
    about Langfuse, including troubleshooting, setup help, and best practices.

    Args:
        query: The search query about Langfuse issues or questions.
    """
    try:
        data = _search_github_discussions(query, first=5)
        return _format_discussion_results(data)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "GitHub API requires authentication for this request. Results may be limited."
        return f"HTTP Error searching discussions: {e.response.status_code}"
    except Exception as e:
        return f"Error searching support discussions: {str(e)}"


@tool
def search_langfuse_support_detailed(query: str) -> str:
    """Search Langfuse GitHub Discussions with more results.

    Returns more detailed results (up to 10 discussions).
    Use this when you need comprehensive information or the initial
    search didn't find what you were looking for.

    Args:
        query: The search query about Langfuse issues or questions.
    """
    try:
        data = _search_github_discussions(query, first=10)
        return _format_discussion_results(data)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "GitHub API requires authentication for this request. Results may be limited."
        return f"HTTP Error searching discussions: {e.response.status_code}"
    except Exception as e:
        return f"Error searching support discussions: {str(e)}"


class LangfuseSupportAgent(BaseAgent):
    """Agent that searches Langfuse GitHub Discussions for support answers."""

    def __init__(self, llm: BaseChatModel, **kwargs: Any):
        """Initialize the Langfuse support agent.

        Args:
            llm: The language model to use.
            **kwargs: Additional configuration.
        """
        super().__init__(llm, **kwargs)
        self._setup_agent()

    def _setup_agent(self) -> None:
        """Set up the ReAct agent with GitHub Discussion search tools."""
        self.tools = [
            search_langfuse_support,
            search_langfuse_support_detailed,
        ]
        self.agent = create_react_agent(self.llm, self.tools)

    @property
    def name(self) -> str:
        return "Langfuse Support Agent"

    @property
    def description(self) -> str:
        return "Searches Langfuse GitHub Discussions for community support answers"

    def run(self, query: str) -> str:
        """Search Langfuse GitHub Discussions for answers.

        Args:
            query: The user's question about Langfuse.

        Returns:
            The agent's response based on discussion search results.
        """
        system_prompt = """You are a helpful assistant that finds answers to Langfuse questions
by searching community discussions on GitHub.

You have access to tools that search the Langfuse GitHub Discussions Support category:
- search_langfuse_support: Quick search for relevant discussions (5 results)
- search_langfuse_support_detailed: More comprehensive search (10 results)

Your goal is to:
1. Search for discussions relevant to the user's question
2. Summarize the most helpful answers and solutions found
3. Provide links to relevant discussions so users can read more
4. If no relevant discussions are found, suggest the user create a new discussion

Always cite the discussion URLs when referencing solutions.
If the search doesn't return useful results, try rephrasing the query."""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]

        result = self.agent.invoke({"messages": messages})

        if result.get("messages"):
            return result["messages"][-1].content
        return "Unable to process the query."
