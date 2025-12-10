"""Langfuse support agent for searching GitHub Discussions."""

import os
from typing import Any, Optional

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.base import BaseAgent
from agents.knowledge_base import (
    get_knowledge_base_stats,
    index_discussions,
    search_knowledge_base,
)

GITHUB_API_URL = "https://api.github.com/graphql"
LANGFUSE_REPO_OWNER = "langfuse"
LANGFUSE_REPO_NAME = "langfuse"
SUPPORT_CATEGORY_SLUG = "support"


def _fetch_discussions_page(
    category_slug: str = SUPPORT_CATEGORY_SLUG,
    first: int = 50,
    after: Optional[str] = None,
) -> dict:
    """Fetch a page of GitHub Discussions using the GraphQL API.

    Args:
        category_slug: Discussion category to filter by.
        first: Number of results per page (max 100).
        after: Cursor for pagination.

    Returns:
        Dictionary containing discussion data and pagination info.
    """
    graphql_query = """
    query ListDiscussions($owner: String!, $repo: String!, $first: Int!, $after: String) {
      repository(owner: $owner, name: $repo) {
        discussions(first: $first, after: $after, orderBy: {field: UPDATED_AT, direction: DESC}) {
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            title
            url
            body
            createdAt
            author {
              login
            }
            category {
              name
              slug
            }
            comments(first: 10) {
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

    variables = {
        "owner": LANGFUSE_REPO_OWNER,
        "repo": LANGFUSE_REPO_NAME,
        "first": first,
    }
    if after:
        variables["after"] = after

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            GITHUB_API_URL,
            json={"query": graphql_query, "variables": variables},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            },
        )
        response.raise_for_status()
        return response.json()


def fetch_all_support_discussions(
    category_slug: str = SUPPORT_CATEGORY_SLUG,
    max_discussions: int = 200,
) -> list[dict]:
    """Fetch all discussions from a category, handling pagination.

    Args:
        category_slug: Discussion category to filter by.
        max_discussions: Maximum number of discussions to fetch.

    Returns:
        List of discussion dictionaries.
    """
    all_discussions = []
    after = None

    while len(all_discussions) < max_discussions:
        remaining = max_discussions - len(all_discussions)
        page_size = min(50, remaining)

        data = _fetch_discussions_page(
            category_slug=category_slug,
            first=page_size,
            after=after,
        )

        if "errors" in data:
            break

        repo_data = data.get("data", {}).get("repository", {})
        discussions_data = repo_data.get("discussions", {})
        discussions = discussions_data.get("nodes", [])

        # Filter by category slug
        for discussion in discussions:
            if discussion:
                disc_category = discussion.get("category", {}).get("slug", "")
                if disc_category == category_slug:
                    all_discussions.append(discussion)

        page_info = discussions_data.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break

        after = page_info.get("endCursor")

    return all_discussions


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


@tool
def search_knowledge_base_tool(query: str) -> str:
    """Search the local knowledge base for Langfuse support answers.

    This searches through previously indexed GitHub Discussions stored locally.
    Use this for fast semantic search when you need to find relevant discussions
    that have already been indexed.

    Args:
        query: The search query about Langfuse issues or questions.
    """
    try:
        docs = search_knowledge_base(query, k=5)

        if not docs:
            return "No relevant discussions found in the knowledge base. Try using search_langfuse_support to search GitHub directly, or sync the knowledge base first."

        output = [f"Found {len(docs)} relevant result(s) in knowledge base:\n"]

        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            title = metadata.get("title", "Untitled")
            url = metadata.get("url", "")
            doc_type = metadata.get("type", "unknown")
            author = metadata.get("author", "Unknown")

            output.append(f"## {i}. [{doc_type.upper()}] {title}")
            output.append(f"**URL:** {url}")
            output.append(f"**Author:** {author}")
            output.append(f"**Content:**\n{doc.page_content[:500]}...")
            output.append("\n---\n")

        return "\n".join(output)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@tool
def sync_knowledge_base(max_discussions: int = 100) -> str:
    """Sync GitHub Discussions to the local knowledge base.

    Fetches discussions from the Langfuse GitHub Support category and indexes
    them in ChromaDB for fast semantic search. Only new discussions are added.

    Args:
        max_discussions: Maximum number of discussions to fetch (default 100).
    """
    try:
        # Fetch discussions from GitHub
        discussions = fetch_all_support_discussions(max_discussions=max_discussions)

        if not discussions:
            return "No discussions found to index."

        # Index into ChromaDB
        indexed_count = index_discussions(discussions)

        # Get updated stats
        stats = get_knowledge_base_stats()

        return (
            f"Sync complete!\n"
            f"- Fetched {len(discussions)} discussions from GitHub\n"
            f"- Indexed {indexed_count} new documents\n"
            f"- Knowledge base now contains:\n"
            f"  - {stats.get('total_documents', 0)} total documents\n"
            f"  - {stats.get('unique_discussions', 0)} unique discussions\n"
            f"  - {stats.get('questions', 0)} questions\n"
            f"  - {stats.get('answers', 0)} answers"
        )
    except httpx.HTTPStatusError as e:
        return f"HTTP Error fetching discussions: {e.response.status_code}"
    except Exception as e:
        return f"Error syncing knowledge base: {str(e)}"


@tool
def get_knowledge_base_status() -> str:
    """Get the current status of the knowledge base.

    Returns statistics about the indexed discussions including total documents,
    number of questions, and number of answers.
    """
    try:
        stats = get_knowledge_base_stats()

        if "error" in stats:
            return f"Error getting knowledge base status: {stats['error']}"

        return (
            f"Knowledge Base Status:\n"
            f"- Total documents: {stats.get('total_documents', 0)}\n"
            f"- Unique discussions: {stats.get('unique_discussions', 0)}\n"
            f"- Questions: {stats.get('questions', 0)}\n"
            f"- Answers: {stats.get('answers', 0)}"
        )
    except Exception as e:
        return f"Error getting knowledge base status: {str(e)}"


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
            search_knowledge_base_tool,
            sync_knowledge_base,
            get_knowledge_base_status,
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

You have access to the following tools:

**Knowledge Base (Local ChromaDB):**
- search_knowledge_base_tool: Fast semantic search through indexed discussions
- sync_knowledge_base: Fetch and index discussions from GitHub into local storage
- get_knowledge_base_status: Check how many discussions are indexed

**GitHub Search (Live API):**
- search_langfuse_support: Quick search for relevant discussions (5 results)
- search_langfuse_support_detailed: More comprehensive search (10 results)

**Recommended workflow:**
1. First, try search_knowledge_base_tool for fast local search
2. If no results or knowledge base is empty, use sync_knowledge_base to populate it
3. Fall back to search_langfuse_support for live GitHub search when needed

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
