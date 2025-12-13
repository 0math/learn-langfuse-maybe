"""Langfuse documentation agent using MCP HTTP endpoint."""

import json
import re
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langfuse import get_client
from langgraph.prebuilt import create_react_agent
from markdownify import markdownify as md

from agents.base import BaseAgent

MCP_ENDPOINT = "https://langfuse.com/api/mcp"
LANGFUSE_DOCS_BASE = "https://langfuse.com"
LANGFUSE_LLMS_TXT = "https://langfuse.com/llms.txt"


def _get_langfuse_client():
    """Get Langfuse client lazily to avoid initialization before env vars are loaded."""
    return get_client()


def _parse_sse_response(text: str) -> dict:
    """Parse Server-Sent Events response to extract JSON data.

    Args:
        text: Raw SSE response text.

    Returns:
        Parsed JSON data from the SSE message.
    """
    for line in text.strip().split("\n"):
        if line.startswith("data: "):
            return json.loads(line[6:])
    return {}


def _track_mcp_error(
    tool_name: str, error_message: str, error_details: Any = None
) -> None:
    """Track MCP error in Langfuse with appropriate tags and level.

    Args:
        tool_name: Name of the MCP tool that failed.
        error_message: The error message.
        error_details: Additional error details.
    """
    try:
        langfuse = _get_langfuse_client()
        # Update current span with error information
        langfuse.update_current_span(
            level="ERROR",
            status_message=f"MCP docs error: {error_message}",
            metadata={
                "mcp_tool": tool_name,
                "error_details": error_details,
            },
        )
        # Update trace with error tag
        langfuse.update_current_trace(
            tags=["MCP docs error"],
        )
    except Exception:
        # Silently fail if not in a trace context
        pass


def _fetch_docs_page_directly(path_or_url: str) -> str:
    """Fetch a Langfuse docs page directly via HTTP as fallback.

    Args:
        path_or_url: The docs path (e.g., "/docs/get-started") or full URL.

    Returns:
        The page content converted to markdown.

    Raises:
        httpx.HTTPError: If the request fails.
    """
    # Normalize the URL
    if path_or_url.startswith("http"):
        url = path_or_url
    else:
        # Ensure path starts with /
        if not path_or_url.startswith("/"):
            path_or_url = "/" + path_or_url
        url = f"{LANGFUSE_DOCS_BASE}{path_or_url}"

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(
            url,
            headers={
                "Accept": "text/html",
                "User-Agent": "LangfuseDocsAgent/1.0",
            },
        )
        response.raise_for_status()

        # Convert HTML to markdown
        html_content = response.text

        # Extract main content area if possible (look for article or main tags)
        main_match = re.search(
            r"<(article|main)[^>]*>(.*?)</\1>", html_content, re.DOTALL | re.IGNORECASE
        )
        if main_match:
            html_content = main_match.group(2)

        # Convert to markdown
        markdown_content = md(
            html_content,
            heading_style="ATX",
            strip=["script", "style", "nav", "footer"],
        )

        # Clean up excessive whitespace
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content.strip()


def _fetch_llms_txt_directly() -> str:
    """Fetch the llms.txt file directly as fallback.

    Returns:
        The content of llms.txt.

    Raises:
        httpx.HTTPError: If the request fails.
    """
    with httpx.Client(timeout=30.0) as client:
        response = client.get(LANGFUSE_LLMS_TXT)
        response.raise_for_status()
        return response.text


def _call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Call a tool on the Langfuse MCP server via HTTP.

    Args:
        tool_name: Name of the MCP tool to call.
        arguments: Arguments to pass to the tool.

    Returns:
        The tool response text.
    """
    request_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                MCP_ENDPOINT,
                json=request_body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            response.raise_for_status()

            # Handle SSE (text/event-stream) or JSON response
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                result = _parse_sse_response(response.text)
            else:
                result = response.json()

            if "result" in result:
                content = result["result"].get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", str(content))
                return str(content)
            elif "error" in result:
                error_msg = f"MCP Error: {result['error']}"
                _track_mcp_error(tool_name, error_msg, result["error"])
                return error_msg

            return str(result)

    except httpx.HTTPStatusError as e:
        error_msg = f"MCP HTTP error: {e.response.status_code}"
        _track_mcp_error(tool_name, error_msg, {"status_code": e.response.status_code})
        raise
    except httpx.RequestError as e:
        error_msg = f"MCP request error: {str(e)}"
        _track_mcp_error(tool_name, error_msg, {"error_type": type(e).__name__})
        raise


@tool
def search_langfuse_docs(query: str) -> str:
    """Semantic search over Langfuse documentation.

    Returns a concise answer synthesized from relevant docs.
    Use this for broader questions about Langfuse.

    Args:
        query: The search query about Langfuse.
    """
    try:
        result = _call_mcp_tool("searchLangfuseDocs", {"query": query})
        # Check if result indicates an error
        if result.startswith("MCP Error:"):
            raise Exception(result)
        return result
    except Exception as e:
        # Fallback: fetch the docs overview and relevant pages directly
        _track_mcp_error(
            "searchLangfuseDocs", str(e), {"query": query, "fallback": True}
        )
        try:
            # Get the llms.txt to find relevant pages
            overview = _fetch_llms_txt_directly()
            # Try to find relevant doc paths from llms.txt based on query keywords
            query_lower = query.lower()
            relevant_paths = []
            for line in overview.split("\n"):
                if line.startswith("- [") or line.startswith("  - ["):
                    # Extract path from markdown link format
                    match = re.search(r"\]\((/[^)]+)\)", line)
                    if match:
                        path = match.group(1)
                        line_lower = line.lower()
                        # Check if any query word appears in the line
                        if any(
                            word in line_lower
                            for word in query_lower.split()
                            if len(word) > 2
                        ):
                            relevant_paths.append(path)

            if relevant_paths:
                # Fetch up to 3 most relevant pages
                contents = []
                for path in relevant_paths[:3]:
                    try:
                        content = _fetch_docs_page_directly(path)
                        contents.append(f"## From {path}\n\n{content[:2000]}...")
                    except Exception:
                        continue
                if contents:
                    return f"[Fallback: Direct docs fetch]\n\n" + "\n\n---\n\n".join(
                        contents
                    )

            # If no relevant paths found, return overview
            return (
                f"[Fallback: Direct docs fetch]\n\nDocumentation overview:\n{overview}"
            )
        except Exception as fallback_error:
            return f"Error searching docs (MCP and fallback both failed): MCP error: {str(e)}, Fallback error: {str(fallback_error)}"


@tool
def get_langfuse_docs_page(pathOrUrl: str) -> str:
    """Get raw Markdown content for a specific Langfuse documentation page.

    Args:
        pathOrUrl: The docs path (e.g., "/docs/get-started") or full URL.
    """
    try:
        result = _call_mcp_tool("getLangfuseDocsPage", {"pathOrUrl": pathOrUrl})
        # Check if result indicates an error
        if result.startswith("MCP Error:"):
            raise Exception(result)
        return result
    except Exception as e:
        # Fallback: fetch the page directly
        _track_mcp_error(
            "getLangfuseDocsPage", str(e), {"pathOrUrl": pathOrUrl, "fallback": True}
        )
        try:
            content = _fetch_docs_page_directly(pathOrUrl)
            return f"[Fallback: Direct docs fetch]\n\n{content}"
        except Exception as fallback_error:
            return f"Error fetching page (MCP and fallback both failed): MCP error: {str(e)}, Fallback error: {str(fallback_error)}"


@tool
def get_langfuse_overview() -> str:
    """Get a high-level index of Langfuse documentation.

    Returns the llms.txt file with key documentation endpoints.
    Call this at the start to discover available documentation.
    """
    try:
        result = _call_mcp_tool("getLangfuseOverview", {})
        # Check if result indicates an error
        if result.startswith("MCP Error:"):
            raise Exception(result)
        return result
    except Exception as e:
        # Fallback: fetch llms.txt directly
        _track_mcp_error("getLangfuseOverview", str(e), {"fallback": True})
        try:
            content = _fetch_llms_txt_directly()
            return f"[Fallback: Direct docs fetch]\n\n{content}"
        except Exception as fallback_error:
            return f"Error fetching overview (MCP and fallback both failed): MCP error: {str(e)}, Fallback error: {str(fallback_error)}"


class LangfuseDocsAgent(BaseAgent):
    """Agent that queries Langfuse documentation via MCP HTTP endpoint."""

    # Tag for identifying this agent's traces in Langfuse
    AGENT_TAG = "docs-agent"

    def __init__(self, llm: BaseChatModel, **kwargs: Any):
        """Initialize the Langfuse docs agent.

        Args:
            llm: The language model to use.
            **kwargs: Additional configuration.
        """
        super().__init__(llm, **kwargs)
        self._setup_agent()

    def _setup_agent(self) -> None:
        """Set up the ReAct agent with MCP tools."""
        self.tools = [
            search_langfuse_docs,
            get_langfuse_docs_page,
            get_langfuse_overview,
        ]
        self.agent = create_react_agent(self.llm, self.tools)
        # Store callbacks for agent invocation
        self._callbacks = [self.langfuse_handler] if self.langfuse_handler else []

    def _get_config(self) -> dict:
        """Build config with callbacks and agent tag for Langfuse tracing."""
        config: dict = {}
        if self._callbacks:
            config["callbacks"] = self._callbacks
        # Add agent tag for filtering in Langfuse UI
        config["metadata"] = {"langfuse_tags": [self.AGENT_TAG]}
        return config

    @property
    def name(self) -> str:
        return "Langfuse Docs Agent"

    @property
    def description(self) -> str:
        return "Searches and answers questions about Langfuse documentation"

    def run(self, query: str) -> str:
        """Execute a query against Langfuse documentation.

        Args:
            query: The user's question about Langfuse.

        Returns:
            The agent's response based on documentation search.
        """
        system_prompt = """You are a helpful assistant that answers questions about Langfuse.
You have access to tools that query the official Langfuse documentation:
- search_langfuse_docs: Semantic search for broader questions
- get_langfuse_docs_page: Get specific documentation pages
- get_langfuse_overview: Get the documentation index

Use these tools to find relevant information before answering.
Always cite the documentation when possible and be accurate with your responses.
If you cannot find relevant information, say so clearly."""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]

        # Invoke agent with Langfuse callback and tags for tracing
        config = self._get_config()
        result = self.agent.invoke({"messages": messages}, config=config)

        if result.get("messages"):
            return result["messages"][-1].content
        return "Unable to process the query."
