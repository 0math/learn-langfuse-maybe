"""Langfuse documentation agent using MCP HTTP endpoint."""

import json
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langfuse import get_client
from langgraph.prebuilt import create_react_agent

from agents.base import BaseAgent

MCP_ENDPOINT = "https://langfuse.com/api/mcp"

# Get Langfuse client for error tracking
_langfuse = get_client()


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
        # Update current span with error information
        _langfuse.update_current_span(
            level="ERROR",
            status_message=f"MCP docs error: {error_message}",
            metadata={
                "mcp_tool": tool_name,
                "error_details": error_details,
            },
        )
        # Update trace with error tag
        _langfuse.update_current_trace(
            tags=["MCP docs error"],
        )
    except Exception:
        # Silently fail if not in a trace context
        pass


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
        return _call_mcp_tool("searchLangfuseDocs", {"query": query})
    except Exception as e:
        return f"Error searching docs: {str(e)}"


@tool
def get_langfuse_docs_page(pathOrUrl: str) -> str:
    """Get raw Markdown content for a specific Langfuse documentation page.

    Args:
        pathOrUrl: The docs path (e.g., "/docs/get-started") or full URL.
    """
    try:
        return _call_mcp_tool("getLangfuseDocsPage", {"pathOrUrl": pathOrUrl})
    except Exception as e:
        return f"Error fetching page: {str(e)}"


@tool
def get_langfuse_overview() -> str:
    """Get a high-level index of Langfuse documentation.

    Returns the llms.txt file with key documentation endpoints.
    Call this at the start to discover available documentation.
    """
    try:
        return _call_mcp_tool("getLangfuseOverview", {})
    except Exception as e:
        return f"Error fetching overview: {str(e)}"


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
