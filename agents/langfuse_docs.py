"""Langfuse documentation agent using MCP HTTP endpoint."""

from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.base import BaseAgent

MCP_ENDPOINT = "https://langfuse.com/api/mcp"


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

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            MCP_ENDPOINT,
            json=request_body,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result = response.json()
        print("RESULT:", result)
        if "result" in result:
            content = result["result"].get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", str(content))
            return str(content)
        elif "error" in result:
            return f"MCP Error: {result['error']}"

        return str(result)


@tool
def search_langfuse_docs(query: str) -> str:
    """Semantic search over Langfuse documentation.

    Returns a concise answer synthesized from relevant docs.
    Use this for broader questions about Langfuse.

    Args:
        query: The search query about Langfuse.
    """
    try:
        print("---Calling MCP ---")
        return _call_mcp_tool("searchLangfuseDocs", {"query": query})
    except Exception as e:
        return f"Error searching docs: {str(e)}"


@tool
def get_langfuse_docs_page(path: str) -> str:
    """Get raw Markdown content for a specific Langfuse documentation page.

    Args:
        path: The docs path (e.g., "/docs/get-started") or full URL.
    """
    try:
        return _call_mcp_tool("getLangfuseDocsPage", {"path": path})
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

        result = self.agent.invoke({"messages": messages})

        if result.get("messages"):
            return result["messages"][-1].content
        return "Unable to process the query."
