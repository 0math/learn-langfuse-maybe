"""Controller agent that routes queries to specialized agents using handoffs."""

from typing import Any, Callable, Generator, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.base import BaseAgent
from agents.langfuse_docs import LangfuseDocsAgent
from agents.langfuse_support import LangfuseSupportAgent


class ControllerAgent(BaseAgent):
    """Controller agent that automatically routes queries to specialized agents.

    This agent uses a tool-calling pattern where specialized agents are wrapped
    as tools. The controller decides which agent to invoke based on the query.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        langfuse_handler: Optional[BaseCallbackHandler] = None,
        **kwargs: Any,
    ):
        """Initialize the controller agent.

        Args:
            llm: The language model to use.
            langfuse_handler: Optional Langfuse callback handler for tracing.
            **kwargs: Additional configuration.
        """
        super().__init__(llm, langfuse_handler, **kwargs)
        self._setup_agent()

    def _setup_agent(self) -> None:
        """Set up the controller agent with subagent tools."""
        self._callbacks = [self.langfuse_handler] if self.langfuse_handler else []

        # Create the specialized agents
        self._docs_agent = LangfuseDocsAgent(
            llm=self.llm, langfuse_handler=self.langfuse_handler
        )
        self._support_agent = LangfuseSupportAgent(
            llm=self.llm, langfuse_handler=self.langfuse_handler
        )

        # Create tools that delegate to the existing agents
        @tool
        def langfuse_docs_agent(query: str) -> str:
            """Query the Langfuse Documentation Agent.

            Use this for questions about:
            - Official Langfuse documentation
            - How to use Langfuse features
            - API references and SDK usage
            - Getting started guides
            - Integration instructions

            Args:
                query: The user's question about Langfuse documentation.
            """
            return self._docs_agent.run(query)

        @tool
        def langfuse_support_agent(query: str) -> str:
            """Query the Langfuse Support Agent.

            Use this for questions about:
            - Troubleshooting and debugging issues
            - Common problems and their solutions
            - Community-discussed questions and answers
            - Real-world usage patterns and workarounds
            - Issues that other users have encountered

            Args:
                query: The user's question about Langfuse support topics.
            """
            return self._support_agent.run(query)

        self.tools = [langfuse_docs_agent, langfuse_support_agent]
        self.agent = create_react_agent(self.llm, self.tools)

    @property
    def name(self) -> str:
        return "Langfuse Controller Agent"

    @property
    def description(self) -> str:
        return "Routes queries to specialized Langfuse agents (Docs or Support)"

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the controller agent."""
        return """You are a helpful Langfuse assistant that routes user queries to specialized agents.

You have access to two specialized agents:

1. **langfuse_docs_agent**: For questions about official Langfuse documentation
   - How to use Langfuse features
   - API references and SDK usage
   - Getting started guides
   - Integration instructions
   - Official best practices

2. **langfuse_support_agent**: For troubleshooting and community questions
   - Debugging issues and errors
   - Common problems and their solutions
   - Real-world usage patterns
   - Issues other users have encountered
   - Workarounds and fixes

**Routing Guidelines:**
- For "how do I..." or "what is..." questions about Langfuse features → use langfuse_docs_agent
- For "I'm getting an error..." or "it's not working..." → use langfuse_support_agent
- For questions that could benefit from both perspectives, call both agents
- Always use at least one agent - never try to answer without consulting them

Synthesize the information from the agents into a clear, helpful response.
If both agents are consulted, combine their insights coherently."""

    def _build_messages(
        self, query: str, history: Optional[list[dict[str, str]]] = None
    ) -> list[BaseMessage]:
        """Build the message list including system prompt and conversation history.

        Args:
            query: The current user query.
            history: Optional list of previous messages with 'role' and 'content' keys.

        Returns:
            List of BaseMessage objects for the agent.
        """
        messages: list[BaseMessage] = [SystemMessage(content=self._get_system_prompt())]

        # Add conversation history
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Add current query
        messages.append(HumanMessage(content=query))

        return messages

    def run(self, query: str, history: Optional[list[dict[str, str]]] = None) -> str:
        """Execute a query by routing to the appropriate specialized agent.

        Args:
            query: The user's question about Langfuse.
            history: Optional conversation history (list of {"role": "user"|"assistant", "content": "..."}).

        Returns:
            The agent's response.
        """
        messages = self._build_messages(query, history)

        config = {"callbacks": self._callbacks} if self._callbacks else {}
        result = self.agent.invoke({"messages": messages}, config=config)

        if result.get("messages"):
            return result["messages"][-1].content
        return "Unable to process the query."

    def stream(
        self,
        query: str,
        history: Optional[list[dict[str, str]]] = None,
        on_tool_start: Optional[Callable[[str], None]] = None,
    ) -> Generator[str, None, None]:
        """Stream the agent execution, yielding tool names as they are called.

        Args:
            query: The user's question about Langfuse.
            history: Optional conversation history (list of {"role": "user"|"assistant", "content": "..."}).
            on_tool_start: Optional callback called when a tool starts executing.

        Yields:
            Tool names as they are invoked, and finally the response.
        """
        messages = self._build_messages(query, history)

        config = {"callbacks": self._callbacks} if self._callbacks else {}

        final_response = None

        for chunk in self.agent.stream({"messages": messages}, config=config):
            # Check for agent actions (tool calls)
            if "agent" in chunk:
                agent_messages = chunk["agent"].get("messages", [])
                for msg in agent_messages:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call.get("name", "unknown")
                            if on_tool_start:
                                on_tool_start(tool_name)
                            yield tool_name
                    elif isinstance(msg, AIMessage) and msg.content:
                        final_response = msg.content

        if final_response:
            yield f"__FINAL__{final_response}"
