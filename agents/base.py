"""Base agent class providing common functionality for all agents."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Provides a common interface for agent implementations.
    Subclasses must implement the `run` method.
    """

    def __init__(self, llm: BaseChatModel, **kwargs: Any):
        """Initialize the agent with a language model.

        Args:
            llm: The language model to use for the agent.
            **kwargs: Additional configuration options.
        """
        self.llm = llm
        self.config = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the agent."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what the agent does."""
        pass

    @abstractmethod
    def run(self, query: str) -> str:
        """Execute the agent with the given query.

        Args:
            query: The user's input query.

        Returns:
            The agent's response as a string.
        """
        pass
