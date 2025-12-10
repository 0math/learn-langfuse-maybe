"""Agents module - flexible agent implementations for various tasks."""

from agents.base import BaseAgent
from agents.langfuse_docs import LangfuseDocsAgent

__all__ = ["BaseAgent", "LangfuseDocsAgent"]
