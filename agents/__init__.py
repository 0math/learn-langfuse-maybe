"""Agents module - flexible agent implementations for various tasks."""

from agents.base import BaseAgent
from agents.controller import ControllerAgent
from agents.knowledge_base import (
    clear_knowledge_base,
    get_knowledge_base_stats,
    index_discussions,
    search_knowledge_base,
)
from agents.langfuse_docs import LangfuseDocsAgent
from agents.langfuse_support import LangfuseSupportAgent

__all__ = [
    "BaseAgent",
    "ControllerAgent",
    "LangfuseDocsAgent",
    "LangfuseSupportAgent",
    "index_discussions",
    "search_knowledge_base",
    "get_knowledge_base_stats",
    "clear_knowledge_base",
]
