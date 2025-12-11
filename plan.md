# Implementation Plan: Community Search Agent

## Overview

Create a new `LangfuseCommunityAgent` that searches Reddit and StackOverflow for Langfuse self-hosted related answers. This agent is invoked **only** when:
1. The question is about self-hosted Langfuse
2. Other agents (docs + support) didn't find satisfactory answers

## Architecture Decision

**Approach: Fallback Agent Pattern**

The ControllerAgent will be updated to:
1. First route self-hosted questions to existing agents (docs/support)
2. If no answer found, automatically fall back to the new CommunityAgent

This requires updating the controller's system prompt in Langfuse UI to instruct it when to use the community agent.

## Implementation Steps

### 1. Create `agents/langfuse_community.py`

New agent file with:

**Tools:**
- `search_reddit(query: str) -> str` - Searches Reddit via web scraping or Reddit API
  - Subreddits: r/selfhosted, r/LangChain, r/LocalLLaMA
  - Adds "langfuse" to search query automatically
- `search_stackoverflow(query: str) -> str` - Searches StackOverflow for [langfuse] tagged questions

**Agent class:**
- `LangfuseCommunityAgent(BaseAgent)` - ReAct agent with the above tools
- System prompt focused on self-hosted deployment questions

### 2. Update `agents/controller.py`

Add the new agent as a third tool:
```python
@tool
def langfuse_community_agent(query: str) -> str:
    """Query the Langfuse Community Agent.
    
    Use this ONLY for self-hosted Langfuse questions when:
    - The docs agent didn't have relevant information
    - The support agent didn't find helpful discussions
    
    Searches Reddit and StackOverflow for community solutions.
    """
```

### 3. Update `agents/__init__.py`

Export the new `LangfuseCommunityAgent`.

### 4. Update Controller System Prompt (Langfuse UI)

Update the prompt to include instructions like:
```
For self-hosted Langfuse questions:
1. First try langfuse_docs_agent for official documentation
2. Then try langfuse_support_agent for GitHub discussions
3. If no satisfactory answer, use langfuse_community_agent for Reddit/StackOverflow
```

### 5. Update `architecture.md`

Add the new agent to the architecture diagram.

## API Considerations

**Reddit Search:**
- Option A: Use Reddit JSON API (append `.json` to search URLs) - no auth required
- Option B: Use web search with site:reddit.com filter
- Recommendation: Option A (simpler, no API key needed)

**StackOverflow Search:**
- Use StackExchange API (no auth required for read-only)
- Filter by `[langfuse]` tag
- Endpoint: `https://api.stackexchange.com/2.3/search`

## File Changes Summary

| File | Action |
|------|--------|
| `agents/langfuse_community.py` | Create new |
| `agents/controller.py` | Add community agent tool |
| `agents/__init__.py` | Export new agent |
| `architecture.md` | Update diagram |
| Langfuse UI | Update controller system prompt |
