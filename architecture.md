# Architecture Overview

This document describes the high-level architecture of the Langfuse learning project, including agent hierarchy and ChromaDB knowledge base integration.

## Part 1: Agent Hierarchy

```
                    ┌─────────────────────────────────────────┐
                    │            STREAMLIT UI                 │
                    │     (Chat Interface + KB Management)    │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │           ControllerAgent               │
                    │         (Master Orchestrator)           │
                    │                                         │
                    │  • LangGraph ReAct Agent                │
                    │  • System prompt from Langfuse UI       │
                    │  • Routes queries to specialized agents │
                    │  • Manages conversation history         │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │         Tool-Based Routing          │
                    └──────────────────┬──────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        │                        ▼
┌─────────────────────────┐            │          ┌─────────────────────────┐
│   LangfuseDocsAgent     │            │          │  LangfuseSupportAgent   │
│   (Documentation)       │            │          │  (Support/Community)    │
│                         │            │          │                         │
│  Tools:                 │            │          │  Tools:                 │
│  • search_langfuse_docs │            │          │  • search_langfuse_     │
│  • get_langfuse_docs_   │            │          │    support              │
│    page                 │            │          │  • search_langfuse_     │
│  • get_langfuse_        │            │          │    support_detailed     │
│    overview             │            │          │  • search_knowledge_    │
└───────────┬─────────────┘            │          │    base_tool            │
            │                          │          │  • sync_knowledge_base  │
            ▼                          │          │  • get_knowledge_base_  │
┌─────────────────────────┐            │          │    status               │
│    Langfuse MCP Server  │            │          └───────────┬─────────────┘
│  https://langfuse.com/  │            │                      │
│  api/mcp                │            │          ┌───────────┴───────────┐
│                         │            │          │                       │
│  • Semantic docs search │            │          ▼                       ▼
│  • Page content fetch   │            │   ┌─────────────┐    ┌─────────────────┐
│  • Overview/index       │            │   │ GitHub API  │    │   ChromaDB      │
└─────────────────────────┘            │   │ (GraphQL)   │    │ (Knowledge Base)│
                                       │   └─────────────┘    └─────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    │           BaseAgent                 │
                    │        (Abstract Class)             │
                    │                                     │
                    │  All agents inherit from this:      │
                    │  • name: str (abstract)             │
                    │  • description: str (abstract)      │
                    │  • run(query) → str (abstract)      │
                    │                                     │
                    └─────────────────────────────────────┘
```

### Agent Descriptions

| Agent | Role | Tag | Key Features |
|-------|------|-----|--------------|
| **ControllerAgent** | Master router/orchestrator | - | Uses LangGraph ReAct pattern; wraps child agents as tools; system prompt fetched from Langfuse |
| **LangfuseDocsAgent** | Official documentation queries | `docs-agent` | Calls Langfuse MCP server; searches and fetches docs pages |
| **LangfuseSupportAgent** | Community support queries | `support-agent` | Searches GitHub discussions; manages ChromaDB knowledge base |
| **BaseAgent** | Abstract base class | - | Defines common interface (`name`, `description`, `run()`) |

> **Filtering by Agent**: Use the agent tags (`docs-agent`, `support-agent`) in Langfuse UI to filter traces by which agent handled the query.

### Data Flow: User Query

```
User Input (Streamlit)
       │
       ▼
ControllerAgent.stream()
       │
       ├─── Langfuse tracing (LangfuseCallbackHandler)
       │
       ▼
LLM decides which tool to call
       │
       ├─── "langfuse_docs_agent" → LangfuseDocsAgent.run()
       │                                    │
       │                                    ▼
       │                           MCP Server → Response
       │
       └─── "langfuse_support_agent" → LangfuseSupportAgent.run()
                                               │
                                               ├─── ChromaDB search
                                               └─── GitHub API search
       │
       ▼
Response streamed back to UI
```

---

## Part 2: ChromaDB Knowledge Base

### Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ChromaDB Vector Store                           │
│                                                                         │
│  Location: ./chroma_db/                                                 │
│  Collection: "langfuse_discussions"                                     │
│  Embedding Model: OpenAI Embeddings                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Documents                                 │   │
│  │                                                                  │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │   │
│  │  │   Question Doc   │  │    Answer Doc    │  │  Comment Doc  │  │   │
│  │  │                  │  │                  │  │               │  │   │
│  │  │ page_content:    │  │ page_content:    │  │ page_content: │  │   │
│  │  │  Title + Body    │  │  "Answer to..."  │  │  Comment text │  │   │
│  │  │                  │  │  + answer body   │  │               │  │   │
│  │  │ metadata:        │  │                  │  │ metadata:     │  │   │
│  │  │  • type: question│  │ metadata:        │  │  • type:answer│  │   │
│  │  │  • url           │  │  • type: answer  │  │  • is_accepted│  │   │
│  │  │  • title         │  │  • is_accepted   │  │    _answer    │  │   │
│  │  │  • author        │  │    _answer: true │  │  • comment_   │  │   │
│  │  │  • created_at    │  │  • url           │  │    index      │  │   │
│  │  │  • category      │  │  • discussion_id │  │  • url        │  │   │
│  │  │  • discussion_id │  │                  │  │               │  │   │
│  │  └──────────────────┘  └──────────────────┘  └───────────────┘  │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Files:                                                                 │
│  ├── chroma.sqlite3          (metadata + document store)               │
│  └── [uuid-directories]/      (HNSW vector indices)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Knowledge Base Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE BASE SYNC PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────┐
    │   GitHub GraphQL    │
    │        API          │
    │  (langfuse/langfuse │
    │   discussions)      │
    └──────────┬──────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  fetch_all_support_          │
    │  discussions(max=100)        │
    │                              │
    │  • Paginated GraphQL queries │
    │  • Filters: Support category │
    │  • Returns: Raw discussion   │
    │    data                      │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  discussion_to_documents()   │
    │                              │
    │  Creates LangChain Documents │
    │  for each discussion:        │
    │                              │
    │  • 1 Question document       │
    │  • 1 Answer doc (if exists)  │
    │  • N Comment docs (marked    │
    │    as answers)               │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  index_discussions()         │
    │                              │
    │  • Checks for duplicates     │
    │    (by discussion_id)        │
    │  • Embeds new documents      │
    │    (OpenAI Embeddings)       │
    │  • Stores in ChromaDB        │
    │  • Reports progress via      │
    │    callback                  │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │        ChromaDB              │
    │  langfuse_discussions        │
    │        collection            │
    └──────────────────────────────┘
```

### Knowledge Base Query Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAG QUERY PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────┘

    User Query: "How do I fix token limit errors?"
               │
               ▼
    ┌──────────────────────────────┐
    │  LangfuseSupportAgent        │
    │                              │
    │  Checks: Is KB empty?        │
    └──────────┬───────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
    [KB Empty]      [KB Has Data]
       │               │
       ▼               ▼
    ┌──────────┐    ┌──────────────────────────────┐
    │ GitHub   │    │  search_knowledge_base()     │
    │ Direct   │    │                              │
    │ Search   │    │  • Embeds query (OpenAI)     │
    │          │    │  • Similarity search         │
    │          │    │  • Returns top k documents   │
    └──────────┘    │  • Filters by type if needed │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  Retrieved Documents         │
                    │                              │
                    │  [Q&A pairs with metadata]   │
                    │  • page_content: Text        │
                    │  • metadata: URL, author,    │
                    │    discussion_id, etc.       │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  LLM Generation              │
                    │                              │
                    │  Synthesizes answer from     │
                    │  retrieved context           │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  Response to User            │
                    │                              │
                    │  "Based on community         │
                    │   discussions, here's how    │
                    │   to fix token limits..."    │
                    └──────────────────────────────┘
```

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `get_chroma_client()` | `knowledge_base.py` | Initialize persistent ChromaDB client |
| `get_vector_store()` | `knowledge_base.py` | Get LangChain Chroma wrapper |
| `discussion_to_documents()` | `knowledge_base.py` | Convert GitHub data to LangChain docs |
| `index_discussions()` | `knowledge_base.py` | Index docs with deduplication |
| `search_knowledge_base()` | `knowledge_base.py` | Semantic similarity search |
| `get_knowledge_base_stats()` | `knowledge_base.py` | Return doc count, Q&A breakdown |
| `clear_knowledge_base()` | `knowledge_base.py` | Wipe all documents |

---

## Part 3: Langfuse Instrumentation

### Trace Attributes

The application tracks the following attributes in Langfuse:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LANGFUSE TRACE ATTRIBUTES                          │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │  Trace Level                                                    │
    │                                                                 │
    │  • session_id    - Groups traces by browser session (UUID)      │
    │  • user_id       - Identifies user (user_{8-char-hex})          │
    │  • trace_id      - Unique trace identifier (from callback)      │
    │  • langfuse_prompt - Links to prompt version for metrics        │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Observation Level (Agent Tags)                                 │
    │                                                                 │
    │  • docs-agent    - Tag for LangfuseDocsAgent traces             │
    │  • support-agent - Tag for LangfuseSupportAgent traces          │
    │  • MCP docs error - Tag for MCP tool failures                   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

### Instrumentation Flow

```
    Streamlit App
         │
         ├─── get_session_id() → UUID per browser session
         ├─── get_user_id()    → "user_{8-char-hex}" per session
         │
         ▼
    ControllerAgent(session_id, user_id)
         │
         ├─── Fetches prompt from Langfuse (with prompt object)
         │
         ▼
    _get_config(prompt_object)
         │
         ├─── metadata["langfuse_session_id"] = session_id
         ├─── metadata["langfuse_user_id"]    = user_id
         ├─── metadata["langfuse_prompt"]     = prompt_object
         │
         ▼
    agent.invoke(..., config=config)
         │
         ▼
    LangfuseCallbackHandler → Langfuse Cloud
         │
         ├─── Trace with session/user/prompt linked
         └─── Observations with agent tags
```

### Prompt Linking

The system prompt is fetched from Langfuse and linked to traces:

| Step | Code Location | Description |
|------|---------------|-------------|
| 1. Fetch | `controller.py:_get_system_prompt()` | Gets prompt from Langfuse by name + label |
| 2. Return | Returns `(content, prompt_object)` | Both compiled content and raw prompt object |
| 3. Pass | `controller.py:_get_config()` | Adds `langfuse_prompt` to metadata |
| 4. Link | LangfuseCallbackHandler | Links prompt version to trace for metrics |

---

## Part 4: Streamlit UI Features

### Empty Chat State

When no chat history exists, the app displays:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EMPTY CHAT STATE                                │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │  Welcome to Learn Langfuse Maybe                                │
    │  Ask me anything about Langfuse...                              │
    └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Popular in the Community                                       │
    │                                                                 │
    │  "Users often ask about {topic} ({count} comments)..."          │
    │  [Link to most active discussion]                               │
    │                                                                 │
    │  ▼ View top discussions                                         │
    │    1. [Discussion Title] (N comments)                           │
    │    2. [Discussion Title] (N comments)                           │
    │    3. [Discussion Title] (N comments)                           │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Learn the Basics                                               │
    │                                                                 │
    │  ┌─────────────────────┐    ┌─────────────────────┐            │
    │  │ Langfuse Variables  │    │ Prompt Management   │            │
    │  │                     │    │                     │            │
    │  │ LANGFUSE_SECRET_KEY │    │ Create, version,    │            │
    │  │ LANGFUSE_PUBLIC_KEY │    │ and deploy prompts  │            │
    │  │ LANGFUSE_HOST       │    │                     │            │
    │  │                     │    │ [Link to docs]      │            │
    │  │ [Quick Start Guide] │    │                     │            │
    │  └─────────────────────┘    └─────────────────────┘            │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

### Data Sources for Empty State

| Component | Source | Caching |
|-----------|--------|---------|
| Popular Discussions | GitHub GraphQL API | 1 hour (st.cache_data) |
| Topic Summary | Keyword analysis of titles | Derived from cached data |
| Documentation Links | Static URLs | None needed |

---

## External Integrations

| Service | Purpose | Endpoint |
|---------|---------|----------|
| **Langfuse MCP** | Official docs search | `https://langfuse.com/api/mcp` |
| **GitHub GraphQL** | Support discussions | `https://api.github.com/graphql` |
| **Langfuse Cloud** | Tracing, prompts, metrics | Configured via env vars |
| **OpenAI API** | LLM (`gpt-4o-mini`) + Embeddings | Configured via API key |
