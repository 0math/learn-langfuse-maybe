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
       ┌───────────────────────────────┼───────────────────────────────┐
       │                               │                               │
       ▼                               ▼                               ▼
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  LangfuseDocsAgent   │    │ LangfuseSupportAgent │    │LangfuseCommunityAgent│
│   (Documentation)    │    │  (GitHub Support)    │    │  (Self-Hosted Help)  │
│                      │    │                      │    │                      │
│  Tools:              │    │  Tools:              │    │  Tools:              │
│  • search_langfuse_  │    │  • search_langfuse_  │    │  • search_reddit     │
│    docs              │    │    support           │    │  • search_           │
│  • get_langfuse_     │    │  • search_langfuse_  │    │    stackoverflow     │
│    docs_page         │    │    support_detailed  │    │                      │
│  • get_langfuse_     │    │  • search_knowledge_ │    │  Searches:           │
│    overview          │    │    base_tool         │    │  • r/selfhosted      │
│                      │    │  • sync_knowledge_   │    │  • r/LangChain       │
└──────────┬───────────┘    │    base              │    │  • r/LocalLLaMA      │
           │                │  • get_knowledge_    │    │  • StackOverflow     │
           ▼                │    base_status       │    │    [langfuse] tag    │
┌──────────────────────┐    └──────────┬───────────┘    └──────────┬───────────┘
│  Langfuse MCP Server │               │                           │
│  https://langfuse.   │    ┌──────────┴───────────┐    ┌──────────┴───────────┐
│  com/api/mcp         │    │                      │    │                      │
│                      │    ▼                      ▼    ▼                      ▼
│  • Semantic search   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  • Page fetch        │  │ GitHub   │  │ ChromaDB │  │  Reddit  │  │ Stack    │
│  • Overview/index    │  │ GraphQL  │  │   (KB)   │  │ JSON API │  │ Overflow │
└──────────────────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘

                    ┌─────────────────────────────────────────┐
                    │           BaseAgent                     │
                    │        (Abstract Class)                 │
                    │                                         │
                    │  All agents inherit from this:          │
                    │  • name: str (abstract)                 │
                    │  • description: str (abstract)          │
                    │  • run(query) → str (abstract)          │
                    └─────────────────────────────────────────┘
```

### Agent Descriptions

| Agent | Role | Key Features |
|-------|------|--------------|
| **ControllerAgent** | Master router/orchestrator | Uses LangGraph ReAct pattern; wraps child agents as tools; system prompt fetched from Langfuse |
| **LangfuseDocsAgent** | Official documentation queries | Calls Langfuse MCP server; searches and fetches docs pages |
| **LangfuseSupportAgent** | Community support queries | Searches GitHub discussions; manages ChromaDB knowledge base |
| **LangfuseCommunityAgent** | Self-hosted deployment help | Searches Reddit and StackOverflow; fallback for self-hosted questions |
| **BaseAgent** | Abstract base class | Defines common interface (`name`, `description`, `run()`) |

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
       ├─── "langfuse_support_agent" → LangfuseSupportAgent.run()
       │                                       │
       │                                       ├─── ChromaDB search
       │                                       └─── GitHub API search
       │
       └─── "langfuse_community_agent" → LangfuseCommunityAgent.run()
                                                 │  (for self-hosted questions
                                                 │   when others fail)
                                                 │
                                                 ├─── Reddit JSON API
                                                 └─── StackOverflow API
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

## External Integrations

| Service | Purpose | Endpoint |
|---------|---------|----------|
| **Langfuse MCP** | Official docs search | `https://langfuse.com/api/mcp` |
| **GitHub GraphQL** | Support discussions | `https://api.github.com/graphql` |
| **Reddit JSON API** | Community discussions | `https://www.reddit.com/r/{subreddit}/search.json` |
| **StackExchange API** | Q&A search | `https://api.stackexchange.com/2.3` |
| **Langfuse Cloud** | Tracing & prompts | Configured via env vars |
| **OpenAI API** | LLM (`gpt-4o-mini`) + Embeddings | Configured via API key |
