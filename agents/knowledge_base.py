"""ChromaDB knowledge base for storing and searching GitHub Discussions."""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Default path for persistent storage
DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"

# Collection name for discussions
DISCUSSIONS_COLLECTION = "langfuse_discussions"


def get_chroma_client(persist_dir: Optional[Path] = None) -> chromadb.ClientAPI:
    """Get a ChromaDB client with persistent storage.

    Args:
        persist_dir: Directory for persistent storage. Uses default if not provided.

    Returns:
        ChromaDB client instance.
    """
    if persist_dir is None:
        persist_dir = DEFAULT_PERSIST_DIR

    persist_dir.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def get_vector_store(
    collection_name: str = DISCUSSIONS_COLLECTION,
    persist_dir: Optional[Path] = None,
) -> Chroma:
    """Get a LangChain Chroma vector store for the discussions collection.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_dir: Directory for persistent storage.

    Returns:
        LangChain Chroma vector store instance.
    """
    if persist_dir is None:
        persist_dir = DEFAULT_PERSIST_DIR

    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def discussion_to_documents(discussion: dict) -> list:
    """Convert a GitHub Discussion to LangChain Documents.

    Creates separate documents for the question and answer to enable
    better semantic search.

    Args:
        discussion: Discussion data from GitHub GraphQL API.

    Returns:
        List of Document objects.
    """
    documents = []

    if not discussion:
        return documents

    url = discussion.get("url", "")
    title = discussion.get("title", "Untitled")
    body = discussion.get("body", "")
    author = discussion.get("author", {}).get("login", "Unknown")
    created_at = discussion.get("createdAt", "")[:10]
    category = discussion.get("category", {}).get("name", "")

    # Create document for the question
    question_content = f"# {title}\n\n{body}"
    question_doc = Document(
        page_content=question_content,
        metadata={
            "type": "question",
            "url": url,
            "title": title,
            "author": author,
            "created_at": created_at,
            "category": category,
            "discussion_id": url.split("/")[-1] if url else "",
        },
    )
    documents.append(question_doc)

    # Create document for the accepted answer if available
    answer = discussion.get("answer")
    if answer:
        answer_body = answer.get("body", "")
        answer_author = answer.get("author", {}).get("login", "Unknown")

        answer_content = f"# Answer to: {title}\n\n{answer_body}"
        answer_doc = Document(
            page_content=answer_content,
            metadata={
                "type": "answer",
                "url": url,
                "title": title,
                "author": answer_author,
                "question_author": author,
                "created_at": created_at,
                "category": category,
                "discussion_id": url.split("/")[-1] if url else "",
                "is_accepted_answer": True,
            },
        )
        documents.append(answer_doc)

    # Also include helpful comments marked as answers
    comments = discussion.get("comments", {}).get("nodes", [])
    for i, comment in enumerate(comments):
        if comment and comment.get("isAnswer"):
            comment_body = comment.get("body", "")
            comment_author = comment.get("author", {}).get("login", "Unknown")

            comment_content = f"# Answer to: {title}\n\n{comment_body}"
            comment_doc = Document(
                page_content=comment_content,
                metadata={
                    "type": "answer",
                    "url": url,
                    "title": title,
                    "author": comment_author,
                    "question_author": author,
                    "created_at": created_at,
                    "category": category,
                    "discussion_id": url.split("/")[-1] if url else "",
                    "is_accepted_answer": True,
                    "comment_index": i,
                },
            )
            documents.append(comment_doc)

    return documents


def index_discussions(
    discussions: list,
    collection_name: str = DISCUSSIONS_COLLECTION,
    persist_dir: Optional[Path] = None,
    on_progress: Optional[callable] = None,
) -> int:
    """Index GitHub Discussions into ChromaDB.

    Args:
        discussions: List of discussion data from GitHub GraphQL API.
        collection_name: Name of the ChromaDB collection.
        persist_dir: Directory for persistent storage.
        on_progress: Optional callback function called after each document is indexed.
                     Receives (current_count, total_count) as arguments.

    Returns:
        Number of documents indexed.
    """
    vector_store = get_vector_store(collection_name, persist_dir)

    all_documents = []
    for discussion in discussions:
        docs = discussion_to_documents(discussion)
        all_documents.extend(docs)

    if all_documents:
        # Get existing IDs to avoid duplicates
        existing_ids = set()
        try:
            collection = vector_store._collection
            existing_data = collection.get()
            if existing_data and existing_data.get("metadatas"):
                for metadata in existing_data["metadatas"]:
                    if metadata:
                        existing_ids.add(metadata.get("discussion_id", ""))
        except Exception:
            pass

        # Filter out already indexed discussions
        new_documents = [
            doc
            for doc in all_documents
            if doc.metadata.get("discussion_id") not in existing_ids
        ]

        if new_documents:
            total = len(new_documents)
            # Index documents one by one to allow progress updates
            for i, doc in enumerate(new_documents, 1):
                vector_store.add_documents([doc])
                if on_progress:
                    on_progress(i, total)
            return total

    return 0


def search_knowledge_base(
    query: str,
    k: int = 5,
    collection_name: str = DISCUSSIONS_COLLECTION,
    persist_dir: Optional[Path] = None,
    filter_type: Optional[str] = None,
) -> list:
    """Search the knowledge base for relevant discussions.

    Args:
        query: Search query.
        k: Number of results to return.
        collection_name: Name of the ChromaDB collection.
        persist_dir: Directory for persistent storage.
        filter_type: Optional filter for document type ("question" or "answer").

    Returns:
        List of relevant Documents.
    """
    vector_store = get_vector_store(collection_name, persist_dir)

    search_kwargs = {"k": k}
    if filter_type:
        search_kwargs["filter"] = {"type": filter_type}

    return vector_store.similarity_search(query, **search_kwargs)


def get_knowledge_base_stats(
    collection_name: str = DISCUSSIONS_COLLECTION,
    persist_dir: Optional[Path] = None,
) -> dict:
    """Get statistics about the knowledge base.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_dir: Directory for persistent storage.

    Returns:
        Dictionary with collection statistics.
    """
    vector_store = get_vector_store(collection_name, persist_dir)

    try:
        collection = vector_store._collection
        count = collection.count()

        # Get unique discussion IDs
        data = collection.get()
        discussion_ids = set()
        questions = 0
        answers = 0

        if data and data.get("metadatas"):
            for metadata in data["metadatas"]:
                if metadata:
                    discussion_ids.add(metadata.get("discussion_id", ""))
                    if metadata.get("type") == "question":
                        questions += 1
                    elif metadata.get("type") == "answer":
                        answers += 1

        return {
            "total_documents": count,
            "unique_discussions": len(discussion_ids),
            "questions": questions,
            "answers": answers,
        }
    except Exception as e:
        return {"error": str(e)}


def clear_knowledge_base(
    collection_name: str = DISCUSSIONS_COLLECTION,
    persist_dir: Optional[Path] = None,
) -> bool:
    """Clear all documents from the knowledge base.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_dir: Directory for persistent storage.

    Returns:
        True if successful, False otherwise.
    """
    try:
        client = get_chroma_client(persist_dir)
        try:
            client.delete_collection(collection_name)
        except Exception:
            # Collection doesn't exist, that's fine
            pass
        return True
    except Exception:
        return False
