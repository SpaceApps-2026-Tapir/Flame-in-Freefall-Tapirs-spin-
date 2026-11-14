"""Persistent Chroma PDF index. AI-assisted implementation."""
import json
import re
from pathlib import Path
from uuid import uuid4

STORE = Path(__file__).resolve().parent / "documents" / "chroma"


def client_and_embedding(path, embedding_function=None):
    import chromadb
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    return chromadb.PersistentClient(path=str(path)), (
        embedding_function if embedding_function is not None else ONNXMiniLM_L6_V2())


def build_index(chunks, path=STORE, embedding_function=None):
    """Publish a full snapshot only after successful embedding of all chunks."""
    if not chunks:
        raise ValueError("No PDF passages to index.")
    client, embedding = client_and_embedding(path, embedding_function)
    name = "pdf-" + uuid4().hex
    collection = client.create_collection(name, embedding_function=embedding,
                                          metadata={"hnsw:space": "cosine"})
    try:
        for start in range(0, len(chunks), 64):
            batch = chunks[start:start + 64]
            collection.add(ids=[str(i) for i in range(start, start + len(batch))],
                           documents=[c["text"] for c in batch],
                           metadatas=[{k: v for k, v in c.items() if k != "text"} for c in batch])
        pointer = path / "active.json.tmp"
        pointer.write_text(json.dumps({"collection": name}), encoding="utf-8")
        pointer.replace(path / "active.json")
    except Exception:
        client.delete_collection(name)
        raise
    # Keep previous snapshots so readers already using them can finish safely.


def search(question, limit=5, path=STORE, embedding_function=None, max_distance=None):
    """Return cosine nearest neighbours; distance is not confidence or proof."""
    pointer = path / "active.json"
    if not question.strip() or limit <= 0 or not pointer.exists():
        return []
    client, embedding = client_and_embedding(path, embedding_function)
    name = json.loads(pointer.read_text())["collection"]
    collection = client.get_collection(name, embedding_function=embedding)
    # Short experiment acronyms are poorly represented by semantic embeddings.
    # Restrict explicit experiment questions to their documented corpus.
    experiments = re.findall(r"\b(?:BASS-II|BASS|SAME-R|SAME|FLEX)\b", question.upper())
    query_options = {}
    if experiments:
        query_options["where"] = {"experiment": {"$in": list(set(experiments))}}
    result = collection.query(query_texts=[question], n_results=min(limit, collection.count()),
                              **query_options,
                              include=["documents", "metadatas", "distances"])
    return [{**metadata, "text": text, "distance": distance}
            for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0],
                                                 result["distances"][0])
            if max_distance is None or distance <= max_distance]
