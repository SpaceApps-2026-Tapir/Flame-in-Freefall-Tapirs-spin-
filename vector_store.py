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
    # Remove repeated experiment labels and question boilerplate from the semantic
    # query so title slides do not outrank actual descriptions and results.
    content_query = re.sub(r"\b(?:BASS-II|BASS|SAME-R|SAME|FLEX)\b", "", question, flags=re.I)
    content_query = re.sub(r"\b(?:what|which|were|was|the|in|experiment|experiments|cite|supporting|passages)\b",
                           " ", content_query, flags=re.I)
    content_query = " ".join(content_query.split()).strip(" ?.,") or question
    result = collection.query(query_texts=[content_query], n_results=min(40, collection.count()),
                              **query_options, include=["documents", "metadatas", "distances"])
    # Combine semantic ranking with lexical evidence. Read text already in Chroma;
    # this requires no new embeddings or PDF downloads.
    corpus = collection.get(**query_options, include=["documents", "metadatas"])
    from collections import Counter
    import math

    def terms(text):
        return [word.rstrip("s") for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2]

    query_terms = set(terms(content_query))
    counts = [Counter(terms(text)) for text in corpus["documents"]]
    frequencies = Counter(word for count in counts for word in count)
    lexical = []
    for ident, count in zip(corpus["ids"], counts):
        score = sum(math.log(1 + len(counts) / (1 + frequencies[word])) *
                    count[word] / (count[word] + 1.2) for word in query_terms if count[word])
        if score:
            lexical.append((ident, score))
    lexical.sort(key=lambda pair: pair[1], reverse=True)
    scores = {}
    distances = dict(zip(result["ids"][0], result["distances"][0]))
    for ranking in (result["ids"][0], [ident for ident, _ in lexical[:40]]):
        for position, ident in enumerate(ranking):
            scores[ident] = scores.get(ident, 0) + 1 / (60 + position)
    records = {ident: (text, metadata) for ident, text, metadata in
               zip(corpus["ids"], corpus["documents"], corpus["metadatas"])}
    passages = []
    pages_seen = set()
    for ident in sorted(scores, key=scores.get, reverse=True):
        text, metadata = records[ident]
        distance = distances.get(ident)
        if max_distance is not None and (distance is None or distance > max_distance):
            continue
        if len(text.split()) < 25:
            continue
        if re.search(r"\.{6,}", text):
            continue  # PDF tables of contents point to evidence but contain none.
        page_key = (metadata.get("document_id", metadata.get("title")), metadata["page"])
        if page_key in pages_seen:
            continue
        pages_seen.add(page_key)
        # Join adjacent overlapping chunks from the same PDF page for context.
        neighbours = [(other_id, other_text) for other_id, (other_text, other_meta) in records.items()
                      if other_meta.get("document_id", other_meta.get("title")) == page_key[0]
                      and other_meta["page"] == page_key[1]
                      and abs(int(other_id) - int(ident)) <= 1]
        neighbours.sort(key=lambda item: int(item[0]))
        passages.append({**metadata, "text": "\n".join(t for _, t in neighbours),
                         "retrieval_score": scores[ident]})
        if len(passages) >= limit:
            break
    return passages
