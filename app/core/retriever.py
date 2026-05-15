import chromadb
import logging
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)

CHROMA_PATH = Path("data/chroma")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

_model = None
_chroma_client = None
_collection = None


def get_model():
    global _model
    if _model is None:
        logger.info("🤗 Loading embedding model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ Embedding model loaded")
    return _model


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        logger.info("📄 Initializing ChromaDB collection...")
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _chroma_client.get_or_create_collection("legal_docs")
        logger.info("✅ ChromaDB collection ready")
    return _collection


def chunk_text(text: str, doc_id: str) -> list[dict]:
    logger.info(f"  📝 Chunking text: {len(text)} chars into chunks of {CHUNK_SIZE} words...")
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    i = 0
    chunk_index = 0

    while i < len(words):
        chunk_words = words[i: i + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "id": f"{doc_id}_chunk_{chunk_index}",
            "text": chunk_text,
            "doc_id": doc_id,
            "chunk_index": chunk_index
        })
        i += step
        chunk_index += 1

    logger.info(f"  ✅ Created {len(chunks)} chunks")
    return chunks


def index_document(doc_id: str, raw_text: str):
    logger.info(f"📇 Indexing document {doc_id[:8]}...")
    collection = get_collection()
    model = get_model()
    chunks = chunk_text(raw_text, doc_id)

    if not chunks:
        logger.warning("  ⚠️ No chunks to index")
        return

    logger.info(f"  🔢 Generating embeddings for {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()
    ids = [c["id"] for c in chunks]
    metadatas = [{"doc_id": c["doc_id"], "chunk_index": c["chunk_index"]} for c in chunks]

    logger.info(f"  💾 Storing in ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    logger.info(f"  ✅ Document indexed successfully")


def retrieve(query: str, doc_id: str = None, top_k: int = TOP_K) -> list[dict]:
    logger.info(f"🔎 Retrieving top {top_k} chunks for query...")
    collection = get_collection()
    model = get_model()

    logger.info(f"  🔤 Encoding query: '{query[:60]}{'...' if len(query) > 60 else ''}'")
    query_embedding = model.encode([query]).tolist()

    where = {"doc_id": doc_id} if doc_id else None
    if doc_id:
        logger.info(f"  📄 Filtering by doc_id: {doc_id[:8]}")

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        score = 1 - results["distances"][0][i]  # Convert distance to similarity
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "score": float(1 - results["distances"][0][i])
        })

    logger.info(f"  ✅ Retrieved {len(chunks)} chunks (avg score: {sum(c['score'] for c in chunks)/max(len(chunks),1):.2f})")
    return chunks


def delete_document_chunks(doc_id: str):
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
