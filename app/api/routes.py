import uuid
import json
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.extractor import extract_text, extract_structured_fields
from app.core.retriever import index_document, retrieve
from app.core.drafter import generate_draft
from app.core.feedback import save_edit
from app.models.database import get_conn

logger = logging.getLogger(__name__)
router = APIRouter()


# ── 1. Upload and process document ──────────────────────────────────────────

@router.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"📄 Upload started: {file.filename}")
    file_bytes = await file.read()
    filename = file.filename or "upload"
    logger.info(f"📊 File size: {len(file_bytes)} bytes")

    try:
        logger.info("🔍 Step 1: Extracting raw text...")
        raw_text = extract_text(filename, file_bytes)
        logger.info(f"✅ Text extracted: {len(raw_text)} characters")
    except Exception as e:
        logger.error(f"❌ Extraction failed: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Extraction failed: {str(e)}")

    if not raw_text.strip():
        logger.error("❌ No text extracted from document")
        raise HTTPException(status_code=422, detail="No text could be extracted from document.")

    logger.info("🔍 Step 2: Extracting structured fields (calling Gemini)...")
    structured_fields = extract_structured_fields(raw_text)
    logger.info(f"✅ Structured fields extracted: {list(structured_fields.keys())}")

    doc_id = str(uuid.uuid4())
    logger.info(f"🆔 Document ID assigned: {doc_id}")

    logger.info("🔍 Step 3: Indexing document in ChromaDB...")
    index_document(doc_id, raw_text)
    logger.info("✅ Document indexed")

    logger.info("🔍 Step 4: Storing in SQLite database...")
    conn = get_conn()
    conn.execute(
        "INSERT INTO documents (id, filename, raw_text, structured_fields) VALUES (?, ?, ?, ?)",
        (doc_id, filename, raw_text, json.dumps(structured_fields))
    )
    conn.commit()
    conn.close()
    logger.info("✅ Document stored in database")
    logger.info(f"🎉 Upload complete: {doc_id}")

    return {
        "document_id": doc_id,
        "filename": filename,
        "char_count": len(raw_text),
        "raw_text": raw_text[:50000] + ("..." if len(raw_text) > 50000 else ""),
        "structured_fields": structured_fields
    }


# ── 2. View extracted content ───────────────────────────────────────────────

@router.get("/documents/{doc_id}/extracted")
def get_extracted(doc_id: str):
    logger.info(f"📄 Fetching extracted content for {doc_id[:8]}...")
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    conn.close()

    if not row:
        logger.error(f"  ❌ Document not found: {doc_id}")
        raise HTTPException(status_code=404, detail="Document not found.")

    logger.info(f"  ✅ Retrieved: {row['filename']}")
    return {
        "document_id": doc_id,
        "filename": row["filename"],
        "char_count": len(row["raw_text"]),
        "raw_text": row["raw_text"][:50000] + ("..." if len(row["raw_text"]) > 50000 else ""),
        "structured_fields": json.loads(row["structured_fields"])
    }


# ── 3. Generate draft ────────────────────────────────────────────────────────

class DraftRequest(BaseModel):
    query: str = "Summarize this legal document and extract key case facts"


@router.post("/documents/{doc_id}/draft")
def create_draft(doc_id: str, body: DraftRequest = DraftRequest()):
    logger.info(f"✍️  Draft request received for {doc_id[:8]}")
    logger.info(f"  Query: '{body.query[:60]}{'...' if len(body.query) > 60 else ''}'")
    
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    conn.close()

    if not row:
        logger.error(f"  ❌ Document not found")
        raise HTTPException(status_code=404, detail="Document not found.")

    raw_text = row["raw_text"]
    structured_fields = json.loads(row["structured_fields"])

    logger.info(f"  🔎 Step 1: Retrieving evidence chunks...")
    evidence_chunks = retrieve(body.query, doc_id=doc_id, top_k=5)
    logger.info(f"  ✅ Retrieved {len(evidence_chunks)} evidence chunks")
    for i, chunk in enumerate(evidence_chunks, 1):
        logger.info(f"    [{i}] Score: {chunk.get('score', 0):.2f} - {chunk['text'][:50]}...")

    if not evidence_chunks:
        logger.error(f"  ❌ No evidence found")
        raise HTTPException(status_code=422, detail="No relevant evidence found in document.")

    logger.info(f"  🔎 Step 2: Generating draft...")
    result = generate_draft(raw_text, structured_fields, evidence_chunks)

    draft_id = str(uuid.uuid4())
    logger.info(f"  ✅ Draft generated: {draft_id[:8]}")

    logger.info(f"  🔎 Step 3: Storing draft in database...")
    conn = get_conn()
    conn.execute(
        "INSERT INTO drafts (id, document_id, content, evidence) VALUES (?, ?, ?, ?)",
        (draft_id, doc_id, result["content"], json.dumps(result["evidence"]))
    )
    conn.commit()
    conn.close()
    logger.info(f"  ✅ Draft stored")
    logger.info(f"🎉 Draft creation complete: {draft_id[:8]}")

    return {
        "draft_id": draft_id,
        "document_id": doc_id,
        "content": result["content"],
        "evidence_count": len(evidence_chunks)
    }


# ── 4. Submit operator edit ──────────────────────────────────────────────────

class EditRequest(BaseModel):
    edited_content: str


@router.post("/drafts/{draft_id}/edit")
def submit_edit(draft_id: str, body: EditRequest):
    logger.info(f"🔧 Edit submission received for draft {draft_id[:8]}")
    
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM drafts WHERE id = ?", (draft_id,)
    ).fetchone()
    conn.close()

    if not row:
        logger.error(f"  ❌ Draft not found")
        raise HTTPException(status_code=404, detail="Draft not found.")

    original = row["content"]
    logger.info(f"  📝 Original draft: {len(original)} chars")
    logger.info(f"  📝 Edited draft: {len(body.edited_content)} chars")
    
    result = save_edit(draft_id, original, body.edited_content)
    logger.info(f"  ✅ Edit processed with {len(result['extracted_rules'])} rules")
    logger.info(f"🎉 Edit complete: {result['edit_id'][:8]}")

    return {
        "edit_id": result["edit_id"],
        "draft_id": draft_id,
        "extracted_rules": result["extracted_rules"],
        "message": f"Captured {len(result['extracted_rules'])} preference rule(s) for future drafts."
    }


# ── 5. Inspect evidence ──────────────────────────────────────────────────────

@router.get("/drafts/{draft_id}/evidence")
def get_evidence(draft_id: str):
    logger.info(f"📋 Fetching evidence for draft {draft_id[:8]}...")
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM drafts WHERE id = ?", (draft_id,)
    ).fetchone()
    conn.close()

    if not row:
        logger.error(f"  ❌ Draft not found")
        raise HTTPException(status_code=404, detail="Draft not found.")

    evidence = json.loads(row["evidence"])
    logger.info(f"  ✅ Retrieved {len(evidence)} evidence chunks")
    return {
        "draft_id": draft_id,
        "evidence_chunks": evidence
    }


# ── 6. List preference rules (bonus) ─────────────────────────────────────────

@router.get("/rules")
def list_rules():
    logger.info(f"📜 Fetching all preference rules...")
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM preference_rules ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    logger.info(f"  ✅ Retrieved {len(rows)} rules")
    return {
        "rules": [{"id": r["id"], "rule": r["rule"], "created_at": r["created_at"]} for r in rows]
    }


# ── 7. List all documents ─────────────────────────────────────────────────────

@router.get("/documents")
def list_documents():
    logger.info(f"📚 Listing all documents...")
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, filename, created_at FROM documents ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    logger.info(f"  ✅ Retrieved {len(rows)} documents")
    return {
        "documents": [dict(r) for r in rows]
    }
