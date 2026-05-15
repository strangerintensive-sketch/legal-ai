import os
import json
import logging
import sqlite3
from google import genai
from app.models.database import get_conn

logger = logging.getLogger(__name__)

# Defer client initialization until needed
_client = None

def get_client():
    global _client
    if _client is None:
        logger.info("🔧 Initializing Gemini client for drafting...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("❌ GOOGLE_API_KEY not set")
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in .env file.")
        _client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client initialized")
    return _client


def get_preference_rules() -> list[str]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT rule FROM preference_rules ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return [row["rule"] for row in rows]


def generate_draft(raw_text: str, structured_fields: dict, evidence_chunks: list[dict]) -> dict:
    logger.info(f"📝 Draft generation started: {len(evidence_chunks)} evidence chunks")
    evidence_text = "\n\n".join([
        f"[Chunk {i+1} | Score: {c['score']:.2f}]\n{c['text']}"
        for i, c in enumerate(evidence_chunks)
    ])
    logger.info(f"  📊 Evidence text prepared: {len(evidence_text)} chars")

    logger.info("🔍 Retrieving preference rules from database...")
    rules = get_preference_rules()
    logger.info(f"  ✅ {len(rules)} rules loaded")
    if rules:
        logger.info(f"  Rules: {[r[:50] + '...' if len(r) > 50 else r for r in rules]}")
    
    rules_section = ""
    if rules:
        rules_section = "\n\nOPERATOR PREFERENCES (apply these to your draft):\n" + "\n".join(
            f"- {r}" for r in rules
        )

    fields_summary = f"""
Case Number: {structured_fields.get('case_number', 'N/A')}
Document Type: {structured_fields.get('document_type', 'N/A')}
Parties: {', '.join(structured_fields.get('parties', [])) or 'N/A'}
Jurisdiction: {structured_fields.get('jurisdiction', 'N/A')}
Dates: {', '.join(structured_fields.get('dates', [])) or 'N/A'}
Key Facts: {'; '.join(structured_fields.get('key_facts', [])) or 'N/A'}
""".strip()

    system_prompt = f"""You are a legal analyst at a law firm. Your job is to produce a clear, structured Case Fact Summary based ONLY on the provided document evidence. 

Rules:
- Every claim must be grounded in the provided evidence chunks.
- Do NOT fabricate facts not present in the evidence.
- Cite evidence chunks using [Chunk N] notation inline.
- Structure the output with these sections: Overview, Key Parties, Timeline of Events, Critical Facts, Open Questions.
- Be concise. This is a first-pass internal memo, not a final document.{rules_section}"""

    user_prompt = f"""STRUCTURED FIELDS EXTRACTED:
{fields_summary}

RETRIEVED EVIDENCE:
{evidence_text}

Generate a Case Fact Summary grounded in the above evidence."""

    logger.info("🤖 Calling Gemini API for draft generation...")
    client = get_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            max_output_tokens=1500,
            system_instruction=system_prompt
        )
    )

    draft_content = response.text.strip()
    logger.info(f"📨 Draft generated: {len(draft_content)} chars")
    logger.info(f"✅ Draft generation complete")

    return {
        "content": draft_content,
        "evidence": evidence_chunks
    }
