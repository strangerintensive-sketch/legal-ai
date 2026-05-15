import os
import json
import re
import uuid
import logging
from google import genai
from app.models.database import get_conn

logger = logging.getLogger(__name__)

# Defer client initialization until needed
_client = None

def get_client():
    global _client
    if _client is None:
        logger.info("🔧 Initializing Gemini client for feedback...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("❌ GOOGLE_API_KEY not set")
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in .env file.")
        _client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client initialized")
    return _client


def extract_rules_from_diff(original: str, edited: str) -> list[str]:
    logger.info(f"📝 Analyzing edits for rules...")
    logger.info(f"  Original: {len(original)} chars, Edited: {len(edited)} chars")
    prompt = f"""You are analyzing how a legal analyst edited a draft document. 
Extract reusable writing preference rules from the differences between the original and edited versions.

ORIGINAL DRAFT:
{original}

EDITED DRAFT:
{edited}

Identify patterns in the edits — things like formatting preferences, tone changes, structural preferences, 
what information the editor added or removed, how they prefer things phrased.

Return ONLY a JSON array of short, actionable rule strings (max 5 rules). 
Each rule should be a single sentence starting with a verb.
Example: ["Always include the filing date in the Overview section", "Use bullet points for Key Facts instead of prose"]

Return only the JSON array, no explanation."""

    logger.info("🤖 Calling Gemini API for diff analysis...")
    client = get_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            max_output_tokens=500
        )
    )

    raw = response.text.strip()
    logger.info(f"  📨 Gemini response received: {len(raw)} chars")
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        rules = json.loads(raw)
        if isinstance(rules, list):
            extracted = [str(r) for r in rules[:5]]
            logger.info(f"  ✅ Rules extracted: {len(extracted)} rules")
            for i, rule in enumerate(extracted, 1):
                logger.info(f"    {i}. {rule[:70]}..." if len(rule) > 70 else f"    {i}. {rule}")
            return extracted
    except Exception as e:
        logger.error(f"  ❌ Failed to parse rules: {str(e)}")
    return []


def save_edit(draft_id: str, original: str, edited: str) -> dict:
    logger.info(f"💾 Saving edit for draft {draft_id[:8]}...")
    rules = extract_rules_from_diff(original, edited)

    edit_id = str(uuid.uuid4())
    logger.info(f"  🆔 Edit ID: {edit_id[:8]}")
    
    logger.info("🔍 Storing in database...")
    conn = get_conn()

    conn.execute(
        "INSERT INTO edits (id, draft_id, original, edited, extracted_rules) VALUES (?, ?, ?, ?, ?)",
        (edit_id, draft_id, original, edited, json.dumps(rules))
    )

    for rule in rules:
        conn.execute(
            "INSERT INTO preference_rules (id, rule, source_edit_id) VALUES (?, ?, ?)",
            (str(uuid.uuid4()), rule, edit_id)
        )

    conn.commit()
    conn.close()
    logger.info(f"  ✅ Edit saved with {len(rules)} new rules")

    return {
        "edit_id": edit_id,
        "extracted_rules": rules
    }
