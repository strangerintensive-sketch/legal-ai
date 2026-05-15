"""
End-to-end demo of the Legal AI pipeline.
Run: python scripts/demo.py
"""

import requests
import json
import time
from pathlib import Path

BASE = "http://localhost:8000/api"


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)


def main():
    step("STEP 1 — Upload document")
    sample = Path("sample_docs/sample_contract.txt")
    with open(sample, "rb") as f:
        resp = requests.post(f"{BASE}/documents", files={"file": (sample.name, f, "text/plain")})
    resp.raise_for_status()
    doc = resp.json()
    doc_id = doc["document_id"]
    print(json.dumps(doc, indent=2))

    step("STEP 2 — View extracted content")
    resp = requests.get(f"{BASE}/documents/{doc_id}/extracted")
    extracted = resp.json()
    print("Structured fields:")
    print(json.dumps(extracted["structured_fields"], indent=2))

    step("STEP 3 — Generate first draft")
    resp = requests.post(f"{BASE}/documents/{doc_id}/draft", json={
        "query": "Summarize the settlement terms, parties, and key obligations"
    })
    resp.raise_for_status()
    draft = resp.json()
    draft_id = draft["draft_id"]
    print(f"\nDraft ID: {draft_id}")
    print(f"\n--- DRAFT CONTENT ---\n{draft['content']}")

    step("STEP 4 — Inspect evidence")
    resp = requests.get(f"{BASE}/drafts/{draft_id}/evidence")
    evidence = resp.json()
    print(f"Evidence chunks used: {len(evidence['evidence_chunks'])}")
    for i, chunk in enumerate(evidence["evidence_chunks"][:2]):
        print(f"\n[Chunk {i+1}] Score: {chunk['score']:.3f}")
        print(chunk["text"][:300] + "...")

    step("STEP 5 — Operator submits edit")
    edited = draft["content"] + "\n\n[OPERATOR NOTE] Always include the total monetary value in the Overview section. Use bullet points for all financial figures."
    resp = requests.post(f"{BASE}/drafts/{draft_id}/edit", json={"edited_content": edited})
    resp.raise_for_status()
    edit_result = resp.json()
    print(json.dumps(edit_result, indent=2))

    step("STEP 6 — View learned preference rules")
    resp = requests.get(f"{BASE}/rules")
    rules = resp.json()
    print(json.dumps(rules, indent=2))

    step("STEP 7 — Generate improved second draft (with rules applied)")
    resp = requests.post(f"{BASE}/documents/{doc_id}/draft", json={
        "query": "Summarize the settlement terms, parties, and key obligations"
    })
    resp.raise_for_status()
    draft2 = resp.json()
    print(f"\n--- IMPROVED DRAFT ---\n{draft2['content']}")

    step("DONE — Full pipeline demonstrated ✅")


if __name__ == "__main__":
    main()
