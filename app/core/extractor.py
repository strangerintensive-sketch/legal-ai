import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document as DocxDocument
import io
import re
import json
import os
import logging
import tempfile
from pathlib import Path
from google import genai
from app.core import extract_pdf_text

logger = logging.getLogger(__name__)

# Defer client initialization until needed
_client = None

def get_client():
    global _client
    if _client is None:
        logger.info("🔧 Initializing Gemini client...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("❌ GOOGLE_API_KEY not set")
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in .env file.")
        _client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client initialized")
    return _client

MIN_TEXT_YIELD = 100  # characters threshold to decide OCR fallback for individual pages


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF with reading-order awareness and multi-column support.
    
    Uses advanced layout analysis to:
    - Preserve reading order across single/multi-column layouts
    - Detect and deduplicate headers
    - Maintain semantic grouping of text blocks
    - Fall back to OCR for low-yield pages
    """
    logger.info("📄 Processing PDF file with advanced extraction...")
    
    # Write bytes to temporary file (extract_pdf_text requires file path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        # Extract with reading-order awareness
        logger.info("🔍 Analyzing PDF structure (columns, headers, reading order)...")
        text_main = extract_pdf_text.extract(tmp_path)
        logger.info(f"✅ Advanced extraction complete: {len(text_main)} chars")
        
        # Check if extraction was successful; if minimal text, try OCR on individual pages
        if len(text_main.strip()) < MIN_TEXT_YIELD:
            logger.warning(f"⚠️ Low extraction yield ({len(text_main)} chars), attempting OCR fallback...")
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages_text = []
            logger.info(f"📖 PDF has {doc.page_count} pages")
            
            for page_num, page in enumerate(doc, 1):
                # Try text-layer first
                text = page.get_text()
                if len(text.strip()) >= MIN_TEXT_YIELD:
                    logger.info(f"  ✅ Page {page_num}: extracted {len(text)} chars via text-layer")
                    pages_text.append(text)
                else:
                    # Low yield, use OCR
                    logger.info(f"  🔍 Page {page_num}: low yield ({len(text)} chars), using OCR...")
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))
                    ocr_text = pytesseract.image_to_string(img)
                    logger.info(f"  ✅ Page {page_num}: OCR extracted {len(ocr_text)} chars")
                    pages_text.append(ocr_text)
            
            doc.close()
            text_main = "\n\n".join(pages_text)
            logger.info(f"📊 OCR fallback complete: {sum(len(p) for p in pages_text)} total chars")
        
        return text_main
        
    except Exception as e:
        logger.error(f"❌ Advanced extraction failed: {str(e)}, falling back to standard PyMuPDF...")
        # Final fallback to standard extraction
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        logger.info(f"📖 PDF has {doc.page_count} pages")
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if len(text.strip()) >= MIN_TEXT_YIELD:
                logger.info(f"  ✅ Page {page_num}: extracted {len(text)} chars")
                pages_text.append(text)
            else:
                logger.info(f"  🔍 Page {page_num}: attempting OCR ({len(text)} chars initial)...")
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                ocr_text = pytesseract.image_to_string(img)
                logger.info(f"  ✅ Page {page_num}: OCR extracted {len(ocr_text)} chars")
                pages_text.append(ocr_text)
        
        doc.close()
        return "\n\n".join(pages_text)
        
    finally:
        # Clean up temporary file
        try:
            Path(tmp_path).unlink()
        except Exception as e:
            logger.debug(f"Could not delete temporary file: {e}")


def extract_text_from_image(file_bytes: bytes) -> str:
    logger.info("🖼️ Processing image file...")
    img = Image.open(io.BytesIO(file_bytes))
    ocr_text = pytesseract.image_to_string(img)
    logger.info(f"✅ Image OCR extracted: {len(ocr_text)} chars")
    return ocr_text


def extract_text_from_txt(file_bytes: bytes) -> str:
    logger.info("📝 Processing text file...")
    text = file_bytes.decode("utf-8", errors="replace")
    logger.info(f"✅ Text file decoded: {len(text)} chars")
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    logger.info("📖 Processing DOCX file...")
    doc = DocxDocument(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    logger.info(f"✅ DOCX extracted: {len(text)} chars")
    return text


def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()
    logger.info(f"🔍 Detecting file type: {ext}")
    
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        return extract_text_from_image(file_bytes)
    elif ext == ".docx":
        return extract_text_from_docx(file_bytes)
    else:
        return extract_text_from_txt(file_bytes)


def extract_structured_fields(raw_text: str) -> dict:
    """Use Gemini to pull key structured fields from raw text."""
    logger.info("🤖 Calling Gemini API for field extraction...")
    prompt = f"""You are a legal document parser. Extract structured fields from the following legal document text.

Return ONLY a JSON object with these fields (use null if not found):
- case_number: string
- parties: list of strings (names of people/companies involved)
- dates: list of strings (any important dates mentioned)
- document_type: string (e.g. contract, motion, notice, memo, deposition)
- key_facts: list of strings (up to 5 most important facts)
- jurisdiction: string

Document text:
{raw_text[:4000]}

Return only the JSON object, no explanation."""

    client = get_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            max_output_tokens=1000
        )
    )

    raw = response.text.strip()
    logger.info(f"📨 Gemini response received ({len(raw)} chars)")
    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        fields = json.loads(raw)
        logger.info(f"✅ Fields parsed: case_number={fields.get('case_number')}, parties={len(fields.get('parties', []))} parties, doc_type={fields.get('document_type')}")
        return fields
    except Exception as e:
        logger.error(f"❌ Failed to parse Gemini response: {str(e)}")
        return {
            "case_number": None,
            "parties": [],
            "dates": [],
            "document_type": "unknown",
            "key_facts": [],
            "jurisdiction": None
        }
