"""Document text and regex-based information extraction."""

import io
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union


def extract_text_from_pdf(file_source: Union[str, Path, bytes, io.BytesIO]) -> str:
    """Extract full text from a PDF file path or byte stream using PyMuPDF."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as e:
            raise ImportError("PyMuPDF is required for PDF extraction. Run: pip install pymupdf") from e

    text_parts = []
    if isinstance(file_source, (str, Path)):
        doc = fitz.open(str(file_source))
    elif isinstance(file_source, bytes):
        doc = fitz.open(stream=file_source, filetype="pdf")
    elif isinstance(file_source, io.BytesIO):
        doc = fitz.open(stream=file_source.getvalue(), filetype="pdf")
    else:
        raise TypeError(f"Unsupported file source type: {type(file_source)}")

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text.strip())
    finally:
        doc.close()

    return "\n\n".join(text_parts).strip()


def extract_text_from_txt(file_source: Union[str, Path, bytes, io.BytesIO], encoding: str = "utf-8") -> str:
    """Extract text from a plain TXT file path or byte stream."""
    if isinstance(file_source, (str, Path)):
        with open(file_source, "r", encoding=encoding, errors="replace") as f:
            return f.read().strip()
    elif isinstance(file_source, bytes):
        return file_source.decode(encoding, errors="replace").strip()
    elif isinstance(file_source, io.BytesIO):
        return file_source.getvalue().decode(encoding, errors="replace").strip()
    else:
        raise TypeError(f"Unsupported file source type: {type(file_source)}")


def extract_text(file_source: Union[str, Path, bytes, io.BytesIO], filename: str = "") -> str:
    """Extract text from a file (PDF or TXT) auto-detected from filename or signature."""
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf(file_source)
    elif lower_name.endswith(".txt"):
        return extract_text_from_txt(file_source)

    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if path.suffix.lower() == ".pdf":
            return extract_text_from_pdf(file_source)
        return extract_text_from_txt(file_source)
    elif isinstance(file_source, (bytes, io.BytesIO)):
        b = file_source if isinstance(file_source, bytes) else file_source.getvalue()
        if b.startswith(b"%PDF"):
            return extract_text_from_pdf(file_source)
        return extract_text_from_txt(file_source)

    return extract_text_from_txt(file_source)


def extract_regex_fields(text: str, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract structured fields using regular expressions based on document category
    or cross-category extraction.
    """
    extracted: Dict[str, Any] = {}
    normalized_cat = category.lower().strip() if category else None

    # 1. Invoice fields
    if normalized_cat in ("invoice", None):
        inv_no_match = re.search(
            r"(?i)\b(?:invoice\s*(?:number|no\.?|#)|inv\s*(?:number|no\.?|#))\s*[:\-#]?\s*([A-Z0-9\-_/]+)",
            text,
        )
        if not inv_no_match:
            inv_no_match = re.search(r"\b(INV[-_]?\d+)\b", text, re.IGNORECASE)

        total_match = re.search(
            r"(?i)\b(?:total\s*(?:amount|due)?|grand\s*total|balance\s*due)[:\s]*[\$€£₹]?\s*([\d,]+\.?\d{0,2})",
            text,
        )
        due_date_match = re.search(
            r"(?i)\b(?:due\s*date|payment\s*due)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b)",
            text,
        )
        if inv_no_match:
            extracted["invoice_number"] = inv_no_match.group(1).strip()
        if total_match:
            extracted["total_amount"] = total_match.group(1).strip()
        if due_date_match:
            extracted["due_date"] = due_date_match.group(1).strip()

    # 2. Resume fields
    if normalized_cat in ("resume", None):
        email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
        phone_match = re.search(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text
        )
        skills_match = re.search(
            r"(?i)\b(?:skills|technologies|technical\s*skills|core\s*competencies)[:\s]*([^\n\r]+)",
            text,
        )
        if email_match:
            extracted["email"] = email_match.group(0).strip()
        if phone_match:
            extracted["phone"] = phone_match.group(0).strip()
        if skills_match:
            skills = [s.strip() for s in re.split(r"[,;|•]", skills_match.group(1)) if s.strip()]
            extracted["skills"] = skills[:10]

    # 3. Bank Statement fields
    if normalized_cat in ("bank_statement", None):
        account_match = re.search(
            r"(?i)\b(?:account\s*(?:number|no\.?|#))[:\s]*([X*\d]{4,18}|\d{4,18})", text
        )
        balance_match = re.search(
            r"(?i)\b(?:closing\s*balance|ending\s*balance|available\s*balance|current\s*balance)[:\s]*[\$€£₹]?\s*([\d,]+\.?\d{0,2})",
            text,
        )
        period_match = re.search(
            r"(?i)\b(?:statement\s*period|period)[:\s]*([A-Za-z0-9\s,\-\/]+(?:\d{4}))",
            text,
        )
        if account_match:
            extracted["account_number"] = account_match.group(1).strip()
        if balance_match:
            extracted["closing_balance"] = balance_match.group(1).strip()
        if period_match:
            extracted["statement_period"] = period_match.group(1).strip()

    # 4. Certificate fields
    if normalized_cat in ("certificate", None):
        recipient_match = re.search(
            r"(?i)(?:awarded\s*to|presented\s*to|certifies\s*that|proudly\s*presented\s*to)[ \t]+([a-zA-Z]+(?:[ \t]+[a-zA-Z]+){1,3})",
            text,
        )
        course_match = re.search(
            r"(?i)(?:for\s*(?:successfully\s*completing|completion\s*of)|course|credential)[:\s]*([^\n\r.]+)",
            text,
        )
        cert_id_match = re.search(
            r"(?i)\b(?:certificate\s*(?:id|number|no\.?|code)|credential\s*id)[:\s]*([A-Z0-9\-_]+)",
            text,
        )
        if recipient_match:
            extracted["recipient_name"] = recipient_match.group(1).strip()
        if course_match:
            extracted["course_name"] = course_match.group(1).strip()
        if cert_id_match:
            extracted["certificate_id"] = cert_id_match.group(1).strip()

    return extracted
