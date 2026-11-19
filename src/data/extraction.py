"""Document text and regex-based information extraction with OCR fallback."""

import io
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def find_tesseract_binary() -> Optional[str]:
    """Locate Tesseract executable on the system across PATH and standard install directories."""
    found = shutil.which("tesseract")
    if found:
        return found

    common_win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in common_win_paths:
        if os.path.exists(p):
            return p
    return None


def find_tessdata_dir() -> Optional[str]:
    """Locate tessdata directory for PyMuPDF OCR or Tesseract engine."""
    env_prefix = os.environ.get("TESSDATA_PREFIX")
    if env_prefix and os.path.exists(env_prefix):
        return env_prefix

    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tessdata"),
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tessdata",
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
    return None


def extract_page_ocr_tesseract(page: Any, dpi: int = 200, lang: str = "eng") -> str:
    """
    Extract text from a PyMuPDF page using Tesseract OCR.
    Prefers pytesseract via high-resolution pixmap, with fallback to page.get_textpage_ocr.
    """
    # 1. Try pytesseract via rendered pixmap
    try:
        import pytesseract
        from PIL import Image

        tess_bin = find_tesseract_binary()
        if tess_bin:
            pytesseract.pytesseract.tesseract_cmd = tess_bin

        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang=lang)
        if text.strip():
            return text.strip()
    except Exception as e:
        logger.debug(f"pytesseract extraction failed: {e}")

    # 2. Try PyMuPDF native OCR (get_textpage_ocr)
    try:
        tessdata = find_tessdata_dir()
        kwargs: Dict[str, Any] = {"language": lang, "dpi": dpi}
        if tessdata:
            kwargs["tessdata"] = tessdata
        tp = page.get_textpage_ocr(**kwargs)
        text = page.get_text(textpage=tp)
        if text.strip():
            return text.strip()
    except Exception as e:
        logger.debug(f"PyMuPDF get_textpage_ocr failed: {e}")

    return ""


def extract_page_ocr_easyocr(page: Any, dpi: int = 200, lang: str = "en") -> str:
    """Extract text from a PyMuPDF page using EasyOCR (if installed)."""
    try:
        import easyocr
        import numpy as np

        pix = page.get_pixmap(dpi=dpi)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            img_np = img_np[:, :, :3]

        easyocr_lang = "en" if lang.startswith("eng") or lang == "en" else lang
        reader = easyocr.Reader([easyocr_lang], gpu=False)
        results = reader.readtext(img_np, detail=0)
        return "\n".join(results).strip()
    except Exception as e:
        logger.debug(f"EasyOCR extraction failed: {e}")
        return ""


def extract_page_ocr(
    page: Any,
    engine: str = "auto",
    dpi: int = 200,
    lang: str = "eng",
) -> str:
    """
    Extract text from a PDF page using the specified or auto-detected OCR engine.
    Supported engines: 'auto', 'tesseract', 'easyocr', 'pymupdf'.
    """
    engine_normalized = (engine or "auto").lower().strip()

    if engine_normalized in ("tesseract", "pymupdf"):
        return extract_page_ocr_tesseract(page, dpi=dpi, lang=lang)
    elif engine_normalized == "easyocr":
        res = extract_page_ocr_easyocr(page, dpi=dpi, lang=lang)
        if res:
            return res
        return extract_page_ocr_tesseract(page, dpi=dpi, lang=lang)
    else:  # auto: try Tesseract/PyMuPDF first, then EasyOCR
        res = extract_page_ocr_tesseract(page, dpi=dpi, lang=lang)
        if res:
            return res
        return extract_page_ocr_easyocr(page, dpi=dpi, lang=lang)


def extract_text_from_image(
    file_source: Union[str, Path, bytes, io.BytesIO],
    engine: str = "auto",
    lang: str = "eng",
) -> str:
    """Extract text from a standalone image (PNG, JPG, TIFF, BMP) using OCR."""
    from PIL import Image

    if isinstance(file_source, (str, Path)):
        img = Image.open(str(file_source))
    elif isinstance(file_source, bytes):
        img = Image.open(io.BytesIO(file_source))
    elif isinstance(file_source, io.BytesIO):
        img = Image.open(file_source)
    else:
        raise TypeError(f"Unsupported file source type: {type(file_source)}")

    # 1. Try pytesseract
    try:
        import pytesseract

        tess_bin = find_tesseract_binary()
        if tess_bin:
            pytesseract.pytesseract.tesseract_cmd = tess_bin
        text = pytesseract.image_to_string(img, lang=lang)
        if text.strip():
            return text.strip()
    except Exception as e:
        logger.debug(f"Image OCR via pytesseract failed: {e}")

    # 2. Try easyocr
    try:
        import easyocr
        import numpy as np

        img_np = np.array(img.convert("RGB"))
        easyocr_lang = "en" if lang.startswith("eng") or lang == "en" else lang
        reader = easyocr.Reader([easyocr_lang], gpu=False)
        return "\n".join(reader.readtext(img_np, detail=0)).strip()
    except Exception as e:
        logger.debug(f"Image OCR via easyocr failed: {e}")

    return ""


def extract_text_from_pdf(
    file_source: Union[str, Path, bytes, io.BytesIO],
    ocr_fallback: bool = True,
    ocr_engine: str = "auto",
    min_chars_per_page: int = 20,
    dpi: int = 200,
    lang: str = "eng",
) -> str:
    """
    Extract full text from a PDF file path or byte stream using PyMuPDF.
    If a page contains scanned images or lacks digital text, automatically falls back to OCR.
    """
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

            # Scanned / Image PDF check: if digital text is empty or too short, invoke OCR fallback
            if ocr_fallback and (not page_text or len(page_text.strip()) < min_chars_per_page):
                ocr_result = extract_page_ocr(page, engine=ocr_engine, dpi=dpi, lang=lang)
                if ocr_result.strip():
                    logger.info(f"OCR fallback applied to PDF page {page_idx + 1} (extracted {len(ocr_result)} chars)")
                    page_text = ocr_result

            if page_text and page_text.strip():
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


def extract_text(
    file_source: Union[str, Path, bytes, io.BytesIO],
    filename: str = "",
    ocr_fallback: bool = True,
    ocr_engine: str = "auto",
    min_chars_per_page: int = 20,
    dpi: int = 200,
    lang: str = "eng",
) -> str:
    """
    Extract text from a file (PDF, TXT, or Image) auto-detected from filename or signature.
    Falls back to OCR for scanned PDFs or pure image documents.
    """
    lower_name = filename.lower()
    image_exts = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")

    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf(
            file_source,
            ocr_fallback=ocr_fallback,
            ocr_engine=ocr_engine,
            min_chars_per_page=min_chars_per_page,
            dpi=dpi,
            lang=lang,
        )
    elif lower_name.endswith(".txt"):
        return extract_text_from_txt(file_source)
    elif any(lower_name.endswith(ext) for ext in image_exts):
        return extract_text_from_image(file_source, engine=ocr_engine, lang=lang)

    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if path.suffix.lower() == ".pdf":
            return extract_text_from_pdf(
                file_source,
                ocr_fallback=ocr_fallback,
                ocr_engine=ocr_engine,
                min_chars_per_page=min_chars_per_page,
                dpi=dpi,
                lang=lang,
            )
        elif path.suffix.lower() in image_exts:
            return extract_text_from_image(file_source, engine=ocr_engine, lang=lang)
        return extract_text_from_txt(file_source)
    elif isinstance(file_source, (bytes, io.BytesIO)):
        b = file_source if isinstance(file_source, bytes) else file_source.getvalue()
        if b.startswith(b"%PDF"):
            return extract_text_from_pdf(
                file_source,
                ocr_fallback=ocr_fallback,
                ocr_engine=ocr_engine,
                min_chars_per_page=min_chars_per_page,
                dpi=dpi,
                lang=lang,
            )
        # Check image signatures (PNG, JPEG, BMP)
        if b.startswith(b"\x89PNG\r\n\x1a\n") or b.startswith(b"\xff\xd8\xff") or b.startswith(b"BM"):
            return extract_text_from_image(file_source, engine=ocr_engine, lang=lang)
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
