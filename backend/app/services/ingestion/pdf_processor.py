"""PDF ingestion: validation, text extraction, and metadata generation.

Everything here operates purely on in-memory bytes so uploaded PDFs are
never written to disk. Extracted page text stays inside this module's
return value only long enough for the caller to compute counts and a short
preview -- it is never logged.
"""

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF

from app.core.exceptions import (
    CorruptedPDFError,
    EmptyFileError,
    EncryptedPDFError,
    InvalidContentTypeError,
    InvalidFileExtensionError,
    InvalidFileSignatureError,
    PolicyLensError,
)

PDF_SIGNATURE = b"%PDF-"
ALLOWED_CONTENT_TYPE = "application/pdf"
PREVIEW_CHAR_LIMIT = 200
MAX_FILENAME_LENGTH = 150
DEFAULT_FILENAME = "document.pdf"
_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._\- ]")


@dataclass
class PageContent:
    """Extracted text and metadata for a single PDF page."""

    page_number: int
    source_filename: str
    text: str
    character_count: int


@dataclass
class ExtractedDocument:
    """Result of successfully validating and processing a PDF."""

    document_id: str
    filename: str
    page_count: int
    character_count: int
    preview: str
    pages: List[PageContent] = field(default_factory=list)


def sanitize_filename(original_filename: str) -> str:
    """Reduce a client-supplied filename to a safe basename for display/storage.

    Strips directory components (defeating path traversal), removes any
    character outside a conservative allow-list, and bounds the length.

    Both "/" and "\\" are treated as separators regardless of host OS: a
    stdlib ``pathlib.Path`` would only strip "\\" on Windows (``PosixPath``
    treats it as a literal character), which would let a backslash-based
    traversal attempt survive sanitization on a Linux deployment even
    though it does not on a Windows dev machine.
    """

    name = (original_filename or "").strip()
    if not name:
        return DEFAULT_FILENAME

    name = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name:
        return DEFAULT_FILENAME

    name = _SAFE_FILENAME_PATTERN.sub("_", name)
    name = name.strip(" .")
    if not name:
        return DEFAULT_FILENAME

    if len(name) > MAX_FILENAME_LENGTH:
        stem, ext = os.path.splitext(name)
        name = stem[: max(MAX_FILENAME_LENGTH - len(ext), 1)] + ext

    return name


def _validate_not_empty(data: bytes) -> None:
    if len(data) == 0:
        raise EmptyFileError("The uploaded file is empty.")


def _validate_extension(filename: str) -> None:
    if not filename or not filename.lower().endswith(".pdf"):
        raise InvalidFileExtensionError("Only files with a .pdf extension are supported.")


def _validate_content_type(content_type: Optional[str]) -> None:
    if content_type != ALLOWED_CONTENT_TYPE:
        raise InvalidContentTypeError(
            f"Invalid content type. Expected '{ALLOWED_CONTENT_TYPE}'."
        )


def _validate_signature(data: bytes) -> None:
    if not data.startswith(PDF_SIGNATURE):
        raise InvalidFileSignatureError("The file does not have a valid PDF signature.")


def _build_preview(pages: List[PageContent]) -> str:
    source = next((page.text for page in pages if page.text.strip()), "")
    stripped = source.strip()
    if len(stripped) <= PREVIEW_CHAR_LIMIT:
        return stripped
    return stripped[:PREVIEW_CHAR_LIMIT] + "..."


def process_pdf(data: bytes, original_filename: str, content_type: Optional[str]) -> ExtractedDocument:
    """Validate and extract a PDF's contents from raw bytes.

    Raises a `PolicyLensError` subclass for any validation failure. Callers
    are expected to translate these into HTTP responses.
    """

    _validate_not_empty(data)
    _validate_extension(original_filename)
    _validate_content_type(content_type)
    _validate_signature(data)

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise CorruptedPDFError("The uploaded file could not be read as a valid PDF.") from exc

    try:
        # Checked deliberately even when no user password was supplied: a
        # PDF that is encrypted for any reason (owner-only restrictions or a
        # required user password) is rejected, not just ones requiring a
        # password to open.
        if doc.is_encrypted:
            raise EncryptedPDFError("Encrypted or password-protected PDFs are not supported.")

        if doc.page_count == 0:
            raise CorruptedPDFError("The PDF does not contain any pages.")

        sanitized_name = sanitize_filename(original_filename)
        pages: List[PageContent] = []
        total_characters = 0

        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = page.get_text("text") or ""
            pages.append(
                PageContent(
                    page_number=index + 1,
                    source_filename=sanitized_name,
                    text=text,
                    character_count=len(text),
                )
            )
            total_characters += len(text)
    except PolicyLensError:
        raise
    except Exception as exc:
        raise CorruptedPDFError("The uploaded file could not be processed as a valid PDF.") from exc
    finally:
        doc.close()

    return ExtractedDocument(
        document_id=hashlib.sha256(data).hexdigest(),
        filename=sanitized_name,
        page_count=len(pages),
        character_count=total_characters,
        preview=_build_preview(pages),
        pages=pages,
    )
