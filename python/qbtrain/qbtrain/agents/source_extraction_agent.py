# qbtrain/agents/source_extraction_agent.py
from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..ai.llm import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ExtractedLink(BaseModel):
    """A single URL the user explicitly wants to read from."""
    url: str = Field(..., min_length=1)
    reason: str = Field(default="", description="Why this link was identified")


class LinkExtractionResult(BaseModel):
    """LLM output: the links parsed out of the user query."""
    links: List[ExtractedLink] = Field(default_factory=list)


class LinkContent(BaseModel):
    link: str
    content: str


class FileContent(BaseModel):
    file: str
    content: str


class SourceExtractionResult(BaseModel):
    files: List[FileContent] = Field(default_factory=list)
    links: List[LinkContent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_LINK_EXTRACTION_SYSTEM_PROMPT = """\
You are a precise link-extraction assistant.
Given a user query, identify ONLY the URLs / links that the user explicitly \
wants to read content from.
Do NOT invent links. Do NOT include links the user merely mentions in passing \
or uses as examples of something else.
Return strict JSON matching the schema provided."""

_LINK_EXTRACTION_USER_PROMPT = """\
User query:
\"\"\"
{query}
\"\"\"

Extract all links the user wants to read from. Return JSON."""


# ---------------------------------------------------------------------------
# Web scraping helpers (Playwright)
# ---------------------------------------------------------------------------

def _scrape_main_content(url: str, timeout_ms: int = 30_000) -> str:
    """
    Use Playwright (sync) to load *url*, then extract the main body text
    as markdown.  Strips nav, header, footer, sidebar, and ad elements.
    """
    from playwright.sync_api import sync_playwright  # lazy import

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            # Remove noisy elements before extracting text
            selectors_to_remove = [
                "nav", "header", "footer",
                "[role='navigation']", "[role='banner']", "[role='contentinfo']",
                ".sidebar", "#sidebar", ".nav", ".navbar",
                ".footer", "#footer", ".header", "#header",
                ".cookie-banner", ".ad", ".ads", ".advertisement",
                ".popup", ".modal", "#cookie-consent",
                "script", "style", "noscript", "iframe",
            ]
            for sel in selectors_to_remove:
                page.evaluate(
                    "(sel) => document.querySelectorAll(sel).forEach(el => el.remove())",
                    sel,
                )

            # Prefer <main> or <article>; fall back to <body>
            main_el = (
                page.query_selector("main")
                or page.query_selector("article")
                or page.query_selector("[role='main']")
                or page.query_selector("body")
            )
            raw_text = main_el.inner_text() if main_el else ""
        finally:
            browser.close()

    # Light cleanup: collapse excessive blank lines
    lines = raw_text.splitlines()
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(stripped)

    return "\n".join(cleaned).strip()


# ---------------------------------------------------------------------------
# File processing helpers
# ---------------------------------------------------------------------------

def _process_file(filename: str, data: bytes) -> str:
    """
    Extract text content from a file given its name and raw bytes.
    Supports: .txt, .md, .pdf, .docx
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in ("txt", "md"):
        return data.decode("utf-8", errors="replace")

    if ext == "pdf":
        return _extract_pdf_text(data)

    if ext == "docx":
        return _extract_docx_text(data)

    raise ValueError(f"Unsupported file type: .{ext} (file: {filename})")


def _extract_pdf_text(data: bytes) -> str:
    import pymupdf  # PyMuPDF — fast, no Java dependency

    pages: list[str] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text()
            if text and text.strip():
                pages.append(text.strip())
    return "\n\n".join(pages)


def _extract_docx_text(data: bytes) -> str:
    from docx import Document  # python-docx

    doc = Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SourceExtractionAgent:
    """
    Takes a user query and an optional list of files, then:
    1. Uses an LLM to extract only the links the user wants to read from.
    2. Scrapes each link for its main body content via Playwright.
    3. Processes each uploaded file (docx, txt, md, pdf) into text.

    Returns ``SourceExtractionResult`` with ``files`` and ``links`` lists.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        scrape_timeout_ms: int = 30_000,
    ):
        self.llm_client = llm_client
        self.scrape_timeout_ms = scrape_timeout_ms

    # ------ link extraction via LLM ------

    def _extract_links(self, query: str, tracer=None) -> List[ExtractedLink]:
        user_prompt = _LINK_EXTRACTION_USER_PROMPT.format(query=query)
        raw: Dict[str, Any] = self.llm_client.json_response(
            prompt=user_prompt,
            system_prompt=_LINK_EXTRACTION_SYSTEM_PROMPT,
            schema=LinkExtractionResult,
            tracer=tracer,
            temperature=0,
            top_k=1,
        )
        parsed = LinkExtractionResult.model_validate(raw)
        return parsed.links

    # ------ web scraping ------

    def _scrape_links(self, links: List[ExtractedLink], tracer=None) -> List[LinkContent]:
        results: List[LinkContent] = []
        for link in links:
            try:
                import time
                t0 = time.time()
                content = _scrape_main_content(link.url, timeout_ms=self.scrape_timeout_ms)
                latency = round((time.time() - t0) * 1000)
                results.append(LinkContent(link=link.url, content=content))
                if tracer:
                    tracer.trace(
                        "SourceExtraction", "scrape",
                        operation="scrape_link",
                        link=link.url,
                        reason=link.reason,
                        latency_ms=latency,
                        content_length=len(content),
                        status="ok",
                    )
            except Exception as exc:
                logger.warning("Failed to scrape %s: %s", link.url, exc)
                results.append(LinkContent(link=link.url, content=f"[error] {exc}"))
                if tracer:
                    tracer.trace(
                        "SourceExtraction", "scrape",
                        operation="scrape_link",
                        link=link.url,
                        reason=link.reason,
                        latency_ms=round((time.time() - t0) * 1000),
                        status="error",
                        error=str(exc),
                    )
        return results

    # ------ file processing ------

    @staticmethod
    def _process_files(files_list: List[Tuple[str, bytes]], tracer=None) -> List[FileContent]:
        results: List[FileContent] = []
        for filename, data in files_list:
            try:
                import time
                t0 = time.time()
                content = _process_file(filename, data)
                latency = round((time.time() - t0) * 1000)
                results.append(FileContent(file=filename, content=content))
                if tracer:
                    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
                    tracer.trace(
                        "SourceExtraction", "file",
                        operation="process_file",
                        filename=filename,
                        file_type=ext,
                        file_size=len(data),
                        latency_ms=latency,
                        content_length=len(content),
                        status="ok",
                    )
            except Exception as exc:
                logger.warning("Failed to process file %s: %s", filename, exc)
                results.append(FileContent(file=filename, content=f"[error] {exc}"))
                if tracer:
                    tracer.trace(
                        "SourceExtraction", "file",
                        operation="process_file",
                        filename=filename,
                        file_size=len(data),
                        latency_ms=round((time.time() - t0) * 1000),
                        status="error",
                        error=str(exc),
                    )
        return results

    # ------ public API ------

    def extract(
        self,
        query: str,
        files_list: Optional[List[Tuple[str, bytes]]] = None,
        tracer=None,
    ) -> SourceExtractionResult:
        """
        Parameters
        ----------
        query : str
            The user's natural-language query that may contain URLs to read.
        files_list : list of (filename, bytes) tuples, optional
            Uploaded files to extract text from.
        tracer : optional
            Tracer instance for observability.

        Returns
        -------
        SourceExtractionResult
            ``{ files: [{file, content}, ...], links: [{link, content}, ...] }``
        """
        import time

        # 1. Extract links from the query via LLM
        t0 = time.time()
        extracted_links = self._extract_links(query, tracer=tracer)
        if tracer:
            tracer.trace(
                "SourceExtraction", "llm",
                operation="extract_links",
                latency_ms=round((time.time() - t0) * 1000),
                links_found=len(extracted_links),
                links=[{"url": l.url, "reason": l.reason} for l in extracted_links],
            )

        # 2. Scrape each link
        link_contents = self._scrape_links(extracted_links, tracer=tracer) if extracted_links else []

        # 3. Process uploaded files
        file_contents = self._process_files(files_list, tracer=tracer) if files_list else []

        # 4. Summary trace
        if tracer:
            successful_files = [fc.file for fc in file_contents if not fc.content.startswith("[error]")]
            failed_files = [fc.file for fc in file_contents if fc.content.startswith("[error]")]
            successful_links = [lc.link for lc in link_contents if not lc.content.startswith("[error]")]
            failed_links = [lc.link for lc in link_contents if lc.content.startswith("[error]")]
            tracer.trace(
                "SourceExtraction", "summary",
                operation="extract_complete",
                files_processed=len(file_contents),
                files_success=successful_files,
                files_failed=failed_files,
                links_scraped=len(link_contents),
                links_success=successful_links,
                links_failed=failed_links,
            )

        return SourceExtractionResult(files=file_contents, links=link_contents)
