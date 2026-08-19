"""
LangGraph OCR routing.

For every page in a PDF we decide, page by page, whether normal text
extraction (PyMuPDF) is enough, or whether the page is scanned/handwritten
and needs to go through an OCR model (TrOCR) instead.

Graph shape:

    START -> classify_page --(needs_ocr=True)--> run_ocr -> END
                            --(needs_ocr=False)-----------> END

Why a graph instead of a plain if/else: it makes the routing decision an
explicit, inspectable step (you can log/trace exactly which path each page
took), and it's easy to extend later - e.g. add a "low-confidence OCR ->
retry with different preprocessing" branch - without rewriting control flow.
"""

from io import BytesIO
from typing import TypedDict

import fitz  # PyMuPDF
from PIL import Image
from langgraph.graph import StateGraph, END

# TrOCR is loaded lazily - only the first time a scanned/handwritten page
# is actually found - since loading the model has a real cost and most
# pages in a normal, digitally-created PDF never need it.
_trocr_processor = None
_trocr_model = None

MIN_TEXT_CHARS = 20  # below this, we treat the page as "no usable text layer"


class PageState(TypedDict):
    pdf_path: str
    page_num: int  # 0-indexed
    text: str
    needs_ocr: bool
    ocr_used: bool


def _load_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        _trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return _trocr_processor, _trocr_model


def classify_page(state: PageState) -> PageState:
    """Try normal text extraction first. If there's basically nothing
    there, flag the page for OCR instead of guessing why."""
    doc = fitz.open(state["pdf_path"])
    page = doc[state["page_num"]]
    text = page.get_text()
    doc.close()

    if len(text.strip()) < MIN_TEXT_CHARS:
        return {**state, "text": "", "needs_ocr": True, "ocr_used": False}

    return {**state, "text": text, "needs_ocr": False, "ocr_used": False}


def run_ocr(state: PageState) -> PageState:
    """Render the page as an image and run TrOCR on it."""
    processor, model = _load_trocr()

    doc = fitz.open(state["pdf_path"])
    page = doc[state["page_num"]]
    pix = page.get_pixmap(dpi=200)
    doc.close()

    image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {**state, "text": text, "ocr_used": True}


def _route(state: PageState) -> str:
    return "run_ocr" if state["needs_ocr"] else END


def build_ocr_graph():
    graph = StateGraph(PageState)
    graph.add_node("classify_page", classify_page)
    graph.add_node("run_ocr", run_ocr)

    graph.set_entry_point("classify_page")
    graph.add_conditional_edges(
        "classify_page",
        _route,
        {"run_ocr": "run_ocr", END: END},
    )
    graph.add_edge("run_ocr", END)

    return graph.compile()


_ocr_graph = None


def get_page_text(pdf_path: str, page_num: int) -> tuple[str, bool]:
    """Run one page through the OCR-routing graph.
    Returns (text, ocr_used)."""
    global _ocr_graph
    if _ocr_graph is None:
        _ocr_graph = build_ocr_graph()

    initial_state: PageState = {
        "pdf_path": pdf_path,
        "page_num": page_num,
        "text": "",
        "needs_ocr": False,
        "ocr_used": False,
    }
    result = _ocr_graph.invoke(initial_state)
    return result["text"], result["ocr_used"]
