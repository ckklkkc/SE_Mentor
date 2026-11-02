"""Run Docling only as a picture-region detector in an isolated environment."""
from __future__ import annotations

import argparse
import json
import logging
from importlib.metadata import version
from pathlib import Path


def export_bundle(doc):
    """Export page geometry only; never export Docling text, headings, or tables."""
    pages = {
        number: {"page": number, "regions": []}
        for number in sorted(doc.pages)
    }
    categories = {"picture": "example_figure", "chart": "diagram"}
    for item, _ in doc.iterate_items():
        category = categories.get(item.label.value)
        if category is None:
            continue
        for prov in getattr(item, "prov", []):
            number = prov.page_no
            page = doc.pages[number]
            box = prov.bbox.to_top_left_origin(page_height=page.size.height)
            bbox = [
                round(box.l / page.size.width * 1000),
                round(box.t / page.size.height * 1000),
                round(box.r / page.size.width * 1000),
                round(box.b / page.size.height * 1000),
            ]
            pages[number]["regions"].append({
                "category": category,
                "bbox": [max(0, min(1000, value)) for value in bbox],
                "reason": "Docling picture/chart layout",
            })
    return {
        "schema_version": 2,
        "strategy": "image_regions_only",
        "docling_version": version("docling"),
        "pages": list(pages.values()),
    }


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("output")
    args = parser.parse_args()

    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    # Marker owns text and table conversion. Disabling these Docling stages both
    # avoids conflicts and prevents OCR/table guesses from changing the lesson.
    options.do_ocr = False
    options.do_table_structure = False
    options.generate_page_images = False
    options.generate_picture_images = False
    result = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    }).convert(Path(args.pdf))
    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(
            f"Docling conversion incomplete: {result.status}; {result.errors}"
        )
    Path(args.output).write_text(
        json.dumps(export_bundle(result.document), ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
