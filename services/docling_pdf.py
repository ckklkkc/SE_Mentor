"""Keep Marker Markdown intact and use Docling only to locate PDF figures."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from services.image_captioning import ExternalMultimodalCaptioner
from services.pdf_visuals import (
    FIGURE_CATEGORIES,
    LayoutRegion,
    PageLayout,
    _find_page_starts,
    _managed,
    _overlap_ratio,
    enrich_pdf_visuals,
)
from services.slide_markdown import rebuild_slide_markdown

PAGE_BREAK = "<!-- PDF_PAGE_BREAK -->"


def run_docling(pdf_path, *, python_path=None):
    root = Path(__file__).resolve().parents[1]
    interpreter = Path(python_path) if python_path else root / ".venv-docling" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if not interpreter.is_file():
        raise RuntimeError(
            "找不到 Docling 環境。請依 README 的 Docling 安裝步驟建立 .venv-docling，"
            "或設定 PDF_DOCLING_PYTHON；使用舊版可設定 PDF_CONVERTER_BACKEND='marker'。"
        )
    with TemporaryDirectory() as directory:
        output = Path(directory) / "bundle.json"
        subprocess.run([
            str(interpreter.resolve()),
            str(root / "services/docling_worker.py"),
            str(Path(pdf_path).resolve()),
            str(output),
        ], check=True)
        return json.loads(output.read_text(encoding="utf-8"))


def _picture_layouts(bundle: dict, page_count: int) -> dict[int, PageLayout]:
    pages = bundle.get("pages", [])
    if [page.get("page") for page in pages] != list(range(1, page_count + 1)):
        raise ValueError("Docling 頁面清單不完整，不能當作成功轉換")
    layouts = {}
    for page in pages:
        regions = []
        for region in page.get("regions", []):
            category = str(region.get("category", "")).strip().lower()
            bbox = region.get("bbox", [])
            if category not in FIGURE_CATEGORIES or len(bbox) != 4:
                continue
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            # Docling can label a large-font slide title as a picture. Small
            # top-band badges/titles are not instructional figures.
            if width * height < 10000 or (
                category == "example_figure" and bbox[3] <= 220
            ):
                continue
            regions.append(region)
        layouts[page["page"]] = PageLayout.model_validate({"regions": regions})
    return layouts


def _merge_native_image_regions(
    layouts: dict[int, PageLayout], slide_layout: dict
) -> dict[str, int]:
    """Merge native raster and composite-vector geometry into Docling layouts."""
    stats = {
        "detected": 0, "added": 0, "overridden": 0,
        "vector_detected": 0, "vector_added": 0,
        "vector_overridden": 0, "vector_matched": 0,
        "image_covered_by_vector": 0,
    }
    for page in slide_layout.get("pages", []):
        layout = layouts.setdefault(page["page"], PageLayout())
        for bbox in page.get("vector_regions", []):
            stats["vector_detected"] += 1
            matches = [
                region for region in layout.regions
                if _overlap_ratio(bbox, region.bbox) > .5
            ]
            if matches:
                # Docling already found this composition. Keep its visual bbox;
                # the native detector is primarily a fallback for missed pages
                # such as page 29 and must not regress an existing good crop.
                stats["vector_matched"] += 1
            else:
                layout.regions.append(LayoutRegion(
                    category="diagram",
                    bbox=bbox,
                    reason="PDF native vector composition (all-content mode)",
                ))
                stats["vector_added"] += 1
        for bbox in page.get("image_regions", []):
            stats["detected"] += 1
            matches = [
                region for region in layout.regions
                if _overlap_ratio(bbox, region.bbox) > .8
            ]
            if matches:
                if "vector composition" in (matches[0].reason or ""):
                    # Small icons embedded inside one vector infographic belong
                    # to the complete composition; never shrink it to one icon.
                    stats["image_covered_by_vector"] += 1
                    continue
                # Native PDF image geometry is more exact than Docling's visual
                # block, so use it to avoid including adjacent bullets/URLs.
                matches[0].bbox = bbox
                matches[0].reason = (
                    "PDF native image geometry overrides Docling region "
                    "(all-content mode)"
                )
                stats["overridden"] += 1
                continue
            layout.regions.append(LayoutRegion(
                category="example_figure",
                bbox=bbox,
                reason="PDF native image geometry (all-content mode)",
            ))
            stats["added"] += 1
    return stats


def _add_page_boundaries(
    markdown: str, starts: list[int | None]
) -> tuple[str, list[int | None], int]:
    """Insert invisible page separators without adding a visible page number."""
    boundaries = sorted({offset for offset in starts[1:] if offset not in (None, 0)})
    insertion = f"\n\n{PAGE_BREAK}\n\n"
    output = markdown
    for offset in reversed(boundaries):
        output = output[:offset] + insertion + output[offset:]
    shifted = [
        None if offset is None else offset + len(insertion) * sum(
            boundary <= offset for boundary in boundaries
        )
        for offset in starts
    ]
    return output, shifted, len(boundaries)


def convert_docling_pdf(
    pdf_path,
    marker_markdown,
    *,
    assets_dir,
    markdown_assets_path,
    settings,
    manifest_path=None,
    python_path=None,
    bundle=None,
    analyzer=None,
    document_type=None,
):
    """Enrich Marker output using Docling figure coordinates, nothing else."""
    import pypdfium2 as pdfium

    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    bundle = bundle if bundle is not None else run_docling(
        pdf_path, python_path=python_path
    )
    analyzer = analyzer or (
        ExternalMultimodalCaptioner(settings, document_type=document_type)
        if settings.enabled else None
    )
    with _managed(pdfium.PdfDocument(str(pdf_path))) as document:
        layouts = _picture_layouts(bundle, len(document))
        starts = _find_page_starts(marker_markdown, document)
    docling_region_count = sum(len(layout.regions) for layout in layouts.values())
    try:
        markdown, shifted_starts, slide_layout = rebuild_slide_markdown(
            pdf_path, marker_markdown, layouts
        )
        boundary_count = slide_layout["page_count"] - 1
        native_image_stats = _merge_native_image_regions(
            layouts, slide_layout
        )
    except ValueError:
        # Non-slide PDFs keep the established Marker document path.
        markdown, shifted_starts, boundary_count = _add_page_boundaries(
            marker_markdown, starts
        )
        slide_layout = None
        native_image_stats = {
            "detected": 0, "added": 0, "overridden": 0,
            "vector_detected": 0, "vector_added": 0,
            "vector_overridden": 0, "vector_matched": 0,
            "image_covered_by_vector": 0,
        }

    # Persist Docling and native-PDF image geometry to make the fallback path auditable.
    geometry = {
        "schema_version": 3,
        "strategy": "all_pdf_image_regions",
        "docling_version": bundle.get("docling_version"),
        "pages": [
            {"page": number, "regions": [r.model_dump() for r in layout.regions]}
            for number, layout in sorted(layouts.items())
        ],
    }
    (assets_dir / "docling_layout.json").write_text(
        json.dumps(geometry, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    markdown, manifest = enrich_pdf_visuals(
        pdf_path,
        markdown,
        assets_dir=assets_dir,
        markdown_assets_path=markdown_assets_path,
        settings=settings,
        analyzer=analyzer,
        page_layouts=layouts,
        page_offsets=shifted_starts,
        verify_text=True,
        require_layout_match=False,
        preserve_all_images=True,
        document_type=document_type,
    )
    visual_region_count = sum(len(layout.regions) for layout in layouts.values())
    manifest.update(
        schema_version=10,
        backend=("native_slides_all_content" if slide_layout
                 else "marker_with_all_detected_images"),
        strategy=("native_slide_document_preserve_all_content" if slide_layout
                  else "marker_document_preserve_all_images"),
        docling_version=bundle.get("docling_version"),
        docling_region_count=docling_region_count,
        native_image_region_count=native_image_stats["detected"],
        native_image_region_added_count=native_image_stats["added"],
        native_image_region_overridden_count=native_image_stats["overridden"],
        native_vector_region_count=native_image_stats["vector_detected"],
        native_vector_region_added_count=native_image_stats["vector_added"],
        native_vector_region_overridden_count=native_image_stats["vector_overridden"],
        native_vector_region_matched_count=native_image_stats["vector_matched"],
        native_images_covered_by_vector_count=native_image_stats["image_covered_by_vector"],
        visual_region_count=visual_region_count,
        text_source="native_pdf_text" if slide_layout else "marker",
        heading_source="native_pdf_slide_title" if slide_layout else "marker",
        table_source="native_pdf_geometry" if slide_layout else "marker",
        page_boundary_count=boundary_count,
        slide_layout=slide_layout,
        content_policy="preserve_all_pdf_content",
    )
    if manifest_path is not None:
        Path(manifest_path).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return markdown, manifest
