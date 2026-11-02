import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from services.docling_pdf import (
    PAGE_BREAK,
    _add_page_boundaries,
    _merge_native_image_regions,
    _picture_layouts,
    convert_docling_pdf,
    run_docling,
)
from services.image_captioning import ImageCaptionSettings
from services.pdf_visuals import PageLayout
from tests.test_pdf_visuals import FakeDocument, FakePage


class DoclingImageOnlyTests(unittest.TestCase):
    def test_docling_text_headings_and_tables_are_ignored(self):
        document = FakeDocument()
        document.pages = [
            FakePage("需求規格\n1", image=True),
            FakePage("Any Question?\n2", image=True),
        ]
        marker = (
            "# 需求規格\n\n"
            "| 設定(Setting) | 編號 | 說明 |\n"
            "|---|---|---|\n"
            "| 個人資料 | SH-ST-01 | 修改密碼 |\n\n"
            "# Any Question?\n"
        )
        bundle = {"docling_version": "fixture", "pages": [
            {
                "page": 1,
                "markdown": "## 錯誤 Docling 標題\n個人賁料\nSH-$T-02",
                "regions": [
                    {"category": "example_figure", "bbox": [100, 200, 900, 800]},
                    {"category": "table", "bbox": [100, 200, 900, 700]},
                    {"category": "heading", "bbox": [0, 0, 900, 100]},
                ],
                "tables": [{"markdown": "亂掉的表格"}],
            },
            {"page": 2, "markdown": "## 錯誤結尾", "regions": [], "tables": []},
        ]}
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary) / "assets"
            with patch("pypdfium2.PdfDocument", return_value=document), patch(
                "services.docling_pdf.rebuild_slide_markdown",
                side_effect=ValueError("not a slide fixture"),
            ):
                output, manifest = convert_docling_pdf(
                    "ignored.pdf",
                    marker,
                    assets_dir=assets,
                    markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=False),
                    bundle=bundle,
                )
            layout = (assets / "docling_layout.json").read_text(encoding="utf-8")

        self.assertIn("| 個人資料 | SH-ST-01 | 修改密碼 |", output)
        self.assertIn("# 需求規格", output)
        self.assertIn("# Any Question?", output)
        self.assertNotIn("錯誤 Docling 標題", output)
        self.assertNotIn("個人賁料", output)
        self.assertNotIn("SH-$T-02", output)
        self.assertNotIn("亂掉的表格", output)
        self.assertIn("page_001_figure_01.png", output)
        self.assertIn("page_002_figure_01.png", output)
        self.assertNotIn('"category": "table"', layout)
        self.assertEqual(manifest["backend"], "marker_with_all_detected_images")
        self.assertTrue(manifest["preserve_all_images"])
        self.assertEqual(manifest["content_policy"], "preserve_all_pdf_content")
        self.assertEqual(manifest["text_source"], "marker")
        self.assertEqual(manifest["table_source"], "marker")
        self.assertEqual(manifest["docling_region_count"], 1)

    def test_native_pdf_image_regions_fill_docling_gaps_without_duplicates(self):
        layouts = {1: PageLayout.model_validate({"regions": [
            {"category": "diagram", "bbox": [100, 200, 800, 800]},
        ]}), 2: PageLayout()}
        stats = _merge_native_image_regions(layouts, {"pages": [
            {"page": 1, "image_regions": [[110, 210, 790, 790]]},
            {"page": 2, "image_regions": [[200, 300, 700, 850]]},
        ]})
        self.assertEqual(stats["detected"], 2)
        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["overridden"], 1)
        self.assertEqual(len(layouts[1].regions), 1)
        self.assertEqual(layouts[1].regions[0].bbox, [110, 210, 790, 790])
        self.assertEqual(len(layouts[2].regions), 1)
        self.assertIn("all-content mode", layouts[2].regions[0].reason)

    def test_composite_vector_region_is_not_shrunk_to_its_small_icons(self):
        layouts = {29: PageLayout()}
        stats = _merge_native_image_regions(layouts, {"pages": [{
            "page": 29,
            "vector_regions": [[80, 220, 820, 980]],
            "image_regions": [[450, 250, 500, 340], [730, 450, 780, 540]],
        }]})
        self.assertEqual(len(layouts[29].regions), 1)
        self.assertEqual(layouts[29].regions[0].bbox, [80, 220, 820, 980])
        self.assertEqual(stats["image_covered_by_vector"], 2)

    def test_boundaries_are_invisible_and_do_not_duplicate_same_offset(self):
        output, shifted, count = _add_page_boundaries(
            "第一頁\n第二頁\n第三頁", [0, 4, 8, 8]
        )
        self.assertEqual(output.count(PAGE_BREAK), 2)
        self.assertEqual(count, 2)
        self.assertNotIn("第 2 頁", output)
        self.assertEqual(shifted[1], 4 + len(f"\n\n{PAGE_BREAK}\n\n"))
        self.assertEqual(shifted[2], 8 + 2 * len(f"\n\n{PAGE_BREAK}\n\n"))

    def test_only_picture_and_chart_regions_are_accepted(self):
        bundle = {"pages": [{"page": 1, "regions": [
            {"category": "heading", "bbox": [0, 0, 100, 100]},
            {"category": "table", "bbox": [0, 100, 500, 500]},
            {"category": "example_figure", "bbox": [70, 80, 170, 160]},
            {"category": "diagram", "bbox": [100, 200, 900, 800]},
        ]}]}
        layouts = _picture_layouts(bundle, 1)
        self.assertEqual([region.category for region in layouts[1].regions], ["diagram"])

    def test_missing_pages_fail_instead_of_losing_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=FakeDocument()):
                with self.assertRaisesRegex(ValueError, "頁面清單不完整"):
                    convert_docling_pdf(
                        "ignored.pdf",
                        "# Marker 原文",
                        assets_dir=Path(temporary),
                        markdown_assets_path="assets",
                        settings=ImageCaptionSettings(enabled=False),
                        bundle={"pages": []},
                    )

    def test_missing_worker_environment_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "Docling 環境"):
            run_docling("ignored.pdf", python_path="/missing/environment/python")

    def test_worker_disables_ocr_tables_and_markdown_export(self):
        source = (Path(__file__).parents[1] / "services/docling_worker.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("options.do_ocr = False", source)
        self.assertIn("options.do_table_structure = False", source)
        self.assertNotIn("export_to_markdown", source)
        self.assertNotIn("TableItem", source)


class FigureVerificationTests(unittest.TestCase):
    def test_verified_text_replaces_wrong_draft(self):
        from services.pdf_visuals import PageLayout, enrich_pdf_visuals

        class Analyzer:
            def query_image_json(self, path, prompt, **kwargs):
                if prompt.startswith("請重新核對"):
                    return dict(
                        image_type="介面圖", description="系統顯示 SH-ST-01。",
                        visible_text=["SH-ST-01"], relationships=[], teaching_points=[],
                    )
                return dict(
                    instructional=True, content_category="ui_figure", evidence="介面按鈕",
                    image_type="介面圖", description="錯誤代號 SH-5T-O1",
                    visible_text=["SH-5T-O1"],
                )

        document = FakeDocument()
        document.pages = [FakePage("使用者介面分析", image=True)]
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=document), patch(
                "services.pdf_text_review.region_text", return_value=""
            ):
                output, manifest = enrich_pdf_visuals(
                    "ignored.pdf", "## 使用者介面分析",
                    assets_dir=Path(temporary), markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=True), analyzer=Analyzer(),
                    page_layouts={1: PageLayout()}, page_offsets=[0], verify_text=True,
                )
        self.assertIn("SH-ST-01", output)
        self.assertNotIn("SH-5T-O1", output)
        self.assertEqual(
            manifest["pages"][0]["figures"][0]["text_verification"], "model_checked"
        )

    def test_caption_failure_retains_docling_candidate_for_review(self):
        from services.pdf_visuals import PageLayout, enrich_pdf_visuals

        document = FakeDocument()
        document.pages = [FakePage("需求工程流程", image=True)]
        analyzer = Mock()
        analyzer.query_image_json.side_effect = RuntimeError("API unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=document):
                output, manifest = enrich_pdf_visuals(
                    "ignored.pdf", "## 需求工程流程",
                    assets_dir=Path(temporary), markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=True), analyzer=analyzer,
                    page_layouts={1: PageLayout()}, page_offsets=[0], verify_text=True,
                )
            self.assertTrue((Path(temporary) / "page_001_figure_01.png").is_file())
        self.assertIn("待查核", output)
        self.assertEqual(manifest["failed_count"], 1)
        self.assertEqual(manifest["review_required_count"], 1)


if __name__ == "__main__":
    unittest.main()
