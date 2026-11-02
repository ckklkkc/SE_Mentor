import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from services.image_captioning import ImageCaptionSettings, ImageDescription
from services.pdf_visuals import (
    PageLayout,
    _embedded_images,
    _grounded_candidates,
    _is_closing_page,
    _strip_slide_page_markers,
    _after_marker_removal,
    _markdown_tables,
    _overlap_ratio,
    _redundant_table,
    _valid_crop,
    enrich_pdf_visuals,
)


class FakeTextPage:
    def __init__(self, text):
        self.text = text

    def get_text_bounded(self):
        return self.text


class FakeBitmap:
    def to_pil(self):
        return Image.new("RGB", (1000, 600), "white")


class PdfImage:
    def get_bounds(self):
        return 100, 60, 900, 520

    def extract(self, stream):
        with Image.new("RGB", (800, 460), "white") as image:
            image.save(stream, format="PNG")


class BrokenPdfImage(PdfImage):
    def extract(self, stream):
        raise RuntimeError("image extraction unavailable")


class FakePage:
    def __init__(self, text, image=False, image_object=None):
        self.text = text
        self.image = image
        self.image_object = image_object

    def get_textpage(self):
        return FakeTextPage(self.text)

    def render(self, scale):
        return FakeBitmap()

    def get_size(self):
        return 1000, 600

    def get_objects(self):
        if self.image_object is not None:
            return [self.image_object]
        return [PdfImage()] if self.image else []


class FakeDocument:
    def __init__(self):
        self.pages = [
            FakePage("課程大綱\n條列"),
            FakePage("需求分析與軟體設計\nWhat\nHow"),
            FakePage("需求的種類\n功能需求\n系統需求"),
            FakePage("MHC-PMC Example\n12", image=True),
            FakePage("需求工程流程\n50"),
        ]

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        return self.pages[index]


class StagedAnalyzer:
    def __init__(self, *, duplicate_table=False, fail_layout=None, fail_figure=None):
        self.layout_pages = []
        self.figure_pages = []
        self.duplicate_table = duplicate_table
        self.fail_layout = fail_layout
        self.fail_figure = fail_figure

    def query_image_json(self, image_path, prompt, *, max_tokens=1400):
        result = self._respond(image_path, prompt, max_tokens=max_tokens)
        if "image_type" in result:
            kind = result["image_type"]
            result.update(instructional=kind != "分類圖",
                          content_category="table" if kind == "分類圖" else
                          "example_figure" if kind == "範例圖" else "diagram",
                          evidence="圖片中的需求對照或箭頭關係")
        return result

    def _respond(self, image_path, prompt, *, max_tokens=1400):
        is_layout = prompt.startswith("請先看完整的教材第 ")
        page = int(re.search(r"第 (\d+) 頁", prompt)[1])
        if is_layout:
            self.layout_pages.append(page)
            if page == self.fail_layout:
                raise RuntimeError("layout unavailable")
            regions = [{"category": "heading", "bbox": [0, 0, 900, 150]}]
            if page == 1:
                regions.append({"category": "bullet_text", "bbox": [100, 200, 900, 800]})
            elif page == 2:
                regions.append({"category": "diagram", "bbox": [100, 210, 850, 800]})
            elif page == 3:
                regions.append({
                    "category": "diagram" if self.duplicate_table else "table",
                    "bbox": [100, 180, 850, 850],
                })
            elif page == 4:
                regions.append({"category": "example_figure", "bbox": [120, 200, 850, 840]})
            elif page == 5:
                regions.append({"category": "diagram", "bbox": [60, 200, 900, 900]})
            return {"regions": regions}

        self.figure_pages.append(page)
        if page == self.fail_figure:
            raise RuntimeError("figure unavailable")
        if page == 2:
            return dict(image_type="流程圖", visible_text=["What", "How"],
                        description="需求分析回答 What，設計回答 How。",
                        relationships=["需求文件導引設計"], teaching_points=[])
        if page == 3:
            return dict(image_type="分類圖",
                        visible_text=["功能需求", "功能性使用者需求", "功能性系統需求"],
                        description="重複的分類表格", relationships=[], teaching_points=[])
        if page == 4:
            return dict(image_type="範例圖", visible_text=["User Requirement Definition",
                        "System Requirements Specification"],
                        description="使用者需求 1 展開成系統需求 1.1–1.5。",
                        relationships=["1 對應 1.1–1.5"], teaching_points=[])
        return dict(image_type="流程圖", visible_text=["可行性分析", "需求擷取"],
                    description="需求工程含確認與回返路徑。", relationships=[],
                    teaching_points=[])


class RecheckAnalyzer(StagedAnalyzer):
    def query_image_json(self, image_path, prompt, *, max_tokens=1400):
        if (prompt.startswith("請先看完整的教材第 4 頁")
                and "複核：" not in prompt):
            self.layout_pages.append(4)
            return {"regions": [{"category": "heading", "bbox": [0, 0, 900, 150]},
                                {"category": "table", "bbox": [120, 200, 850, 840]}]}
        return super().query_image_json(image_path, prompt, max_tokens=max_tokens)


class PdfVisualTests(unittest.TestCase):
    def test_broken_embedded_image_uses_page_region_fallback(self):
        document = FakeDocument()
        document.pages = [FakePage(
            "線框圖1\n1", image_object=BrokenPdfImage()
        )]
        layout = PageLayout.model_validate({"regions": [
            {"category": "ui_figure", "bbox": [100, 100, 900, 850]},
        ]})
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=document):
                output, manifest = enrich_pdf_visuals(
                    "ignored.pdf", "# 線框圖1\n<!-- PDF_FIGURES_PAGE_1 -->",
                    assets_dir=Path(temporary) / "assets",
                    markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=False),
                    page_layouts={1: layout}, page_offsets=[0],
                    preserve_all_images=True,
                )

        self.assertEqual(manifest["failed_count"], 0)
        self.assertEqual(manifest["described_count"], 1)
        self.assertIn("page_001_figure_01.png", output)
        self.assertEqual(manifest["pages"][0]["figures"][0]["source"], "page_region")
        self.assertTrue(any(
            "改用頁面座標裁切" in warning
            for warning in manifest["pages"][0]["warnings"]
        ))

    def test_embedded_image_extraction_error_is_a_warning_not_page_failure(self):
        document = FakeDocument()
        document.pages = [FakePage("範例圖\n1", image_object=BrokenPdfImage())]
        warnings = []
        with tempfile.TemporaryDirectory() as temporary:
            images = _embedded_images(document, 0, Path(temporary), warnings)
        self.assertEqual(images, [])
        self.assertTrue(any("改用頁面座標裁切" in warning for warning in warnings))

    def test_native_image_bounds_override_full_slide_or_partial_model_boxes(self):
        native = [{"id": 9, "bbox": [440, 305, 665, 948], "path": Path("map.png"),
                   "pixel_size": [430, 756]}]
        for model_box in ([0, 0, 1000, 1000], [450, 410, 620, 800]):
            layout = PageLayout.model_validate({"regions": [
                {"category": "ui_figure", "bbox": model_box},
            ]})
            candidates = _grounded_candidates(layout, native, [20, 195, 40, 215], [])
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["bbox"], native[0]["bbox"])
            self.assertEqual(candidates[0]["image_path"], Path("map.png"))
        self.assertEqual(len(_grounded_candidates(PageLayout(), native, None, [])), 1)

    def test_full_slide_and_page_number_crops_are_rejected_without_native_image(self):
        for box in ([0, 0, 1000, 1000], [0, 180, 900, 800]):
            layout = PageLayout.model_validate({"regions": [{"category": "diagram", "bbox": box}]})
            warnings = []
            self.assertEqual(_grounded_candidates(layout, [], [20, 195, 40, 215], warnings), [])
            self.assertTrue(warnings)

    def test_closing_page_never_exports_decorative_image(self):
        self.assertTrue(_is_closing_page("Any Question?\n53"))
        self.assertFalse(_is_closing_page("需求工程流程\n50"))
        document = FakeDocument()
        document.pages = [FakePage("Any Question?\n53", image=True)]
        analyzer = StagedAnalyzer()
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=document):
                output, manifest = enrich_pdf_visuals(
                    "ignored.pdf", "Any Question?\n![](foo_page_0_picture.png)",
                    assets_dir=Path(temporary) / "assets", markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=True), analyzer=analyzer,
                )
        self.assertEqual(analyzer.figure_pages, [])
        self.assertEqual(manifest["described_count"], 0)
        self.assertEqual(manifest["pages"][0]["status"], "excluded_closing")
        self.assertNotIn("![", output)

    def test_all_content_mode_keeps_closing_page_image(self):
        document = FakeDocument()
        document.pages = [FakePage("Any Question?\n1", image=True)]
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=document):
                output, manifest = enrich_pdf_visuals(
                    "ignored.pdf", "# Any Question?\n<!-- PDF_FIGURES_PAGE_1 -->",
                    assets_dir=Path(temporary) / "assets",
                    markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=False),
                    page_layouts={1: PageLayout()},
                    page_offsets=[0], preserve_all_images=True,
                )
        self.assertIn("page_001_figure_01.png", output)
        self.assertEqual(manifest["excluded_closing_count"], 0)
        self.assertTrue(manifest["preserve_all_images"])

    def test_important_native_diagram_survives_misclassification_without_invented_caption(self):
        class RejectingAnalyzer:
            def query_image_json(self, image_path, prompt, *, max_tokens=1400):
                if prompt.startswith("請先看完整"):
                    return {"regions": [{"category": "bullet_text", "bbox": [0, 100, 950, 900]}]}
                return dict(instructional=False, content_category="decoration", evidence="",
                            image_type="裝飾圖", description="不可採用的推測圖說", visible_text=["1"])
        document = FakeDocument()
        document.pages = [FakePage("需求工程流程\n1", image=True)]
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=document):
                output, manifest = enrich_pdf_visuals(
                    "ignored.pdf", "需求工程流程\n", assets_dir=Path(temporary) / "assets",
                    markdown_assets_path="assets", settings=ImageCaptionSettings(enabled=True),
                    analyzer=RejectingAnalyzer(),
                )
            with Image.open(Path(temporary) / "assets/page_001_figure_01.png") as result:
                self.assertEqual(result.size, (800, 460))
        self.assertEqual(manifest["review_required_count"], 1)
        self.assertIn("待查核", output)
        self.assertNotIn("不可採用的推測圖說", output)

    def test_slide_page_markers_removed_without_removing_instructional_numbers(self):
        text = "#### **44**\n- **46**\n50\n- 系統支援 50 人\n| 範例 | 46 |\n1. 步驟一\n99\n"
        clean = _strip_slide_page_markers(text, {44, 46, 50})
        self.assertNotIn("**44**", clean)
        self.assertNotIn("**46**", clean)
        self.assertIn("- 系統支援 50 人", clean)
        self.assertIn("| 範例 | 46 |", clean)
        self.assertIn("1. 步驟一", clean)
        self.assertIn("99", clean)

    def test_embedded_table_is_preserved_but_plain_text_is_not_an_image(self):
        class TextAnalyzer:
            def query_image_json(self, image_path, prompt, *, max_tokens=1400):
                if prompt.startswith("請先看完整"):
                    return {"regions": [{"category": "heading", "bbox": [0, 0, 900, 150]}]}
                return dict(instructional=False, content_category=category, evidence="文字內容",
                            image_type="文字", description="一般文字或表格", visible_text=[])

        document = FakeDocument()
        document.pages = [FakePage("使用者介面分析\n1", image=True)]
        for category in ("table", "plain_text"):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as temporary:
                with patch("pypdfium2.PdfDocument", return_value=document):
                    output, manifest = enrich_pdf_visuals(
                        "ignored.pdf", "使用者介面分析\n", assets_dir=Path(temporary) / "assets",
                        markdown_assets_path="assets", settings=ImageCaptionSettings(enabled=True),
                        analyzer=TextAnalyzer(),
                    )
                if category == "table":
                    self.assertEqual(manifest["described_count"], 1)
                    self.assertIn("![", output)
                    self.assertEqual(manifest["pages"][0]["figures"][0]["category"], "table_image")
                else:
                    self.assertEqual(manifest["described_count"], 0)
                    self.assertNotIn("![", output)

    def test_docling_grounded_slide_image_survives_plain_text_misclassification(self):
        class RejectingAnalyzer:
            def query_image_json(self, image_path, prompt, *, max_tokens=1400):
                return dict(
                    instructional=False, content_category="plain_text",
                    evidence="", image_type="一般文字",
                    description="模型誤判後不可採用的內容", visible_text=[],
                )

        document = FakeDocument()
        document.pages = [FakePage("功能使用者需求範例2\n1", image=True)]
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=document):
                output, manifest = enrich_pdf_visuals(
                    "ignored.pdf", "<!-- PDF_FIGURES_PAGE_1 -->\n",
                    assets_dir=Path(temporary) / "assets",
                    markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=True),
                    analyzer=RejectingAnalyzer(),
                    page_layouts={1: PageLayout()},
                    page_offsets=[0],
                )

        self.assertEqual(manifest["described_count"], 1)
        self.assertEqual(manifest["review_required_count"], 1)
        self.assertIn("page_001_figure_01.png", output)
        self.assertIn("待查核教材圖例", output)
        self.assertNotIn("模型誤判後不可採用的內容", output)

    def test_layout_requires_heading_bullets_table_and_figures_as_separate_regions(self):
        layout = PageLayout.model_validate({"regions": [
            {"category": "heading", "bbox": [0, 0, 1000, 150]},
            {"category": "bullet_text", "bbox": [40, 180, 900, 450]},
            {"category": "table", "bbox": [40, 470, 900, 850]},
            {"category": "diagram", "bbox": [100, 200, 800, 850]},
        ]})
        self.assertEqual(len(layout.regions), 4)

    def test_every_page_is_scanned_before_local_figures_are_described(self):
        markdown = (
            "# 課程大綱\n![](banner_page_0_Picture_1.jpeg)\n"
            "需求的種類\n| 類別 | 使用者需求 | 系統需求 |\n"
            "|---|---|---|\n"
            "| 功能需求 | 功能性使用者需求 | 功能性系統需求 |\n"
            "MHC-PMC Example\n需求工程流程\n"
            "![](figure_page_4_Picture_1.jpeg)\n"
        )
        analyzer = StagedAnalyzer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("pypdfium2.PdfDocument", return_value=FakeDocument()):
                enriched, manifest = enrich_pdf_visuals(
                    "ignored.pdf", markdown, assets_dir=root / "assets",
                    markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=True), analyzer=analyzer,
                    manifest_path=root / "images.json",
                )
            self.assertTrue((root / "assets/page_004_figure_01.png").is_file())
            self.assertTrue((root / "images.json").is_file())

        self.assertEqual(analyzer.layout_pages, [1, 2, 3, 4, 5])
        self.assertEqual(analyzer.figure_pages, [2, 4, 5])
        self.assertEqual(manifest["scanned_count"], 5)
        self.assertEqual(manifest["selected_count"], 3)
        self.assertEqual(manifest["described_count"], 3)
        self.assertEqual(manifest["pages"][2]["status"], "text_only")
        self.assertNotIn("banner_page_0", enriched)
        self.assertIn("使用者需求 1 展開成系統需求 1.1–1.5", enriched)
        self.assertLess(enriched.index("page_002_figure_01.png"), enriched.index("需求的種類"))
        self.assertLess(enriched.index("MHC-PMC Example"), enriched.index("page_004_figure_01.png"))

    def test_mislabeled_duplicate_table_is_rejected_after_second_stage(self):
        markdown = (
            "需求的種類\n| 類別 | 使用者需求 | 系統需求 |\n"
            "|---|---|---|\n"
            "| 功能需求 | 功能性使用者需求 | 功能性系統需求 |\n"
        )
        analyzer = StagedAnalyzer(duplicate_table=True)
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=FakeDocument()):
                enriched, manifest = enrich_pdf_visuals(
                    "ignored.pdf", markdown,
                    assets_dir=Path(temporary) / "assets",
                    markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=True), analyzer=analyzer,
                )
        self.assertIn(3, analyzer.figure_pages)
        self.assertNotIn("重複的分類表格", enriched)
        self.assertEqual(manifest["pages"][2]["status"], "filtered")

    def test_sparse_text_screenshot_receives_layout_recheck(self):
        analyzer = RecheckAnalyzer()
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=FakeDocument()):
                enriched, manifest = enrich_pdf_visuals(
                    "ignored.pdf", "MHC-PMC Example\n",
                    assets_dir=Path(temporary) / "assets",
                    markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=True), analyzer=analyzer,
                )
        self.assertEqual(analyzer.layout_pages.count(4), 2)
        self.assertEqual(manifest["rechecked_count"], 1)
        self.assertIn("使用者需求 1 展開成系統需求 1.1–1.5", enriched)

    def test_limit_applies_to_figures_not_page_scan(self):
        analyzer = StagedAnalyzer()
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=FakeDocument()):
                _, manifest = enrich_pdf_visuals(
                    "ignored.pdf", "課程大綱\n需求的種類\nMHC-PMC Example\n",
                    assets_dir=Path(temporary) / "assets",
                    markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=True, max_images=1),
                    analyzer=analyzer,
                )
        self.assertEqual(analyzer.layout_pages, [1, 2, 3, 4, 5])
        self.assertEqual(analyzer.figure_pages, [2])
        self.assertEqual(manifest["described_count"], 1)
        self.assertEqual(manifest["limited_count"], 2)

    def test_description_failure_does_not_hide_other_pages(self):
        analyzer = StagedAnalyzer(fail_figure=2, fail_layout=3)
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=FakeDocument()):
                _, manifest = enrich_pdf_visuals(
                    "ignored.pdf", "MHC-PMC Example\n",
                    assets_dir=Path(temporary) / "assets",
                    markdown_assets_path="assets",
                    settings=ImageCaptionSettings(enabled=True), analyzer=analyzer,
                )
        self.assertEqual(manifest["failed_count"], 2)
        self.assertEqual(manifest["pages"][1]["errors"][0]["stage"], "description")
        self.assertEqual(manifest["pages"][2]["errors"][0]["stage"], "layout")
        # 失敗頁仍保留原圖與 0% 信心的待查核說明，不再靜默丟失。
        self.assertEqual(manifest["described_count"], 3)
        self.assertTrue(manifest["pages"][1]["figures"][0]["needs_review"])
        with tempfile.TemporaryDirectory() as temporary:
            with patch("pypdfium2.PdfDocument", return_value=FakeDocument()):
                with self.assertRaisesRegex(RuntimeError, "figure unavailable"):
                    enrich_pdf_visuals(
                        "ignored.pdf", "MHC-PMC Example\n",
                        assets_dir=Path(temporary) / "strict_assets",
                        markdown_assets_path="strict_assets",
                        settings=ImageCaptionSettings(enabled=True, fail_on_error=True),
                        analyzer=StagedAnalyzer(fail_figure=2),
                    )

    def test_crop_table_comparison_and_offset(self):
        self.assertIsNone(_valid_crop([0, 0, 1000, 10], 1000, 600))
        self.assertIsNotNone(_valid_crop([100, 100, 700, 700], 1000, 600))
        self.assertAlmostEqual(_overlap_ratio([0, 0, 100, 100], [0, 0, 100, 100]), 1)
        tables = _markdown_tables(
            "| 類別 | 使用者需求 | 系統需求 |\n|---|---|---|\n"
            "| 功能需求 | 功能性使用者需求 | 功能性系統需求 |\n"
        )
        figure = ImageDescription(image_type="分類圖", description="需求分類",
                                  visible_text=["功能需求", "功能性使用者需求",
                                                "功能性系統需求"])
        self.assertTrue(_redundant_table(figure, tables))
        markdown = "前文 ![](foo.png) 後文"
        self.assertEqual(_after_marker_removal(markdown, markdown.index("後文")), len("前文  "))


if __name__ == "__main__":
    unittest.main()
