import json
import tempfile
import unittest
from pathlib import Path

from services.image_captioning import (
    ExternalMultimodalCaptioner,
    ImageCaptionSettings,
    ImageDescription,
    _description_quality_reasons,
    _model_prompt,
    _normalize_description_payload,
    enrich_markdown_with_marker_images,
)


class FakeImage:
    size = (800, 600)

    def save(self, path, format=None):
        Path(path).write_bytes(b"fake-image-content")


class FakeCaptioner:
    def __init__(self, fail=False):
        self.fail = fail
        self.contexts = []

    def describe(self, image_path, *, context, source_name):
        self.contexts.append(context)
        if self.fail:
            raise RuntimeError("temporary api failure")
        return ImageDescription(
            image_type="流程圖",
            visible_text=["需求擷取", "需求分析"],
            description="流程由需求擷取進入需求分析。",
            meaning="此圖呈現需求工程活動的先後銜接。",
            relationships=["需求擷取 → 需求分析"],
            teaching_points=["兩個活動具有先後關係"],
            context_connection="本圖把前文的需求擷取具體化為後續分析流程。",
            document_facts=["需求擷取先於需求分析"],
            confidence=0.92,
        )


class SequenceCaptioner(ExternalMultimodalCaptioner):
    def __init__(self, payloads):
        super().__init__(
            ImageCaptionSettings(
                enabled=True,
                max_attempts=len(payloads),
                retry_base_seconds=0,
            ),
            document_type="SRD",
        )
        self.payloads = iter(payloads)

    def query_image_json(self, image_path, prompt, *, max_tokens=1400):
        payload = next(self.payloads)
        if isinstance(payload, Exception):
            raise payload
        return payload


class ImageCaptioningTests(unittest.TestCase):
    def test_preserves_original_and_inserts_structured_caption(self):
        markdown = "# 需求工程\n擷取後進行分析。\n![流程](page_22_Picture_1.png)\n下一步是規格化。"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            captioner = FakeCaptioner()
            enriched, manifest = enrich_markdown_with_marker_images(
                markdown,
                {"page_22_Picture_1.png": FakeImage()},
                assets_dir=root / "book_assets",
                markdown_assets_path="book_assets",
                settings=ImageCaptionSettings(enabled=True),
                captioner=captioner,
                manifest_path=root / "book_images.json",
            )

            self.assertIn("![流程](book_assets/001_page_22_Picture_1.png)", enriched)
            self.assertIn("AI 圖片轉述", enriched)
            self.assertIn("需求擷取 → 需求分析", enriched)
            self.assertIn("與本文的連接", enriched)
            self.assertIn("可供文件分析的事實", enriched)
            self.assertIn("92%", enriched)
            self.assertTrue((root / "book_assets/001_page_22_Picture_1.png").is_file())
            self.assertEqual(manifest["described_count"], 1)
            self.assertIn("擷取後進行分析", captioner.contexts[0])
            saved = json.loads((root / "book_images.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["images"][0]["status"], "described")

    def test_api_failure_keeps_image_and_records_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            enriched, manifest = enrich_markdown_with_marker_images(
                "![圖](figure.png)",
                {"figure.png": FakeImage()},
                assets_dir=root / "assets",
                markdown_assets_path="assets",
                settings=ImageCaptionSettings(enabled=True, fail_on_error=False),
                captioner=FakeCaptioner(fail=True),
            )

            self.assertIn("![圖](assets/001_figure.png)", enriched)
            self.assertIn("待查核圖片", enriched)
            self.assertIn("不納入文件分析", enriched)
            self.assertEqual(manifest["failed_count"], 1)
            self.assertEqual(manifest["images"][0]["status"], "description_failed")
            self.assertTrue(manifest["images"][0]["needs_review"])
            self.assertEqual(
                manifest["images"][0]["errors"][0]["type"], "RuntimeError"
            )
            self.assertIn("已嘗試 1 次", enriched)

    def test_disabled_mode_still_exports_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            enriched, manifest = enrich_markdown_with_marker_images(
                "![原圖](figure.png)",
                {"figure.png": FakeImage()},
                assets_dir=root / "assets",
                markdown_assets_path="assets",
                settings=ImageCaptionSettings(enabled=False),
                captioner=FakeCaptioner(),
            )

            self.assertIn("assets/001_figure.png", enriched)
            self.assertEqual(manifest["described_count"], 0)
            self.assertEqual(
                manifest["images"][0]["status"],
                "preserved_description_disabled",
            )

    def test_context_keeps_heading_and_both_sides_without_crossing_page(self):
        markdown = (
            "# 前一頁\n不應帶入\n<!-- PDF_PAGE_BREAK -->\n"
            "# 登入需求\n使用者輸入帳密。\n"
            "![流程](login.png)\n系統驗證失敗時顯示錯誤。\n"
            "<!-- PDF_PAGE_BREAK -->\n# 下一頁\n也不應帶入"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            captioner = FakeCaptioner()
            enrich_markdown_with_marker_images(
                markdown,
                {"login.png": FakeImage()},
                assets_dir=root / "assets",
                markdown_assets_path="assets",
                settings=ImageCaptionSettings(enabled=True),
                captioner=captioner,
                document_type="SRD",
            )

        context = captioner.contexts[0]
        self.assertIn("[文件類型] SRD", context)
        self.assertIn("[所在章節] 登入需求", context)
        self.assertIn("[圖片前文]", context)
        self.assertIn("使用者輸入帳密", context)
        self.assertIn("[圖片後文]", context)
        self.assertIn("系統驗證失敗", context)
        self.assertNotIn("不應帶入", context)

    def test_same_image_in_different_sections_gets_context_specific_descriptions(self):
        markdown = (
            "# 需求\n![共用圖](same.png)\n"
            "<!-- PDF_PAGE_BREAK -->\n"
            "# 測試\n![共用圖](same.png)\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            captioner = FakeCaptioner()
            enriched, manifest = enrich_markdown_with_marker_images(
                markdown,
                {"same.png": FakeImage()},
                assets_dir=root / "assets",
                markdown_assets_path="assets",
                settings=ImageCaptionSettings(enabled=True),
                captioner=captioner,
                document_type="SRD",
            )

        self.assertEqual(len(captioner.contexts), 2)
        self.assertIn("[所在章節] 需求", captioner.contexts[0])
        self.assertIn("[所在章節] 測試", captioner.contexts[1])
        self.assertEqual(manifest["occurrence_count"], 2)
        self.assertEqual(enriched.count("AI 圖片轉述"), 2)

    def test_standalone_trello_section_preserves_image_without_description(self):
        markdown = (
            "# 專案文件\n"
            "## Trello\n"
            "![任務看板](trello.png)\n"
            "---\n"
            "### Figma\nhttps://example.test/design"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            captioner = FakeCaptioner()
            enriched, manifest = enrich_markdown_with_marker_images(
                markdown,
                {"trello.png": FakeImage()},
                assets_dir=root / "assets",
                markdown_assets_path="assets",
                settings=ImageCaptionSettings(enabled=True),
                captioner=captioner,
                document_type="SRD",
            )

        record = manifest["images"][0]
        self.assertIn("![任務看板](assets/001_trello.png)", enriched)
        self.assertNotIn("AI 圖片轉述", enriched)
        self.assertEqual(captioner.contexts, [])
        self.assertEqual(record["status"], "preserved_description_excluded")
        self.assertEqual(
            record["occurrences"][0]["status"], "description_excluded"
        )
        self.assertFalse(record["blocks_ingestion"])
        self.assertEqual(manifest["description_excluded_count"], 1)
        self.assertEqual(manifest["description_excluded_occurrence_count"], 1)
        self.assertEqual(manifest["described_count"], 0)
        self.assertEqual(manifest["described_occurrence_count"], 0)

    def test_trello_image_inside_requirements_section_is_still_described(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            captioner = FakeCaptioner()
            enriched, manifest = enrich_markdown_with_marker_images(
                "# 功能需求\n![Trello 看板](trello.png)",
                {"trello.png": FakeImage()},
                assets_dir=Path(temp_dir) / "assets",
                markdown_assets_path="assets",
                settings=ImageCaptionSettings(enabled=True),
                captioner=captioner,
                document_type="SRD",
            )

        self.assertIn("AI 圖片轉述", enriched)
        self.assertEqual(len(captioner.contexts), 1)
        self.assertEqual(manifest["description_excluded_count"], 0)
        self.assertEqual(manifest["described_count"], 1)

    def test_trello_section_in_textbook_is_not_excluded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            captioner = FakeCaptioner()
            enriched, manifest = enrich_markdown_with_marker_images(
                "# Trello\n![看板](trello.png)",
                {"trello.png": FakeImage()},
                assets_dir=Path(temp_dir) / "assets",
                markdown_assets_path="assets",
                settings=ImageCaptionSettings(enabled=True),
                captioner=captioner,
                document_type=None,
            )

        self.assertIn("AI 圖片轉述", enriched)
        self.assertEqual(manifest["description_excluded_count"], 0)

    def test_normalizes_common_json_variations_and_corrects_decoration(self):
        description = _normalize_description_payload({
            "image_type": "裝飾圖",
            "visible_text": "登入後台",
            "description": "後台登入介面包含帳號欄位與登入按鈕。",
            "meaning": "此畫面呈現管理者進入後台前的身分驗證入口。",
            "relationships": [{"from": "管理者", "to": "登入後台"}],
            "teaching_points": None,
            "context_connection": None,
            "document_facts": "畫面提供帳號輸入欄位。",
            "confidence": "95%",
            "uncertainty": "",
        })

        self.assertEqual(description.image_type, "介面截圖")
        self.assertEqual(description.visible_text, ["登入後台"])
        self.assertEqual(description.document_facts, ["畫面提供帳號輸入欄位。"])
        self.assertEqual(description.relationships, ["管理者 → 登入後台"])
        self.assertEqual(description.confidence, 0.95)

    def test_schema_failure_is_retried_with_diagnostics(self):
        captioner = SequenceCaptioner([
            {
                "image_type": "介面截圖",
                "description": "登入畫面",
                "confidence": None,
            },
            {
                "image_type": "介面截圖",
                "visible_text": ["登入"],
                "description": "登入畫面提供登入按鈕。",
                "meaning": "此畫面呈現管理者身分驗證入口。",
                "relationships": [],
                "teaching_points": [],
                "context_connection": "具體化登入需求。",
                "document_facts": ["畫面提供登入按鈕。"],
                "confidence": 0.9,
                "uncertainty": None,
            },
        ])

        description = captioner.describe(
            Path("login.png"), context="登入需求", source_name="login.png"
        )

        self.assertEqual(description.confidence, 0.9)
        self.assertEqual(captioner.last_diagnostics["attempts"], 2)
        self.assertTrue(captioner.last_diagnostics["recovered_after_retry"])

    def test_missing_project_meaning_is_retried_then_safely_filled(self):
        incomplete = {
            "image_type": "介面截圖",
            "visible_text": ["線上點餐系統", "查看購物車"],
            "description": (
                "此介面截圖顯示線上點餐系統的菜單頁面，"
                "包含餐點、價格及數量調整按鈕。"
            ),
            "relationships": [],
            "teaching_points": [],
            "context_connection": "此圖片具體化菜單瀏覽與點餐頁面。",
            "document_facts": ["畫面顯示菜單及查看購物車按鈕。"],
            "confidence": 0.95,
            "uncertainty": None,
        }
        captioner = SequenceCaptioner([incomplete, incomplete])

        description = captioner.describe(
            Path("menu.png"), context="菜單瀏覽與點餐頁", source_name="menu.png"
        )

        self.assertEqual(
            description.meaning,
            "此介面呈現線上點餐系統的菜單頁面，包含餐點、價格及數量調整按鈕。",
        )
        self.assertTrue(captioner.last_diagnostics["meaning_fallback_applied"])
        self.assertEqual(captioner.last_diagnostics["attempts"], 2)
        self.assertFalse(captioner.last_diagnostics["blocking_reasons"])
        self.assertTrue(any(
            "自動補全" in reason
            for reason in captioner.last_diagnostics["review_reasons"]
        ))

    def test_pipeline_fills_missing_project_meaning_without_blocking_ingest(self):
        class MissingMeaningCaptioner:
            def describe(self, image_path, *, context, source_name):
                return ImageDescription(
                    image_type="介面截圖",
                    visible_text=["菜單", "查看購物車"],
                    description="此介面截圖顯示菜單與查看購物車按鈕。",
                    document_facts=["畫面顯示查看購物車按鈕。"],
                    confidence=0.95,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            enriched, manifest = enrich_markdown_with_marker_images(
                "# 菜單\n![圖](menu.png)",
                {"menu.png": FakeImage()},
                assets_dir=Path(temp_dir) / "assets",
                markdown_assets_path="assets",
                settings=ImageCaptionSettings(enabled=True),
                captioner=MissingMeaningCaptioner(),
                document_type="SRD",
            )

        record = manifest["images"][0]
        self.assertIn("圖片表達的意義", enriched)
        self.assertIn("此介面呈現菜單與查看購物車按鈕。", enriched)
        self.assertTrue(record["review_recommended"])
        self.assertFalse(record["blocks_ingestion"])
        self.assertTrue(record["analysis_eligible"])
        self.assertIn("自動補全", record["review_reasons"][0])

    def test_minor_uncertainty_is_reviewable_but_does_not_block_project_ingest(self):
        class MinorUncertaintyCaptioner:
            def describe(self, image_path, *, context, source_name):
                return ImageDescription(
                    image_type="介面截圖",
                    description="畫面顯示帳號、密碼欄位與登入按鈕。",
                    meaning="此畫面呈現管理者進入後台前的身分驗證入口。",
                    visible_text=["帳號", "密碼", "登入後台"],
                    teaching_points=["不應出現在專案文件圖說"],
                    document_facts=["畫面提供帳號與密碼輸入欄位。"],
                    confidence=0.9,
                    uncertainty="角落的小型說明文字無法完整辨識。",
                    uncertainty_severity="minor",
                    affects_core_meaning=False,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            enriched, manifest = enrich_markdown_with_marker_images(
                "# 登入\n![圖](login.png)",
                {"login.png": FakeImage()},
                assets_dir=Path(temp_dir) / "assets",
                markdown_assets_path="assets",
                settings=ImageCaptionSettings(enabled=True),
                captioner=MinorUncertaintyCaptioner(),
                document_type="SRD",
            )

        record = manifest["images"][0]
        self.assertTrue(record["review_recommended"])
        self.assertFalse(record["blocks_ingestion"])
        self.assertTrue(record["analysis_eligible"])
        self.assertIn("圖片表達的意義", enriched)
        self.assertIn("不確定處（輕微）", enriched)
        self.assertNotIn("教學重點", enriched)
        self.assertNotIn("不應出現在專案文件圖說", enriched)

    def test_project_caption_prompt_keeps_analysis_in_project_qa(self):
        prompt = _model_prompt("登入章節", "login.png", "SRD")

        self.assertIn("不要在 meaning 或其他欄位分析缺漏、矛盾、風險", prompt)
        self.assertIn("teaching_points 必須回傳空陣列", prompt)
        self.assertIn("不得寫入 document_facts", prompt)
        self.assertIn("相輔相成", prompt)

    def test_project_caption_rejects_analysis_and_ungrounded_technology(self):
        description = ImageDescription(
            image_type="介面截圖",
            visible_text=["菜單管理", "新增餐點"],
            description="畫面提供菜單管理與新增餐點功能。",
            meaning="此畫面的不足是缺少搜尋，建議加入篩選功能。",
            document_facts=["後端使用 Python，資料庫使用 MySQL。"],
            confidence=0.9,
        )

        _, blocking = _description_quality_reasons(
            description, min_confidence=0.75, document_type="SRD"
        )

        self.assertTrue(any("專案問答" in reason for reason in blocking))
        self.assertTrue(any("python" in reason for reason in blocking))
        self.assertTrue(any("mysql" in reason for reason in blocking))

    def test_project_caption_rejects_unsupported_context_relationship(self):
        description = ImageDescription(
            image_type="專案看板",
            visible_text=["To Do", "Doing", "Done"],
            description="看板顯示待辦、進行中與完成任務。",
            meaning="此專案看板呈現任務進度與狀態。",
            context_connection="此看板與非功能需求相輔相成。",
            document_facts=["看板包含待辦、進行中與完成欄位。"],
            confidence=0.95,
        )

        _, blocking = _description_quality_reasons(
            description, min_confidence=0.75, document_type="SRD"
        )

        self.assertTrue(any("專案問答" in reason for reason in blocking))

    def test_informative_project_image_without_facts_needs_review(self):
        class IncompleteCaptioner:
            def describe(self, image_path, *, context, source_name):
                return ImageDescription(
                    image_type="使用者故事地圖",
                    description="圖中呈現使用者故事地圖。",
                    visible_text=["登入"],
                    confidence=0.9,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            _, manifest = enrich_markdown_with_marker_images(
                "# 故事地圖\n![圖](story.png)",
                {"story.png": FakeImage()},
                assets_dir=Path(temp_dir) / "assets",
                markdown_assets_path="assets",
                settings=ImageCaptionSettings(enabled=True),
                captioner=IncompleteCaptioner(),
                document_type="SRD",
            )

        record = manifest["images"][0]
        self.assertTrue(record["needs_review"])
        self.assertFalse(record["analysis_eligible"])
        self.assertIn("原子事實", record["review_reasons"][0])


if __name__ == "__main__":
    unittest.main()
