import json
import tempfile
import unittest
from pathlib import Path

from services.document_image_quality import (
    ProjectImageQualityError,
    validate_project_image_conversion,
)


class ProjectImageQualityTests(unittest.TestCase):
    def _write(self, root: Path, manifest: dict) -> Path:
        markdown = root / "document.md"
        markdown.write_text("# 文件", encoding="utf-8")
        markdown.with_name("document_images.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        return markdown

    def test_verified_figure_passes_and_decoration_does_not_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            markdown = self._write(Path(temporary), {"pages": [{
                "page": 1,
                "errors": [],
                "figures": [
                    {"review_content_category": "diagram", "analysis_eligible": True},
                    {"review_content_category": "decoration", "analysis_eligible": False},
                ],
            }]})
            validate_project_image_conversion(markdown)

    def test_low_confidence_informative_figure_blocks_kg(self):
        with tempfile.TemporaryDirectory() as temporary:
            markdown = self._write(Path(temporary), {"pages": [{
                "page": 7,
                "errors": [],
                "figures": [{
                    "review_content_category": "ui_figure",
                    "analysis_eligible": False,
                    "confidence": 0.42,
                }],
            }]})
            with self.assertRaisesRegex(
                ProjectImageQualityError, "第 7 頁 ui_figure.*42%"
            ):
                validate_project_image_conversion(markdown)

    def test_missing_manifest_blocks_pdf_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            markdown = Path(temporary) / "document.md"
            markdown.write_text("# 文件", encoding="utf-8")
            with self.assertRaisesRegex(ProjectImageQualityError, "找不到圖片轉換紀錄"):
                validate_project_image_conversion(markdown)

    def test_marker_failure_reports_reason_without_broken_discord_url(self):
        with tempfile.TemporaryDirectory() as temporary:
            markdown = self._write(Path(temporary), {"images": [{
                "source_key": "https://hackmd.io/_uploads/a.png",
                "status": "description_failed",
                "description": {"confidence": 0},
                "needs_review": True,
                "review_reasons": ["圖片模型回應欄位不完整或格式錯誤"],
            }]})
            with self.assertRaises(ProjectImageQualityError) as raised:
                validate_project_image_conversion(markdown)

        message = str(raised.exception)
        self.assertIn("`https://hackmd.io/_uploads/a.png`", message)
        self.assertIn("欄位不完整", message)

    def test_minor_uncertainty_does_not_block_new_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            markdown = self._write(Path(temporary), {"images": [{
                "source_key": "https://hackmd.io/_uploads/a.png",
                "status": "described",
                "description": {"confidence": 0.9},
                "needs_review": True,
                "review_recommended": True,
                "blocks_ingestion": False,
                "review_reasons": ["minor 不確定處：角落小字不清楚"],
                "blocking_reasons": [],
            }]})

            validate_project_image_conversion(markdown)

    def test_explicitly_excluded_trello_evidence_image_does_not_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            markdown = self._write(Path(temporary), {"images": [{
                "source_key": "https://hackmd.io/_uploads/trello.png",
                "status": "preserved_description_excluded",
                "description": None,
                "occurrences": [{
                    "status": "description_excluded",
                    "description": None,
                    "exclusion_reason": "獨立 Trello 章節中的專案任務規劃展示圖片",
                    "blocks_ingestion": False,
                }],
                "description_exclusions": [
                    "獨立 Trello 章節中的專案任務規劃展示圖片"
                ],
                "review_recommended": False,
                "blocks_ingestion": False,
            }]})

            validate_project_image_conversion(markdown)

    def test_major_uncertainty_blocks_new_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            markdown = self._write(Path(temporary), {"images": [{
                "source_key": "https://hackmd.io/_uploads/flow.png",
                "status": "described",
                "description": {"confidence": 0.85},
                "needs_review": True,
                "review_recommended": True,
                "blocks_ingestion": True,
                "review_reasons": ["major 不確定處：箭頭方向無法辨識"],
                "blocking_reasons": ["major 不確定處：箭頭方向無法辨識"],
            }]})
            with self.assertRaisesRegex(
                ProjectImageQualityError, "箭頭方向無法辨識"
            ):
                validate_project_image_conversion(markdown)

    def test_blocking_message_reports_attempts_and_manifest_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            markdown = self._write(Path(temporary), {"images": [{
                "source_key": "https://hackmd.io/_uploads/menu.png",
                "status": "described",
                "description": {"confidence": 0.95},
                "blocks_ingestion": True,
                "blocking_reasons": ["資訊圖片未擷取到可驗證文字、關係或事實"],
                "occurrences": [{
                    "description": {"confidence": 0.95},
                    "diagnostics": {"attempts": 3},
                }],
            }]})

            with self.assertRaises(ProjectImageQualityError) as raised:
                validate_project_image_conversion(markdown)

        message = str(raised.exception)
        self.assertIn("結構化轉述", message)
        self.assertIn("已嘗試 3 次", message)
        self.assertIn("`document_images.json`", message)


if __name__ == "__main__":
    unittest.main()
