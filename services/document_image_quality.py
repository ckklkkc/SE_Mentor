"""Quality gate for image-aware SRD/SDD/STD ingestion."""

from __future__ import annotations

import json
from pathlib import Path


class ProjectImageQualityError(RuntimeError):
    """The converted document still contains images unsafe for analysis."""


def _inline_code(value: object) -> str:
    """Keep Discord from extending a raw URL into the following punctuation."""
    return f"`{str(value).replace('`', '')}`"


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def validate_project_image_conversion(markdown_path: str | Path) -> None:
    """Reject conversion results that would silently omit informative images."""
    markdown_path = Path(markdown_path)
    manifest_path = markdown_path.with_name(f"{markdown_path.stem}_images.json")
    if not manifest_path.is_file():
        raise ProjectImageQualityError(
            "找不到圖片轉換紀錄，已停止文件分析；請重新執行含圖片描述的 PDF 轉換。"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blocking: list[str] = []

    for page in manifest.get("pages", []):
        page_number = page.get("page", "?")
        errors = page.get("errors", [])
        if errors and (page.get("embedded_images") or page.get("regions")):
            stages = ", ".join(
                str(error.get("stage", "unknown")) for error in errors
            )
            blocking.append(f"第 {page_number} 頁（{stages}）")
        for figure in page.get("figures", []):
            category = figure.get(
                "review_content_category", figure.get("category", "unknown")
            )
            if category in {"decoration", "plain_text"}:
                continue
            if not figure.get("analysis_eligible", False):
                blocking.append(
                    f"第 {page_number} 頁 {category}"
                    f"（信心 {float(figure.get('confidence', 0.0)):.0%}）"
                )

    # Marker/HackMD manifests distinguish a review suggestion from a true
    # ingestion blocker. Older manifests did not have blocks_ingestion, so keep
    # their stricter needs_review behavior for backward compatibility.
    for image in manifest.get("images", []):
        description = image.get("description") or {}
        occurrence_confidences = [
            float((occurrence.get("description") or {}).get("confidence", 0.0))
            for occurrence in image.get("occurrences", [])
        ]
        confidence = min(
            occurrence_confidences
            or [float(description.get("confidence", 0.0))]
        )
        hard_failure = image.get("status") in {
                "description_failed",
                "preserved_description_disabled",
                "preserved_description_limit",
                "preserved_too_small",
            }
        if "blocks_ingestion" in image:
            should_block = hard_failure or bool(image.get("blocks_ingestion"))
        else:
            should_block = hard_failure or bool(image.get("needs_review"))
        if should_block:
            reasons = [
                str(reason).strip()
                for reason in (
                    image.get("blocking_reasons")
                    or image.get("review_reasons", [])
                )
                if str(reason).strip()
            ]
            if not reasons and image.get("error"):
                reasons.append("圖片描述服務失敗")
            reason_text = f"；原因：{'、'.join(reasons[:3])}" if reasons else ""
            attempts = [
                attempt
                for occurrence in image.get("occurrences", [])
                if (
                    attempt := _positive_int(
                        (occurrence.get("diagnostics") or {}).get("attempts")
                    )
                ) is not None
            ]
            attempt_text = f"；已嘗試 {max(attempts)} 次" if attempts else ""
            blocking.append(
                f"圖片 {_inline_code(image.get('source_key', '?'))}"
                f"（信心 {confidence:.0%}{attempt_text}{reason_text}）"
            )

    if blocking:
        unique = list(dict.fromkeys(blocking))
        preview = "；".join(unique[:12])
        remainder = len(unique) - 12
        if remainder > 0:
            preview += f"；另有 {remainder} 個"
        raise ProjectImageQualityError(
            "圖片的結構化轉述尚未通過自動檢查，因此尚未建立知識圖譜"
            f"與向量索引：{preview}。請查看 `{manifest_path.name}` 中的"
            " blocking_reasons 與 diagnostics。"
        )
