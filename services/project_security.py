"""Security boundaries for student-owned project documents and questions."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit, urlunsplit


PROJECT_DOCUMENT_TYPES = {"SRD", "SDD", "STD"}
PROJECT_QUESTION_MAX_CHARS = 500
PROJECT_DOCUMENT_MAX_CHARS = 200_000


class ProjectAccessError(ValueError):
    """Raised before retrieval or LLM use when project access is not allowed."""


class ProjectDocumentRejectedError(ValueError):
    """Raised before image captioning when a source is not a project artifact."""


@dataclass(frozen=True)
class ProjectDocumentRelevance:
    document_type: str
    character_count: int
    heading_count: int
    table_row_count: int
    image_count: int
    identifier_count: int
    matched_signals: tuple[str, ...]
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


_TYPE_SIGNALS = {
    "SRD": {
        "需求文件": r"(?:軟體|系統)?需求(?:規格|文件|說明)|\bSRD\b|requirements? specification",
        "功能需求": r"功能需求|functional requirements?",
        "非功能需求": r"非功能需求|non[- ]functional requirements?",
        "使用者需求": r"使用者故事|user stor(?:y|ies)|使用案例|use cases?",
        "驗收與角色": r"驗收|接受準則|acceptance criteria|利害關係人|stakeholders?|actors?",
    },
    "SDD": {
        "設計文件": r"軟體設計(?:文件|規格)?|系統設計|\bSDD\b|software design",
        "架構": r"系統架構|軟體架構|architecture|deployment",
        "元件模組": r"元件|模組|component|module|service",
        "介面": r"\bAPI\b|介面規格|interface|endpoint|request|response",
        "資料設計": r"資料庫|資料模型|database|schema|\bERD\b|entity relationship",
        "行為設計": r"循序圖|類別圖|sequence diagram|class diagram|design pattern",
    },
    "STD": {
        "測試文件": r"軟體測試(?:文件|計畫|規格)?|\bSTD\b|software test",
        "測試案例": r"測試案例|測試個案|test cases?",
        "測試步驟": r"測試步驟|前置條件|test steps?|preconditions?",
        "預期實際": r"預期結果|實際結果|expected results?|actual results?",
        "測試結果": r"通過|失敗|pass(?:ed)?|fail(?:ed)?|缺陷|defects?|bugs?",
        "測試層級": r"單元測試|整合測試|系統測試|驗收測試|unit test|integration test|system test|acceptance test",
    },
}

_PROJECT_ANCHOR = re.compile(
    r"(?:本|此)?(?:系統|專案|平台|應用程式|軟體|服務|網站|後端|前端)"
    r"|\b(?:system|project|application|software|platform|backend|frontend)\b",
    re.IGNORECASE,
)
_HEADING = re.compile(r"(?m)^#{1,6}\s+\S")
_TABLE_ROW = re.compile(r"(?m)^\s*\|.+\|\s*$")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^\)]+\)")
_IDENTIFIERS = {
    "SRD": re.compile(
        r"\b(?:REQ|FR|NFR|US|USR|NF|MEN|MAN|STA|REV)[-_]?[A-Z0-9][A-Z0-9_.-]*\b",
        re.IGNORECASE,
    ),
    "SDD": re.compile(
        r"\b(?:API|CMP|COMP|MOD|SVC|DB|ARCH)[-_]?[A-Z0-9][A-Z0-9_.-]*\b",
        re.IGNORECASE,
    ),
    "STD": re.compile(
        r"\b(?:TC|TEST|UT|IT|ST|AT)[-_]?[A-Z0-9][A-Z0-9_.-]*\b",
        re.IGNORECASE,
    ),
}

_CROSS_GROUP_REQUEST = re.compile(
    r"(?:別組|其他組|他組|另一組|別的小組|其他小組)"
    r"|\b(?:another|other)\s+(?:group|team)s?\b",
    re.IGNORECASE,
)
_DIRECT_PROMPT_OVERRIDE = re.compile(
    r"忽略.{0,24}(?:規則|指令|提示|限制)"
    r"|(?:顯示|輸出|洩漏).{0,24}(?:系統提示|system prompt|developer message)"
    r"|\bignore\b.{0,40}\b(?:previous|above|system|developer)\b.{0,24}\b(?:instructions?|prompts?|messages?)\b",
    re.IGNORECASE | re.DOTALL,
)


def canonical_hackmd_note_url(note_url: str) -> str:
    parsed = urlsplit(note_url.strip())
    host = (parsed.hostname or "").rstrip(".").lower()
    if parsed.scheme.lower() != "https" or host not in {"hackmd.io", "www.hackmd.io"}:
        raise ProjectAccessError("只允許教師核准的 https://hackmd.io 專案文件。")
    if parsed.username or parsed.password or not parsed.path.strip("/"):
        raise ProjectAccessError("HackMD 專案文件連結格式不正確。")
    path = re.sub(
        r"/(?:edit|view|publish|slide|download)/?$", "", parsed.path
    ).rstrip("/")
    return urlunsplit(("https", "hackmd.io", path, "", ""))


def validate_project_document_source(
    note_url: str,
    group: str,
    document_type: str,
    allowed_sources: object,
    *,
    require_allowlist: bool = True,
) -> str:
    """Authorize an exact HackMD note before downloading or captioning it."""
    normalized_type = document_type.upper()
    if normalized_type not in PROJECT_DOCUMENT_TYPES:
        raise ProjectAccessError("文件類型必須是 SRD、SDD 或 STD。")
    canonical = canonical_hackmd_note_url(note_url)
    rules = allowed_sources if isinstance(allowed_sources, dict) else {}
    group_rules = rules.get(group)
    if not isinstance(group_rules, dict):
        if require_allowlist:
            raise ProjectAccessError(
                f"組別「{group}」尚未由教師登記可上傳的專案文件連結。"
            )
        return canonical
    values = group_rules.get(normalized_type, [])
    if isinstance(values, str):
        values = [values]
    approved = {
        canonical_hackmd_note_url(value)
        for value in values
        if isinstance(value, str) and value.strip()
    }
    if canonical not in approved:
        raise ProjectAccessError(
            f"這個 HackMD 並非組別「{group}」核准的 {normalized_type} 文件。"
        )
    return canonical


def assess_project_document_relevance(
    markdown: str,
    document_type: str,
) -> ProjectDocumentRelevance:
    """Perform a zero-token structural relevance check for SRD/SDD/STD."""
    normalized_type = document_type.upper()
    if normalized_type not in PROJECT_DOCUMENT_TYPES:
        raise ProjectDocumentRejectedError("文件類型必須是 SRD、SDD 或 STD。")
    character_count = len(markdown)
    heading_count = len(_HEADING.findall(markdown))
    table_row_count = len(_TABLE_ROW.findall(markdown))
    image_count = len(_IMAGE.findall(markdown))
    identifier_count = len(_IDENTIFIERS[normalized_type].findall(markdown))
    matched = tuple(
        name
        for name, pattern in _TYPE_SIGNALS[normalized_type].items()
        if re.search(pattern, markdown, re.IGNORECASE)
    )
    reasons: list[str] = []
    if character_count < 300:
        reasons.append("內容過短，無法構成可分析的專案文件")
    if character_count > PROJECT_DOCUMENT_MAX_CHARS:
        reasons.append(
            f"內容超過 {PROJECT_DOCUMENT_MAX_CHARS:,} 字的單次專案文件上限"
        )
    if not _PROJECT_ANCHOR.search(markdown):
        reasons.append("缺少系統、軟體、平台或專案等明確分析對象")
    if len(matched) < 2:
        reasons.append(
            f"缺少 {normalized_type} 應有的兩類以上結構性內容"
        )
    structure_points = sum((
        heading_count >= 2,
        table_row_count >= 3,
        identifier_count >= 2,
        image_count >= 1,
    ))
    if structure_points < 2:
        reasons.append("缺少足夠的章節、表格、文件編號或專案圖例結構")
    return ProjectDocumentRelevance(
        document_type=normalized_type,
        character_count=character_count,
        heading_count=heading_count,
        table_row_count=table_row_count,
        image_count=image_count,
        identifier_count=identifier_count,
        matched_signals=matched,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def validate_project_document_relevance(
    markdown: str,
    document_type: str,
) -> ProjectDocumentRelevance:
    report = assess_project_document_relevance(markdown, document_type)
    if not report.passed:
        raise ProjectDocumentRejectedError(
            "文件未通過軟體工程專案文件檢查，且尚未呼叫圖片或知識圖譜模型："
            + "；".join(report.reasons)
        )
    return report


def validate_project_question(question: str) -> str:
    """Reject cross-group and direct instruction-override requests early."""
    normalized = question.strip()
    if not normalized:
        raise ProjectAccessError("問題不得為空白。")
    if len(normalized) > PROJECT_QUESTION_MAX_CHARS:
        raise ProjectAccessError(
            f"專案問題最多 {PROJECT_QUESTION_MAX_CHARS} 字，請縮小問題範圍。"
        )
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", normalized):
        raise ProjectAccessError("問題不得包含控制字元。")
    if _CROSS_GROUP_REQUEST.search(normalized):
        raise ProjectAccessError("專案問答只能查詢你所屬組別的文件，不能查詢或比較其他組別內容。")
    if _DIRECT_PROMPT_OVERRIDE.search(normalized):
        raise ProjectAccessError("問題包含要求改變系統規則或揭露內部提示的內容，已拒絕處理。")
    return normalized
