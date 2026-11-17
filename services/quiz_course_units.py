"""Human-facing chapter numbers for the curated 11-unit question bank.

Keep the MongoDB identifiers unchanged: old quiz attempts and target evidence
can still refer to the original bank IDs.
"""

MANUAL_BANK_PREFIX = "se_manual_20261117_v1"

COURSE_UNIT_TITLES = {
    1: "[01]軟體危機與軟體流程",
    2: "[02]基礎需求工程",
    3: "[03]使用者故事分析",
    4: "[04]敏捷開發方法",
    5: "[05]基礎專案管理與看板",
    6: "[06]版本控制",
    7: "[07]軟體設計-系統設計",
    8: "[08]軟體設計-模組設計",
    9: "[09]軟體測試",
    10: "[10]進階軟體測試",
    11: "[11]DevOps自動化建置管理",
}


def resolve_course_unit(course_unit: str | None) -> str | None:
    """Resolve a chapter number while preserving exact title/ID lookups.

    Numeric requests deliberately use the curated bank ID, not the title:
    a title lookup would also select older approved questions of that chapter.
    """
    if course_unit is None:
        return None
    value = course_unit.strip()
    if not value:
        return None
    if value.isdecimal():
        # Bound the input before integer conversion; API calls need no giant ints.
        if len(value) > 2 or int(value) not in COURSE_UNIT_TITLES:
            raise ValueError("章節數字必須介於 1～11，或從章節選單選擇。")
        return f"{MANUAL_BANK_PREFIX}:unit:{int(value):02d}"
    return value
