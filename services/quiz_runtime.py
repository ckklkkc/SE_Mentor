"""Discord 教學端唯讀題庫服務；不在學生互動期間呼叫 LLM。"""

import random
from collections import defaultdict

from database.mongo_controller import DiagnosisQuiz


async def get_approved_quizzes(
    count: int = 5,
    course_unit: str | None = None,
) -> list[DiagnosisQuiz]:
    """同一次小測驗優先讓每題對應不同評量目標。"""
    if not 5 <= count <= 10:
        raise ValueError("小測驗題數必須介於 5 到 10 題")

    query: dict = {"review_status": "approved"}
    if course_unit:
        query["$or"] = [
            {"course_unit_id": course_unit},
            {"course_unit_title": course_unit},
            # 相容舊題庫以 chapter 保存教材單元名稱的資料。
            {"chapter": course_unit},
        ]
    approved = await DiagnosisQuiz.find(query).to_list()

    if not course_unit:
        by_unit: dict[str, list[DiagnosisQuiz]] = defaultdict(list)
        for question in approved:
            unit_key = (
                question.course_unit_id
                or question.course_unit_title
                or question.chapter
            )
            by_unit[unit_key].append(question)
        eligible_units = []
        for unit_key, unit_questions in by_unit.items():
            target_ids = {
                question.assessment_target_id or f"legacy:{question.concept}"
                for question in unit_questions
            }
            if len(target_ids) >= count:
                eligible_units.append(unit_key)
        if not eligible_units:
            raise ValueError(
                f"目前沒有單一 CourseUnit 擁有 {count} 個具核准題目的不同評量目標"
            )
        selected_unit = random.choice(eligible_units)
        approved = by_unit[selected_unit]

    by_target: dict[str, list[DiagnosisQuiz]] = defaultdict(list)
    for question in approved:
        key = question.assessment_target_id or f"legacy:{question.concept}"
        by_target[key].append(question)

    if len(by_target) < count:
        scope = f"CourseUnit「{course_unit}」" if course_unit else "全部 CourseUnit"
        raise ValueError(
            f"{scope}只有 {len(by_target)} 個具核准題目的不同評量目標，"
            f"無法組成 {count} 題小測驗"
        )

    selected_targets = random.sample(list(by_target), count)
    questions = [random.choice(by_target[target_id]) for target_id in selected_targets]
    random.shuffle(questions)
    return questions


async def get_quizzes(
    user_id: int,
    count: int = 5,
    course_unit: str | None = None,
) -> tuple[list[DiagnosisQuiz], str]:
    """Teaching boundary：只讀取教師核准題目。"""
    del user_id
    questions = await get_approved_quizzes(count, course_unit)
    selected_title = questions[0].course_unit_title or questions[0].chapter
    scope = course_unit or selected_title
    return (
        questions,
        f"小測驗範圍：{scope}；共 {count} 題，且每題對應不同評量目標。",
    )
