"""教材導向的 CourseUnit／Section 細粒度知識分析。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field, field_validator

from config import CLAUDE_API_KEY
from database.mongo_controller import (
    AssessmentTargetCandidate,
    KnowledgePointCandidate,
    KnowledgeRelationCandidate,
    SectionAnalysis,
    SourceEvidence,
    TextbookAnalysis,
)


RELATION_TYPES = (
    "prerequisite", "part_of", "contrasts_with", "similar_to",
    "causes", "used_for", "example_of",
)

NON_KNOWLEDGE_TITLE_RULES: tuple[tuple[str, str, str], ...] = (
    (r"^(封面|title\s*page)$", "front_matter", "封面不包含可評量知識"),
    (r"^(目錄|大綱|課程大綱|contents?|outline|agenda)\s*[:：]?$", "front_matter", "目錄或大綱不作為知識來源"),
    (r"^(習題|練習題|課後練習)(\s*\([^)]*\))?\s*[:：]?$|^(exercises?|quiz)\s*[:：]?$", "exercise", "習題區段不抽取新的知識命題"),
    (r"^(參考資料|參考文獻)(\s*\([^)]*\))?\s*[:：]?$|^(references?|bibliography)\s*[:：]?$", "reference", "參考資料區段不作為評量目標"),
    (r"^(any\s+questions?|q\s*&\s*a|questions?)\s*[?？!！:：]*$", "closing", "問答或收尾頁不包含新的課程知識"),
    (r"^(謝謝|感謝聆聽|thank\s+you)\s*[!！.。]*$", "closing", "致謝或收尾頁不包含新的課程知識"),
)


def stable_id(prefix: str, *parts: str) -> str:
    normalized = "|".join(re.sub(r"\s+", " ", p.strip().lower()) for p in parts)
    return f"{prefix}_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]}"


def course_unit_title_from_source(source_name: str) -> str:
    """移除 prepare_markdown 加上的時間戳，保留例如 [02]基礎需求工程。"""
    title = Path(source_name).stem
    return re.sub(r"^\d{8}_\d{6}_\d+_", "", title).strip()


class TextbookChunk(BaseModel):
    id: str
    section_id: str
    section_title: str
    heading_path: list[str]
    content: str


class _MarkdownSection(BaseModel):
    level: int
    heading_path: list[str]
    content: str


class SectionClassification(BaseModel):
    content_type: Literal[
        "knowledge", "front_matter", "exercise", "reference", "closing", "other"
    ]
    included_in_assessment: bool
    reason: str | None = None


def classify_section(title: str) -> SectionClassification:
    normalized = re.sub(r"\s+", " ", title.strip()).lower()
    for pattern, content_type, reason in NON_KNOWLEDGE_TITLE_RULES:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return SectionClassification(
                content_type=content_type,
                included_in_assessment=False,
                reason=reason,
            )
    return SectionClassification(content_type="knowledge", included_in_assessment=True)


class RawKnowledgePoint(BaseModel):
    key: str = Field(description="本區段唯一、簡短且穩定的英文或數字代碼")
    name: str
    description: str
    evidence_chunk_ids: list[str]
    evidence_quotes: list[str] = Field(default_factory=list)


class RawRelation(BaseModel):
    source_key: str
    target_key: str
    relation_type: Literal[
        "prerequisite", "part_of", "contrasts_with", "similar_to",
        "causes", "used_for", "example_of",
    ]
    description: str


class RawAssessmentTarget(BaseModel):
    title: str
    objective: str
    knowledge_point_keys: list[str] = Field(min_length=1)
    discrimination: str
    common_misconceptions: list[str] = Field(min_length=2, max_length=5)
    evidence_chunk_ids: list[str] = Field(min_length=1)

    @field_validator("title", "objective", "discrimination")
    @classmethod
    def reject_embedded_target_sections(cls, value: str) -> str:
        value = value.strip()
        if re.search(r"(?:^|\n)\s*(?:能力|鑑別|迷思)\s*[：:]", value):
            raise ValueError("單一欄位中不可嵌入另一個評量目標區段")
        return re.sub(r"\s+", " ", value)

    @field_validator("common_misconceptions")
    @classmethod
    def validate_misconceptions(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if re.search(r"(?:^|\n)\s*(?:能力|鑑別|迷思)\s*[：:]", value):
                raise ValueError("迷思欄位中不可嵌入另一個評量目標")
            cleaned.append(re.sub(r"\s+", " ", value))
        return cleaned


class RawSectionAnalysis(BaseModel):
    knowledge_points: list[RawKnowledgePoint] = Field(default_factory=list)
    relations: list[RawRelation] = Field(default_factory=list)
    assessment_targets: list[RawAssessmentTarget] = Field(default_factory=list)


class RawMergeGroup(BaseModel):
    member_ids: list[str] = Field(min_length=2)
    rationale: str


class ConsolidationPlan(BaseModel):
    knowledge_point_groups: list[RawMergeGroup] = Field(default_factory=list)
    assessment_target_groups: list[RawMergeGroup] = Field(default_factory=list)


def textbook_analysis_validation_errors(analysis: TextbookAnalysis) -> list[str]:
    """檢查 CourseUnit 是否足以建立五題小測驗，不再強迫每個 Section 生五題。"""
    errors: list[str] = []
    if analysis.consolidation_status != "completed":
        errors.append("跨 Section 知識點／評量目標整併未完成，請重新匯入教材")
    marker = re.compile(r"(?:^|\n)\s*(?:能力|鑑別|迷思)\s*[：:]")
    usable_targets = [
        target for target in analysis.assessment_targets
        if target.review_status != "rejected"
    ]
    if len(usable_targets) < 5:
        errors.append(
            f"整個 CourseUnit 只有 {len(usable_targets)} 個未退回評量目標，至少需要 5 個"
        )
    known_points = {point.id for point in analysis.knowledge_points}
    for target in usable_targets:
        values = [
            target.title,
            target.objective,
            target.discrimination,
            *target.common_misconceptions,
        ]
        if any(marker.search(value or "") for value in values):
            errors.append(f"評量目標 {target.id} 的欄位混入其他目標內容")
        if len(target.common_misconceptions) < 2:
            errors.append(f"評量目標 {target.id} 的具體迷思少於 2 個")
        if not set(target.knowledge_point_ids) <= known_points:
            errors.append(f"評量目標 {target.id} 引用了不存在的知識點")
    for section in analysis.sections:
        if not section.included_in_assessment and (
            section.knowledge_point_ids or section.assessment_target_ids
        ):
            errors.append(f"已排除區段 {section.title} 仍包含知識點或評量目標")
    return errors


def split_textbook_markdown(
    markdown: str,
    source_name: str,
    max_chunk_chars: int = 5000,
) -> tuple[str, list[tuple[str, str, list[TextbookChunk]]]]:
    """依 Markdown 標題保留路徑並切成 Section；名稱保留以相容舊呼叫。"""
    course_unit_id = stable_id("course_unit", source_name, markdown)
    heading_stack: dict[int, str] = {}
    parsed_sections: list[_MarkdownSection] = []
    heading_counts: Counter[int] = Counter()
    buffer: list[str] = []
    current_level = 0
    current_path: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            parsed_sections.append(
                _MarkdownSection(
                    level=current_level,
                    heading_path=current_path.copy(),
                    content=text,
                )
            )
        buffer.clear()

    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            current_level = len(match.group(1))
            heading_counts[current_level] += 1
            heading_stack[current_level] = match.group(2).strip()
            for level in list(heading_stack):
                if level > current_level:
                    del heading_stack[level]
            current_path = [heading_stack[level] for level in sorted(heading_stack)]
        else:
            buffer.append(line)
    flush()

    if not parsed_sections:
        parsed_sections = [
            _MarkdownSection(
                level=1,
                heading_path=[course_unit_title_from_source(source_name)],
                content=markdown,
            )
        ]

    section_level = next(
        (level for level in sorted(heading_counts) if heading_counts[level] >= 2),
        None,
    )
    if section_level is None:
        section_level = min(heading_counts, default=1)

    grouped: dict[str, list[_MarkdownSection]] = {}
    order: list[str] = []
    for parsed in parsed_sections:
        path_index = min(max(section_level - 1, 0), len(parsed.heading_path) - 1)
        title = (
            parsed.heading_path[path_index]
            if parsed.heading_path
            else course_unit_title_from_source(source_name)
        )
        if title not in grouped:
            grouped[title] = []
            order.append(title)
        grouped[title].append(parsed)

    result: list[tuple[str, str, list[TextbookChunk]]] = []
    for section_order, title in enumerate(order, start=1):
        section_id = stable_id("section", course_unit_id, str(section_order), title)
        chunks: list[TextbookChunk] = []
        chunk_number = 0
        for parsed in grouped[title]:
            content = parsed.content.strip()
            while content:
                split_at = min(len(content), max_chunk_chars)
                if split_at < len(content):
                    boundary = content.rfind("\n", 0, split_at)
                    if boundary > max_chunk_chars // 2:
                        split_at = boundary
                piece, content = content[:split_at].strip(), content[split_at:].strip()
                if not piece:
                    continue
                chunk_number += 1
                chunks.append(
                    TextbookChunk(
                        id=stable_id("chunk", section_id, str(chunk_number), piece),
                        section_id=section_id,
                        section_title=title,
                        heading_path=parsed.heading_path,
                        content=piece,
                    )
                )
        result.append((section_id, title, chunks))
    return course_unit_id, result


class _UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {item_id: item_id for item_id in ids}

    def find(self, item_id: str) -> str:
        parent = self.parent[item_id]
        if parent != item_id:
            self.parent[item_id] = self.find(parent)
        return self.parent[item_id]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _merge_map(ids: list[str], groups: list[RawMergeGroup], prefix: str, unit_id: str) -> dict[str, str]:
    union = _UnionFind(ids)
    known = set(ids)
    for group in groups:
        members = list(dict.fromkeys(item for item in group.member_ids if item in known))
        for member in members[1:]:
            union.union(members[0], member)
    grouped: dict[str, list[str]] = defaultdict(list)
    for item_id in ids:
        grouped[union.find(item_id)].append(item_id)
    mapping: dict[str, str] = {}
    for members in grouped.values():
        canonical = (
            members[0]
            if len(members) == 1
            else stable_id(prefix, unit_id, *sorted(members))
        )
        mapping.update({member: canonical for member in members})
    return mapping


def consolidate_course_unit(
    course_unit_id: str,
    sections: list[SectionAnalysis],
    points: list[KnowledgePointCandidate],
    relations: list[KnowledgeRelationCandidate],
    targets: list[AssessmentTargetCandidate],
    plan: ConsolidationPlan | None = None,
) -> tuple[
    list[SectionAnalysis],
    list[KnowledgePointCandidate],
    list[KnowledgeRelationCandidate],
    list[AssessmentTargetCandidate],
]:
    """依跨 Section 合併計畫去重，並保留所有來源 Section 與教材證據。"""
    plan = plan or ConsolidationPlan()
    point_map = _merge_map(
        [point.id for point in points],
        plan.knowledge_point_groups,
        "kp",
        course_unit_id,
    )
    point_groups: dict[str, list[KnowledgePointCandidate]] = defaultdict(list)
    for point in points:
        point_groups[point_map[point.id]].append(point)

    merged_points: list[KnowledgePointCandidate] = []
    for canonical_id, members in point_groups.items():
        best = max(members, key=lambda item: len(item.description))
        evidence_by_chunk = {
            evidence.chunk_id: evidence
            for member in members
            for evidence in member.evidence
        }
        merged_points.append(KnowledgePointCandidate(
            id=canonical_id,
            name=best.name,
            description=best.description,
            section_ids=list(dict.fromkeys(
                section_id for member in members for section_id in member.section_ids
            )),
            evidence=list(evidence_by_chunk.values()),
        ))

    remapped_targets: list[AssessmentTargetCandidate] = []
    for target in targets:
        remapped_targets.append(target.model_copy(update={
            "knowledge_point_ids": list(dict.fromkeys(
                point_map[point_id]
                for point_id in target.knowledge_point_ids
                if point_id in point_map
            )),
        }))
    target_map = _merge_map(
        [target.id for target in remapped_targets],
        plan.assessment_target_groups,
        "target",
        course_unit_id,
    )
    target_groups: dict[str, list[AssessmentTargetCandidate]] = defaultdict(list)
    for target in remapped_targets:
        target_groups[target_map[target.id]].append(target)

    merged_targets: list[AssessmentTargetCandidate] = []
    for canonical_id, members in target_groups.items():
        best = max(members, key=lambda item: len(item.objective) + len(item.discrimination))
        merged_targets.append(AssessmentTargetCandidate(
            id=canonical_id,
            section_ids=list(dict.fromkeys(
                section_id for member in members for section_id in member.section_ids
            )),
            knowledge_point_ids=list(dict.fromkeys(
                point_id for member in members for point_id in member.knowledge_point_ids
            )),
            title=best.title,
            objective=best.objective,
            discrimination=best.discrimination,
            common_misconceptions=list(dict.fromkeys(
                misconception
                for member in members
                for misconception in member.common_misconceptions
            ))[:5],
            evidence_chunk_ids=list(dict.fromkeys(
                chunk_id for member in members for chunk_id in member.evidence_chunk_ids
            )),
        ))

    relation_by_key: dict[tuple[str, str, str], KnowledgeRelationCandidate] = {}
    for relation in relations:
        source_id = point_map.get(relation.source_id)
        target_id = point_map.get(relation.target_id)
        if not source_id or not target_id or source_id == target_id:
            continue
        key = (source_id, target_id, relation.relation_type)
        relation_by_key.setdefault(key, relation.model_copy(update={
            "source_id": source_id,
            "target_id": target_id,
        }))

    for section in sections:
        section.knowledge_point_ids = list(dict.fromkeys(
            point_map[point_id]
            for point_id in section.knowledge_point_ids
            if point_id in point_map
        ))
        section.assessment_target_ids = list(dict.fromkeys(
            target_map[target_id]
            for target_id in section.assessment_target_ids
            if target_id in target_map
        ))

    return sections, merged_points, list(relation_by_key.values()), merged_targets


class TextbookAnalyzer:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    def _analyze_section(self, title: str, chunks: list[TextbookChunk]) -> RawSectionAnalysis:
        context = "\n\n".join(
            f"<chunk id=\"{chunk.id}\" headings=\"{' > '.join(chunk.heading_path)}\">\n"
            f"{chunk.content}\n</chunk>"
            for chunk in chunks
        )
        prompt = f"""
你是軟體工程課程的教材分析與評量設計專家。分析 CourseUnit 內的 Section「{title}」。

1. knowledge_points 必須是可判斷對錯的最小知識命題，不可只抽取大主題或高頻關鍵字。
2. 同義詞不要拆開；Section 標題本身不能直接當知識點。
3. relations 只能使用 {', '.join(RELATION_TYPES)}，兩端必須引用 knowledge point key。
4. assessment_targets 優先評量知識點之間的差異、關聯、前置條件或情境選用。
5. common_misconceptions 必須能直接形成錯誤選項，不能只寫「觀念不清」。
6. 每個知識點與目標都必須引用實際 chunk id；引文只能摘自教材。
7. 依內容自然產生必要數量，不得為了湊數重複、拆碎或杜撰目標；沒有可評量內容時可回傳空陣列。
8. 各欄位不可混入「能力：」「鑑別：」「迷思：」等其他欄位內容。
9. 只使用下列教材，使用台灣繁體中文。

教材：
{context}
"""
        response = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=10000,
            messages=[{"role": "user", "content": prompt}],
            output_format=RawSectionAnalysis,
        )
        return response.parsed_output

    def _build_consolidation_plan(
        self,
        course_unit_title: str,
        points: list[KnowledgePointCandidate],
        targets: list[AssessmentTargetCandidate],
    ) -> ConsolidationPlan:
        if len(points) < 2 and len(targets) < 2:
            return ConsolidationPlan()
        payload = {
            "knowledge_points": [
                {"id": item.id, "name": item.name, "description": item.description}
                for item in points
            ],
            "assessment_targets": [
                {
                    "id": item.id,
                    "title": item.title,
                    "objective": item.objective,
                    "discrimination": item.discrimination,
                }
                for item in targets
            ],
        }
        prompt = f"""
你正在整併同一 CourseUnit「{course_unit_title}」不同 Section 的分析結果。
找出語意實質相同、只是措辭或例子不同的知識命題與評量目標。

規則：
1. 只輸出確定應合併的群組；每組至少兩個既有 ID，不可創造 ID。
2. 相關但可分別判斷對錯的命題不可合併。
3. 上位概念與下位概念、定義與範例、功能與非功能需求不可因文字相似而合併。
4. 評量目標只有在可觀察能力與鑑別目的相同時才合併。

候選資料：
{json.dumps(payload, ensure_ascii=False)}
"""
        response = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=6000,
            messages=[{"role": "user", "content": prompt}],
            output_format=ConsolidationPlan,
        )
        return response.parsed_output

    def analyze(self, markdown: str, source_file: str, uploader: str) -> TextbookAnalysis:
        course_unit_id, section_chunks = split_textbook_markdown(markdown, source_file)
        course_unit_title = course_unit_title_from_source(source_file)
        sections: list[SectionAnalysis] = []
        all_points: list[KnowledgePointCandidate] = []
        all_relations: list[KnowledgeRelationCandidate] = []
        all_targets: list[AssessmentTargetCandidate] = []

        for order, (section_id, title, chunks) in enumerate(section_chunks, start=1):
            classification = classify_section(title)
            section = SectionAnalysis(
                id=section_id,
                title=title,
                order=order,
                content_type=classification.content_type,
                included_in_assessment=classification.included_in_assessment,
                exclusion_reason=classification.reason,
            )
            sections.append(section)
            if not classification.included_in_assessment:
                continue

            raw = self._analyze_section(title, chunks)
            chunk_by_id = {chunk.id: chunk for chunk in chunks}
            point_by_key: dict[str, KnowledgePointCandidate] = {}
            for point in raw.knowledge_points:
                valid_chunk_ids = [cid for cid in point.evidence_chunk_ids if cid in chunk_by_id]
                evidence: list[SourceEvidence] = []
                for index, chunk_id in enumerate(valid_chunk_ids):
                    chunk = chunk_by_id[chunk_id]
                    quote = point.evidence_quotes[index] if index < len(point.evidence_quotes) else ""
                    evidence.append(SourceEvidence(
                        chunk_id=chunk_id,
                        heading_path=chunk.heading_path,
                        quote=quote[:500],
                    ))
                if not evidence:
                    continue
                point_id = stable_id(
                    "kp_candidate", course_unit_id, section_id, point.name, point.description
                )
                point_by_key[point.key] = KnowledgePointCandidate(
                    id=point_id,
                    name=point.name,
                    description=point.description,
                    section_ids=[section_id],
                    evidence=evidence,
                )

            section.knowledge_point_ids = [point.id for point in point_by_key.values()]
            all_points.extend(point_by_key.values())
            for relation in raw.relations:
                source = point_by_key.get(relation.source_key)
                target = point_by_key.get(relation.target_key)
                if not source or not target or relation.relation_type not in RELATION_TYPES:
                    continue
                all_relations.append(KnowledgeRelationCandidate(
                    source_id=source.id,
                    target_id=target.id,
                    relation_type=relation.relation_type,
                    description=relation.description,
                ))

            known_evidence_ids = {
                evidence.chunk_id
                for point in point_by_key.values()
                for evidence in point.evidence
            }
            section_targets: list[AssessmentTargetCandidate] = []
            for target in raw.assessment_targets:
                point_ids = [
                    point_by_key[key].id
                    for key in target.knowledge_point_keys
                    if key in point_by_key
                ]
                evidence_ids = [
                    cid for cid in target.evidence_chunk_ids if cid in known_evidence_ids
                ]
                if not point_ids or not evidence_ids:
                    continue
                section_targets.append(AssessmentTargetCandidate(
                    id=stable_id(
                        "target_candidate", course_unit_id, section_id,
                        target.title, *point_ids,
                    ),
                    section_ids=[section_id],
                    knowledge_point_ids=list(dict.fromkeys(point_ids)),
                    title=target.title,
                    objective=target.objective,
                    discrimination=target.discrimination,
                    common_misconceptions=target.common_misconceptions,
                    evidence_chunk_ids=list(dict.fromkeys(evidence_ids)),
                ))
            section.assessment_target_ids = [target.id for target in section_targets]
            all_targets.extend(section_targets)

        try:
            plan = self._build_consolidation_plan(
                course_unit_title,
                all_points,
                all_targets,
            )
            consolidation_status = "completed"
            consolidation_note = None
        except Exception as exc:
            # Section 級分析仍可供教師審核；不因第二階段 API 暫時失敗丟失全部成果。
            print(f"[textbook_analyzer] 跨 Section 整併失敗，保留未合併候選：{exc}")
            plan = ConsolidationPlan()
            consolidation_status = "fallback"
            consolidation_note = f"跨 Section 整併失敗，保留未合併候選：{exc}"

        sections, points, relations, targets = consolidate_course_unit(
            course_unit_id,
            sections,
            all_points,
            all_relations,
            all_targets,
            plan,
        )
        return TextbookAnalysis(
            textbook_id=course_unit_id,
            course_unit_id=course_unit_id,
            course_unit_title=course_unit_title,
            title=course_unit_title,
            source_file=Path(source_file).name,
            uploader=uploader,
            sections=sections,
            knowledge_points=points,
            relations=relations,
            assessment_targets=targets,
            consolidation_status=consolidation_status,
            consolidation_note=consolidation_note,
        )
