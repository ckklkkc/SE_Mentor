"""教材導向的細粒度知識點與評量目標分析。

這個模組刻意不沿用一般 Entity/Triple 抽取：教材的章節標題不是學生弱點，
可診斷的最小單位應是「可判斷對錯的知識命題」及其與其他命題的關係。
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from config import CLAUDE_API_KEY
from database.mongo_controller import (
    AssessmentTargetCandidate,
    ChapterAnalysis,
    KnowledgePointCandidate,
    KnowledgeRelationCandidate,
    SourceEvidence,
    TextbookAnalysis,
)


RELATION_TYPES = (
    "prerequisite", "part_of", "contrasts_with", "similar_to",
    "causes", "used_for", "example_of",
)


def stable_id(prefix: str, *parts: str) -> str:
    normalized = "|".join(re.sub(r"\s+", " ", p.strip().lower()) for p in parts)
    return f"{prefix}_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]}"


class TextbookChunk(BaseModel):
    id: str
    chapter_id: str
    chapter_title: str
    heading_path: list[str]
    content: str


class _Section(BaseModel):
    level: int
    heading_path: list[str]
    content: str


class RawKnowledgePoint(BaseModel):
    key: str = Field(description="本章唯一、簡短且穩定的英文或數字代碼")
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
    knowledge_point_keys: list[str]
    discrimination: str
    common_misconceptions: list[str]
    evidence_chunk_ids: list[str]


class RawChapterAnalysis(BaseModel):
    knowledge_points: list[RawKnowledgePoint]
    relations: list[RawRelation]
    assessment_targets: list[RawAssessmentTarget]


def split_textbook_markdown(
    markdown: str,
    source_name: str,
    max_chunk_chars: int = 5000,
) -> tuple[str, list[tuple[str, str, list[TextbookChunk]]]]:
    """依 Markdown 標題保留章節路徑並切塊，回傳 textbook_id 與章節。"""
    textbook_id = stable_id("book", source_name, markdown)
    heading_stack: dict[int, str] = {}
    sections: list[_Section] = []
    heading_counts: Counter[int] = Counter()
    buffer: list[str] = []
    current_level = 0
    current_path: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(
                _Section(level=current_level, heading_path=current_path.copy(), content=text)
            )
        buffer.clear()

    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            current_level = len(match.group(1))
            heading_counts[current_level] += 1
            title = match.group(2).strip()
            heading_stack[current_level] = title
            for level in list(heading_stack):
                if level > current_level:
                    del heading_stack[level]
            current_path = [heading_stack[level] for level in sorted(heading_stack)]
        else:
            buffer.append(line)
    flush()

    if not sections:
        sections = [_Section(level=1, heading_path=[Path(source_name).stem], content=markdown)]

    chapter_level = next(
        (level for level in sorted(heading_counts) if heading_counts[level] >= 2),
        None,
    )
    if chapter_level is None:
        chapter_level = min(heading_counts, default=1)

    grouped: dict[str, list[_Section]] = {}
    order: list[str] = []
    for section in sections:
        path_index = min(max(chapter_level - 1, 0), len(section.heading_path) - 1)
        chapter_title = (
            section.heading_path[path_index]
            if section.heading_path
            else Path(source_name).stem
        )
        if chapter_title not in grouped:
            grouped[chapter_title] = []
            order.append(chapter_title)
        grouped[chapter_title].append(section)

    chapters: list[tuple[str, str, list[TextbookChunk]]] = []
    for chapter_order, title in enumerate(order, start=1):
        chapter_id = stable_id("chapter", textbook_id, str(chapter_order), title)
        chunks: list[TextbookChunk] = []
        chunk_number = 0
        for section in grouped[title]:
            content = section.content.strip()
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
                        id=stable_id("chunk", chapter_id, str(chunk_number), piece),
                        chapter_id=chapter_id,
                        chapter_title=title,
                        heading_path=section.heading_path,
                        content=piece,
                    )
                )
        chapters.append((chapter_id, title, chunks))
    return textbook_id, chapters


class TextbookAnalyzer:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    def _analyze_chapter(self, title: str, chunks: list[TextbookChunk]) -> RawChapterAnalysis:
        context = "\n\n".join(
            f"<chunk id=\"{chunk.id}\" headings=\"{' > '.join(chunk.heading_path)}\">\n"
            f"{chunk.content}\n</chunk>"
            for chunk in chunks
        )
        prompt = f"""
你是軟體工程課程的教材分析與評量設計專家。分析章節「{title}」。

目標不是抽取大主題或高頻關鍵字，而是建立能診斷學生理解差異的細粒度結構：
1. knowledge_points 必須是可單獨判斷理解正確與否的最小知識命題。例如不要只寫「使用者故事」，
   應寫「使用者故事以使用者價值描述需求，而非正式需求規格」或「使用者故事與驗收條件的功能不同」。
2. 同義詞不要拆成不同知識點；章節標題本身不能直接當知識點。
3. relations 只能使用 {', '.join(RELATION_TYPES)}，且兩端必須引用 knowledge point 的 key。
4. assessment_targets 應優先評量兩個以上知識點之間的差異、關聯、前置條件或情境選用；
   單一知識點目標也必須評量可觀察的理解，不可只是名詞背誦。
5. common_misconceptions 要描述可對應到錯誤選項的具體錯誤推理。
6. 每個知識點與評量目標都必須引用實際 chunk id。evidence_quotes 只能短摘教材原文，不可杜撰。
7. 只使用下列教材內容，使用台灣繁體中文。

教材：
{context}
"""
        response = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=10000,
            messages=[{"role": "user", "content": prompt}],
            output_format=RawChapterAnalysis,
        )
        return response.parsed_output

    def analyze(self, markdown: str, source_file: str, uploader: str) -> TextbookAnalysis:
        textbook_id, chapter_chunks = split_textbook_markdown(markdown, source_file)
        chapter_results: list[ChapterAnalysis] = []

        for order, (chapter_id, title, chunks) in enumerate(chapter_chunks, start=1):
            raw = self._analyze_chapter(title, chunks)
            chunk_by_id = {chunk.id: chunk for chunk in chunks}
            point_by_key: dict[str, KnowledgePointCandidate] = {}

            for point in raw.knowledge_points:
                valid_chunk_ids = [cid for cid in point.evidence_chunk_ids if cid in chunk_by_id]
                evidence: list[SourceEvidence] = []
                for index, chunk_id in enumerate(valid_chunk_ids):
                    chunk = chunk_by_id[chunk_id]
                    quote = point.evidence_quotes[index] if index < len(point.evidence_quotes) else ""
                    evidence.append(
                        SourceEvidence(
                            chunk_id=chunk_id,
                            heading_path=chunk.heading_path,
                            quote=quote[:500],
                        )
                    )
                if not evidence:
                    continue
                kp_id = stable_id("kp", textbook_id, chapter_id, point.name, point.description)
                point_by_key[point.key] = KnowledgePointCandidate(
                    id=kp_id,
                    name=point.name,
                    description=point.description,
                    chapter_id=chapter_id,
                    evidence=evidence,
                )

            relations: list[KnowledgeRelationCandidate] = []
            for relation in raw.relations:
                source = point_by_key.get(relation.source_key)
                target = point_by_key.get(relation.target_key)
                if not source or not target or relation.relation_type not in RELATION_TYPES:
                    continue
                relations.append(
                    KnowledgeRelationCandidate(
                        source_id=source.id,
                        target_id=target.id,
                        relation_type=relation.relation_type,
                        description=relation.description,
                    )
                )

            targets: list[AssessmentTargetCandidate] = []
            known_evidence_ids = {
                evidence.chunk_id
                for point in point_by_key.values()
                for evidence in point.evidence
            }
            for target in raw.assessment_targets:
                point_ids = [
                    point_by_key[key].id for key in target.knowledge_point_keys
                    if key in point_by_key
                ]
                evidence_ids = [
                    cid for cid in target.evidence_chunk_ids if cid in known_evidence_ids
                ]
                if not point_ids or not evidence_ids:
                    continue
                targets.append(
                    AssessmentTargetCandidate(
                        id=stable_id("target", textbook_id, chapter_id, target.title, *point_ids),
                        chapter_id=chapter_id,
                        knowledge_point_ids=list(dict.fromkeys(point_ids)),
                        title=target.title,
                        objective=target.objective,
                        discrimination=target.discrimination,
                        common_misconceptions=target.common_misconceptions,
                        evidence_chunk_ids=list(dict.fromkeys(evidence_ids)),
                    )
                )

            chapter_results.append(
                ChapterAnalysis(
                    id=chapter_id,
                    title=title,
                    order=order,
                    knowledge_points=list(point_by_key.values()),
                    relations=relations,
                    assessment_targets=targets,
                )
            )

        return TextbookAnalysis(
            textbook_id=textbook_id,
            title=Path(source_file).stem,
            source_file=Path(source_file).name,
            uploader=uploader,
            chapters=chapter_results,
        )
