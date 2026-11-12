"""Offline content authoring workflows.

Discord runtime code must not import this module.  PDF conversion, knowledge
graph construction, question generation, and teacher review are intentionally
kept outside the interaction path so long-running jobs cannot block the bot.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from datetime import datetime
from pathlib import Path

import common
import config
import prompts
from database.mongo_controller import DiagnosisQuiz, TextbookAnalysis
from database.neo4j_importer import Neo4jImporter, TripleList
from services.kg_constructor import KGConstructor
from services.document_image_quality import (
    validate_project_image_conversion, validate_textbook_image_conversion,
)
from services.hackmd_document import prepare_hackmd_markdown
from services.project_security import validate_project_document_source
from services.textbook_analyzer import (
    TextbookAnalyzer,
    textbook_analysis_validation_errors,
)
from utils import file_processor


SUPPORTED_SOURCE_SUFFIXES = {".pdf", ".md", ".markdown"}
PROJECT_DOCUMENT_TYPES = {"SRD", "SDD", "STD"}


def normalize_project_group(group: str) -> str:
    normalized = group.strip()
    if (
        not normalized
        or len(normalized) > 80
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
        or re.search(r"[\x00-\x1f]", normalized)
    ):
        raise ValueError("group 必須是 1～80 字且不得包含路徑或控制字元")
    return normalized


def prepare_markdown(
    source: str | Path,
    output_dir: str | Path,
    *,
    describe_images: bool | None = None,
    image_provider: str | None = None,
    image_model: str | None = None,
    fail_on_image_error: bool | None = None,
    pdf_backend: str | None = None,
    document_type: str | None = None,
) -> tuple[Path, str]:
    """Archive a source document and return a Markdown path plus its content."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到檔案：{source_path}")
    if source_path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
        raise ValueError("只支援 PDF、.md 或 .markdown")

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived_source = destination_dir / f"{timestamp}_{source_path.name}"
    shutil.copy2(source_path, archived_source)

    if source_path.suffix.lower() == ".pdf":
        markdown_path = archived_source.with_suffix(".md")
        assets_dir = markdown_path.with_name(f"{markdown_path.stem}_assets")
        manifest_path = markdown_path.with_name(
            f"{markdown_path.stem}_images.json"
        )
        markdown = file_processor.pdf2md(
            str(archived_source),
            image_output_dir=assets_dir,
            image_markdown_prefix=assets_dir.name,
            image_manifest_path=manifest_path,
            describe_images=describe_images,
            image_provider=image_provider,
            image_model=image_model,
            fail_on_image_error=fail_on_image_error,
            pdf_backend=pdf_backend,
            document_type=document_type,
        )
        markdown_path.write_text(markdown, encoding="utf-8")
    else:
        markdown_path = archived_source
        markdown = markdown_path.read_text(encoding="utf-8")
    return markdown_path, markdown


def _neo4j_importer() -> Neo4jImporter:
    return Neo4jImporter(
        uri=common.NEO4J_URI,
        username="neo4j",
        password=config.NEO4J_PASSWORD,
    )


def upload_textbook_analysis_to_neo4j(analysis: TextbookAnalysis) -> None:
    importer = _neo4j_importer()
    try:
        if not importer.connect():
            raise RuntimeError("無法連線 Neo4j")
        if not importer.upload_textbook_analysis(analysis, str(analysis.id)):
            raise RuntimeError("Neo4j 教材分析寫入失敗")
    finally:
        importer.close()


def update_textbook_review_in_neo4j(analysis_id: str, status: str) -> None:
    importer = _neo4j_importer()
    try:
        if not importer.connect():
            raise RuntimeError("無法連線 Neo4j")
        if not importer.update_textbook_analysis_status(analysis_id, status):
            raise RuntimeError("Neo4j 教材審核狀態更新失敗")
    finally:
        importer.close()


def update_assessment_target_review_in_neo4j(target_id: str, status: str) -> None:
    importer = _neo4j_importer()
    try:
        if not importer.connect():
            raise RuntimeError("無法連線 Neo4j")
        if not importer.update_assessment_target_status(target_id, status):
            raise RuntimeError("Neo4j 找不到指定評量目標")
    finally:
        importer.close()


async def ingest_textbook(
    source: str | Path,
    uploader: str,
    *,
    describe_images: bool | None = None,
    image_provider: str | None = None,
    image_model: str | None = None,
    fail_on_image_error: bool | None = None,
    pdf_backend: str | None = None,
) -> TextbookAnalysis:
    markdown_path, markdown = await asyncio.to_thread(
        prepare_markdown,
        source,
        Path("md_files") / "textbooks",
        describe_images=describe_images,
        image_provider=image_provider,
        image_model=image_model,
        fail_on_image_error=fail_on_image_error,
        pdf_backend=pdf_backend,
    )
    if Path(source).suffix.lower() == ".pdf":
        validate_textbook_image_conversion(markdown_path)
    analyzer = TextbookAnalyzer()
    analysis = await asyncio.to_thread(
        analyzer.analyze,
        markdown,
        markdown_path.name,
        uploader,
    )
    await analysis.insert()
    try:
        await asyncio.to_thread(upload_textbook_analysis_to_neo4j, analysis)
    except Exception:
        analysis.review_note = "Neo4j 寫入失敗；修復服務後請重新執行教材匯入。"
        await analysis.save()
        raise
    return analysis


def format_textbook_analysis(
    analysis: TextbookAnalysis,
    section_order: int,
) -> str:
    section = next(
        (item for item in analysis.sections if item.order == section_order),
        None,
    )
    if not section:
        raise ValueError(
            f"找不到第 {section_order} 區段；這個 CourseUnit 共有 {len(analysis.sections)} 個區段"
        )

    points_by_id = {point.id: point for point in analysis.knowledge_points}
    targets_by_id = {target.id: target for target in analysis.assessment_targets}
    points = [
        points_by_id[point_id]
        for point_id in section.knowledge_point_ids
        if point_id in points_by_id
    ]
    targets = [
        targets_by_id[target_id]
        for target_id in section.assessment_target_ids
        if target_id in targets_by_id
    ]

    lines = [
        f"CourseUnit：{analysis.course_unit_title or analysis.title}",
        f"CourseUnit ID：{analysis.course_unit_id or analysis.textbook_id}",
        f"分析 ID：{analysis.id}",
        f"審核狀態：{analysis.review_status}",
        f"跨區段整併：{analysis.consolidation_status}",
        f"第 {section.order} 區段：{section.title} ({section.id})",
        f"內容類型：{section.content_type}；納入評量：{'是' if section.included_in_assessment else '否'}",
        f"知識點：{len(points)}；評量目標：{len(targets)}",
        "",
    ]
    if analysis.consolidation_note:
        lines.append(f"整併警告：{analysis.consolidation_note}")
    if not section.included_in_assessment:
        lines.append(f"排除原因：{section.exclusion_reason or '非知識性區段'}")
        return "\n".join(lines)

    lines.append("【細粒度知識點】")
    for point in points:
        evidence = "；".join(item.quote for item in point.evidence if item.quote)
        lines.append(f"- {point.id}｜{point.name}\n  {point.description}")
        if len(point.section_ids) > 1:
            lines.append(f"  跨區段來源：{', '.join(point.section_ids)}")
        if evidence:
            lines.append(f"  證據：{evidence[:500]}")

    lines.append("\n【評量目標】")
    for target in targets:
        lines.extend([
            f"- {target.id}｜{target.title}｜{target.review_status}",
            f"  能力：{target.objective}",
            f"  鑑別：{target.discrimination}",
            f"  迷思：{'；'.join(target.common_misconceptions)}",
        ])
    return "\n".join(lines)


async def review_textbook_analysis(
    analysis_id: str,
    decision: str,
    reviewer: str,
    note: str | None = None,
) -> TextbookAnalysis:
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision 必須是 approved 或 rejected")
    analysis = await TextbookAnalysis.get(analysis_id)
    if not analysis:
        raise ValueError("找不到指定的教材分析")
    if decision == "approved":
        errors = textbook_analysis_validation_errors(analysis)
        if errors:
            raise ValueError("教材分析品質檢查未通過：" + "；".join(errors))
        # 整體核准代表教師接受所有尚未個別退回的目標；已 rejected 者不會復活。
        now = datetime.now()
        for target in analysis.assessment_targets:
            if target.review_status == "pending":
                target.review_status = "approved"
                target.reviewed_by = reviewer
                target.reviewed_at = now

    await asyncio.to_thread(
        update_textbook_review_in_neo4j,
        str(analysis.id),
        decision,
    )
    analysis.review_status = decision
    analysis.review_note = note
    analysis.reviewed_by = reviewer
    analysis.reviewed_at = datetime.now()
    await analysis.save()
    return analysis


async def review_assessment_target(
    analysis_id: str,
    target_id: str,
    decision: str,
    reviewer: str,
    note: str | None = None,
) -> TextbookAnalysis:
    """個別核准或退回 CourseUnit 中的評量目標。"""
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision 必須是 approved 或 rejected")
    analysis = await TextbookAnalysis.get(analysis_id)
    if not analysis:
        raise ValueError("找不到指定的教材分析")
    target = next(
        (item for item in analysis.assessment_targets if item.id == target_id),
        None,
    )
    if not target:
        raise ValueError("找不到指定評量目標")
    await asyncio.to_thread(
        update_assessment_target_review_in_neo4j,
        target_id,
        decision,
    )
    target.review_status = decision
    target.review_note = note
    target.reviewed_by = reviewer
    target.reviewed_at = datetime.now()
    await analysis.save()
    return analysis


def format_quiz(question: DiagnosisQuiz) -> str:
    diagnosis_by_index = {
        item.option_index: item for item in (question.option_diagnoses or [])
    }
    option_lines: list[str] = []
    for index, option in enumerate(question.options):
        diagnosis = diagnosis_by_index.get(index)
        marker = "（正確）" if index == question.answer else ""
        option_lines.append(f"{chr(65 + index)}. {option} {marker}")
        if diagnosis and diagnosis.misconception:
            option_lines.append(f"   迷思：{diagnosis.misconception}")

    return "\n".join([
        f"題目 ID：{question.id}",
        f"狀態：{question.review_status}",
        f"評量目標：{question.assessment_target or question.concept}",
        f"評量目標 ID：{question.assessment_target_id or 'legacy'}",
        f"教材分析 ID：{question.source_analysis_id or 'legacy'}",
        f"CourseUnit：{question.course_unit_title or 'legacy'}",
        f"平行題版本：{question.variant_number}",
        "",
        f"題目：{question.question}",
        *option_lines,
        "",
        f"解析：{question.analysis}",
        f"知識點 IDs：{', '.join(question.knowledge_point_ids)}",
        f"來源 chunks：{', '.join(question.source_chunk_ids)}",
    ])


async def review_quiz(
    question_id: str,
    decision: str,
    reviewer: str,
    note: str | None = None,
) -> DiagnosisQuiz:
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision 必須是 approved 或 rejected")
    question = await DiagnosisQuiz.get(question_id)
    if not question:
        raise ValueError("找不到指定題目")
    question.review_status = decision
    question.review_note = note
    question.reviewed_by = reviewer
    question.reviewed_at = datetime.now()
    await question.save()
    return question


def build_project_knowledge_graph(
    markdown_path: str | Path,
    doc_type: str,
    group: str,
    uploader: str,
) -> dict:
    doc_type = doc_type.upper()
    if doc_type not in PROJECT_DOCUMENT_TYPES:
        raise ValueError("doc_type 必須是 SRD、SDD 或 STD")

    markdown_path = Path(markdown_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    cleaned = file_processor.remove_specific_sections(
        file_processor.clean_markdown(markdown)
    )
    documents = [
        file_processor.replace_tables_in_text(document.page_content)
        for document in file_processor.md_splitter(cleaned)
    ]

    constructor = KGConstructor()
    importer = _neo4j_importer()
    source_file = markdown_path.name
    try:
        if not importer.connect():
            raise RuntimeError("無法連線 Neo4j")

        if doc_type == "SDD":
            triples = constructor.kg_construction_pipeline(
                documents,
                prompts.ENTITY_PROMPT_4_SDD,
                prompts.TRIPLE_PROMPT_4_SDD,
                group,
            )
            success = importer.upload_doc_triples(
                triples, source_file, doc_type, group, uploader
            )
            success = success and importer.link_references_to_requirements(
                "API", doc_type, group, "實作需求"
            )
        elif doc_type == "SRD":
            entities = constructor.entities_extraction_pipeline(
                documents,
                doc_type,
                prompts.ENTITY_PROMPT_4_SRD,
                group,
            )
            entities = constructor.match_fr_to_us_pipeline(entities, group)
            success = importer.upload_entities(
                entities, source_file, doc_type, group, uploader
            )
            if success:
                # SRD relationships are derived from the current import. Remove
                # previous inferred edges first, otherwise a corrected re-import
                # would leave the old (incorrect) traceability in the graph.
                success = importer.replace_project_relationships(
                    group, ["滿足", "操作"]
                )
            if success:
                success = importer.link_references_to_requirements(
                    "UserStory", doc_type, group, "滿足"
                )
            if success:
                actor_relations = TripleList(
                    triples=constructor.create_actor_relationships(entities)
                )
                success = importer.upload_doc_triples(
                    actor_relations, source_file, doc_type, group, uploader
                )
        else:
            entities = constructor.entities_extraction_pipeline(
                documents,
                doc_type,
                prompts.ENTITY_PROMPT_4_STD,
                group,
            )
            success = importer.upload_entities(
                entities, source_file, doc_type, group, uploader
            )
            success = success and importer.link_references_to_requirements(
                "TestCase", doc_type, group, "驗證"
            )

        if not success:
            raise RuntimeError(f"{doc_type} 知識圖譜寫入失敗")
        graph_stats = importer.project_graph_stats(
            group, doc_type, source_file
        )
        if graph_stats["node_count"] < 1:
            raise RuntimeError(
                f"{doc_type} 知識圖譜驗證失敗：沒有找到本次匯入的節點"
            )
        print(
            "[專案知識圖譜] 寫入驗證：",
            graph_stats,
        )
        return graph_stats
    finally:
        importer.close()


async def ingest_project_document(
    source: str | Path,
    doc_type: str,
    group: str,
    uploader: str,
    upload_vectors: bool = True,
    describe_images: bool | None = None,
    image_provider: str | None = None,
    image_model: str | None = None,
    fail_on_image_error: bool | None = None,
    pdf_backend: str | None = None,
) -> Path:
    normalized_doc_type = doc_type.upper()
    if normalized_doc_type not in PROJECT_DOCUMENT_TYPES:
        raise ValueError("doc_type 必須是 SRD、SDD 或 STD")
    source_is_pdf = Path(source).suffix.lower() == ".pdf"
    if source_is_pdf:
        if describe_images is None:
            describe_images = bool(
                getattr(config, "PROJECT_DOCUMENT_DESCRIBE_IMAGES", True)
            )
        if pdf_backend is None:
            pdf_backend = str(
                getattr(config, "PROJECT_DOCUMENT_PDF_BACKEND", "docling")
            )

    output_dir = Path("md_files") / "groups" / group
    markdown_path, _ = await asyncio.to_thread(
        prepare_markdown,
        source,
        output_dir,
        describe_images=describe_images,
        image_provider=image_provider,
        image_model=image_model,
        fail_on_image_error=fail_on_image_error,
        pdf_backend=pdf_backend,
        document_type=normalized_doc_type,
    )
    if source_is_pdf and bool(
        getattr(config, "PROJECT_DOCUMENT_REQUIRE_VERIFIED_IMAGES", True)
    ):
        await asyncio.to_thread(
            validate_project_image_conversion,
            markdown_path,
        )
    await asyncio.to_thread(
        build_project_knowledge_graph,
        markdown_path,
        normalized_doc_type,
        group,
        uploader,
    )
    if upload_vectors:
        # Lazy import: teaching-time bot startup must not load authoring code.
        from services import haystack_service

        await haystack_service.upload_doc_2_vectordb(
            file_path=str(markdown_path),
            doc_type=normalized_doc_type,
            group_name=group,
            uploader=uploader,
        )
    return markdown_path


async def ingest_hackmd_project_document(
    note_url: str,
    doc_type: str,
    group: str,
    uploader: str,
    *,
    upload_vectors: bool = True,
    image_provider: str | None = None,
    image_model: str | None = None,
) -> tuple[Path, dict]:
    """Ingest one public HackMD note with contextual per-image descriptions."""
    normalized_doc_type = doc_type.upper()
    if normalized_doc_type not in PROJECT_DOCUMENT_TYPES:
        raise ValueError("doc_type 必須是 SRD、SDD 或 STD")
    normalized_group = normalize_project_group(group)

    # Authorize the exact teacher-approved note before any download, image
    # captioning, tagging or graph-construction model can be invoked.
    validate_project_document_source(
        note_url,
        normalized_group,
        normalized_doc_type,
        getattr(config, "PROJECT_DOCUMENT_ALLOWED_SOURCES", {}),
        require_allowlist=bool(
            getattr(config, "PROJECT_DOCUMENT_REQUIRE_SOURCE_ALLOWLIST", True)
        ),
    )

    output_dir = Path("md_files") / "groups" / normalized_group
    markdown_path, _, manifest = await asyncio.to_thread(
        prepare_hackmd_markdown,
        note_url,
        output_dir,
        document_type=normalized_doc_type,
        image_provider=image_provider,
        image_model=image_model,
    )
    await asyncio.to_thread(validate_project_image_conversion, markdown_path)
    graph_stats = await asyncio.to_thread(
        build_project_knowledge_graph,
        markdown_path,
        normalized_doc_type,
        normalized_group,
        uploader,
    )
    manifest["knowledge_graph"] = graph_stats
    if upload_vectors:
        from services import haystack_service

        vector_chunk_count = await haystack_service.upload_doc_2_vectordb(
            file_path=str(markdown_path),
            doc_type=normalized_doc_type,
            group_name=normalized_group,
            uploader=uploader,
            source_url=note_url,
            source_kind="hackmd",
            source_sha256=manifest.get("source_sha256"),
        )
        manifest["vector_chunk_count"] = vector_chunk_count
    return markdown_path, manifest
