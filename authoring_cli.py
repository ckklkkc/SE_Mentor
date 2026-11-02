"""Command-line entry point for offline SE Mentor content authoring."""

from __future__ import annotations

import argparse
import asyncio
import sys

from database.mongo_controller import DiagnosisQuiz, TextbookAnalysis, init_mongo
from services import quiz_generator_kg
from services.content_authoring import (
    format_quiz,
    format_textbook_analysis,
    ingest_project_document,
    ingest_textbook,
    prepare_markdown,
    review_quiz,
    review_assessment_target,
    review_textbook_analysis,
)


def add_image_description_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--pdf-backend", choices=["marker", "docling"],
                         help="PDF 解析器；Docling 需先安裝獨立環境")
    image_mode = command.add_mutually_exclusive_group()
    image_mode.add_argument(
        "--describe-images",
        dest="describe_images",
        action="store_true",
        default=None,
        help="用外部多模態模型產生圖片轉述",
    )
    image_mode.add_argument(
        "--skip-image-descriptions",
        dest="describe_images",
        action="store_false",
        help="只保存原圖，不呼叫多模態模型",
    )
    command.add_argument(
        "--image-provider",
        choices=["openai", "anthropic"],
        help="覆寫 config.py 的圖片模型服務商",
    )
    command.add_argument("--image-model", help="覆寫圖片描述模型名稱")
    command.add_argument(
        "--fail-on-image-error",
        action="store_true",
        default=None,
        help="任一圖片描述失敗就停止整次匯入；預設保留原圖並繼續",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SE Mentor 離線教材、知識圖譜與題庫製作工具"
    )
    parser.add_argument("--database", default="TABotAI_quiz", help="MongoDB database")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest-textbook", help="分析教材並建立待審 KG")
    ingest.add_argument("file")
    ingest.add_argument("--uploader", required=True)
    add_image_description_arguments(ingest)

    convert = commands.add_parser(
        "convert-pdf",
        help="只執行 PDF 轉換與圖片轉述，預覽 Markdown，不連線資料庫",
    )
    convert.add_argument("file")
    convert.add_argument("--output-dir", default="md_files/preview")
    add_image_description_arguments(convert)

    show_textbook = commands.add_parser("show-textbook", help="顯示教材分析")
    show_textbook.add_argument("analysis_id")
    show_textbook.add_argument("--section-order", type=int)
    show_textbook.add_argument(
        "--chapter-order", type=int,
        help="舊版相容參數；等同 --section-order",
    )

    review_textbook = commands.add_parser("review-textbook", help="審核教材分析")
    review_textbook.add_argument("analysis_id")
    review_textbook.add_argument("decision", choices=["approved", "rejected"])
    review_textbook.add_argument("--reviewer", required=True)
    review_textbook.add_argument("--note")

    list_targets = commands.add_parser(
        "list-assessment-targets", help="列出 CourseUnit 的評量目標"
    )
    list_targets.add_argument("analysis_id")
    list_targets.add_argument(
        "--status", choices=["pending", "approved", "rejected"]
    )

    review_target = commands.add_parser(
        "review-assessment-target", help="個別核准或退回評量目標"
    )
    review_target.add_argument("analysis_id")
    review_target.add_argument("target_id")
    review_target.add_argument("decision", choices=["approved", "rejected"])
    review_target.add_argument("--reviewer", required=True)
    review_target.add_argument("--note")

    generate = commands.add_parser("generate-quiz-drafts", help="產生待審題目")
    generate.add_argument("--analysis-id", required=True)
    generate.add_argument("--count", type=int, choices=range(1, 11), default=5)
    generate.add_argument("--section-id", "--chapter-id", dest="section_id")

    generate_bank = commands.add_parser(
        "generate-course-unit-bank",
        help="將每個核准評量目標補到 2～3 題平行題",
    )
    generate_bank.add_argument("--analysis-id", required=True)
    generate_bank.add_argument("--items-per-target", type=int, choices=[2, 3], default=2)

    bank_status = commands.add_parser(
        "course-unit-bank-status", help="顯示各評量目標的題庫覆蓋狀態"
    )
    bank_status.add_argument("--analysis-id", required=True)

    list_quizzes = commands.add_parser("list-quizzes", help="列出題目與 MongoDB ID")
    list_quizzes.add_argument(
        "--status",
        choices=["pending", "approved", "rejected"],
        default="pending",
    )
    list_quizzes.add_argument("--analysis-id")
    list_quizzes.add_argument("--limit", type=int, default=20)

    show_quiz = commands.add_parser("show-quiz", help="顯示完整題目與迷思映射")
    show_quiz.add_argument("question_id")

    review_question = commands.add_parser("review-quiz", help="審核單一題目")
    review_question.add_argument("question_id")
    review_question.add_argument("decision", choices=["approved", "rejected"])
    review_question.add_argument("--reviewer", required=True)
    review_question.add_argument("--note")

    project = commands.add_parser(
        "ingest-project-document",
        help="離線匯入 SRD/SDD/STD 並建立專案 KG",
    )
    project.add_argument("file")
    project.add_argument("doc_type", choices=["SRD", "SDD", "STD"])
    project.add_argument("group")
    project.add_argument("--uploader", required=True)
    project.add_argument(
        "--skip-vectors",
        action="store_true",
        help="只建立 KG，不建立向量索引",
    )
    add_image_description_arguments(project)
    return parser


async def run(args: argparse.Namespace) -> None:
    # 專案文件匯入只操作 Neo4j／向量索引，不應被 MongoDB 可用性綁住。
    if args.command not in {"ingest-project-document", "convert-pdf"}:
        await init_mongo(args.database)

    if args.command == "convert-pdf":
        markdown_path, _ = await asyncio.to_thread(
            prepare_markdown,
            args.file,
            args.output_dir,
            describe_images=args.describe_images,
            image_provider=args.image_provider,
            image_model=args.image_model,
            fail_on_image_error=args.fail_on_image_error,
            pdf_backend=args.pdf_backend,
        )
        print(f"Markdown：{markdown_path}")
        print(f"圖片目錄：{markdown_path.with_name(markdown_path.stem + '_assets')}")
        print(f"圖片紀錄：{markdown_path.with_name(markdown_path.stem + '_images.json')}")
        return

    if args.command == "ingest-textbook":
        analysis = await ingest_textbook(
            args.file,
            args.uploader,
            describe_images=args.describe_images,
            image_provider=args.image_provider,
            image_model=args.image_model,
            fail_on_image_error=args.fail_on_image_error,
            pdf_backend=args.pdf_backend,
        )
        included = sum(section.included_in_assessment for section in analysis.sections)
        excluded = len(analysis.sections) - included
        print(f"分析 ID：{analysis.id}")
        print(f"狀態：{analysis.review_status}")
        print(f"CourseUnit：{analysis.course_unit_title}")
        print(f"跨 Section 整併：{analysis.consolidation_status}")
        if analysis.consolidation_note:
            print(f"警告：{analysis.consolidation_note}")
        print(
            f"Section：{len(analysis.sections)}（納入 {included}；排除 {excluded}）；"
            f"去重後知識點：{len(analysis.knowledge_points)}；"
            f"去重後評量目標：{len(analysis.assessment_targets)}"
        )
        return

    if args.command == "show-textbook":
        analysis = await TextbookAnalysis.get(args.analysis_id)
        if not analysis:
            raise ValueError("找不到指定的教材分析")
        section_order = args.section_order or args.chapter_order or 1
        print(format_textbook_analysis(analysis, section_order))
        return

    if args.command == "review-textbook":
        analysis = await review_textbook_analysis(
            args.analysis_id,
            args.decision,
            args.reviewer,
            args.note,
        )
        print(f"教材分析 {analysis.id} 已更新為 {analysis.review_status}")
        return

    if args.command == "list-assessment-targets":
        analysis = await TextbookAnalysis.get(args.analysis_id)
        if not analysis:
            raise ValueError("找不到指定的教材分析")
        section_by_id = {section.id: section.title for section in analysis.sections}
        targets = [
            target for target in analysis.assessment_targets
            if not args.status or target.review_status == args.status
        ]
        if not targets:
            print("找不到符合條件的評量目標")
            return
        for target in targets:
            section_names = [
                section_by_id[section_id]
                for section_id in target.section_ids
                if section_id in section_by_id
            ]
            print(
                f"{target.id}\t{target.review_status}\t"
                f"{' / '.join(section_names)}\t{target.title}"
            )
        return

    if args.command == "review-assessment-target":
        analysis = await review_assessment_target(
            args.analysis_id,
            args.target_id,
            args.decision,
            args.reviewer,
            args.note,
        )
        target = next(
            item for item in analysis.assessment_targets if item.id == args.target_id
        )
        print(f"評量目標 {target.id} 已更新為 {target.review_status}")
        return

    if args.command == "generate-quiz-drafts":
        drafts = await quiz_generator_kg.generate_quiz_drafts(
            analysis_id=args.analysis_id,
            count=args.count,
            chapter_id=args.section_id,
        )
        for draft in drafts:
            print(f"{draft.id}\t{draft.question}")
        return

    if args.command == "generate-course-unit-bank":
        drafts = await quiz_generator_kg.generate_course_unit_bank(
            analysis_id=args.analysis_id,
            items_per_target=args.items_per_target,
        )
        if not drafts:
            print("題庫已達指定覆蓋數量，未新增題目")
            return
        print(f"新增 {len(drafts)} 題待審平行題")
        for draft in drafts:
            print(
                f"{draft.id}\t{draft.assessment_target_id}\t"
                f"variant={draft.variant_number}\t{draft.question}"
            )
        return

    if args.command == "course-unit-bank-status":
        analysis = await TextbookAnalysis.get(args.analysis_id)
        if not analysis:
            raise ValueError("找不到指定的教材分析")
        questions = await DiagnosisQuiz.find({
            "source_analysis_id": str(analysis.id),
        }).to_list()
        for target in analysis.assessment_targets:
            target_questions = [
                question for question in questions
                if question.assessment_target_id == target.id
            ]
            approved = sum(q.review_status == "approved" for q in target_questions)
            pending = sum(q.review_status == "pending" for q in target_questions)
            rejected = sum(q.review_status == "rejected" for q in target_questions)
            print(
                f"{target.id}\t{target.review_status}\t"
                f"approved={approved}\tpending={pending}\trejected={rejected}\t"
                f"{target.title}"
            )
        return

    if args.command == "list-quizzes":
        filters = {"review_status": args.status}
        if args.analysis_id:
            filters["source_analysis_id"] = args.analysis_id
        questions = await DiagnosisQuiz.find(filters).sort("-_id").limit(
            max(1, args.limit)
        ).to_list()
        if not questions:
            print("找不到符合條件的題目")
            return
        for question in questions:
            print(
                f"{question.id}\t{question.review_status}\t"
                f"{question.assessment_target or question.concept}\t"
                f"{question.question}"
            )
        return

    if args.command == "show-quiz":
        question = await DiagnosisQuiz.get(args.question_id)
        if not question:
            raise ValueError("找不到指定題目")
        print(format_quiz(question))
        return

    if args.command == "review-quiz":
        question = await review_quiz(
            args.question_id,
            args.decision,
            args.reviewer,
            args.note,
        )
        print(f"題目 {question.id} 已更新為 {question.review_status}")
        return

    if args.command == "ingest-project-document":
        markdown_path = await ingest_project_document(
            args.file,
            args.doc_type,
            args.group,
            args.uploader,
            upload_vectors=not args.skip_vectors,
            describe_images=args.describe_images,
            image_provider=args.image_provider,
            image_model=args.image_model,
            fail_on_image_error=args.fail_on_image_error,
            pdf_backend=args.pdf_backend,
        )
        print(f"專案文件已完成離線匯入：{markdown_path}")
        return

    raise ValueError(f"未知指令：{args.command}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        asyncio.run(run(args))
        return 0
    except Exception as exc:
        print(f"authoring 失敗：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
