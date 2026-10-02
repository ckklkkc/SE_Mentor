import os
import ast
import asyncio
import json
import random
import threading
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from database.neo4j_importer import Neo4jImporter
from database.mongo_controller import (
    DiagnosisQuiz,
    OptionDiagnosis,
    init_mongo,
    LearningProfile,
    ChatLogs,
)
from config import NEO4J_PASSWORD, OPENAI_API_KEY
from common import NEO4J_URI

from haystack.utils import Secret
from haystack import Pipeline, Document
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from neo4j import GraphDatabase

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

CHAPTERS = [
    "[01]軟體危機與軟體流程",
    "[02]基礎需求工程",
    "[03]使用者故事分析",
    "[04]敏捷開發方法",
    "[05]基礎專案管理與看板",
    "[06]版本控制",
    "[07]軟體設計-系統設計",
    "[08]軟體設計-模組設計",
    "[09]軟體測試",
    "[10]進階軟體測試",
    "[11]DevOps自動化建置管理",
]


class Question(BaseModel):
    question: str
    options: list[str]
    answer: int = Field(ge=0, le=3)
    analysis: str
    concept: str
    knowledge_point_ids: list[str] = Field(default_factory=list)
    assessment_target_id: str | None = None
    assessment_target: str | None = None
    option_diagnoses: list[OptionDiagnosis] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str]) -> list[str]:
        if len(value) != 4:
            raise ValueError("每題必須恰好有 4 個選項")
        return value

    @field_validator("option_diagnoses")
    @classmethod
    def validate_option_diagnoses(cls, value: list[OptionDiagnosis]) -> list[OptionDiagnosis]:
        if value and sorted(item.option_index for item in value) != [0, 1, 2, 3]:
            raise ValueError("option_diagnoses 必須完整對應 0、1、2、3 四個選項")
        return value


class QuestionList(BaseModel):
    questions: list[Question]


_quiz_pipeline: Optional[Pipeline] = None
_pipeline_lock = threading.Lock()


def get_core_nodes(chapter: str, limit: int = 5) -> list[str]:
    """取得某章節在教材 KG 中連結度最高的核心 Concept。"""
    cypher = f"""
        MATCH (n:Concept)
        WHERE '{chapter}.pdf' IN n.source_files
        WITH n, count {{ (n)--() }} AS degree
        ORDER BY degree DESC
        LIMIT {int(limit)}
        RETURN n.name AS name
    """

    importer = Neo4jImporter(
        uri=NEO4J_URI,
        username="neo4j",
        password=NEO4J_PASSWORD,
    )
    try:
        if importer.connect():
            return importer.run_cypher(cypher) or []
    except Exception as e:
        print(f"[quiz_generator] get_core_nodes error: {e}")
    finally:
        importer.close()
    return []


def _kg_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=("neo4j", NEO4J_PASSWORD),
    )


def get_kg_documents(
    concept_names: list[str],
    source_files: Optional[list[str]] = None,
    relations_per_entity: int = 5,
) -> list[Document]:
    """
    直接從既有教材 Knowledge Graph 組成出題 Context。

    不依賴 Haystack document-embeddings；使用 KG 中已存在的
    name / description / source_files / relationships。
    """
    names = [name.strip() for name in concept_names if name and name.strip()]
    names = list(dict.fromkeys(names))
    if not names:
        return []

    cypher = """
    UNWIND $names AS target_name
    MATCH (entity)
    WHERE entity.name = target_name
      AND any(label IN labels(entity)
              WHERE label IN ['Concept', 'Technology', 'Methodology'])
      AND (
        $source_files IS NULL
        OR any(sf IN coalesce(entity.source_files, []) WHERE sf IN $source_files)
      )
    OPTIONAL MATCH (entity)-[rel]-(neighbor)
    WHERE rel IS NULL
       OR $source_files IS NULL
       OR any(sf IN coalesce(rel.source_files, []) WHERE sf IN $source_files)
       OR any(sf IN coalesce(neighbor.source_files, []) WHERE sf IN $source_files)
    WITH entity,
         collect({
             relation: type(rel),
             relation_description: rel.description,
             neighbor_name: neighbor.name,
             neighbor_description: neighbor.description,
             neighbor_label: head(labels(neighbor))
         })[0..$relations_per_entity] AS relations
    RETURN head(labels(entity)) AS label,
           entity.name AS name,
           entity.description AS description,
           entity.source_files AS source_files,
           relations
    """

    docs: list[Document] = []
    with _kg_driver() as driver:
        with driver.session(database="neo4j") as session:
            records = session.run(
                cypher,
                names=names,
                source_files=source_files,
                relations_per_entity=max(1, int(relations_per_entity)),
            )
            for record in records:
                lines = [f"【{record['label'] or '概念'}】{record['name']}"]
                if record['description']:
                    lines.append(f"說明：{record['description']}")

                relations = record['relations'] or []
                relation_lines = []
                for rel in relations:
                    if not rel or not rel.get('neighbor_name'):
                        continue
                    relation_text = f"- {record['name']} --{rel.get('relation') or '相關'}--> {rel['neighbor_name']}"
                    if rel.get('relation_description'):
                        relation_text += f"：{rel['relation_description']}"
                    if rel.get('neighbor_description'):
                        relation_text += f"；{rel['neighbor_name']}：{rel['neighbor_description']}"
                    relation_lines.append(relation_text)

                if relation_lines:
                    lines.append("相關知識：")
                    lines.extend(relation_lines)

                docs.append(
                    Document(
                        content="\n".join(lines),
                        meta={
                            "content_type": "textbook_kg",
                            "entity_name": record['name'],
                            "source_files": record['source_files'] or [],
                        },
                    )
                )

    print(f"[quiz_generator] KG context: {len(docs)} documents for {len(names)} concepts")
    return docs


def find_relevant_kg_concepts(keywords: list[str], limit: int = 10) -> list[str]:
    """從學生問句/弱點中找出實際存在於教材 KG 的概念名稱。"""
    keywords = [k.strip() for k in keywords if k and k.strip()]
    if not keywords:
        return []

    cypher = """
    MATCH (entity)
    WHERE any(label IN labels(entity)
              WHERE label IN ['Concept', 'Technology', 'Methodology'])
      AND entity.name IS NOT NULL
      AND any(keyword IN $keywords WHERE
            (size(entity.name) >= 2 AND keyword CONTAINS entity.name)
            OR (size(keyword) >= 2 AND entity.name CONTAINS keyword))
    WITH DISTINCT entity, count { (entity)--() } AS degree
    ORDER BY degree DESC
    LIMIT $limit
    RETURN entity.name AS name
    """

    with _kg_driver() as driver:
        with driver.session(database="neo4j") as session:
            result = session.run(cypher, keywords=keywords, limit=max(1, int(limit)))
            names = [record['name'] for record in result if record['name']]

    print(f"[quiz_generator] personalized KG concepts: {names}")
    return names


def get_approved_assessment_documents(
    count: int = 5,
    chapter_id: str | None = None,
) -> list[Document]:
    """取得教師核准的評量目標，以及其知識點、關係與教材證據。"""
    cypher = """
    MATCH (assessment:AssessmentTarget {review_status: 'approved'})
    MATCH (chapter:Chapter)-[:HAS_ASSESSMENT_TARGET]->(assessment)
    WHERE $chapter_id IS NULL OR chapter.id = $chapter_id
    MATCH (assessment)-[:ASSESSES]->(point:KnowledgePoint {review_status: 'approved'})
    OPTIONAL MATCH (point)-[knowledge_rel:KNOWLEDGE_RELATION]-(related:KnowledgePoint)
    WHERE knowledge_rel.review_status = 'approved'
    OPTIONAL MATCH (point)-[evidence_rel:EVIDENCED_BY]->(chunk:SourceChunk)
    WITH assessment, chapter,
         collect(DISTINCT {
             id: point.id, name: point.name, description: point.description
         }) AS points,
         collect(DISTINCT {
             source: startNode(knowledge_rel).name,
             relation_type: knowledge_rel.relation_type,
             target: endNode(knowledge_rel).name,
             description: knowledge_rel.description
         }) AS relations,
         collect(DISTINCT {
             chunk_id: chunk.id, quote: evidence_rel.quote,
             heading_path: chunk.heading_path
         }) AS evidence
    RETURN assessment.id AS target_id,
           assessment.title AS title,
           assessment.objective AS objective,
           assessment.discrimination AS discrimination,
           assessment.common_misconceptions AS misconceptions,
           chapter.id AS chapter_id,
           chapter.title AS chapter_title,
           points, relations, evidence
    ORDER BY rand()
    LIMIT $limit
    """
    docs: list[Document] = []
    with _kg_driver() as driver:
        with driver.session(database="neo4j") as session:
            records = session.run(
                cypher,
                chapter_id=chapter_id,
                limit=max(1, int(count)),
            )
            for record in records:
                points = [point for point in (record["points"] or []) if point.get("id")]
                relations = [
                    relation for relation in (record["relations"] or [])
                    if relation.get("target") and relation.get("relation_type")
                ]
                evidence = [
                    item for item in (record["evidence"] or []) if item.get("chunk_id")
                ]
                lines = [
                    f"【評量目標 ID】{record['target_id']}",
                    f"【章節】{record['chapter_title']}",
                    f"【目標】{record['title']}",
                    f"【可觀察能力】{record['objective']}",
                    f"【鑑別重點】{record['discrimination']}",
                    f"【常見迷思】{'；'.join(record['misconceptions'] or [])}",
                    "【知識點】",
                ]
                lines.extend(
                    f"- {point['id']}｜{point['name']}：{point.get('description') or ''}"
                    for point in points
                )
                if relations:
                    lines.append("【知識關係】")
                    lines.extend(
                        f"- {rel['source']} --{rel['relation_type']}--> {rel['target']}："
                        f"{rel.get('description') or ''}"
                        for rel in relations
                    )
                if evidence:
                    lines.append("【教材證據】")
                    lines.extend(
                        f"- {item['chunk_id']}｜{item.get('quote') or ''}"
                        for item in evidence
                    )
                docs.append(
                    Document(
                        content="\n".join(lines),
                        meta={
                            "assessment_target_id": record["target_id"],
                            "chapter_id": record["chapter_id"],
                            "knowledge_point_ids": [point["id"] for point in points],
                            "source_chunk_ids": [item["chunk_id"] for item in evidence],
                        },
                    )
                )
    return docs


def _get_quiz_pipeline() -> Pipeline:
    """Lazy init，避免每次 /quiz 都重新載入 embedding model。"""
    global _quiz_pipeline
    if _quiz_pipeline is not None:
        return _quiz_pipeline

    template = [
        ChatMessage.from_user(
            """
你是一位專業的「軟體工程」課程教授。請嚴格根據 Context 中的教材內容，生成 {{count}} 題單選題。

【出題焦點】
{{focus}}

【學生過往 QA / 學習線索】
{{student_context}}

【規則】
1. QA / 學習線索只用來判斷學生可能正在學什麼、哪裡需要複習；正確知識必須以 Context 為準。
2. 題目要測理解、比較、應用或情境判斷，不要只是背名詞定義。
3. 每題必須恰好 4 個選項。
4. answer 必須使用 0、1、2、3 表示正確選項索引，分別對應 A、B、C、D。
5. 每題都要提供 analysis，說明為何正確答案正確，並簡要指出其他選項的問題。
6. 每題只能對應一個 Context 中明列的評量目標。assessment_target_id 必須原樣填入；
   assessment_target 填目標名稱；knowledge_point_ids 與 source_chunk_ids 也只能填 Context 中的 ID。
7. 四個選項應用來辨識學生對知識點關係／差異的理解。option_diagnoses 必須依序對應四個選項；
   正確選項的 misconception 填 null，錯誤選項則填具體錯誤推理，不可只寫「觀念不清」。
8. concept 保留作為相容欄位，填入精確的知識點名稱，不可只填章節大標題。
9. 不可把學生過去問過的問題原句直接改成選擇題；要根據教材重新設計具有鑑別度的新題目。
10. 不可使用 Context 無法支持的知識，且使用台灣繁體中文。

Context:
{% for document in documents %}
---
{{ document.content }}
{% endfor %}
            """
        )
    ]

    prompt_builder = ChatPromptBuilder(
        template=template,
        required_variables=["documents", "count", "focus", "student_context"],
    )
    gpt_chat = OpenAIChatGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        generation_kwargs={
            "response_format": QuestionList,
            "temperature": 0.7,
        },
    )

    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", gpt_chat)
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    _quiz_pipeline = pipeline
    return _quiz_pipeline


def _parse_questions(result: dict, expected_count: int) -> list[Question]:
    text = result["llm"]["replies"][0]._content[0].text

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # 相容原專案的 Python dict-like structured output
        payload = ast.literal_eval(text)

    parsed = QuestionList.model_validate(payload)
    if len(parsed.questions) < expected_count:
        raise ValueError(
            f"LLM 只生成 {len(parsed.questions)} 題，少於要求的 {expected_count} 題"
        )
    return parsed.questions[:expected_count]


def _run_generation(
    *,
    documents: list[Document],
    focus: str,
    student_context: str,
    count: int,
) -> list[Question]:
    if not documents:
        raise ValueError("教材知識圖譜沒有取得任何可用 Context，無法即時出題")

    data = {
        "prompt_builder": {
            "documents": documents,
            "count": count,
            "focus": focus,
            "student_context": student_context or "無",
        },
    }

    # 同時保護 lazy init 與 Pipeline.run，避免多個 Discord command 同時初始化/執行。
    with _pipeline_lock:
        pipeline = _get_quiz_pipeline()
        result = pipeline.run(data=data, include_outputs_from=["llm"])

    return _parse_questions(result, count)


def _generate_chapter_quiz_sync(chapter: str, count: int = 5) -> list[Question]:
    core_nodes = get_core_nodes(chapter, limit=max(5, count))
    if not core_nodes:
        raise ValueError(f"教材 KG 找不到章節核心概念：{chapter}")

    documents = get_kg_documents(
        core_nodes,
        source_files=[f"{chapter}.pdf"],
        relations_per_entity=5,
    )
    focus = "、".join(core_nodes)
    return _run_generation(
        documents=documents,
        focus=f"章節：{chapter}\n核心概念：{focus}",
        student_context="無，這是一般即時測驗。",
        count=count,
    )


async def generate_quiz_kg(chapter: str, count: int = 5) -> list[dict]:
    """保留原 API：依單一章節生成題目，回傳 dict list。"""
    questions = await asyncio.to_thread(_generate_chapter_quiz_sync, chapter, count)
    return [q.model_dump() for q in questions]


def _generate_realtime_quizzes_sync(count: int = 5) -> list[Question]:
    """每次 /quiz 都直接從教材 KG 取得 Context，再由 LLM 生成新題。"""
    selected_chapters = random.sample(CHAPTERS, k=min(count, len(CHAPTERS)))

    concepts: list[str] = []
    for chapter in selected_chapters:
        nodes = get_core_nodes(chapter, limit=2)
        concepts.extend(nodes)

    concepts = list(dict.fromkeys(concepts))
    if not concepts:
        raise ValueError("教材 KG 找不到任何核心概念")

    source_files = [f"{chapter}.pdf" for chapter in selected_chapters]
    documents = get_kg_documents(
        concepts,
        source_files=source_files,
        relations_per_entity=5,
    )

    focus = "、".join(concepts)
    chapter_text = "、".join(selected_chapters)
    return _run_generation(
        documents=documents,
        focus=f"從下列章節與核心概念出題：\n章節：{chapter_text}\n核心概念：{focus}",
        student_context="無，這是每次重新生成的一般診斷題。",
        count=count,
    )


async def generate_realtime_quizzes(count: int = 5) -> list[Question]:
    return await asyncio.to_thread(_generate_realtime_quizzes_sync, count)


def _generate_quiz_drafts_sync(count: int = 5, chapter_id: str | None = None) -> list[Question]:
    documents = get_approved_assessment_documents(count=count, chapter_id=chapter_id)
    if len(documents) < count:
        raise ValueError(
            f"只有 {len(documents)} 個已核准評量目標，少於要求的 {count} 個；"
            "請先審核教材分析"
        )
    target_ids = {doc.meta["assessment_target_id"] for doc in documents}
    questions = _run_generation(
        documents=documents,
        focus="每個評量目標各生成一題，用選項區分具體迷思與正確理解。",
        student_context="無；這是教師審核前的題目草稿。",
        count=count,
    )
    for question in questions:
        if question.assessment_target_id not in target_ids:
            raise ValueError("LLM 回傳了 Context 以外的 assessment_target_id")
        source = next(
            doc for doc in documents
            if doc.meta["assessment_target_id"] == question.assessment_target_id
        )
        valid_points = set(source.meta["knowledge_point_ids"])
        valid_chunks = set(source.meta["source_chunk_ids"])
        if not question.knowledge_point_ids or not set(question.knowledge_point_ids) <= valid_points:
            raise ValueError("LLM 回傳了評量目標以外的 knowledge_point_ids")
        if not question.source_chunk_ids or not set(question.source_chunk_ids) <= valid_chunks:
            raise ValueError("LLM 回傳了評量目標以外的 source_chunk_ids")
        if len(question.option_diagnoses) != 4:
            raise ValueError("題目缺少四個選項的診斷映射")
        for diagnosis in question.option_diagnoses:
            if diagnosis.option_index == question.answer and diagnosis.misconception:
                raise ValueError("正確選項不可標記 misconception")
            if diagnosis.option_index != question.answer and not diagnosis.misconception:
                raise ValueError("每個錯誤選項都必須標記具體 misconception")
    return questions


async def generate_quiz_drafts(
    count: int = 5,
    chapter_id: str | None = None,
) -> list[DiagnosisQuiz]:
    """依核准評量目標產生待審題目，不會直接提供給學生。"""
    questions = await asyncio.to_thread(_generate_quiz_drafts_sync, count, chapter_id)
    drafts = [
        DiagnosisQuiz(
            **question.model_dump(),
            chapter=chapter_id or "cross-chapter",
            review_status="pending",
        )
        for question in questions
    ]
    if drafts:
        await DiagnosisQuiz.insert_many(drafts)
    return drafts


async def _get_student_quiz_context(user_id: int, recent_limit: int = 8) -> tuple[str, str, str, int]:
    """取得最近課程 QA、目前學習弱點，以及使用者已主動標記克服的弱點。"""
    chat_logs = await ChatLogs.find_one(
        ChatLogs.student.discord_id == user_id,
        fetch_links=True,
    )
    profile = await LearningProfile.find_one(
        LearningProfile.student.discord_id == user_id,
        fetch_links=True,
    )

    recent_logs = []
    if chat_logs and chat_logs.course_logs:
        recent_logs = chat_logs.course_logs[-recent_limit:]

    qa_lines = []
    for idx, log in enumerate(recent_logs, start=1):
        # A 只提供少量文字作為「學習線索」，不當成出題知識來源。
        answer_preview = (log.chatbot_response or "").strip().replace("\n", " ")[:350]
        qa_lines.append(
            f"{idx}. Q: {log.user_content.strip()}\n   A(僅供辨識主題): {answer_preview}"
        )

    pain_points = profile.pain_points if profile and profile.pain_points else []
    resolved_pain_points = (
        profile.resolved_pain_points
        if profile and profile.resolved_pain_points
        else []
    )
    pain_text = "、".join(pain_points)
    resolved_text = "、".join(resolved_pain_points)
    qa_text = "\n".join(qa_lines)
    return qa_text, pain_text, resolved_text, len(recent_logs)


def _generate_personalized_quizzes_sync(
    qa_text: str,
    pain_text: str,
    resolved_text: str = "",
    count: int = 5,
) -> list[Question]:
    if not qa_text and not pain_text:
        raise ValueError("目前沒有足夠的過往課程 QA 或學習弱點可供個人化出題")

    question_lines = []
    for line in qa_text.splitlines():
        stripped = line.strip()
        if "Q:" in stripped:
            question_lines.append(stripped.split("Q:", 1)[1].strip())

    keywords = question_lines[-8:]
    if pain_text:
        keywords.extend([p.strip() for p in pain_text.split("、") if p.strip()])

    concepts = find_relevant_kg_concepts(keywords, limit=max(8, count * 2))

    # 使用者在 /learning_profile 主動標記為已克服的弱點，
    # 不因舊的 Course QA 紀錄再次被當成個人化弱點焦點。
    resolved = {item.strip() for item in resolved_text.split("、") if item.strip()}
    if resolved:
        concepts = [concept for concept in concepts if concept not in resolved]

    if not concepts:
        if resolved:
            raise ValueError("可用的個人化概念皆已由使用者標記為已克服")
        raise ValueError("過往 QA / 學習弱點目前無法對應到教材 KG 概念")

    documents = get_kg_documents(
        concepts,
        source_files=None,
        relations_per_entity=5,
    )

    student_context = (
        f"過往課程 QA：\n{qa_text or '無'}\n\n"
        f"目前學習弱點：{pain_text or '無'}\n"
        f"使用者已主動標記克服（請勿作為弱點出題焦點）：{resolved_text or '無'}"
    )
    focus = (
        f"優先評量學生最近主動詢問或答錯的教材概念：{'、'.join(concepts)}。"
        "題目必須是新題，正確知識以 Knowledge Graph Context 為準。"
    )

    return _run_generation(
        documents=documents,
        focus=focus,
        student_context=student_context,
        count=count,
    )


async def generate_personalized_quizzes(
    user_id: int,
    count: int = 5,
) -> tuple[list[Question], int]:
    qa_text, pain_text, resolved_text, qa_count = await _get_student_quiz_context(user_id)
    questions = await asyncio.to_thread(
        _generate_personalized_quizzes_sync,
        qa_text,
        pain_text,
        resolved_text,
        count,
    )
    return questions, qa_count


async def upload_quiz_to_mongo():
    """初始化題庫：保留原本一次生成後寫入 MongoDB 的用途。"""
    await init_mongo("TABotAI_quiz")
    diagnosis_quizzes = []

    for chapter in CHAPTERS:
        print(f"正在生成 {chapter} 的題目...")
        quizzes = await generate_quiz_kg(chapter, count=5)
        for quiz in quizzes:
            diagnosis_quizzes.append(
                DiagnosisQuiz(
                    question=quiz["question"],
                    options=quiz["options"],
                    answer=quiz["answer"],
                    analysis=quiz["analysis"],
                    concept=quiz["concept"],
                    chapter=chapter,
                )
            )

    if diagnosis_quizzes:
        await DiagnosisQuiz.insert_many(diagnosis_quizzes)


async def get_bank_quizzes(count: int = 5) -> list[DiagnosisQuiz]:
    """只從教師已核准的 MongoDB 題庫抽題。"""
    selected_chapters = random.sample(CHAPTERS, k=min(count, len(CHAPTERS)))
    questions: list[DiagnosisQuiz] = []

    for chapter in selected_chapters:
        quiz_list = await DiagnosisQuiz.find({
            "chapter": chapter,
            "review_status": "approved",
        }).to_list()
        if quiz_list:
            questions.append(random.choice(quiz_list))

    # 若某章沒有題目，從所有現有題庫補滿。
    if len(questions) < count:
        all_quizzes = await DiagnosisQuiz.find({"review_status": "approved"}).to_list()
        used_ids = {str(q.id) for q in questions}
        candidates = [q for q in all_quizzes if str(q.id) not in used_ids]
        need = min(count - len(questions), len(candidates))
        if need > 0:
            questions.extend(random.sample(candidates, need))

    if len(questions) < count:
        raise ValueError(
            f"教師已核准題庫只有 {len(questions)} 題，無法組成 {count} 題測驗"
        )

    return questions[:count]


async def get_quizzes(mode: str, user_id: int, count: int = 5) -> tuple[list, str]:
    """
    mode:
      - bank: MongoDB 固定題庫
      - realtime: 即時 RAG + LLM
      - personalized: 根據學生 course QA + pain points
      - mixed: 題庫 2 題 + 個人化/即時 3 題
    """
    mode = (mode or "reviewed").lower()

    if mode == "reviewed":
        return await get_bank_quizzes(count), "審核題庫：所有題目皆已經教師核准"

    if mode == "bank":
        return await get_bank_quizzes(count), "題庫模式：從 MongoDB 既有題庫抽題"

    if mode == "realtime":
        try:
            return await generate_realtime_quizzes(count), "即時模式：本次由教材 RAG + LLM 重新生成"
        except Exception as e:
            print(f"[quiz_generator] realtime failed, fallback to bank: {e}")
            return await get_bank_quizzes(count), "即時生成失敗，已自動改用 MongoDB 題庫"

    if mode == "personalized":
        try:
            questions, qa_count = await generate_personalized_quizzes(user_id, count)
            return questions, f"個人化模式：依最近 {qa_count} 筆課程 QA／學習弱點生成"
        except Exception as e:
            print(f"[quiz_generator] personalized failed, fallback to realtime: {e}")
            try:
                questions = await generate_realtime_quizzes(count)
                return questions, "目前缺少可用的個人化資料，已改用即時生成模式"
            except Exception as rt_e:
                print(f"[quiz_generator] realtime fallback failed: {rt_e}")
                return await get_bank_quizzes(count), "個人化與即時生成失敗，已改用 MongoDB 題庫"

    if mode == "mixed":
        bank_count = min(2, count)
        generated_count = count - bank_count
        bank_questions = await get_bank_quizzes(bank_count)

        generated_questions: list = []
        source_text = ""
        if generated_count > 0:
            try:
                generated_questions, qa_count = await generate_personalized_quizzes(
                    user_id, generated_count
                )
                source_text = f"其中 {generated_count} 題依最近 {qa_count} 筆課程 QA／學習弱點生成"
            except Exception as e:
                print(f"[quiz_generator] mixed personalized failed, use realtime: {e}")
                try:
                    generated_questions = await generate_realtime_quizzes(generated_count)
                    source_text = f"其中 {generated_count} 題為本次即時生成"
                except Exception as rt_e:
                    print(f"[quiz_generator] mixed realtime failed, use all bank: {rt_e}")
                    return await get_bank_quizzes(count), "個人化與即時生成失敗，已全部改用 MongoDB 題庫"

        questions = bank_questions + generated_questions
        random.shuffle(questions)
        return questions, f"混合模式：{bank_count} 題 MongoDB 題庫；{source_text}"

    raise ValueError(f"未知的 quiz mode: {mode}")


# 相容舊名稱與舊呼叫方式
async def get_quizes() -> list:
    return await get_bank_quizzes(5)


if __name__ == "__main__":
    asyncio.run(upload_quiz_to_mongo())
