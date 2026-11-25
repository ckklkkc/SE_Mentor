import pymongo
from pymongo import AsyncMongoClient
from beanie import Document, init_beanie
from beanie.odm.fields import Link
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal
from datetime import datetime, timedelta

ReviewStatus = Literal["pending", "approved", "rejected"]


class SourceEvidence(BaseModel):
  """教材中的可追溯證據；題目與知識點都應能回到這個範圍。"""
  chunk_id: str
  heading_path: list[str] = Field(default_factory=list)
  quote: str


class KnowledgePointCandidate(BaseModel):
  id: str
  name: str
  description: str
  section_ids: list[str] = Field(default_factory=list)
  # 舊資料相容欄位；新分析以 section_ids 為準。
  chapter_id: str | None = None
  evidence: list[SourceEvidence] = Field(default_factory=list)


class KnowledgeRelationCandidate(BaseModel):
  source_id: str
  target_id: str
  relation_type: Literal[
    "prerequisite", "part_of", "contrasts_with", "similar_to",
    "causes", "used_for", "example_of"
  ]
  description: str


class AssessmentTargetCandidate(BaseModel):
  id: str
  section_ids: list[str] = Field(default_factory=list)
  # 舊資料相容欄位；新分析以 section_ids 為準。
  chapter_id: str | None = None
  knowledge_point_ids: list[str]
  title: str
  objective: str
  discrimination: str
  common_misconceptions: list[str] = Field(default_factory=list)
  evidence_chunk_ids: list[str] = Field(default_factory=list)
  review_status: ReviewStatus = "pending"
  review_note: str | None = None
  reviewed_by: str | None = None
  reviewed_at: datetime | None = None


class SectionAnalysis(BaseModel):
  id: str
  title: str
  order: int
  content_type: Literal[
    "knowledge", "front_matter", "exercise", "reference", "closing", "other"
  ] = "knowledge"
  included_in_assessment: bool = True
  exclusion_reason: str | None = None
  knowledge_point_ids: list[str] = Field(default_factory=list)
  assessment_target_ids: list[str] = Field(default_factory=list)


# 保留舊 import 名稱，避免外部程式立即失效。
ChapterAnalysis = SectionAnalysis


class TextbookAnalysis(Document):
  """CourseUnit 教材分析草稿。教師核准後才可成為出題依據。"""
  textbook_id: str
  course_unit_id: str = ""
  course_unit_title: str = ""
  title: str
  source_file: str
  uploader: str
  sections: list[SectionAnalysis] = Field(default_factory=list)
  knowledge_points: list[KnowledgePointCandidate] = Field(default_factory=list)
  relations: list[KnowledgeRelationCandidate] = Field(default_factory=list)
  assessment_targets: list[AssessmentTargetCandidate] = Field(default_factory=list)
  consolidation_status: Literal["completed", "fallback"] = "completed"
  consolidation_note: str | None = None
  review_status: ReviewStatus = "pending"
  review_note: str | None = None
  reviewed_by: str | None = None
  reviewed_at: datetime | None = None
  created_at: datetime = Field(default_factory=datetime.now)

  @model_validator(mode="before")
  @classmethod
  def migrate_legacy_chapters(cls, value):
    """讀取舊版 chapters 內嵌結構並轉為 CourseUnit/Section 結構。"""
    if not isinstance(value, dict):
      return value
    data = dict(value)
    data.setdefault("course_unit_id", data.get("textbook_id", ""))
    data.setdefault("course_unit_title", data.get("title", ""))
    if "sections" in data or "chapters" not in data:
      return data

    sections = []
    points = []
    relations = []
    targets = []
    for chapter in data.get("chapters") or []:
      chapter_data = (
        chapter.model_dump() if hasattr(chapter, "model_dump") else dict(chapter)
      )
      section_id = chapter_data["id"]
      point_ids = []
      for point in chapter_data.get("knowledge_points") or []:
        point_data = point.model_dump() if hasattr(point, "model_dump") else dict(point)
        point_data.setdefault("section_ids", [section_id])
        point_ids.append(point_data["id"])
        points.append(point_data)
      target_ids = []
      for target in chapter_data.get("assessment_targets") or []:
        target_data = target.model_dump() if hasattr(target, "model_dump") else dict(target)
        target_data.setdefault("section_ids", [section_id])
        # 舊版只有整份分析審核；target 狀態由外層狀態繼承。
        target_data.setdefault("review_status", data.get("review_status", "pending"))
        target_ids.append(target_data["id"])
        targets.append(target_data)
      relations.extend(chapter_data.get("relations") or [])
      sections.append({
        "id": section_id,
        "title": chapter_data["title"],
        "order": chapter_data["order"],
        "content_type": "knowledge",
        "included_in_assessment": True,
        "knowledge_point_ids": point_ids,
        "assessment_target_ids": target_ids,
      })
    data["sections"] = sections
    data["knowledge_points"] = points
    data["relations"] = relations
    data["assessment_targets"] = targets
    return data

  class Settings:
    name = "textbook_analyses"


class OptionDiagnosis(BaseModel):
  option_index: int = Field(ge=0, le=3)
  knowledge_point_id: str | None = None
  misconception: str | None = None


class DiagnosisQuiz(Document):
  question: str
  options: list[str]
  answer: int
  analysis: str
  concept: str
  chapter: str
  knowledge_point_ids: list[str] = Field(default_factory=list)
  assessment_target_id: str | None = None
  assessment_target: str | None = None
  option_diagnoses: list[OptionDiagnosis] = Field(default_factory=list)
  source_chunk_ids: list[str] = Field(default_factory=list)
  source_analysis_id: str | None = None
  course_unit_id: str | None = None
  course_unit_title: str | None = None
  section_ids: list[str] = Field(default_factory=list)
  variant_number: int = Field(default=1, ge=1)
  review_status: ReviewStatus = "pending"
  review_note: str | None = None
  reviewed_by: str | None = None
  reviewed_at: datetime | None = None
  version: int = 1

  class Settings:
    name = "diagnosis_quiz"

class StudentProfile(Document):
  """學生基本資訊"""
  discord_id: int
  name: str
  group:  str | None = None
  joined_at: datetime = Field(default_factory=datetime.now)
  
  class Settings:
      name = "student_profiles"


class TargetEvidence(BaseModel):
  assessment_target_id: str
  label: str
  correct_count: int = 0
  wrong_count: int = 0
  misconceptions: list[str] = Field(default_factory=list)
  updated_at: datetime = Field(default_factory=datetime.now)


class QuizResponse(BaseModel):
  question_id: str | None = None
  assessment_target_id: str | None = None
  assessment_target: str
  selected_option: int = Field(ge=0, le=3)
  correct_option: int = Field(ge=0, le=3)
  is_correct: bool
  misconception: str | None = None

class LearningProfile(Document):
  student: Link[StudentProfile]
  pain_points: list[str] = Field(default_factory=list)
  resolved_pain_points: list[str] = Field(default_factory=list)
  learned: list[str] = Field(default_factory=list)
  target_evidence: list[TargetEvidence] = Field(default_factory=list)
  created_at: datetime = Field(default_factory=datetime.now)

  class Settings:
    name = "learning_profiles"

class QuizAttempt(Document):
  """學生每次完成診斷測驗的摘要紀錄。"""
  student: Link[StudentProfile]
  mode: str
  score: int
  total_questions: int
  correct_concepts: list[str] = Field(default_factory=list)
  wrong_concepts: list[str] = Field(default_factory=list)
  correct_targets: list[str] = Field(default_factory=list)
  wrong_targets: list[str] = Field(default_factory=list)
  responses: list[QuizResponse] = Field(default_factory=list)
  completed_at: datetime = Field(default_factory=datetime.now)

  class Settings:
    name = "quiz_attempts"
class LogInfo(BaseModel):
  user_content: str
  chatbot_response: str
  user_timestamp: datetime
  chatbot_timestamp: datetime

class ChatLogs(Document):
  student: Link[StudentProfile]
  project_logs: list[LogInfo] = Field(default_factory=list)
  course_logs: list[LogInfo] = Field(default_factory=list)

  class Settings:
    name = "chat_logs"

async def init_mongo(database_name):
  client = AsyncMongoClient("mongodb://localhost:27017/")
  database = client.get_database(database_name) 
  await init_beanie(database=database, document_models=[DiagnosisQuiz, TextbookAnalysis, StudentProfile, LearningProfile, QuizAttempt, ChatLogs])
  # collection = client["<collection name>"]
  return client

async def get_recent_chat_history_str(
    discord_id: int,
    agent_type: Literal["course", "project"],
    limit: int = 10,
    timeout_minutes: int = 720,
    max_reply_chars: int = 500
) -> str:
    """
    透過 discord_id 讀取最近 N 筆對話紀錄，並直接格式化為給 LLM 參考的純文字
    """
    # 查學生檔案
    student = await StudentProfile.find_one(StudentProfile.discord_id == discord_id)
    if not student:
        return "（無先前對話紀錄）"

    # $slice: -limit 直接在資料庫層截取最後 10 筆，避免撈出整包歷史
    log_field = "course_logs" if agent_type == "course" else "project_logs"
    collection = ChatLogs.get_pymongo_collection()
    doc = await collection.find_one(
        {"student.$id": student.id},
        {log_field: {"$slice": -limit}, "_id": 0}
    )

    if not doc or not doc.get(log_field):
        return "（無先前對話紀錄）"

    # 過濾超時對話並截斷過長的機器人舊回答
    cutoff_time = datetime.now() - timedelta(minutes=timeout_minutes)
    lines = []
    turn = 1

    for log in doc[log_field]:
        user_ts = log.get("user_timestamp")
        if isinstance(user_ts, datetime) and user_ts < cutoff_time:
            continue

        u_text = log.get("user_content", "").strip()
        b_text = log.get("chatbot_response", "").strip()
        if len(b_text) > max_reply_chars:
            b_text = b_text[:max_reply_chars] + "...（後略）"

        lines.append(f"[第 {turn} 輪]\n學生：{u_text}\n助教：{b_text}\n時間：{user_ts}")
        turn += 1

    return "\n".join(lines) if lines else "（無近期有效對話紀錄）"

