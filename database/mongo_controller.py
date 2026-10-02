import pymongo
from pymongo import AsyncMongoClient
from beanie import Document, init_beanie
from beanie.odm.fields import Link
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

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
  chapter_id: str
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
  chapter_id: str
  knowledge_point_ids: list[str]
  title: str
  objective: str
  discrimination: str
  common_misconceptions: list[str] = Field(default_factory=list)
  evidence_chunk_ids: list[str] = Field(default_factory=list)


class ChapterAnalysis(BaseModel):
  id: str
  title: str
  order: int
  knowledge_points: list[KnowledgePointCandidate] = Field(default_factory=list)
  relations: list[KnowledgeRelationCandidate] = Field(default_factory=list)
  assessment_targets: list[AssessmentTargetCandidate] = Field(default_factory=list)


class TextbookAnalysis(Document):
  """教材分析草稿。教師核准後才可成為出題依據。"""
  textbook_id: str
  title: str
  source_file: str
  uploader: str
  chapters: list[ChapterAnalysis] = Field(default_factory=list)
  review_status: ReviewStatus = "pending"
  review_note: str | None = None
  reviewed_by: str | None = None
  reviewed_at: datetime | None = None
  created_at: datetime = Field(default_factory=datetime.now)

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
