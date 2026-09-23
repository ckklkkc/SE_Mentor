import pymongo
from pymongo import AsyncMongoClient
from beanie import Document, init_beanie
from beanie.odm.fields import Link
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class DiagnosisQuiz(Document):
  question: str
  options: list[str]
  answer: int
  analysis: str
  concept: str
  chapter: str

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

class LearningProfile(Document):
  student: Link[StudentProfile]
  pain_points: list[str] = Field(default_factory=list)
  resolved_pain_points: list[str] = Field(default_factory=list)
  learned: list[str] = Field(default_factory=list)
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
  await init_beanie(database=database, document_models=[DiagnosisQuiz, StudentProfile, LearningProfile, QuizAttempt, ChatLogs])
  # collection = client["<collection name>"]
  return client
