"""Discord teaching runtime and student-owned HackMD project ingestion.

Heavy authoring modules are imported lazily only when ``/upload_document`` is
used, so normal QA and quiz startup remain independent from conversion models.
"""

import discord
import asyncio, os
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from discord.ext import commands
from discord import app_commands
from opencc import OpenCC

from services import haystack_service, quiz_runtime
from services.project_security import (
    ProjectAccessError,
    ProjectDocumentRejectedError,
    validate_project_question,
)
from database.mongo_controller import (
    LearningProfile, StudentProfile, QuizAttempt, QuizResponse,
    TargetEvidence, ChatLogs, LogInfo, init_mongo,
)
import config, prompts

# client 是跟 discord 連接，intents 是要求機器人的權限
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# client = discord.Client(intents = intents)
bot = commands.Bot(command_prefix="!", intents=intents)

welcomed_users = list()
developers = [905814062850510889]

cc = OpenCC('s2twp')

executor = ThreadPoolExecutor(max_workers=15)
document_ingest_semaphore = asyncio.Semaphore(2)
logger = logging.getLogger(__name__)
# 把同步阻塞函式包成 async，避免卡住 event loop
async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args, **kwargs))


async def _authorized_project_student(
    discord_id: int,
) -> tuple[StudentProfile | None, str | None]:
    """Resolve project scope from server-side identity, never command input."""
    student = await StudentProfile.find_one(
        StudentProfile.discord_id == discord_id
    )
    normalized_group = (student.group or "").strip() if student else ""
    if (
        not normalized_group
        or len(normalized_group) > 80
        or normalized_group in {".", ".."}
        or "/" in normalized_group
        or "\\" in normalized_group
        or any(ord(char) < 32 for char in normalized_group)
    ):
        return student, None
    return student, normalized_group

QUIZ_MODE_LABELS = {
    "reviewed": "教師審核題庫",
    "bank": "題庫",
    "realtime": "即時",
    "personalized": "個人化",
    "mixed": "混合",
}

MIN_TARGET_OBSERVATIONS = 2
WEAKNESS_WRONG_RATE = 0.60
MASTERY_CORRECT_RATE = 0.80


def target_evidence_state(evidence: TargetEvidence) -> str:
    """多次證據達門檻後才判定弱點或已掌握，單次作答只累積證據。"""
    attempts = evidence.correct_count + evidence.wrong_count
    if attempts < MIN_TARGET_OBSERVATIONS:
        return "insufficient"
    if evidence.wrong_count >= 2 and evidence.wrong_count / attempts >= WEAKNESS_WRONG_RATE:
        return "weak"
    if evidence.correct_count / attempts >= MASTERY_CORRECT_RATE:
        return "mastered"
    return "developing"

def _format_datetime(value: datetime | None) -> str:
    if not value:
        return "未知時間"
    return value.strftime("%Y/%m/%d %H:%M")

def _compact_list(items: list[str], empty_text: str = "目前沒有紀錄", limit: int = 20) -> str:
    if not items:
        return empty_text
    unique_items = list(dict.fromkeys(items))
    shown = unique_items[:limit]
    text = "\n".join(f"• {cc.convert(str(item))}" for item in shown)
    if len(unique_items) > limit:
        text += f"\n…另有 {len(unique_items) - limit} 項"
    return text[:1024]


async def build_learning_profile_embed(user_id: int) -> tuple[discord.Embed, list[str]]:
    """整理學生目前的學習歷程，用於 /learning_profile 與弱點更新後刷新畫面。"""
    student = await StudentProfile.find_one(StudentProfile.discord_id == user_id)
    if not student:
        embed = discord.Embed(
            title="我的學習紀錄",
            description="目前尚未建立你的學習紀錄。先使用 `/course_qa` 或 `/quiz` 後再查看。",
        )
        return embed, []

    profile = await LearningProfile.find_one(
        LearningProfile.student.discord_id == user_id,
        fetch_links=True,
    )
    chat_logs = await ChatLogs.find_one(
        ChatLogs.student.discord_id == user_id,
        fetch_links=True,
    )

    attempts_query = QuizAttempt.find(
        QuizAttempt.student.discord_id == user_id,
        fetch_links=True,
    )
    attempt_count = await attempts_query.count()
    recent_attempts = await QuizAttempt.find(
        QuizAttempt.student.discord_id == user_id,
        fetch_links=True,
    ).sort("-completed_at").limit(5).to_list()

    pain_points = list(dict.fromkeys(profile.pain_points)) if profile and profile.pain_points else []
    resolved_pain_points = (
        list(dict.fromkeys(profile.resolved_pain_points))
        if profile and profile.resolved_pain_points
        else []
    )
    learned = list(dict.fromkeys(profile.learned)) if profile and profile.learned else []
    course_logs = chat_logs.course_logs if chat_logs and chat_logs.course_logs else []

    embed = discord.Embed(
        title=f"{student.name} 的學習紀錄",
        description=(
            "這裡顯示目前仍需要複習的概念與近期學習活動。"
            "若你已透過學習筆記或 QA 理解某個弱點，可從下方選單將它標記為已克服。"
        ),
    )

    embed.add_field(
        name=f"目前學習弱點（{len(pain_points)}）",
        value=_compact_list(pain_points, "目前沒有需要複習的學習弱點。"),
        inline=False,
    )
    embed.add_field(
        name=f"已主動標記克服（{len(resolved_pain_points)}）",
        value=_compact_list(resolved_pain_points, "目前尚無手動標記為已克服的弱點。"),
        inline=False,
    )
    embed.add_field(
        name=f"已掌握／已處理概念（{len(learned)}）",
        value=_compact_list(learned, "目前尚無已掌握概念紀錄。"),
        inline=False,
    )

    if recent_attempts:
        quiz_lines = []
        for attempt in recent_attempts:
            mode_label = QUIZ_MODE_LABELS.get(attempt.mode, attempt.mode)
            wrong = "、".join(dict.fromkeys(attempt.wrong_concepts)) if attempt.wrong_concepts else "無"
            if len(wrong) > 100:
                wrong = wrong[:97] + "…"
            quiz_lines.append(
                f"• {_format_datetime(attempt.completed_at)}｜{mode_label}｜"
                f"{attempt.score}/{attempt.total_questions}｜弱點：{cc.convert(wrong)}"
            )
        quiz_text = "\n".join(quiz_lines)
    else:
        quiz_text = "目前尚無已保存的測驗紀錄；更新此版本後完成的 Quiz 會開始記錄。"
    embed.add_field(
        name=f"近期 Quiz（累計 {attempt_count} 次）",
        value=quiz_text[:1024],
        inline=False,
    )

    if course_logs:
        qa_lines = []
        for log in course_logs[-5:][::-1]:
            question = (log.user_content or "").replace("\n", " ").strip()
            if len(question) > 90:
                question = question[:87] + "…"
            qa_lines.append(f"• {_format_datetime(log.user_timestamp)}｜{cc.convert(question)}")
        qa_text = "\n".join(qa_lines)
    else:
        qa_text = "目前尚無 Course QA 紀錄。"
    embed.add_field(
        name=f"近期 Course QA（累計 {len(course_logs)} 次）",
        value=qa_text[:1024],
        inline=False,
    )

    if len(pain_points) > 25:
        embed.set_footer(text="Discord 選單一次最多顯示 25 個弱點；移除後會自動載入後續項目。")
    elif pain_points:
        embed.set_footer(text="從下方選單選擇弱點，即可標記為已克服並更新後續教學建議。")
    else:
        embed.set_footer(text="目前沒有 active learning weakness。")

    return embed, pain_points

class WeaknessSelect(discord.ui.Select):
    def __init__(self, parent_view: "LearningProfileView", pain_points: list[str]):
        self.parent_view = parent_view
        self.visible_pain_points = pain_points[:25]
        options = [
            discord.SelectOption(
                label=cc.convert(concept)[:100],
                value=str(index),
                description="標記已克服並從目前弱點移除",
            )
            for index, concept in enumerate(self.visible_pain_points)
        ]
        super().__init__(
            placeholder="選擇已克服的學習弱點（可複選）",
            min_values=1,
            max_values=max(1, len(options)),
            options=options,
        )

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.parent_view.user_id:
            await interaction.response.send_message("這不是你的學習紀錄。", ephemeral=True)
            return

        await interaction.response.defer()
        selected = [
            self.visible_pain_points[int(index)]
            for index in self.values
            if index.isdigit() and int(index) < len(self.visible_pain_points)
        ]

        profile = await LearningProfile.find_one(
            LearningProfile.student.discord_id == interaction.user.id,
            fetch_links=True,
        )
        if not profile:
            await interaction.followup.send("目前找不到你的 LearningProfile。", ephemeral=True)
            return

        selected_set = set(selected)
        removed = [concept for concept in profile.pain_points if concept in selected_set]
        profile.pain_points = [concept for concept in profile.pain_points if concept not in selected_set]

        # 使用者主動確認已理解：從 active weakness 移除。
        # resolved_pain_points 用來記錄「由使用者主動清除」的概念，
        # personalized quiz 會避開這些概念，避免僅因舊 QA 又反覆聚焦。
        profile.resolved_pain_points = list(
            dict.fromkeys([*(profile.resolved_pain_points or []), *removed])
        )
        profile.learned = list(dict.fromkeys([*(profile.learned or []), *removed]))
        await profile.save()

        embed, pain_points = await build_learning_profile_embed(interaction.user.id)
        self.parent_view.refresh(pain_points)
        await interaction.edit_original_response(embed=embed, view=self.parent_view)

class LearningProfileView(discord.ui.View):
    def __init__(self, user_id: int, pain_points: list[str]):
        super().__init__(timeout=300)
        self.user_id = user_id
        self.refresh(pain_points)

    def refresh(self, pain_points: list[str]):
        self.clear_items()
        if pain_points:
            self.add_item(WeaknessSelect(self, pain_points))

# ----- Button UI -----
class QuizView(discord.ui.View):
    def __init__(self, questions: list, user_id: int, user_name: str, mode: str = "reviewed"):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.user_name = user_name
        self.mode = mode
        self.questions = questions
        self.answer_history = ""
        self.learning_pp = list()
        self.learned_concepts = list()
        self.responses: list[QuizResponse] = []
        self.index = 0
        self.score = 0

    async def on_timeout(self):
        # timeout 後禁用按鈕，避免殭屍 View 佔記憶體
        for item in self.children:
            item.disabled = True

    def get_question(self):
        question = self.questions[self.index]
        return f"""
        **第{self.index+1}題)** {cc.convert(question.question)}

        A. {question.options[0]}
        B. {question.options[1]}
        C. {question.options[2]}
        D. {question.options[3]}

        """

    def split_message(self, text: str, limit: int = 1500):
        return [
            text[i:i+limit]
            for i in range(0, len(text), limit)
        ]
    
    async def get_note(self):
        # 根據不熟的概念生成筆記
        misconception = '、'.join(self.learning_pp)
        res = await run_blocking(haystack_service.neo4j_generate_notes, misconception)
        # res = await neo4j_generate_notes(misconception)
        return res["llm"]["replies"][0]._content[0].text

    async def save_quiz_attempt(self) -> list[str]:
        """保存本次 Quiz 摘要，供 /learning_profile 顯示學習紀錄。"""
        student = await StudentProfile.find_one(StudentProfile.discord_id == self.user_id)
        if not student:
            student = await StudentProfile(
                discord_id=self.user_id,
                name=self.user_name,
            ).insert()

        correct_targets = [
            response.assessment_target for response in self.responses if response.is_correct
        ]
        wrong_targets = [
            response.assessment_target for response in self.responses if not response.is_correct
        ]
        await QuizAttempt(
            student=student,
            mode=self.mode,
            score=self.score,
            total_questions=len(self.questions),
            correct_concepts=list(dict.fromkeys(self.learned_concepts)),
            wrong_concepts=list(dict.fromkeys(self.learning_pp)),
            correct_targets=list(dict.fromkeys(correct_targets)),
            wrong_targets=list(dict.fromkeys(wrong_targets)),
            responses=self.responses,
        ).insert()

        profile = await LearningProfile.find_one(
            LearningProfile.student.discord_id == self.user_id,
            fetch_links=True,
        )
        if not profile:
            profile = await LearningProfile(student=student).insert()

        evidence_by_id = {
            item.assessment_target_id: item for item in (profile.target_evidence or [])
        }
        touched_ids: set[str] = set()
        wrong_touched_ids: set[str] = set()
        for response in self.responses:
            target_id = response.assessment_target_id or f"legacy:{response.assessment_target}"
            touched_ids.add(target_id)
            evidence = evidence_by_id.get(target_id)
            if not evidence:
                evidence = TargetEvidence(
                    assessment_target_id=target_id,
                    label=response.assessment_target,
                )
                evidence_by_id[target_id] = evidence
            if response.is_correct:
                evidence.correct_count += 1
            else:
                evidence.wrong_count += 1
                wrong_touched_ids.add(target_id)
                if response.misconception:
                    evidence.misconceptions = list(dict.fromkeys(
                        evidence.misconceptions + [response.misconception]
                    ))
            evidence.updated_at = datetime.now()
        profile.target_evidence = list(evidence_by_id.values())
        confirmed_weaknesses: list[str] = []
        pain_points = list(profile.pain_points or [])
        learned = list(profile.learned or [])
        resolved = list(profile.resolved_pain_points or [])
        for target_id in touched_ids:
            evidence = evidence_by_id[target_id]
            state = target_evidence_state(evidence)
            # 舊的同目標弱點可能帶有不同選項迷思，先統一移除再依狀態重建。
            pain_points = [
                item for item in pain_points
                if item != evidence.label and not item.startswith(f"{evidence.label}：")
            ]
            if state == "weak":
                detail = evidence.misconceptions[-1] if evidence.misconceptions else None
                weakness = f"{evidence.label}：{detail}" if detail else evidence.label
                pain_points.append(weakness)
                if target_id in wrong_touched_ids:
                    confirmed_weaknesses.append(weakness)
                learned = [item for item in learned if item != evidence.label]
                resolved = [
                    item for item in resolved
                    if item != evidence.label and not item.startswith(f"{evidence.label}：")
                ]
            elif state == "mastered":
                learned.append(evidence.label)

        profile.pain_points = list(dict.fromkeys(pain_points))
        profile.learned = list(dict.fromkeys(learned))
        profile.resolved_pain_points = list(dict.fromkeys(resolved))
        await profile.save()
        return confirmed_weaknesses

    @staticmethod
    def _target_label(question) -> str:
        return getattr(question, "assessment_target", None) or question.concept

    @staticmethod
    def _option_misconception(question, choice: int) -> str | None:
        for diagnosis in getattr(question, "option_diagnoses", []) or []:
            option_index = (
                diagnosis.option_index if hasattr(diagnosis, "option_index")
                else diagnosis.get("option_index")
            )
            if option_index == choice:
                return (
                    diagnosis.misconception if hasattr(diagnosis, "misconception")
                    else diagnosis.get("misconception")
                )
        return None

    async def check_answer(self, interaction: discord.Interaction, choice: int):
        # Discord component interaction 必須在約 3 秒內 ACK。
        # 一進 callback 就先 defer，避免後續任何 I/O 或 event loop 延遲造成
        # 「SE_Mentor 未及時回應」。
        await interaction.response.defer()

        question = self.questions[self.index]
        target_label = self._target_label(question)
        target_id = getattr(question, "assessment_target_id", None)
        misconception = self._option_misconception(question, choice)
        is_correct = choice == question.answer
        self.responses.append(
            QuizResponse(
                question_id=str(question.id) if getattr(question, "id", None) else None,
                assessment_target_id=target_id,
                assessment_target=target_label,
                selected_option=choice,
                correct_option=question.answer,
                is_correct=is_correct,
                misconception=misconception if not is_correct else None,
            )
        )
        # 檢查答案並記錄答錯題目
        if is_correct:
            self.score += 1
            self.answer_history = self.answer_history + f"- 第{self.index+1}題：✓\n    - 題目：{question.question}\n    - 正確答案：{question.options[question.answer]}\n"
            self.learned_concepts.append(target_label)
        else:
            analysis = self.questions[self.index].analysis
            self.answer_history = self.answer_history + f"- 第{self.index+1}題：✕\n    - 題目：{question.question}\n    - 正確答案：{question.options[question.answer]}\n    - 你的答案：{question.options[choice]}\n    - 解析：{analysis}\n"
            weakness = target_label
            if misconception:
                weakness = f"{target_label}：{misconception}"
            self.learning_pp.append(weakness)

        self.index += 1

        if self.index >= len(self.questions):
            # 測驗結束：先停用按鈕，避免完成後仍可重複作答。
            for item in self.children:
                item.disabled = True
            self.stop()

            # component 已經 defer，因此不能再使用 interaction.response.*；
            # 直接編輯原本的測驗訊息。
            await interaction.edit_original_response(view=self)

            await interaction.followup.send(
                f"\n測驗結束q(≧▽≦q) 你的分數：{self.score}/{len(self.questions)}\n答題記錄：\n{self.answer_history}\n"
            )
            print(f"學生學習弱項：{self.learning_pp}")

            # 無論題目來源為何，都保存本次測驗摘要作為學習紀錄。
            confirmed_weaknesses = await self.save_quiz_attempt()

            # 學習筆記與「正式弱點判定」是兩件事：
            # - 本次只要有錯題，就立即依題目的評量目標／選項迷思提供複習筆記。
            # - 是否寫入 learning profile 的弱點，仍由多次作答證據門檻決定。
            if self.learning_pp:
                msg = await interaction.followup.send("正在生成本次錯題的複習筆記…")
                generated_note = await self.get_note()
                review_points = '、'.join(dict.fromkeys(self.learning_pp))
                if confirmed_weaknesses:
                    confirmed = '、'.join(dict.fromkeys(confirmed_weaknesses))
                    evidence_message = (
                        f"多次作答證據顯示你可能對這些評量目標較弱：{confirmed}"
                    )
                else:
                    evidence_message = (
                        "本次錯題已記入評量目標證據；單次答錯不會立即判定為學習弱點。"
                        "同一目標累積至少兩次作答後，系統才會依正確率更新學習輪廓。"
                    )
                note = (
                    f"{evidence_message}\n"
                    f"本次錯題複習範圍：{review_points}\n"
                    f"以下是依本次錯題產生的學習筆記：\n\n{generated_note}"
                )
                chunks = self.split_message(note)

                await msg.edit(content=chunks[0])

                for chunk in chunks[1:]:
                    await interaction.followup.send(chunk)
        else:
            # 已在函式開頭 defer；用 edit_original_response 顯示下一題。
            await interaction.edit_original_response(
                content=self.get_question(),
                view=self,
            )

    @discord.ui.button(label="A", style=discord.ButtonStyle.primary)
    async def option_a(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.check_answer(interaction, 0)

    @discord.ui.button(label="B", style=discord.ButtonStyle.primary)
    async def option_b(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.check_answer(interaction, 1)

    @discord.ui.button(label="C", style=discord.ButtonStyle.primary)
    async def option_c(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.check_answer(interaction, 2)

    @discord.ui.button(label="D", style=discord.ButtonStyle.primary)
    async def option_d(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.check_answer(interaction, 3)

# ----- Slash Command -----

@bot.tree.command(name="quiz", description="開始測驗")
@app_commands.describe(
    course_unit="教材單元名稱或 CourseUnit ID，例如 [02]基礎需求工程",
    questions="本次小測驗題數（5～10）",
)
async def quiz(
    interaction: discord.Interaction,
    course_unit: str | None = None,
    questions: app_commands.Range[int, 5, 10] = 5,
):
    await interaction.response.defer(ephemeral=True)
    try:
        question_list, source_message = await quiz_runtime.get_quizzes(
            user_id=interaction.user.id,
            count=questions,
            course_unit=course_unit,
        )

        print(
            f"使用者【{interaction.user.name}】已使用診斷測驗！"
            " mode=bank"
        )
        view = QuizView(
            question_list,
            interaction.user.id,
            interaction.user.name,
            mode="bank",
        )
        await interaction.followup.send(
            f"{source_message}\n\n{view.get_question()}",
            view=view,
        )

    except Exception as e:
        await interaction.followup.send(f"無法載入已審核題目：{e}")

@bot.tree.command(name="learning_profile", description="查看自己的學習紀錄與學習弱點")
async def learning_profile(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        embed, pain_points = await build_learning_profile_embed(interaction.user.id)
        view = LearningProfileView(interaction.user.id, pain_points)
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)
    except Exception as e:
        print(f"讀取學習紀錄失敗：{e}")
        await interaction.followup.send(f"讀取學習紀錄失敗：{e}", ephemeral=True)


@bot.tree.command(
    name="upload_document",
    description="從公開 HackMD 連結匯入需求、設計或測試文件",
)
@app_commands.describe(
    hackmd_url="公開的 HackMD 筆記連結",
    document_type="文件類型",
)
@app_commands.choices(document_type=[
    app_commands.Choice(name="需求文件（SRD）", value="SRD"),
    app_commands.Choice(name="設計文件（SDD）", value="SDD"),
    app_commands.Choice(name="測試文件（STD）", value="STD"),
])
async def upload_document(
    interaction: discord.Interaction,
    hackmd_url: str,
    document_type: app_commands.Choice[str],
):
    await interaction.response.defer(ephemeral=True)
    student, normalized_group = await _authorized_project_student(
        interaction.user.id
    )
    if not student or not normalized_group:
        await interaction.followup.send(
            "你的帳號尚未由教師綁定專案組別，不能上傳專案文件。",
            ephemeral=True,
        )
        return

    await interaction.followup.send(
        "已開始檢查教師核准來源與專案文件結構；通過後才會逐張分析圖片；"
        "圖片較多時會需要數分鐘。",
        ephemeral=True,
    )
    quality_error_types: tuple[type[Exception], ...] = ()
    source_error_types: tuple[type[Exception], ...] = ()
    security_error_types: tuple[type[Exception], ...] = (
        ProjectAccessError,
        ProjectDocumentRejectedError,
    )
    try:
        # Lazy import keeps the normal Discord teaching runtime lightweight.
        from services.content_authoring import ingest_hackmd_project_document
        from services.document_image_quality import ProjectImageQualityError
        from services.hackmd_document import HackMDImportError

        quality_error_types = (ProjectImageQualityError,)
        source_error_types = (HackMDImportError,)

        async with document_ingest_semaphore:
            markdown_path, manifest = await ingest_hackmd_project_document(
                hackmd_url,
                document_type.value,
                normalized_group,
                interaction.user.name,
            )

        image_count = int(manifest.get("image_count", 0))
        described_count = int(manifest.get("described_count", 0))
        excluded_count = int(manifest.get("description_excluded_count", 0))
        review_count = int(manifest.get("review_recommended_count", 0))
        exclusion_text = (
            f"，免轉述 {excluded_count} 張"
            if excluded_count
            else ""
        )
        review_line = (
            f"- 圖片複核提醒：{review_count} 張（輕微提醒，不影響本次匯入）\n"
            if review_count
            else ""
        )
        graph_stats = manifest.get("knowledge_graph", {})
        graph_nodes = int(graph_stats.get("node_count", 0))
        graph_relationships = int(graph_stats.get("relationship_count", 0))
        vector_chunks = int(manifest.get("vector_chunk_count", 0))
        relevance = manifest.get("project_relevance", {})
        relevance_signals = "、".join(
            str(value) for value in relevance.get("matched_signals", [])
        )
        await interaction.followup.send(
            "文件匯入完成。\n"
            f"- 類型：{document_type.value}\n"
            f"- 組別：{normalized_group}\n"
            f"- 來源授權與文件結構檢查：通過"
            f"{f'（{relevance_signals}）' if relevance_signals else ''}\n"
            f"- 圖片：{image_count} 張，完成描述 {described_count} 張"
            f"{exclusion_text}\n"
            f"{review_line}"
            f"- 知識圖譜驗證：本文件 {graph_nodes} 個節點，"
            f"本組目前 {graph_relationships} 條關聯\n"
            f"- Markdown 向量索引：本文件 {vector_chunks} 個片段\n"
            f"- 轉換檔：`{markdown_path.name}`\n\n"
            "現在可使用 `/project_qa` 詢問文件內容、圖片流程、介面或需求問題。",
            ephemeral=True,
        )
    except Exception as exc:
        if isinstance(exc, security_error_types):
            logger.warning("專案文件安全檢查拒絕匯入：%s", exc)
            response_text = str(exc)
        elif quality_error_types and isinstance(exc, quality_error_types):
            logger.warning("文件匯入未完成（圖片轉述檢查）：%s", exc)
            response_text = str(exc)
        elif source_error_types and isinstance(exc, source_error_types):
            logger.warning("HackMD 文件或圖片讀取失敗：%s", exc)
            response_text = f"HackMD 文件或圖片讀取失敗：{exc}"
        else:
            logger.exception("文件匯入過程發生未預期錯誤")
            response_text = (
                f"文件匯入過程發生錯誤（{type(exc).__name__}）：{exc}"
            )
        await interaction.followup.send(
            response_text[:1800],
            ephemeral=True,
        )


def _discord_message_chunks(text: str, limit: int = 1900) -> list[str]:
    """Split a response without cutting normal paragraphs when possible."""
    chunks = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n\n", 0, limit + 1)
        if cut < limit // 2:
            cut = remaining.rfind("\n", 0, limit + 1)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit + 1)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    return chunks


@bot.tree.command(name="project_qa", description="專案問答")
@app_commands.describe(question="請輸入你的問題")
async def project_qa(interaction: discord.Interaction, question: str):
    await interaction.response.defer(ephemeral=True)
    user_timestamp = datetime.now()
    print(f"使用者【{interaction.user.name}】已使用專案問答！")

    student, normalized_group = await _authorized_project_student(
        interaction.user.id
    )
    if not student or not normalized_group:
        await interaction.followup.send(
            "你的帳號尚未由教師綁定專案組別，不能使用專案問答。",
            ephemeral=True,
        )
        return
    try:
        question = validate_project_question(question)
    except ProjectAccessError as exc:
        await interaction.followup.send(
            str(exc),
            ephemeral=True,
        )
        return

    try:
        llm_response = await run_blocking(
            haystack_service.neo4j_doc_retriever,
            question,
            normalized_group,
        )
        # content = f"> {question}\n\n{response['answer_llm']['replies'][0]}"
        response = cc.convert(llm_response)
        content = f"> {question}\n\n{response}"
        chatbot_timestamp = datetime.now()
        for chunk in _discord_message_chunks(content):
            await interaction.followup.send(content=chunk, ephemeral=True)
    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"回答生成失敗：{e}", ephemeral=True)
        return

    chat_logs = await ChatLogs.find_one(ChatLogs.student.discord_id == student.discord_id, fetch_links=True)
    if chat_logs:
        chat_logs.project_logs.append(
            LogInfo(
                user_content=question,
                chatbot_response=response,
                user_timestamp=user_timestamp,
                chatbot_timestamp=chatbot_timestamp
            )
        )
        await chat_logs.save()
    else:
        await ChatLogs(
            student=student,
            project_logs=[
                LogInfo(
                    user_content=question,
                    chatbot_response=response,
                    user_timestamp=user_timestamp,
                    chatbot_timestamp=chatbot_timestamp
                )
            ]
        ).insert()

@bot.tree.command(name="course_qa", description="課程問答")
@app_commands.describe(question="請輸入你的問題")
async def course_qa(interaction: discord.Interaction, question: str):
    await interaction.response.defer(ephemeral=True)    #ephemeral=True 表示這個回應原則上只有提問者看得到。
    user_timestamp = datetime.now()
    print(f"使用者【{interaction.user.name}】已使用課程問答！")

    try:

        llm_keywords = await run_blocking(haystack_service.handle_user_question, question)

        llm_result = await run_blocking(haystack_service.neo4j_textbook_kg_retriever, question) #不要讓一個學生的問題卡住整個 Discord Bot
        response = cc.convert(llm_result['answer_llm']['replies'][0])

        crit_score = await run_blocking(haystack_service.CRIT_check, question, response)
        print(f"CRIT分數為：{crit_score}")



        content = f"> {question}\n\n{response}"
        chatbot_timestamp = datetime.now()
        await interaction.followup.send(content=content)
        await interaction.followup.send(f"CRIT分數為：{crit_score}")
    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"回答生成失敗：{e}")
        return

    student = await StudentProfile.find_one(StudentProfile.discord_id == interaction.user.id)
    if not student:
        student = await StudentProfile(
            discord_id=interaction.user.id,
            name=interaction.user.name
        ).insert()

    chat_logs = await ChatLogs.find_one(ChatLogs.student.discord_id == student.discord_id, fetch_links=True)
    if chat_logs:
        chat_logs.course_logs.append(
            LogInfo(
                user_content=question,
                chatbot_response=response,
                user_timestamp=user_timestamp,
                chatbot_timestamp=chatbot_timestamp
            )
        )
        await chat_logs.save()
    else:
        await ChatLogs(
            student=student,
            course_logs=[
                LogInfo(
                    user_content=question,
                    chatbot_response=response,
                    user_timestamp=user_timestamp,
                    chatbot_timestamp=chatbot_timestamp
                )
            ]
        ).insert()

@bot.event
async def setup_hook():
    await init_mongo("TABotAI_quiz")

    # 全域同步不依賴特定伺服器；正式環境只需要這一步。
    try:
        global_commands = await bot.tree.sync()
        print(f"已同步 {len(global_commands)} 個全域教學指令")
    except discord.HTTPException as exc:
        print(f"全域指令同步失敗，將沿用 Discord 現有指令：{exc}")

    # 開發時可在 config.py 設定 TEST_GUILD_ID 取得立即更新的 guild commands。
    # Bot 不在該伺服器或 ID 填錯時只警告，不阻止教學 Bot 上線。
    test_guild_id = getattr(config, "TEST_GUILD_ID", None)
    if test_guild_id:
        try:
            guild = discord.Object(id=int(test_guild_id))
            bot.tree.copy_global_to(guild=guild)
            guild_commands = await bot.tree.sync(guild=guild)
            print(
                f"測試伺服器 {test_guild_id} 已同步 "
                f"{len(guild_commands)} 個教學指令"
            )
        except (discord.Forbidden, discord.NotFound) as exc:
            print(
                f"略過測試伺服器 {test_guild_id} 的指令同步：{exc}\n"
                "請確認 TEST_GUILD_ID 正確，且 Bot 已用 bot + "
                "applications.commands scopes 加入該伺服器。"
            )
        except (TypeError, ValueError):
            print("略過測試伺服器同步：TEST_GUILD_ID 必須是 Discord 伺服器的數字 ID")

# 調用event函式庫
@bot.event
async def on_ready():
    print(f"目前登入身份：{bot.user}")
    print("Discord 教學 Bot 已上線")

@bot.event
async def on_member_join(member):
    # DM 給進入指定頻道的使用者
    print(f"{member.name} has joined the server!")

    await StudentProfile(
        discord_id=member.id,
        name=member.name
    ).insert()

    try:
        await member.send(prompts.DCCHATBOT_WELCOME_MESSAGE)
        welcomed_users.append(member.id)

    except discord.Forbidden:
        print(f"無法私訊 {member.name}")

    print(welcomed_users)
    
@bot.event
async def on_message(message):
    # 如果是機器人本身傳的訊息就忽略
    if message.author.bot:
        return
    # 如果訊息不是在私人訊息 (DM 頻道) 
    if not isinstance(message.channel, discord.DMChannel):
        if message.author.id not in developers:
            return
        else:
            await message.channel.send("你好！若要使用 SE Mentor 的其他功能，請使用斜線指令")

    else:
        user_id = message.author.id
        if user_id not in welcomed_users:   # 如果是新的使用者就傳送歡迎訊息
            welcomed_users.append(user_id)
            await message.channel.send(prompts.DCCHATBOT_WELCOME_MESSAGE)
        else:
            await message.channel.send("你好！若要使用 SE Mentor 的其他功能，請使用斜線指令")

    await bot.process_commands(message)
    print(welcomed_users)

# @bot.event
# # # 當頻道有新訊息
# async def on_message(message):
#     # 排除機器人本身的訊息，避免無限循環
#     if message.author == bot.user:
#         return
    
#     print("Message received: ", message.content)
#     response_text = await neo4j_retriever(message.content)
#     await message.channel.send(response_text)

if __name__ == "__main__":
    try:
        bot.run(config.DISCORD_TOKEN)
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Ctrl+C may surface as CancelledError while the Discord gateway waits
        # for the next WebSocket message, depending on the Python runtime.
        print("SE Mentor 已停止。")
