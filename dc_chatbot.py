import discord
import asyncio, os
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from discord.ext import commands
from discord import app_commands
from opencc import OpenCC

from utils import file_processor
from services import haystack_service, quiz_generator_kg
from services.kg_constructor import KGConstructor
from database.mongo_controller import LearningProfile, StudentProfile, QuizAttempt, ChatLogs, LogInfo, init_mongo
from database.neo4j_importer import Neo4jImporter, TripleList, EntityList, Entity
import config, prompts, common

# client 是跟 discord 連接，intents 是要求機器人的權限
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# client = discord.Client(intents = intents)
bot = commands.Bot(command_prefix="!", intents=intents)

GUILD_ID = discord.Object(id=1550908830336950322) 
welcomed_users = list()
developers = [905814062850510889]

cc = OpenCC('s2twp')

executor = ThreadPoolExecutor(max_workers=15)
# 把同步阻塞函式包成 async，避免卡住 event loop
async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args, **kwargs))

QUIZ_MODE_LABELS = {
    "bank": "題庫",
    "realtime": "即時",
    "personalized": "個人化",
    "mixed": "混合",
}

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
        embed.set_footer(text="從下方選單選擇弱點，即可標記為已克服並停止後續個人化出題持續聚焦該概念。")
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
    def __init__(self, questions: list, user_id: int, user_name: str, mode: str = "mixed"):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.user_name = user_name
        self.mode = mode
        self.questions = questions
        self.answer_history = ""
        self.learning_pp = list()
        self.learned_concepts = list()
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
    
    async def update_profile(self, user_name: str, user_id: int, all_correct: bool, concepts: list):
        student = await StudentProfile.find_one(StudentProfile.discord_id == user_id)
        if not student:
            student = await StudentProfile(
                discord_id=user_id,
                name=user_name
            ).insert()
        profile = await LearningProfile.find_one(LearningProfile.student.discord_id == user_id, fetch_links=True)

        if profile:
            pain_points = profile.pain_points
            learned_concepts = profile.learned

            # 全對的話，concepts 就是已經學會的概念
            if all_correct:
                print("全對：", concepts)
                # 學習檔案中有學習弱點
                if pain_points:
                    print("原本的弱點：", pain_points)
                    # 取得原本不會但已經會的概念
                    learned = [c for c in concepts if c in pain_points]
                    print("原本不會但已經會：", learned)

                    # 從學習弱點中移除已學會的概念
                    pain_points = [pp for pp in pain_points if pp not in learned]

                    print("修正後的弱點：", pain_points)
                    learned_concepts.extend(learned)
                    profile.pain_points = pain_points
                    profile.learned = list(set(learned_concepts))

                else:
                    print("原本已經會的：", learned_concepts)
                    learned_concepts.extend(concepts)
                    learned_concepts = list(set(learned_concepts)) # 移除重複的概念
                    print("更新後的已學會：", learned_concepts)
                    profile.learned = learned_concepts
            
            # 有答錯的話，concepts 就是答錯的概念
            else:
                print("有錯：", concepts)
                if pain_points:
                    print("原本的弱點：", pain_points)
                    pain_points.extend(concepts)
                    pain_points = list(set(pain_points))
                    profile.pain_points = pain_points
                
                else:
                    profile.pain_points = concepts

                # 如果之後的測驗再次答錯同一概念，代表弱點重新出現，
                # 從 resolved 清單移除並重新啟用。
                resolved = profile.resolved_pain_points or []
                concept_set = set(concepts)
                profile.resolved_pain_points = [
                    item for item in resolved if item not in concept_set
                ]

            await profile.save()
        
        else:
            if all_correct:
                await LearningProfile(
                    student=student,
                    learned=concepts
                ).insert()
            else:
                await LearningProfile(
                    student=student,
                    pain_points=concepts
                ).insert()

    async def get_note(self):
        # 根據不熟的概念生成筆記
        misconception = '、'.join(self.learning_pp)
        res = await run_blocking(haystack_service.neo4j_generate_notes, misconception)
        # res = await neo4j_generate_notes(misconception)
        return res["llm"]["replies"][0]._content[0].text

    async def save_quiz_attempt(self):
        """保存本次 Quiz 摘要，供 /learning_profile 顯示學習紀錄。"""
        student = await StudentProfile.find_one(StudentProfile.discord_id == self.user_id)
        if not student:
            student = await StudentProfile(
                discord_id=self.user_id,
                name=self.user_name,
            ).insert()

        await QuizAttempt(
            student=student,
            mode=self.mode,
            score=self.score,
            total_questions=len(self.questions),
            correct_concepts=list(dict.fromkeys(self.learned_concepts)),
            wrong_concepts=list(dict.fromkeys(self.learning_pp)),
        ).insert()

    async def check_answer(self, interaction: discord.Interaction, choice: int):
        # Discord component interaction 必須在約 3 秒內 ACK。
        # 一進 callback 就先 defer，避免後續任何 I/O 或 event loop 延遲造成
        # 「SE_Mentor 未及時回應」。
        await interaction.response.defer()

        question = self.questions[self.index]
        # 檢查答案並記錄答錯題目
        if choice == question.answer:
            self.score += 1
            self.answer_history = self.answer_history + f"- 第{self.index+1}題：✓\n    - 題目：{question.question}\n    - 正確答案：{question.options[question.answer]}\n"
            self.learned_concepts.append(self.questions[self.index].concept)
        else:
            analysis = self.questions[self.index].analysis
            self.answer_history = self.answer_history + f"- 第{self.index+1}題：✕\n    - 題目：{question.question}\n    - 正確答案：{question.options[question.answer]}\n    - 你的答案：{question.options[choice]}\n    - 解析：{analysis}\n"
            self.learning_pp.append(self.questions[self.index].concept)

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
            await self.save_quiz_attempt()

            if len(self.learning_pp) > 0:
                await self.update_profile(self.user_name, self.user_id, False, self.learning_pp)

                msg = await interaction.followup.send("正在生成筆記…")
                genetared_note = await self.get_note()
                pp = '、'.join(self.learning_pp)
                note = f"你可能對這些概念比較弱：{pp}\n以下是你的專屬筆記！\n\n{genetared_note}"
                chunks = self.split_message(note)

                await msg.edit(content=chunks[0])

                for chunk in chunks[1:]:
                    await interaction.followup.send(chunk)
            else:
                await self.update_profile(self.user_name, self.user_id, True, self.learned_concepts)
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

def printout_questions(question_list):
    for question in question_list:
        print(f"Question: {question['question']}")
        print(f"Options: {question['options']}")
        print(f"Answer: {question['answer']}")
        print(f"Analysis: {question['analysis']}")
        print(f"Concept: {question['concept']}")
        print("\n")

def build_knowledge_graph(source_file: str, doc_type: str, group: str, uploader: str):
    try:
        with open(source_file, 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")

    # 資料前處理
    cleaned_md = file_processor.clean_markdown(md_content)
    cleaned_md = file_processor.remove_specific_sections(cleaned_md)
    md_documents = file_processor.md_splitter(cleaned_md)

    document_contents = list()
    for doc in md_documents:
        # 替換表格
        table_extracted_string = file_processor.replace_tables_in_text(doc.page_content)
        document_contents.append(table_extracted_string)
    
    print(f"markdown 處理完成，開始建立【{doc_type}】圖譜")

    constructor = KGConstructor()
    importer = Neo4jImporter(uri=common.NEO4J_URI, username="neo4j", password=config.NEO4J_PASSWORD)
    try:
        if importer.connect():
            # =====建立知識圖譜 (實體+關係)=====
            if doc_type == "SDD":
                triple_list = constructor.kg_construction_pipeline(document_contents, group)
                print(f"【{doc_type}】三元組抽取完成")
                is_success = importer.upload_doc_triples(triple_list, source_file, doc_type, group, uploader)
                is_success = importer.link_references_to_requirements("API", doc_type, group, "實作需求")

            # =====提取需求文件實體&配對=====
            elif doc_type == "SRD":
                # 抽實體
                entity_list = constructor.entities_extraction_pipeline(document_contents, doc_type, prompts.ENTITY_PROMPT_4_SRD, group)
                print(f"【{doc_type}】實體抽取完成")
                # 配對
                entity_list = constructor.match_fr_to_us_pipeline(entity_list, group)
                print(f"【{doc_type}】實體配對完成")
                is_success = importer.upload_entities(entity_list, source_file, doc_type, group, uploader)
                is_success = importer.link_references_to_requirements("UserStory", doc_type, group, "滿足")
                # 操作角色
                actor_relations = constructor.create_actor_relationships(entity_list)
                print(f"【{doc_type}】角色抽取完成")
                triple_list = TripleList(triples=actor_relations)
                is_success = importer.upload_doc_triples(triple_list, source_file, doc_type, group, uploader)
            
            # =====提取測試文件實體&連接需求文件=====    
            elif doc_type == "STD":
                # 抽實體
                entity_list = constructor.entities_extraction_pipeline(document_contents, doc_type, prompts.ENTITY_PROMPT_4_STD, group)
                print(f"【{doc_type}】實體抽取完成")
                is_success = importer.upload_entities(entity_list, source_file, doc_type, group, uploader)
                is_success = importer.link_references_to_requirements("TestCase", doc_type, group, "驗證")
                
            print(f"上傳結果：{is_success}")
    except Exception as e:
        print(f"建立【{doc_type}】知識圖譜時遇到錯誤：{e}")
    finally:
        importer.close()

# ----- Slash Command -----

@bot.tree.command(name="quiz", description="開始測驗")
@app_commands.describe(mode="選擇題目來源；未指定時使用混合模式")
@app_commands.choices(
    mode=[
        app_commands.Choice(name="混合：題庫 + 個人化/即時", value="mixed"),
        app_commands.Choice(name="題庫：MongoDB 既有題目", value="bank"),
        app_commands.Choice(name="即時：教材 RAG + LLM 重新出題", value="realtime"),
        app_commands.Choice(name="個人化：依過往課程 QA / 學習弱點", value="personalized"),
    ]
)
async def quiz(
    interaction: discord.Interaction,
    mode: app_commands.Choice[str] | None = None,
):
    await interaction.response.defer(ephemeral=True)
    try:
        selected_mode = mode.value if mode else "mixed"
        question_list, source_message = await quiz_generator_kg.get_quizzes(
            mode=selected_mode,
            user_id=interaction.user.id,
            count=5,
        )

        print(
            f"使用者【{interaction.user.name}】已使用診斷測驗！"
            f" mode={selected_mode}"
        )
        view = QuizView(
            question_list,
            interaction.user.id,
            interaction.user.name,
            mode=selected_mode,
        )
        await interaction.followup.send(
            f"{source_message}\n\n{view.get_question()}",
            view=view,
        )

    except Exception as e:
        # 萬一生成失敗，發送錯誤訊息給使用者
        await interaction.followup.send(f"題目生成失敗：{e}")

@bot.tree.command(name="learning_profile", description="查看自己的學習紀錄與學習弱點")
async def learning_profile(interaction: discord.Interaction):
    await interaction.response.deferreturn(ephemeral=True)
    try:
        embed, pain_points = await build_learning_profile_embed(interaction.user.id)
        view = LearningProfileView(interaction.user.id, pain_points)
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)
    except Exception as e:
        print(f"讀取學習紀錄失敗：{e}")
        await interaction.followup.send(f"讀取學習紀錄失敗：{e}", ephemeral=True)

@bot.tree.command(name="project_qa", description="專案問答")
@app_commands.describe(question="請輸入你的問題", group="請輸入你的組別或代號")
async def project_qa(interaction: discord.Interaction, question: str, group: str):
    await interaction.response.defer(ephemeral=True)
    user_timestamp = datetime.now()
    print(f"使用者【{interaction.user.name}】已使用專案問答！")

    try:
        llm_response = await run_blocking(haystack_service.neo4j_doc_retriever, question, group)
        # content = f"> {question}\n\n{response['answer_llm']['replies'][0]}"
        response = cc.convert(llm_response)
        content = f"> {question}\n\n{response}"
        chatbot_timestamp = datetime.now()
        await interaction.followup.send(content=content)    
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
        llm_result = await run_blocking(haystack_service.neo4j_textbook_kg_retriever, question) #不要讓一個學生的問題卡住整個 Discord Bot
        response = cc.convert(llm_result['answer_llm']['replies'][0])
        content = f"> {question}\n\n{response}"
        chatbot_timestamp = datetime.now()
        await interaction.followup.send(content=content)    
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

@bot.tree.command(name="upload_document", description="上傳專案文件")
@app_commands.describe(file="請選擇文件", doc_type="請選擇文件類型", group="請輸入組別或代號")
@app_commands.choices(
    doc_type=[
        app_commands.Choice(name="需求文件", value="SRD"),
        app_commands.Choice(name="設計文件", value="SDD"),
        app_commands.Choice(name="測試文件", value="STD"),
    ]
)
async def upload_document(interaction: discord.Interaction, file: discord.Attachment, doc_type: app_commands.Choice[str], group: str):
    # await interaction.response.defer(ephemeral=True)
    await interaction.response.send_message("文件處理中，約需 2-10 分鐘，請稍後。\n處理完成會進行通知。")
    user = interaction.user

    student = await StudentProfile.find_one(StudentProfile.discord_id == user.id)
    if not student:
        student = await StudentProfile(
            discord_id=user.id,
            name=user.name,
            group=group
        ).insert()
    else:
        student.group = group
        await student.save()


    # 儲存檔案
    file_path = f"md_files\\groups\\{group}"
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    save_path = f"md_files\\groups\\{group}\\{file.filename}"
    await file.save(save_path)

    suffix = Path(save_path).suffix.lower()
    if suffix == ".pdf":
        mk_file = file_processor.pdf2md(save_path)
        save_path = f"md_files\\groups\\{group}\\{file.filename}".replace(".pdf", ".md")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(mk_file)

    # 建圖
    await run_blocking(build_knowledge_graph, source_file=save_path, doc_type=doc_type.value, group=group, uploader=interaction.user.name)
    # 向量
    await haystack_service.upload_doc_2_vectordb(file_path=save_path, doc_type=doc_type.value, group_name=group, uploader=interaction.user.name)

    await user.send(f"【{file.filename}】已成功匯入")

@bot.event
async def setup_hook():
    await init_mongo("TABotAI_quiz")

# 調用event函式庫
@bot.event
async def on_ready():
    
    # bot.tree.clear_commands(guild=GUILD_ID)
    bot.tree.copy_global_to(guild=GUILD_ID)
    slash = await bot.tree.sync()

    print(f"目前登入身份：{bot.user}")
    print(f"在測試伺服器載入 {len(slash)} 個斜線指令")

    for command in slash:
        print(f"- /{command.name}")

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

bot.run(config.DISCORD_TOKEN)
