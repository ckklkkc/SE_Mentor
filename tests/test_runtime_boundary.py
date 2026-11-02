import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DiscordRuntimeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "dc_chatbot.py").read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_discord_exposes_only_teaching_commands(self):
        command_names = set()
        for node in self.tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                function = decorator.func
                if not (
                    isinstance(function, ast.Attribute)
                    and function.attr == "command"
                    and isinstance(function.value, ast.Attribute)
                    and function.value.attr == "tree"
                ):
                    continue
                for keyword in decorator.keywords:
                    if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                        command_names.add(keyword.value.value)

        self.assertEqual(
            command_names,
            {
                "quiz", "learning_profile", "upload_document",
                "project_qa", "course_qa",
            },
        )

    def test_discord_does_not_import_authoring_modules_at_startup(self):
        forbidden = {
            "utils.file_processor",
            "services.kg_constructor",
            "services.textbook_analyzer",
            "services.content_authoring",
            "services.quiz_generator_kg",
            "database.neo4j_importer",
        }
        imported = set()
        # Only inspect module-level imports. /upload_document deliberately uses
        # one function-local lazy import after the interaction is accepted.
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        self.assertTrue(forbidden.isdisjoint(imported), imported & forbidden)
        self.assertIn(
            "from services.content_authoring import ingest_hackmd_project_document",
            self.source,
        )

    def test_test_guild_sync_is_optional_and_non_fatal(self):
        self.assertNotIn("1550908830336950322", self.source)
        self.assertIn('getattr(config, "TEST_GUILD_ID", None)', self.source)
        self.assertIn("discord.Forbidden, discord.NotFound", self.source)

    def test_offline_cli_owns_authoring_commands(self):
        source = (ROOT / "authoring_cli.py").read_text(encoding="utf-8")
        expected = {
            "ingest-textbook",
            "show-textbook",
            "review-textbook",
            "list-assessment-targets",
            "review-assessment-target",
            "generate-quiz-drafts",
            "generate-course-unit-bank",
            "course-unit-bank-status",
            "list-quizzes",
            "show-quiz",
            "review-quiz",
            "ingest-project-document",
        }
        for command in expected:
            self.assertIn(f'"{command}"', source)

    def test_runtime_quiz_service_has_no_generation_dependency(self):
        source = (ROOT / "services" / "quiz_runtime.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)

        forbidden_prefixes = ("openai", "anthropic", "services.quiz_generator_kg")
        self.assertFalse(
            any(name.startswith(forbidden_prefixes) for name in imported),
            imported,
        )
        self.assertIn('"review_status": "approved"', source)
        self.assertIn("每題對應不同評量目標", source)

    def test_course_qa_only_retrieves_approved_textbook_graph(self):
        source = (ROOT / "services" / "haystack_service.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("entity.review_status = 'approved'", source)
        self.assertIn("entity:KnowledgePoint OR entity:AssessmentTarget", source)

    def test_project_qa_preserves_source_precedence_and_partial_availability(self):
        source = (ROOT / "services" / "haystack_service.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("兩者衝突時，以 Markdown 明文為準", source)
        self.assertIn("專案知識圖譜檢索失敗", source)
        self.assertIn("專案 Markdown 向量檢索失敗", source)
        self.assertIn("coalesce(entity.req_id, '')", source)
        self.assertIn("_remove_stale_project_vectors", source)

    def test_project_commands_derive_group_from_student_profile(self):
        project_qa = next(
            node for node in self.tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "project_qa"
        )
        upload = next(
            node for node in self.tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "upload_document"
        )
        self.assertNotIn("group", [arg.arg for arg in project_qa.args.args])
        self.assertNotIn("group", [arg.arg for arg in upload.args.args])
        self.assertIn("_authorized_project_student", self.source)
        self.assertIn("尚未由教師綁定專案組別", self.source)

    def test_project_qa_uses_role_separated_prompt_and_hard_budget(self):
        source = (ROOT / "services" / "haystack_service.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("ChatMessage.from_system(system_prompt)", source)
        self.assertIn("PROJECT_QA_MAX_OUTPUT_TOKENS", source)
        self.assertIn("PROJECT_QA_MAX_MARKDOWN_CHARS", source)
        self.assertIn("不得查詢、比較、推測或揭露其他組別內容", source)

    def test_upload_checks_allowlist_and_relevance_before_captioning(self):
        authoring = (ROOT / "services" / "content_authoring.py").read_text(
            encoding="utf-8"
        )
        hackmd = (ROOT / "services" / "hackmd_document.py").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            authoring.index("validate_project_document_source("),
            authoring.index("prepare_hackmd_markdown,"),
        )
        self.assertLess(
            hackmd.index("validate_project_document_relevance(markdown"),
            hackmd.index("_image_map(markdown, note_url)"),
        )

    def test_srd_reimport_replaces_derived_relationships(self):
        source = (ROOT / "services" / "content_authoring.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("replace_project_relationships", source)
        self.assertIn('["滿足", "操作"]', source)
        self.assertIn("project_graph_stats", source)
        self.assertIn('graph_stats["node_count"] < 1', source)

    def test_project_qa_splits_long_discord_responses(self):
        self.assertIn("def _discord_message_chunks", self.source)
        self.assertIn("for chunk in _discord_message_chunks(content)", self.source)
        self.assertIn("知識圖譜驗證：本文件", self.source)

    def test_quiz_authoring_is_scoped_and_returns_persisted_ids(self):
        source = (ROOT / "services" / "quiz_generator_kg.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("assessment.analysis_id = $analysis_id", source)
        self.assertIn("source_analysis_id=str(analysis.id)", source)
        self.assertIn("await draft.insert()", source)
        self.assertIn("QUIZ_GENERATION_BATCH_SIZE = 5", source)
        self.assertIn("改為逐評量目標重試", source)
        self.assertIn("generate_course_unit_bank", source)

    def test_quiz_provenance_comes_from_approved_target_metadata(self):
        source = (ROOT / "services" / "quiz_generator_kg.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def canonicalize_question_provenance", source)
        self.assertIn('"knowledge_point_ids": point_ids', source)
        self.assertIn('"source_chunk_ids": chunk_ids', source)
        self.assertIn("normalize_and_validate_batch", source)
        self.assertNotIn(
            'raise ValueError("LLM 回傳了評量目標以外的 knowledge_point_ids")',
            source,
        )

    def test_learning_weakness_requires_repeated_evidence(self):
        self.assertIn("MIN_TARGET_OBSERVATIONS = 2", self.source)
        self.assertIn('return "insufficient"', self.source)
        self.assertIn("單次答錯不會立即判定為學習弱點", self.source)

    def test_wrong_answers_still_generate_immediate_review_notes(self):
        self.assertIn("if self.learning_pp:", self.source)
        self.assertIn("正在生成本次錯題的複習筆記", self.source)
        self.assertIn("以下是依本次錯題產生的學習筆記", self.source)
        self.assertIn("if confirmed_weaknesses:", self.source)


if __name__ == "__main__":
    unittest.main()
