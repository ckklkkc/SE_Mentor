import unittest

from database.mongo_controller import (
    AssessmentTargetCandidate,
    KnowledgePointCandidate,
    SectionAnalysis,
    SourceEvidence,
    TextbookAnalysis,
)
from services.textbook_analyzer import (
    ConsolidationPlan,
    RawMergeGroup,
    classify_section,
    consolidate_course_unit,
    course_unit_title_from_source,
    split_textbook_markdown,
    stable_id,
)


class TextbookSplitterTests(unittest.TestCase):
    def test_keeps_subsections_under_chapter_heading(self):
        markdown = """# 軟體工程
## 使用者故事
### 定義
使用者故事以非正式語言描述使用者價值。
### 驗收條件
驗收條件用來確認故事何時完成。
## 敏捷方法
Scrum 透過短週期迭代交付。
"""

        textbook_id, chapters = split_textbook_markdown(
            markdown,
            "sample.md",
            max_chunk_chars=80,
        )

        self.assertTrue(textbook_id.startswith("course_unit_"))
        self.assertEqual(
            [title for _, title, _ in chapters],
            ["使用者故事", "敏捷方法"],
        )
        first_chapter_chunks = chapters[0][2]
        self.assertEqual(len(first_chapter_chunks), 2)
        self.assertEqual(
            first_chapter_chunks[0].heading_path,
            ["軟體工程", "使用者故事", "定義"],
        )

    def test_excludes_non_knowledge_sections_before_llm_analysis(self):
        for title in ["封面", "目錄", "習題 (Exercises)", "參考文獻", "Any Question?"]:
            with self.subTest(title=title):
                self.assertFalse(classify_section(title).included_in_assessment)
        self.assertTrue(classify_section("使用者需求與系統需求").included_in_assessment)

    def test_archived_filename_becomes_course_unit_title(self):
        self.assertEqual(
            course_unit_title_from_source(
                "20261005_144146_155533_[02]基礎需求工程.md"
            ),
            "[02]基礎需求工程",
        )

    def test_consolidates_duplicate_points_and_targets_across_sections(self):
        sections = [
            SectionAnalysis(
                id="s1", title="概論", order=1,
                knowledge_point_ids=["p1"], assessment_target_ids=["t1"],
            ),
            SectionAnalysis(
                id="s2", title="詳細說明", order=2,
                knowledge_point_ids=["p2"], assessment_target_ids=["t2"],
            ),
        ]
        points = [
            KnowledgePointCandidate(
                id="p1", name="使用者需求層次", description="高階描述",
                section_ids=["s1"],
                evidence=[SourceEvidence(chunk_id="c1", quote="高階", heading_path=[])],
            ),
            KnowledgePointCandidate(
                id="p2", name="使用者需求的抽象層次",
                description="使用者需求是面向企業目的的高階描述",
                section_ids=["s2"],
                evidence=[SourceEvidence(chunk_id="c2", quote="企業目的", heading_path=[])],
            ),
        ]
        targets = [
            AssessmentTargetCandidate(
                id="t1", section_ids=["s1"], knowledge_point_ids=["p1"],
                title="區分需求層次", objective="判斷使用者需求層次",
                discrimination="能否辨認高階描述",
                common_misconceptions=["詳細就是使用者需求", "兩者沒有差別"],
                evidence_chunk_ids=["c1"],
            ),
            AssessmentTargetCandidate(
                id="t2", section_ids=["s2"], knowledge_point_ids=["p2"],
                title="辨識使用者需求層次", objective="依抽象程度判斷需求層次",
                discrimination="能否區分使用者與系統需求",
                common_misconceptions=["有技術名詞才是需求", "兩者沒有差別"],
                evidence_chunk_ids=["c2"],
            ),
        ]
        plan = ConsolidationPlan(
            knowledge_point_groups=[RawMergeGroup(member_ids=["p1", "p2"], rationale="同義")],
            assessment_target_groups=[RawMergeGroup(member_ids=["t1", "t2"], rationale="同能力")],
        )

        merged_sections, merged_points, _, merged_targets = consolidate_course_unit(
            "unit", sections, points, [], targets, plan
        )

        self.assertEqual(len(merged_points), 1)
        self.assertEqual(set(merged_points[0].section_ids), {"s1", "s2"})
        self.assertEqual({item.chunk_id for item in merged_points[0].evidence}, {"c1", "c2"})
        self.assertEqual(len(merged_targets), 1)
        self.assertEqual(
            merged_sections[0].assessment_target_ids,
            merged_sections[1].assessment_target_ids,
        )

    def test_legacy_chapter_document_is_read_as_course_unit_sections(self):
        analysis = TextbookAnalysis.model_validate({
            "textbook_id": "legacy-book",
            "title": "[02]基礎需求工程",
            "source_file": "legacy.pdf",
            "uploader": "teacher",
            "review_status": "approved",
            "chapters": [{
                "id": "old-chapter",
                "title": "需求分類",
                "order": 1,
                "knowledge_points": [{
                    "id": "old-point",
                    "name": "功能需求",
                    "description": "系統做什麼",
                    "chapter_id": "old-chapter",
                    "evidence": [],
                }],
                "relations": [],
                "assessment_targets": [{
                    "id": "old-target",
                    "chapter_id": "old-chapter",
                    "knowledge_point_ids": ["old-point"],
                    "title": "區分需求",
                    "objective": "能區分需求",
                    "discrimination": "是否混淆",
                    "common_misconceptions": ["混淆功能與品質", "只看關鍵字"],
                    "evidence_chunk_ids": ["old-chunk"],
                }],
            }],
        })

        self.assertEqual(analysis.course_unit_id, "legacy-book")
        self.assertEqual(analysis.sections[0].id, "old-chapter")
        self.assertEqual(analysis.knowledge_points[0].section_ids, ["old-chapter"])
        self.assertEqual(analysis.assessment_targets[0].review_status, "approved")

    def test_stable_ids_are_deterministic_and_normalized(self):
        self.assertEqual(
            stable_id("kp", "使用者  故事", " VALUE "),
            stable_id("kp", "使用者 故事", "value"),
        )


if __name__ == "__main__":
    unittest.main()
