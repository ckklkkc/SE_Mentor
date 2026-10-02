import unittest

from services.textbook_analyzer import split_textbook_markdown, stable_id


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

        self.assertTrue(textbook_id.startswith("book_"))
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

    def test_stable_ids_are_deterministic_and_normalized(self):
        self.assertEqual(
            stable_id("kp", "使用者  故事", " VALUE "),
            stable_id("kp", "使用者 故事", "value"),
        )


if __name__ == "__main__":
    unittest.main()
