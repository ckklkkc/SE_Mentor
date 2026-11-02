import unittest

from services.slide_markdown import (
    NativeLine,
    _fill_missing_titles,
    _is_page_number_line,
    _join_items,
    _page_number_band,
    _repair_two_column_table,
    _table_markdown,
    _title_from_lines,
    _vector_figure_boxes,
)


class SlideMarkdownTests(unittest.TestCase):
    def test_page_number_uses_shared_vertical_band_not_left_position(self):
        pages = [
            [NativeLine(str(page), [21.5, 105.5, 34.5, 114.7])]
            for page in range(1, 38)
        ]
        pages[28].append(NativeLine("29", [134.5, 82.6, 147.5, 91.8]))
        pages[35] = [NativeLine("36", [134.5, 105.5, 147.6, 114.7])]
        pages[36] = [NativeLine("37", [134.5, 105.5, 147.5, 114.7])]
        band = _page_number_band(pages)
        self.assertTrue(_is_page_number_line(pages[35][0], 36, band))
        self.assertTrue(_is_page_number_line(pages[36][0], 37, band))
        self.assertTrue(_is_page_number_line(pages[28][1], 29, band))

    def test_dense_vector_composition_becomes_one_figure(self):
        class Page:
            width = 960
            height = 540
            page_number = 29
            curves = [
                {"x0": 166, "top": 130, "x1": 700, "bottom": 500}
                for _ in range(6)
            ]
            lines = []
            rects = []
            images = [
                {"x0": 726, "top": 249, "x1": 769, "bottom": 297}
            ]

        boxes = _vector_figure_boxes(
            Page(),
            [NativeLine("iterative", [90, 293, 153, 305])],
            (104, 116),
        )
        self.assertEqual(len(boxes), 1)
        self.assertLessEqual(boxes[0][0], 82)
        self.assertGreaterEqual(boxes[0][2], 777)

    def test_two_column_table_without_middle_rule_uses_header_alignment(self):
        class Row:
            def __init__(self, bbox):
                self.bbox = bbox

        class Table:
            bbox = (65, 145, 796, 463)
            rows = [Row((65, 145, 796, 183)), Row((65, 183, 796, 308))]

        class Crop:
            def __init__(self, bbox):
                self.bbox = bbox

            def extract_text(self, **kwargs):
                header = self.bbox[1] == 145
                left = self.bbox[2] < 500
                if header:
                    return "Question" if left else "Answer"
                return "What is software?" if left else "Computer programs."

        class Page:
            def extract_words(self):
                return [
                    {"text": "Question", "x0": 71, "x1": 147,
                     "top": 155, "bottom": 175},
                    {"text": "Answer", "x0": 384, "x1": 447,
                     "top": 155, "bottom": 175},
                ]

            def crop(self, bbox):
                return Crop(bbox)

        rows = _repair_two_column_table(
            Page(), Table(),
            [["Question", "Answer"], ["What is software? Computer programs.", None]],
        )
        self.assertEqual(rows[1], ["What is software?", "Computer programs."])

    def test_title_band_accepts_powerpoint_left_margin_just_below_fifty(self):
        title = NativeLine(
            "軟體的本質問題: 複雜性(Complexity)",
            [49.8926, 38.99, 758.03, 79.82],
        )
        value, boxes = _title_from_lines([title], 11)
        self.assertEqual(value, title.text)
        self.assertEqual(boxes, [title.bbox])

    def test_title_band_ignores_standalone_slide_number(self):
        title = NativeLine(
            "軟體開發步驟、文件與工具", [49.8926, 39.69, 500, 79.82]
        )
        slide_number = NativeLine("29", [134.47, 82.58, 148, 92.0])
        value, boxes = _title_from_lines([title, slide_number], 28)
        self.assertEqual(value, title.text)
        self.assertEqual(boxes, [title.bbox])

    def test_wrapped_lines_remain_one_bullet_and_indent_is_preserved(self):
        lines = [
            NativeLine("\uf0a8 使用者介面分析(User", [73, 135, 360, 161]),
            NativeLine("Interface Analysis)", [99, 170, 304, 194]),
            NativeLine("\uf0a4 使用者介面的說明與分", [102, 208, 361, 229]),
            NativeLine("析可以讓需求更清楚、明瞭。", [123, 236, 353, 259]),
            NativeLine("右側補充段落。", [650, 166, 870, 184]),
            NativeLine("範例：環保地圖", [419, 132, 595, 153]),
        ]
        self.assertEqual(_join_items(lines), [
            (0, "使用者介面分析(User Interface Analysis)"),
            (1, "使用者介面的說明與分析可以讓需求更清楚、明瞭。"),
            (0, "右側補充段落。"),
            (0, "範例：環保地圖"),
        ])

    def test_table_cell_lines_are_preserved_without_splitting_words(self):
        output = _table_markdown([
            ["Property", "Measure"],
            ["Speed", "Processed transactions/second\nUser/event response time"],
        ])
        self.assertIn(
            "| Speed | Processed transactions/second<br>User/event response time |",
            output,
        )

    def test_missing_native_title_uses_marker_title_between_adjacent_slides(self):
        headings = "\n".join([
            "# 線框圖(Wireframe)1",
            "# 線框圖(Wireframe)2",
            "# 線框圖繪製原則:",
            "# 線框圖(Wireframe)3",
        ])
        self.assertEqual(
            _fill_missing_titles(
                ["線框圖(Wireframe)1", None, "線框圖(Wireframe)3"], headings
            )[1],
            "線框圖(Wireframe)2",
        )


if __name__ == "__main__":
    unittest.main()
