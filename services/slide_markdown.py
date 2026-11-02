"""Rebuild landscape teaching slides page by page from the native PDF layer."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from services.pdf_visuals import PageLayout, _managed


PAGE_BREAK = "<!-- PDF_PAGE_BREAK -->"
FIGURE_PLACEHOLDER = "<!-- PDF_FIGURES_PAGE_{page} -->"
BULLETS = {"\uf0a8": 0, "\uf0a4": 1, "◼": 2, "▪": 2, "•": 1}


@dataclass
class NativeLine:
    text: str
    bbox: list[float]


@dataclass
class NativeTable:
    bbox: list[float]
    rows: list[list[str | None]]


def _native_lines(document, page_index: int) -> list[NativeLine]:
    """Read PDF text-stream lines with their original top-left coordinates."""
    with _managed(document[page_index]) as page:
        _, height = page.get_size()
        with _managed(page.get_textpage()) as textpage:
            result: list[NativeLine] = []
            current: list[tuple[str, tuple[float, float, float, float]]] = []
            for index in range(textpage.count_chars()):
                char = textpage.get_text_range(index, 1)
                if char in "\r\n":
                    if current:
                        boxes = [box for _, box in current]
                        result.append(NativeLine(
                            text="".join(value for value, _ in current).strip(),
                            bbox=[
                                min(box[0] for box in boxes),
                                height - max(box[3] for box in boxes),
                                max(box[2] for box in boxes),
                                height - min(box[1] for box in boxes),
                            ],
                        ))
                        current = []
                    continue
                try:
                    current.append((char, textpage.get_charbox(index)))
                except Exception:
                    continue
            if current:
                boxes = [box for _, box in current]
                result.append(NativeLine(
                    text="".join(value for value, _ in current).strip(),
                    bbox=[min(box[0] for box in boxes), height - max(box[3] for box in boxes),
                          max(box[2] for box in boxes), height - min(box[1] for box in boxes)],
                ))
            return [line for line in result if line.text]


def _inside(bbox: list[float], regions: list[list[float]], *, overlap: float = .45) -> bool:
    area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    for region in regions:
        intersection = max(0.0, min(bbox[2], region[2]) - max(bbox[0], region[0])) * max(
            0.0, min(bbox[3], region[3]) - max(bbox[1], region[1])
        )
        if intersection / area >= overlap:
            return True
    return False


def _clean_cell(value: str | None) -> str:
    if not value:
        return ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
    return "<br>".join(line.replace("|", "&#124;") for line in lines if line)


def _table_markdown(rows: list[list[str | None]]) -> str:
    width = max(len(row) for row in rows)
    clean = [[_clean_cell(cell) for cell in row] + [""] * (width - len(row)) for row in rows]
    if len(clean) == 1:
        clean.insert(0, [f"欄 {index + 1}" for index in range(width)])
    return "\n".join([
        "| " + " | ".join(clean[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
        *("| " + " | ".join(row) + " |" for row in clean[1:]),
    ])


def _repair_two_column_table(page, table, rows: list[list[str | None]]) -> list[list[str | None]]:
    """Split a visually two-column table even when its middle rule is absent."""
    if (
        len(rows) < 2
        or max((len(row) for row in rows), default=0) != 2
        or len(rows[0]) < 2
        or not rows[0][0]
        or not rows[0][1]
        or any(len(row) > 1 and bool(row[1] and row[1].strip()) for row in rows[1:])
        or len(getattr(table, "rows", [])) != len(rows)
    ):
        return rows

    header_box = table.rows[0].bbox
    second_header = re.sub(r"\s+", " ", rows[0][1]).strip().split()[0].lower()
    header_words = [
        word for word in page.extract_words()
        if word["top"] >= header_box[1] - 2
        and word["bottom"] <= header_box[3] + 2
    ]
    anchor = next(
        (word for word in header_words if word["text"].strip().lower() == second_header),
        None,
    )
    if anchor is None:
        return rows

    split_x = anchor["x0"] - 4
    if not table.bbox[0] + 40 < split_x < table.bbox[2] - 40:
        return rows

    repaired: list[list[str | None]] = []
    for row in table.rows:
        x0, top, x1, bottom = row.bbox
        left = page.crop((x0, top, split_x, bottom)).extract_text(
            x_tolerance=3, y_tolerance=3
        ) or ""
        right = page.crop((split_x, top, x1, bottom)).extract_text(
            x_tolerance=3, y_tolerance=3
        ) or ""
        repaired.append([left.strip(), right.strip()])
    if any(not left or not right for left, right in repaired):
        return rows
    return repaired


def _table_candidates(page) -> list[NativeTable]:
    """Accept actual multi-column grids, never the slide header/background."""
    tables: list[NativeTable] = []
    for table in page.find_tables():
        rows = table.extract()
        rows = _repair_two_column_table(page, table, rows)
        width = max((len(row) for row in rows), default=0)
        nonempty = sum(bool(cell and cell.strip()) for row in rows for cell in row)
        if len(rows) >= 2 and width >= 2 and nonempty >= 4:
            tables.append(NativeTable(list(table.bbox), rows))

    if tables:
        return tables

    # PowerPoint sometimes draws one bordered row with several multiline
    # columns (for example a two-column document outline). Detect its explicit
    # borders because the default table finder expects internal row lines.
    vertical = [line for line in page.lines
                if abs(line["x1"] - line["x0"]) < 1.5 and line["height"] > 80]
    horizontal = [line for line in page.lines
                  if abs(line["bottom"] - line["top"]) < 1.5 and line["width"] > 100]
    for top_line in horizontal:
        for bottom_line in horizontal:
            if bottom_line["top"] - top_line["top"] < 80:
                continue
            xs = sorted({round(line["x0"], 1) for line in vertical
                         if abs(line["top"] - top_line["top"]) < 3
                         and abs(line["bottom"] - bottom_line["top"]) < 3})
            if len(xs) < 3:
                continue
            rows = page.extract_table({
                "vertical_strategy": "explicit", "explicit_vertical_lines": xs,
                "horizontal_strategy": "explicit",
                "explicit_horizontal_lines": [top_line["top"], bottom_line["top"]],
            })
            if rows and len(rows[0]) >= 2:
                return [NativeTable([xs[0], top_line["top"], xs[-1], bottom_line["top"]], rows)]
    return []


def _image_boxes(page) -> list[list[float]]:
    boxes = []
    for image in page.images:
        bbox = [image["x0"], image["top"], image["x1"], image["bottom"]]
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        area = width * height
        # 這裡只排除無效物件與幾乎覆蓋整頁的掃描背景；圖片是否重要不在
        # PDF→Markdown 階段判斷。降低尺寸門檻，保留小型教材圖示與裝飾。
        if width >= 16 and height >= 16 and 256 <= area < page.width * page.height * .92:
            boxes.append(bbox)
    return boxes


def _page_number_band(pages: list[list[NativeLine]]) -> tuple[float, float] | None:
    """Learn the shared horizontal band used by slide numbers in this deck."""
    centers = []
    for index, lines in enumerate(pages):
        number = str(index + 1)
        for line in lines:
            height = line.bbox[3] - line.bbox[1]
            if (
                line.text.strip() == number
                and height <= 24
                and line.bbox[1] < 160
            ):
                centers.append((line.bbox[1] + line.bbox[3]) / 2)
    if not centers:
        return None
    center = median(centers)
    deviation = median(abs(value - center) for value in centers)
    tolerance = max(6.0, min(18.0, deviation * 4 + 2))
    return center - tolerance, center + tolerance


def _is_page_number_line(
    line: NativeLine,
    page_number: int,
    page_number_band: tuple[float, float] | None,
) -> bool:
    if line.text.strip() != str(page_number):
        return False
    center = (line.bbox[1] + line.bbox[3]) / 2
    if page_number_band is not None:
        if page_number_band[0] <= center <= page_number_band[1]:
            return True
        # Some exported slides contain a second hidden copy of the page number
        # in the same header decoration (page 29). It is still safely
        # identifiable by exact page value, small size and top-band position.
        return line.bbox[1] < 125 and line.bbox[3] - line.bbox[1] <= 24
    return line.bbox[0] < 60


def _vector_figure_boxes(
    page,
    lines: list[NativeLine],
    page_number_band: tuple[float, float] | None,
) -> list[list[float]]:
    """Group a dense vector composition into one figure instead of tiny icons."""
    curves = []
    for item in page.curves:
        bbox = [item["x0"], item["top"], item["x1"], item["bottom"]]
        area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
        if bbox[1] >= 115 and area >= 24:
            curves.append(bbox)
    if len(curves) < 6:
        return []

    graphics = list(curves)
    for collection, minimum_area in (
        (page.lines, 24), (page.images, 24), (page.rects, 100)
    ):
        for item in collection:
            bbox = [item["x0"], item["top"], item["x1"], item["bottom"]]
            area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            if bbox[1] >= 115 and area >= minimum_area:
                graphics.append(bbox)
    x0 = min(box[0] for box in graphics)
    top = min(box[1] for box in graphics)
    x1 = max(box[2] for box in graphics)
    bottom = max(box[3] for box in graphics)
    area_ratio = (x1 - x0) * (bottom - top) / (page.width * page.height)
    if area_ratio < .08 or x1 - x0 < page.width * .25:
        return []

    # Include labels that belong to the composition, such as the left-side
    # "iterative" label on page 29, without absorbing the slide title/footer.
    base_x0, base_top, base_x1, base_bottom = x0, top, x1, bottom
    for line in lines:
        if (
            line.bbox[1] < 115
            or line.bbox[1] >= page.height * .9
            or _is_page_number_line(
                line, page.page_number, page_number_band
            )
        ):
            continue
        vertical_gap = max(
            base_top - line.bbox[3], line.bbox[1] - base_bottom, 0
        )
        horizontal_gap = max(
            base_x0 - line.bbox[2], line.bbox[0] - base_x1, 0
        )
        if vertical_gap <= 32 and horizontal_gap <= 120:
            x0 = min(x0, line.bbox[0])
            top = min(top, line.bbox[1])
            x1 = max(x1, line.bbox[2])
            bottom = max(bottom, line.bbox[3])

    pad = 8
    bottom_pad = 2
    return [[
        max(0, x0 - pad), max(115, top - pad),
        min(page.width, x1 + pad), min(page.height, bottom + bottom_pad),
    ]]


def _title_from_lines(lines: list[NativeLine], page_index: int) -> tuple[str | None, list[list[float]]]:
    if page_index == 0:
        visible = [line for line in lines if not line.text.isdigit()]
        if len(visible) >= 3:
            title = f"{visible[0].text.rstrip(':：')}：{visible[1].text} {visible[2].text}"
            return title, [line.bbox for line in visible[:3]]
    # PowerPoint theme variants in the same course place the actual title at
    # x=49.89, while other slides start at x=58–90.  The old strict x>50
    # condition therefore treated many real titles as missing and fell back to
    # Marker's repeated course header.  Accept the title band with a small
    # left-margin tolerance, but never promote a standalone slide number (page
    # 29 in the sample deck places its number inside this vertical band).
    title_lines = [
        line for line in lines
        if line.bbox[1] < 95
        and line.bbox[0] >= 35
        and not re.fullmatch(r"\s*\d+\s*", line.text)
    ]
    if title_lines:
        return " ".join(line.text for line in title_lines), [line.bbox for line in title_lines]
    return None, []


def _marker_headings(markdown: str) -> list[str]:
    return [match.group(1).strip().strip("*") for match in re.finditer(
        r"(?m)^#{1,6}\s+(.+?)\s*$", markdown
    ) if not re.fullmatch(r"\**\d+\**", match.group(1).strip())]


def _norm(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.lower(), flags=re.UNICODE)


def _fill_missing_titles(titles: list[str | None], marker_markdown: str) -> list[str]:
    headings = _marker_headings(marker_markdown)
    heading_norms = [_norm(value) for value in headings]
    output = list(titles)
    for index, title in enumerate(titles):
        if title is not None:
            continue
        previous = next((value for value in reversed(titles[:index]) if value), None)
        following = next((value for value in titles[index + 1:] if value), None)
        start = 0
        if previous:
            matches = [i for i, value in enumerate(heading_norms) if _norm(previous) == value]
            if matches:
                start = matches[-1] + 1
        end = len(headings)
        if following:
            match = next((i for i in range(start, len(headings))
                          if _norm(following) == heading_norms[i]), None)
            if match is not None:
                end = match
        output[index] = headings[start] if start < end else f"第 {index + 1} 頁"
    return [value or f"第 {index + 1} 頁" for index, value in enumerate(output)]


def _join_items(lines: list[NativeLine]) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    last_bbox: list[float] | None = None
    for line in lines:
        text = re.sub(r"\s+", " ", line.text).strip()
        symbol = text[0] if text else ""
        if symbol in BULLETS:
            items.append((BULLETS[symbol], text[1:].strip()))
        elif items and last_bbox is not None:
            horizontal_jump = abs(line.bbox[0] - last_bbox[0]) > 230
            vertical_gap = line.bbox[1] - last_bbox[3] > 18
            backwards = line.bbox[1] < last_bbox[1] - 20
            if horizontal_jump or vertical_gap or backwards:
                items.append((0, text))
            else:
                depth, previous = items[-1]
                spacer = " " if re.search(r"[A-Za-z0-9)]$", previous) and re.match(r"[A-Za-z0-9(]", text) else ""
                items[-1] = (depth, previous + spacer + text)
        else:
            items.append((0, text))
        last_bbox = line.bbox
    return [(depth, text) for depth, text in items if text]


def _page_markdown(
    page,
    lines: list[NativeLine],
    title: str,
    title_boxes: list[list[float]],
    layout: PageLayout,
    page_number_band: tuple[float, float] | None,
) -> tuple[str, dict[str, Any]]:
    tables = _table_candidates(page)
    image_boxes = _image_boxes(page)
    vector_boxes = _vector_figure_boxes(page, lines, page_number_band)
    layout_boxes = [[value / 1000 * (page.width if i % 2 == 0 else page.height)
                     for i, value in enumerate(region.bbox)] for region in layout.regions]
    excluded = (
        title_boxes + [table.bbox for table in tables]
        + image_boxes + vector_boxes + layout_boxes
    )

    hyperlinks = []
    link_boxes = []
    for link in page.hyperlinks:
        uri = link.get("uri")
        if not uri:
            continue
        bbox = [link["x0"], link["top"], link["x1"], link["bottom"]]
        hyperlinks.append(uri)
        link_boxes.append(bbox)

    content_lines = []
    page_number_removed = False
    for line in lines:
        if _is_page_number_line(line, page.page_number, page_number_band):
            page_number_removed = True
            continue
        if _inside(line.bbox, excluded) or _inside(line.bbox, link_boxes, overlap=.2):
            continue
        if "http://" in line.text or "https://" in line.text:
            continue
        content_lines.append(line)

    parts = [f"# {title}"]
    items = _join_items(content_lines)
    if items:
        parts.append("\n".join(f"{'  ' * depth}- {text}" for depth, text in items))
    parts.extend(_table_markdown(table.rows) for table in tables)
    parts.append(FIGURE_PLACEHOLDER.format(page=page.page_number))
    if hyperlinks:
        parts.append("\n".join(f"- <{uri}>" for uri in dict.fromkeys(hyperlinks)))
    return "\n\n".join(parts), {
        "page": page.page_number,
        "title": title,
        "bullet_count": len(items),
        "table_count": len(tables),
        "image_count": len(image_boxes),
        "vector_count": len(vector_boxes),
        "page_number_removed": page_number_removed,
        "image_regions": [
            [
                max(0, min(1000, round(box[0] / page.width * 1000))),
                max(0, min(1000, round(box[1] / page.height * 1000))),
                max(0, min(1000, round(box[2] / page.width * 1000))),
                max(0, min(1000, round(box[3] / page.height * 1000))),
            ]
            for box in image_boxes
        ],
        "vector_regions": [
            [
                max(0, min(1000, round(box[0] / page.width * 1000))),
                max(0, min(1000, round(box[1] / page.height * 1000))),
                max(0, min(1000, round(box[2] / page.width * 1000))),
                max(0, min(1000, round(box[3] / page.height * 1000))),
            ]
            for box in vector_boxes
        ],
        "url_count": len(set(hyperlinks)),
    }


def rebuild_slide_markdown(
    pdf_path: str | Path,
    marker_markdown: str,
    layouts: dict[int, PageLayout],
) -> tuple[str, list[int], dict[str, Any]]:
    """Create one deterministic Markdown section per physical slide."""
    import pdfplumber
    import pypdfium2 as pdfium

    with pdfplumber.open(pdf_path) as source, _managed(pdfium.PdfDocument(str(pdf_path))) as document:
        if not source.pages or sum(page.width > page.height * 1.25 for page in source.pages) < len(source.pages) * .8:
            raise ValueError("不是橫式投影片教材，不能使用逐頁投影片重建")
        native = [_native_lines(document, index) for index in range(len(document))]
        page_number_band = _page_number_band(native)
        title_results = [_title_from_lines(lines, index) for index, lines in enumerate(native)]
        titles = _fill_missing_titles([result[0] for result in title_results], marker_markdown)

        output = ""
        offsets = []
        records = []
        for index, (page, lines, title, (_, title_boxes)) in enumerate(
            zip(source.pages, native, titles, title_results)
        ):
            if index:
                offsets.append(len(output))
                output += f"\n\n{PAGE_BREAK}\n\n"
            else:
                offsets.append(0)
            markdown, record = _page_markdown(
                page, lines, title, title_boxes, layouts[index + 1], page_number_band
            )
            output += markdown
            records.append(record)
        return output.strip() + "\n", offsets, {
            "strategy": "native_slide_page_rebuild",
            "page_count": len(records),
            "pages": records,
            "bullet_count": sum(record["bullet_count"] for record in records),
            "table_count": sum(record["table_count"] for record in records),
            "image_count": sum(record["image_count"] for record in records),
            "vector_count": sum(record["vector_count"] for record in records),
            "page_number_band": list(page_number_band) if page_number_band else None,
            "url_count": sum(record["url_count"] for record in records),
        }
