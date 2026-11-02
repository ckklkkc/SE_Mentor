"""Preserve table spans separately from the rectangular Markdown used for retrieval."""
from __future__ import annotations

import html
import json
import copy
import re
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


def restore_native_linebreaks(document, page_index, data):
    """Restore cell line breaks only when all non-whitespace characters agree."""
    from services.pdf_visuals import _managed
    result = copy.deepcopy(data)
    with _managed(document[page_index]) as page:
        _, height = page.get_size()
        with _managed(page.get_textpage()) as textpage:
            for cell in result['table_cells']:
                box = cell.get('bbox')
                if not box or not cell.get('text'):
                    continue
                if box.get('coord_origin') == 'BOTTOMLEFT':
                    bottom,top=box['b'],box['t']
                else:
                    bottom,top=height-box['b'],height-box['t']
                native=textpage.get_text_bounded(left=box['l']-1,bottom=bottom-1,
                                                right=box['r']+1,top=top+1).strip()
                if native and re.sub(r'\s+','',native)==re.sub(r'\s+','',cell['text']):
                    cell['text']=native.replace('\r\n','\n')
    return result


class TableCell(BaseModel):
    text: str
    start_row_offset_idx: int = Field(ge=0)
    end_row_offset_idx: int = Field(ge=1)
    start_col_offset_idx: int = Field(ge=0)
    end_col_offset_idx: int = Field(ge=1)
    column_header: bool = False


class TableGrid(BaseModel):
    num_rows: int = Field(ge=1, le=300)
    num_cols: int = Field(ge=1, le=40)
    table_cells: list[TableCell]

    @model_validator(mode="after")
    def check_cells(self):
        occupied = set()
        for cell in self.table_cells:
            r1, r2 = cell.start_row_offset_idx, cell.end_row_offset_idx
            c1, c2 = cell.start_col_offset_idx, cell.end_col_offset_idx
            if not (r1 < r2 <= self.num_rows and c1 < c2 <= self.num_cols):
                raise ValueError("表格儲存格超出範圍或方向錯誤")
            for r in range(r1, r2):
                for c in range(c1, c2):
                    if (r, c) in occupied:
                        raise ValueError("表格合併儲存格重疊")
                    occupied.add((r, c))
        if not any(cell.text.strip() for cell in self.table_cells):
            raise ValueError("空白表格不能視為完成辨識")
        return self


def render_table(data: dict):
    table = TableGrid.model_validate(data)
    grid = [[""] * table.num_cols for _ in range(table.num_rows)]
    anchors, occupied = {}, set()
    for cell in table.table_cells:
        r1, r2 = cell.start_row_offset_idx, cell.end_row_offset_idx
        c1, c2 = cell.start_col_offset_idx, cell.end_col_offset_idx
        anchors[r1, c1] = cell
        for r in range(r1, r2):
            for c in range(c1, c2):
                occupied.add((r, c))
                # Repeat a row-spanning label for row-wise retrieval, but do
                # not turn a section spanning 3 columns into 3 duplicate fields.
                grid[r][c] = cell.text if c == c1 else ""
    # Only genuine single-column headers form Markdown headers. A spanning
    # section label such as 設定(Setting) must remain a data row, not 3 duplicate keys.
    first = [c for c in table.table_cells if c.start_row_offset_idx == 0]
    use_header = bool(first) and len(first) == table.num_cols and all(
        c.column_header and c.end_col_offset_idx-c.start_col_offset_idx == 1
        and c.end_row_offset_idx == 1 for c in first
    ) and len(set(grid[0])) == table.num_cols and all(grid[0])
    headers = grid[0] if use_header else [f"欄 {i+1}" for i in range(table.num_cols)]
    def md(value):
        return html.escape(value, quote=False).replace("|", "&#124;").replace("\n", "<br> ")
    def row(values):
        return "| " + " | ".join(md(v) for v in values) + " |"
    markdown = "\n".join([row(headers), row(["---"]*table.num_cols)] +
                         [row(r) for r in grid[int(use_header):]])
    html_rows = []
    for r in range(table.num_rows):
        cells = []
        for c in range(table.num_cols):
            cell = anchors.get((r, c))
            if cell:
                tag = "th" if cell.column_header else "td"
                cells.append(f'<{tag} rowspan="{cell.end_row_offset_idx-r}" '
                             f'colspan="{cell.end_col_offset_idx-c}">'
                             + html.escape(cell.text).replace("\n", "<br>") + f"</{tag}>")
            elif (r, c) not in occupied:
                cells.append("<td></td>")
        html_rows.append("<tr>"+"".join(cells)+"</tr>")
    html_table = '<table border="1" style="border-collapse:collapse">'+"\n".join(html_rows)+"</table>"
    return markdown, html_table


def save_table(data, assets: Path, prefix: str, name: str):
    markdown, table_html = render_table(data)
    (assets/f"{name}.html").write_text(
        '<!doctype html><html lang="zh-Hant"><meta charset="utf-8">'
        '<title>教材表格</title><body>'+table_html+'</body></html>', encoding="utf-8")
    (assets/f"{name}.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    return markdown + f"\n\n[檢視合併儲存格表格]({prefix}/{name}.html)"


def transcribe_table(analyzer, image_path, native_text):
    prompt = f"""只轉寫這張局部教材表格，不解釋教學意義，也不補寫表頭。
辨認行、欄、合併儲存格。文字逐字保留原繁體中文、英文、需求編號與數值；
同格換行用 \\n，不把多項指標接成一個詞。看不清楚回傳 complete=false 並說明 uncertainty。
非表格也回傳 complete=false。cell 的起訖座標由0起算，end不包含，合併格只出現一次。
只回傳 JSON：{{"complete":true,"uncertainty":null,"num_rows":2,"num_cols":2,
"table_cells":[{{"text":"原字","start_row_offset_idx":0,"end_row_offset_idx":1,
"start_col_offset_idx":0,"end_col_offset_idx":2,"column_header":true}}]}}。
局部 PDF 文字層供核對（可能不含圖中文字）：{native_text[:7000]}"""
    draft = analyzer.query_image_json(image_path, prompt, max_tokens=6500)
    if draft.get("complete") is not True or draft.get("uncertainty"):
        raise ValueError("表格仍有無法辨認內容，保留原結果："+str(draft.get("uncertainty", "未確認為表格")))
    table = TableGrid.model_validate(draft)
    verification = analyzer.query_image_json(image_path,
        "獨立核對圖片與下面的表格 JSON。逐格檢查有無漏行、漏欄、合併位置錯誤、"
        "新增原圖沒有的文字，尤其英文/需求代號/數字/單位。不能只確認 JSON 合法。"
        "任一字無法確認即 approved=false。只回傳 {\"approved\":true,\"issues\":[]}。\n"
        + table.model_dump_json(), max_tokens=1200)
    if verification.get("approved") is not True or verification.get("issues"):
        raise ValueError("表格核對未通過："+str(verification.get("issues", [])))
    return table.model_dump()
