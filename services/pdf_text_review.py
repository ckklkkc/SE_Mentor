"""Ground figure transcription in the source crop and its native PDF text."""
from __future__ import annotations

import json
from services.image_captioning import ImageDescription
from services.pdf_visuals import _managed


def region_text(document, page_index, bbox):
    with _managed(document[page_index]) as page:
        width, height = page.get_size()
        x1, y1, x2, y2 = bbox
        with _managed(page.get_textpage()) as textpage:
            return textpage.get_text_bounded(
                left=x1 * width / 1000, bottom=(1000-y2) * height / 1000,
                right=x2 * width / 1000, top=(1000-y1) * height / 1000,
            )


def verify_figure(analyzer, path, draft, document, page_index, bbox):
    native = region_text(document, page_index, bbox)
    prompt = f"""請重新核對這張原始圖片及先前的轉述，這是獨立的逐字核對步驟。
不要改畫圖片，不要用語言習慣修正教材原字。visible_text 僅保留確實可讀的字串，
英文原字、專案代號、需求編號、數字及單位必須逐字一致，不要翻譯或補字。
無法辨認的字移出 visible_text，寫入 uncertainty，絕不依上下文猜測。
依校正結果同步刪除 description、relationships、teaching_points、context_connection、
document_facts 中不受原圖支持的推論。confidence 為 0～1；有任何字、箭頭方向、
元素對應或裁切疑問時必須降低，並在 uncertainty 說明。
PDF 局部文字層可能有字型編碼問題或不含點陣圖文字，須與圖片交叉核對，不可盲目抄錄。
只回傳 JSON：{{"image_type":"圖例類型","visible_text":[],"description":"經核對的圖意",
"relationships":[],"teaching_points":[],"context_connection":"與本文的連接",
"document_facts":[],"confidence":0.9,"uncertainty":null}}。
局部 PDF 文字層：{native[:5000]}
待核對草稿：{json.dumps(draft.model_dump(), ensure_ascii=False)}"""
    verified = ImageDescription.model_validate(analyzer.query_image_json(path, prompt, max_tokens=2600))
    if not verified.description.strip():
        raise ValueError("圖片核對未回傳圖意，不採用未核對的草稿")
    return verified
