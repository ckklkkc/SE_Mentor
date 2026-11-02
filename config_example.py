OPENAI_API_KEY = "<openai-api-key>"
GEMINI_API_KEY = "<gemini-api-key>"
CLAUDE_API_KEY = "<claude-api-key>"

LANGCHAIN_API_KEY = "<langchain-api-key>"

DISCORD_TOKEN = "<discord-bot-token>"

# 可選：開發／測試伺服器 ID。正式環境或不需要 guild 即時同步時保持 None。
TEST_GUILD_ID = None

NEO4J_PASSWORD = "<your-neo4j-password>"

# 啟用後：以 PDF 頁級視覺判讀篩選教材圖例；停用時只保存 Marker 原圖。
IMAGE_DESCRIPTION_ENABLED = True
IMAGE_DESCRIPTION_PROVIDER = "openai"  # openai 或 anthropic
IMAGE_DESCRIPTION_MODEL = "gpt-4o-mini"  # anthropic 可改為 claude-sonnet-4-6
IMAGE_DESCRIPTION_MAX_IMAGES = 100  # 逐頁掃描不受限；最多轉述的圖例數
IMAGE_DESCRIPTION_MIN_WIDTH = 64
IMAGE_DESCRIPTION_MIN_HEIGHT = 64
IMAGE_DESCRIPTION_CONTEXT_CHARS = 1200
IMAGE_DESCRIPTION_MIN_CONFIDENCE = 0.75
IMAGE_DESCRIPTION_TIMEOUT_SECONDS = 90
# 單張圖若遇到暫時性 API／JSON／欄位格式問題，採 1.5、3 秒指數退避重試。
IMAGE_DESCRIPTION_MAX_ATTEMPTS = 3
IMAGE_DESCRIPTION_RETRY_BASE_SECONDS = 1.5
# False：單張圖說失敗時保留原圖、記錄錯誤並繼續；True：停止整次匯入。
IMAGE_DESCRIPTION_FAIL_ON_ERROR = False

#test 11111

# PDF 解析器：docling 模式仍以 Marker 產生文字／標題／表格，只借用圖片定位。
PDF_CONVERTER_BACKEND = "docling"
# None 自動使用專案 .venv-docling 的 Python。Windows 自訂路徑請用 r"C:\...\python.exe"。
PDF_DOCLING_PYTHON = None

# /upload_document（SRD／SDD／STD）採更嚴格的預設：保留 Marker 文字結構、
# 以 Docling/PDF 幾何定位圖片，再由多模態模型轉述。圖片失敗或低信心時，
# 先停止建立 KG，避免把內容不完整的文件當成完整輸入。
PROJECT_DOCUMENT_DESCRIBE_IMAGES = True
PROJECT_DOCUMENT_PDF_BACKEND = "docling"
PROJECT_DOCUMENT_REQUIRE_VERIFIED_IMAGES = True

# 學生只能上傳教師事先核准的 HackMD。鍵必須與 StudentProfile.group 完全相同；
# 網址會先移除 ?view、/edit、/download 等顯示差異後，再以完整筆記網址比對。
# 安全預設為 True：若組別或文件類型沒有登記，/upload_document 會在任何 LLM
# 或圖片描述呼叫前拒絕，避免用無關文件消耗課程 Token。
PROJECT_DOCUMENT_REQUIRE_SOURCE_ALLOWLIST = True
PROJECT_DOCUMENT_ALLOWED_SOURCES = {
    "第二組": {
        "SRD": ["https://hackmd.io/@se-project-team02/replace-with-srd-note-id"],
        "SDD": ["https://hackmd.io/@se-project-team02/replace-with-sdd-note-id"],
        "STD": ["https://hackmd.io/@se-project-team02/replace-with-std-note-id"],
    },
}
