# SE Mentor

## Setup

### 1. 安裝依賴

```
docker compose up

uv sync

.venv\Scripts\activate
```

### 2. 設定 config.py

```
copy config_example.py config.py
```

編輯 `config.py`，填入以下設定：

```python!
# API keys
OPENAI_API_KEY = "<openai-api-key>"
GEMINI_API_KEY = "<gemini-api-key>"
CLAUDE_API_KEY = "<claude-api-key>"
LANGCHAIN_API_KEY = "<langchain-api-key>"

# dc 機器人的 token
DISCORD_TOKEN = "<discord-bot-token>"

# Neo4j 資料庫連線密碼
NEO4J_PASSWORD = "<your-neo4j-password>"
```

## 使用說明

### dc_chatbot.py

跑機器人，直接執行即可。

```
uv run dc_chatbot.py
```

#### 教材診斷題工作流

新版教材流程以「可評量知識點」而不是大範圍 Entity 作為弱點單位：

1. 教師用 `/upload_textbook` 上傳 PDF 或 Markdown。
2. 系統保留章節階層與來源 chunk，抽取細粒度知識點、受控知識關係、評量目標與常見迷思。
3. 教師用 `/inspect_textbook` 檢視內容，再用 `/review_textbook` 核准整批分析；核准前的 Neo4j 節點不會成為出題依據。
4. 教師用 `/generate_quiz_drafts` 依核准評量目標產生四選一題目草稿。
5. 教師用 `/inspect_quiz` 檢查題幹、選項、解析與迷思映射，再用 `/review_quiz` 逐題核准或退回。
6. 學生用 `/quiz` 作答時只會取得已核准題目。系統保存每題的評量目標、選項迷思與教材來源，
   因此弱點可顯示為「評量目標：具體錯誤推理」，而非只顯示章節大概念。

教師權限目前為 `developers` 清單中的帳號，或具有 Discord `Manage Server` 權限的成員。

現有題庫若沒有 `review_status`，不會直接提供給學生；請重新生成並審核，或先完成資料遷移。

### services/kg_constructor.py

讀取檔案並建立知識圖譜。

#### 教材

```
uv run services/kg_constructor.py --textbook <file_path>
```

#### 開發文件

```
uv run services/kg_constructor.py <file_path> <doc_type> <group> <uploader>
```

- 'doc_type' 必須是 SRD、SDD、STD 其中一個

範例：

```
uv run services/kg_constructor.py files\test_file.md SRD 測試組 dev
```
