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

# 可選：測試伺服器 ID；Bot 必須已加入該伺服器
TEST_GUILD_ID = None

# Neo4j 資料庫連線密碼
NEO4J_PASSWORD = "<your-neo4j-password>"

# PDF 圖片處理（PDF 頁級圖例判讀＋Marker Markdown）
IMAGE_DESCRIPTION_ENABLED = True
IMAGE_DESCRIPTION_PROVIDER = "openai"  # openai 或 anthropic
IMAGE_DESCRIPTION_MODEL = "gpt-4o-mini"
IMAGE_DESCRIPTION_MAX_IMAGES = 100  # 逐頁掃描不受限；最多轉述的圖例數
IMAGE_DESCRIPTION_MIN_WIDTH = 64
IMAGE_DESCRIPTION_MIN_HEIGHT = 64
IMAGE_DESCRIPTION_CONTEXT_CHARS = 1200
IMAGE_DESCRIPTION_TIMEOUT_SECONDS = 90
IMAGE_DESCRIPTION_MAX_ATTEMPTS = 3
IMAGE_DESCRIPTION_RETRY_BASE_SECONDS = 1.5
IMAGE_DESCRIPTION_FAIL_ON_ERROR = False
```

## 使用說明

### Discord 教學 Bot

Discord runtime 負責學生上課期間會使用的功能，以及學生自行提交的公開 HackMD
專案文件：

- `/course_qa`
- `/project_qa`
- `/quiz`
- `/learning_profile`
- `/upload_document`

教材上傳、PDF 轉換、題庫生成與教師審核仍透過離線 CLI 完成。只有學生專案的
HackMD 文件可從 Discord 匯入；重型轉換模組會在使用指令後才延遲載入，且同時
最多處理兩份文件，避免 Bot 啟動或一般問答被轉換工作阻塞。

| 執行程序 | 負責內容 | 不負責內容 |
| --- | --- | --- |
| `dc_chatbot.py` | 課程／專案問答、HackMD 專案文件匯入、已審題庫作答、學習歷程 | PDF 教材轉檔、LLM 出題、教師審核 |
| `authoring_cli.py` | 教材與專案文件匯入、KG 建構、題庫草稿、教師審核 | Discord 互動與學生作答 |

因此 Discord Bot 可獨立重啟；耗時的教材處理失敗也不會卡住 Discord Interaction。

```
uv run dc_chatbot.py
```

正式使用不必設定 `TEST_GUILD_ID`，Bot 會同步全域指令。若希望開發期間讓斜線指令
立即出現在單一測試伺服器，請開啟 Discord 開發者模式、複製該伺服器 ID，並設定：

```python
TEST_GUILD_ID = 123456789012345678
```

同時必須在 Discord Developer Portal 的 OAuth2 URL Generator 勾選 `bot` 與
`applications.commands`，再用產生的網址將 Bot 加入該伺服器。ID 填錯或 Bot
無法存取時，程式會略過 guild 同步並繼續啟動，不再因 `50001 Missing Access` 中止。

### 離線教材與題庫製作

新版教材流程以 `CourseUnit → Section → 細粒度知識點／評量目標` 為主：

1. 一份如 `[02]基礎需求工程.pdf` 的檔案是一個 `CourseUnit`；PDF 內標題切片稱為 `Section`。
2. 封面、目錄、習題、參考資料、`Any Question?` 等 Section 在 LLM 分析前排除，並保存排除原因。
3. 對知識 Section 抽取可判斷對錯的命題、受控關係、評量目標與教材證據；不再強迫每個 Section 湊 5～8 個目標。
4. 第二階段在整個 CourseUnit 範圍合併語意重複的知識點與評量目標，保留所有來源 Section 與證據。
5. 教師可個別退回不適當的評量目標，再核准整個 CourseUnit。
6. 每個核准目標建立 2～3 題平行題，題目經教師逐題核准後才進入學生題庫。
7. `/quiz` 每次抽 5～10 個不同評量目標；單次答錯會立即提供錯題複習筆記並記錄證據，至少累積兩次才判定弱點。

先啟動 MongoDB 與 Neo4j：

```bash
docker compose up -d
```

匯入教材：

```bash
uv run python authoring_cli.py ingest-textbook "教材.pdf" --uploader teacher
```

### PDF 圖片轉述

PDF 經 Marker 取得文字後，逐頁分析標題下方的條列、表格、範例圖、示意圖
和介面圖。整頁影像只用於分析，不會當作教材圖例輸出；簡短的 Any Question／
致謝結尾頁直接排除。PDF 內嵌圖片另行逐張檢查，避免整頁分類漏掉條列旁的
手機操作範例、地圖介面或流程圖。內嵌圖採用 PDF 原始圖片，不使用模型猜測
的裁切範圍；向量圖則依局部範圍裁切，拒絕整頁及含左上角頁碼的範圍。

第二階段要求模型提供圖片具教學意義的可見依據，排除裝飾、一般條列與表格。
明確教學主題下的內嵌圖若仍被拒判為裝飾，保留原圖並標示「待查核」，不採用
模型推測的圖說；表格與普通文字不適用此保留規則。圖片內的需求對照範例仍可
保留。圖例 PNG 存於 Markdown 同名 `_assets`，連同繁體中文圖說回填到對應頁
附近。左上角數字頁碼不列為教材文字，來源頁次仍保留供查核。

`_images.json` 記錄版面區塊、實際圖片來源、尺寸、SHA-256、可見依據、
`needs_review` 及錯誤；終端另列待查核圖數與排除結尾頁數。一般頁面至少需要
一次模型呼叫，文字層稀少但含大圖的頁面可能複核一次，每個候選圖另需一次
教學意義判讀（包括最後被排除的圖）；`IMAGE_DESCRIPTION_MAX_IMAGES` 僅限制
輸出的圖例數，不限制頁面掃描。

建議先只預覽轉換結果，不連線 MongoDB／Neo4j：

```bash
uv run python authoring_cli.py convert-pdf "教材.pdf" --describe-images --image-provider openai
```

確認 Markdown 與圖片說明後，再執行完整教材分析：

```bash
uv run python authoring_cli.py ingest-textbook "教材.pdf" --uploader teacher --describe-images
```

如果只想保存原圖、不產生圖說，可使用：

```bash
uv run python authoring_cli.py convert-pdf "教材.pdf" --skip-image-descriptions
```

頁級模式若某頁 API 失敗，該頁不會被當作已成功轉換；失敗及頁碼會記在
`_images.json`。正式教材建議加上 `--fail-on-image-error`，避免帶著未分析的
重要圖例進入後續知識圖譜。若 `--skip-image-descriptions`，則退回僅保存
Marker 原圖的舊模式，不會判斷教學意義，也不會發現純向量流程圖。

圖片轉述會把擷取出的圖片與附近教材文字傳給所選的外部服務，可能產生 API
費用；含有敏感資料的專案文件請先確認組織的資料政策。若不允許外傳，請使用
`--skip-image-descriptions`，此模式僅保存 Marker 已輸出的原圖及相對位置。

使用上一個指令輸出的 analysis ID 檢視 Section。`--chapter-order` 仍可使用，
但只是舊版相容名稱：

```bash
uv run python authoring_cli.py show-textbook <analysis_id> --section-order 1
uv run python authoring_cli.py list-assessment-targets <analysis_id>
```

先個別退回不適當的評量目標，再核准 CourseUnit。整體核准會核准所有尚未被退回的目標：

```bash
uv run python authoring_cli.py review-assessment-target <analysis_id> <target_id> rejected --reviewer teacher --note "目標重複或不具教學意義"
uv run python authoring_cli.py review-textbook <analysis_id> approved --reviewer teacher
```

把每個核准目標補到 2 題平行題（可改成 3），並檢查題庫覆蓋狀態：

```bash
uv run python authoring_cli.py generate-course-unit-bank --analysis-id <analysis_id> --items-per-target 2
uv run python authoring_cli.py course-unit-bank-status --analysis-id <analysis_id>
uv run python authoring_cli.py list-quizzes --status pending --analysis-id <analysis_id>
uv run python authoring_cli.py show-quiz <question_id>
uv run python authoring_cli.py review-quiz <question_id> approved --reviewer teacher
```

若只想產生少量草稿，仍可使用：

```bash
uv run python authoring_cli.py generate-quiz-drafts --analysis-id <analysis_id> --count 5
```

大量出題會自動以每批最多 5 個評量目標生成；若某批未完整
覆蓋所有目標，系統會將該批改為逐目標重試，全部完成並驗證後才寫入 MongoDB。
題目的 `knowledge_point_ids` 與 `source_chunk_ids` 不採信 LLM 抄寫結果，而是依
`assessment_target_id` 從已核准 KG metadata 回填，避免模型帶入關聯但非直接受評的 ID。

至少要有 5 個不同評量目標各自擁有核准題目，Discord 才能組成小測驗：

```text
/quiz course_unit:[02]基礎需求工程 questions:5
```

`questions` 可設為 5～10。系統會在每個被抽到的評量目標中隨機選一題平行題。
省略 `course_unit` 時，系統會隨機挑選一個題庫充足的 CourseUnit，但不會把不同
CourseUnit 混在同一次小測驗中。

舊分析資料可讀取，但不會自動獲得 Section 排除與跨 Section 去重結果；要套用新版
邏輯，必須用新版程式重新執行 `ingest-textbook`，取得新的 analysis ID。

Neo4j 可使用下列查詢檢視新版圖譜：

```cypher
MATCH path=(unit:CourseUnit)-[:HAS_SECTION]->(section:Section)
OPTIONAL MATCH path2=(section)-[:COVERS|HAS_ASSESSMENT_TARGET]->(item)
RETURN path, path2
LIMIT 200
```

現有題庫若沒有 `review_status`，不會直接提供給學生；請重新生成並審核，或先完成資料遷移。

若舊版 `generate-quiz-drafts` 曾顯示 `None` 題目 ID，可用下列指令找回已寫入
MongoDB 的待審題目；這些題目在核准前不會提供給學生：

```bash
uv run python authoring_cli.py list-quizzes --status pending --limit 20
```

### 離線專案文件匯入

SRD、SDD、STD 也改由同一個 CLI 預先建立 KG 與向量索引：

```bash
uv run python authoring_cli.py ingest-project-document "SRD.pdf" SRD team-a --uploader teacher
```

若只需要 KG、不建立向量索引：

```bash
uv run python authoring_cli.py ingest-project-document "SRD.md" SRD team-a --uploader teacher --skip-vectors
```

PDF 專案文件預設使用 `docling` 圖片定位並啟用多模態圖片轉述。每張圖片會連同
文件類型、所在章節、圖片前文與後文一起判讀；輸出的圖說保留「與本文的連接」、
圖片本身表達的意義、可供文件分析的原子事實、信心分數及分級不確定處，並插回
原頁章節，而不是集中附加在文件末尾。SRD、SDD、STD 分別使用需求、設計與測試
文件的判讀規則。匯入階段只忠實轉述圖片，不分析文件缺漏、矛盾、風險或改善方法；
這些工作由 `/project_qa` 結合全文、圖片與知識圖譜處理。

匯入專案文件時會先檢查同名 `_images.json`：流程圖、UI、需求範例或圖片表格若
描述失敗、文字核對失敗、低於 `IMAGE_DESCRIPTION_MIN_CONFIDENCE`，會停止建立 KG
及向量索引。只有會影響角色、數值、箭頭、狀態或核心意思的重大不確定才會阻擋；
非關鍵小字模糊等輕微不確定只保留複核提醒。裝飾圖與誤裁的一般文字不會阻擋。這項品質閘門可由
`PROJECT_DOCUMENT_REQUIRE_VERIFIED_IMAGES` 設定，但正式的 `/upload_document`
流程建議保持啟用，避免圖片內容在後續分析中靜默遺失。

### Discord 匯入 HackMD 專案文件

學生先把 HackMD 筆記設為「所有人可閱讀」，再使用：

```text
/upload_document hackmd_url:https://hackmd.io/@team/note document_type:需求文件（SRD） group:team-a
```

圖片模型若遇到暫時性 API、JSON 或欄位格式問題，會依
`IMAGE_DESCRIPTION_MAX_ATTEMPTS` 自動重試；常見的 `"95%"`、`95`、單一字串及
`null` 陣列也會先安全正規化。每張圖片的嘗試次數、失敗階段、錯誤摘要及語意品質
原因會寫入同名 `_images.json`。介面截圖、使用者故事地圖、流程／架構圖與專案看板
不會再被當成裝飾圖直接略過；資訊圖片若沒有可驗證文字、關係或文件事實，品質閘門
仍會停止建立 KG 與向量索引。`review_recommended` 代表可稍後人工複核；只有
`blocks_ingestion` 才會中止匯入。

若 SRD／SDD／STD 圖片已成功取得內容說明、可見文字及文件事實，只有
`meaning` 欄位在重試後仍偶發漏回，系統會以已驗證的內容說明產生保守的描述性
意義，並列為複核提醒而非中止整份匯入。這個補全不讀取周邊文字、不判斷缺漏、
風險或改善方式；低信心、重大不確定、缺少圖片事實等問題仍會阻擋建立索引。

系統會下載 HackMD 原始 Markdown，保留標題、條列及表格順序，並處理 Markdown
圖片與 `<img>` 圖片。每張圖片分別下載、依所在章節及前後文產生結構化描述，再把
圖說插回原圖位置。任何圖片下載失敗、圖說失敗或低於信心門檻時，整份文件不會
進入 KG 或向量索引，避免產生看似成功但缺圖的分析結果。

專案文件中若圖片位於名稱完全等於 `Trello` 的獨立標題下，系統將它視為小組使用
任務看板的展示證明：原圖及來源仍會保存於資產與 `_images.json`，但不呼叫圖片
描述模型、不插入 AI 圖說，也不參與圖片品質阻擋。只有這個明確章節例外會被排除；
需求、設計或測試章節內出現的專案看板仍會照常轉述與檢查。

安全限制：只接受 `https://hackmd.io` 公開筆記；拒絕內部網路圖片網址、含帳密
網址、SVG、超大檔案及單篇超過 60 張圖片。若學生帳號已綁定組別，不能透過此
指令把文件寫入其他組別。

`/project_qa` 每次都會同時檢索本組的專案 KG 與 Markdown 向量片段。圖片描述中的
圖中文字、元素關係、圖片表達的意義、與本文連接及文件事實會跟原文一起被檢索。
圖片圖說本身不預先下分析結論；回答時再跨全文與多張圖片分析需求缺漏、流程分支、
設計追溯及測試證據。重大不確定或待查核內容不會視為確定事實。

### Marker＋Docling 圖片定位

`docling` 後端不會拿 Docling Markdown 取代教材。橫式投影片會直接依原 PDF 的
53 個實體頁面逐頁重建：原生文字層負責標題與條列，PDF 表格線負責表格，連結註解
負責完整網址；PDF 原生圖片座標負責跨平台裁切所有內嵌圖片，Docling 補足沒有
內嵌點陣圖的向量圖範圍。
一般非投影片 PDF 才回退至 Marker 文件結果。後續 Section、知識點、評量目標及
題庫流程不變。

Windows PowerShell 在專案根目錄執行一次：

```powershell
uv venv .venv-docling --python 3.11
uv pip install --python .venv-docling/Scripts/python.exe -r requirements-docling.txt --torch-backend cpu
```

Docling 固定使用 2.140.0，與 Marker／Surya 分開安裝，避免 Transformers 版本衝突。
CPU 可執行但較慢；首次執行會下載版面模型，需要網路。Docling OCR 與表格模型固定
關閉，避免錯字、錯誤欄列或標題判定進入 Markdown。
Linux/macOS 的安裝指令將 Python 路徑改為 `.venv-docling/bin/python`。
已有其他 Docling 環境時，可在 `config.py` 設定 `PDF_DOCLING_PYTHON` 為其 Python
執行檔完整路徑；不要將整條命令填入此欄位。

```powershell
uv run python authoring_cli.py convert-pdf "[02]基礎需求工程.pdf" --pdf-backend docling --describe-images --image-provider openai
```

執行主程式時若 PowerShell 提示目前是 `(.venv-docling)`，先執行 `deactivate`（或開啟
新的終端）再執行上述命令。`uv run` 使用專案的 `.venv` 是預期行為；Docling 會由
程式另外呼叫 `.venv-docling/Scripts/python.exe`，不需要先啟用 Docling 環境。

確認結果後，以相同後端匯入：

```powershell
uv run python authoring_cli.py ingest-textbook "[02]基礎需求工程.pdf" --uploader teacher --pdf-backend docling --describe-images
```

也可以在既有 `config.py` 加上 `PDF_CONVERTER_BACKEND = "docling"` 作為預設。
`--pdf-backend marker` 隨時可切回原流程，不會自動切換而隱藏 Docling 安裝錯誤。

處理方式：

- 橫式教材逐頁處理，每個實體頁固定只有一個 `#` 標題；標題下原生文字統一整理成
  Markdown 階層條列，跨行文字合併回同一項，不會把內文誤升為標題或跨頁合併。
- H1 優先取原生 PDF 頂端標題區並排除純數字頁碼；只有確實找不到原生標題時才以
  Marker 標題補位，避免固定課程名稱覆蓋各頁真正標題。
- 頁碼依同一份投影片中多數頁的共同水平帶辨識，不要求固定在最左側；因此範本將
  頁碼向右移動時，也不會把頁碼寫成條列內容。
- 每頁之間加入不可見的 `<!-- PDF_PAGE_BREAK -->`，左上角頁碼不輸出。
- 有實際列欄線的原生表格依幾何位置重建；同一儲存格的多行項目使用 `<br>`，不再
  把每個英文單字拆行。影像型表格保留高解析原圖，避免未核對 OCR 改壞內容。
- 雙欄表格若只有外框與水平列線，會依表頭兩欄的實際 X 位置補回缺少的中間分界，
  避免 Question／Answer 全部落在第一欄。
- 網址直接讀取 PDF link annotation 的 URI，所以原檔即使將網址折成兩行，Markdown
  仍會得到一條完整可點擊網址。
- `docling_layout.json` 保存 Docling 圖片／圖表 bbox 及 PDF 原生圖片座標，方便查核
  每張輸出圖片的來源。
- PDF→Markdown 採完整內容模式：所有 PDF 原生圖片與 Docling 圖像區域都輸出，
  包含資訊圖、線框圖、UI、圖片型表格、Logo、裝飾及 Any Question／Q&A 結尾圖片；
  此階段不判斷它們是否重要。只有幾乎覆蓋整頁的掃描背景不重複輸出為圖片。
- 由多個向量框、箭頭、文字與小圖示共同構成的流程圖會合併為一張完整圖例，不會
  只留下零碎的小圖示或把圖中文字錯當成普通條列。
- 若 Windows 的 pypdfium2 無法列出／抽出圖片物件，系統會改用 pdfplumber 的原生
  圖片座標從頁面渲染結果裁切，避免整批圖片只剩 Docling 向量圖；這會列為警告而
  不會把該頁計入「失敗」。真正失敗時，終端會另列出頁碼、階段與錯誤訊息。
- 啟用圖說時，外部多模態模型只產生圖片說明與文字核對；即使分類為純文字、表格
  或裝飾，也不會在轉檔階段刪除。是否形成知識點或評量目標交由後續分析決定。
- PNG 與 `docling_layout.json` 存入同名 `_assets`；`_images.json` 記錄圖例、來源頁、
  bbox、核對狀態與錯誤。Markdown 使用相對路徑，不嵌入 Base64。
- `--skip-image-descriptions` 不呼叫外部模型，但仍保存所有 PDF 圖片及 Docling 向量圖。
  `--fail-on-image-error` 可讓圖片判讀失敗時停止匯入。

技術依據：
[Docling 圖片匯出](https://docling-project.github.io/docling/_generated/examples/export_figures/)。
