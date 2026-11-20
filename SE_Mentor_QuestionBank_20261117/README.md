# SE Mentor：11 章教材選擇題庫

共 **179 題四選一單選題**，每章 15～20 題。依儲存庫 `textbook/` 的 11 份 PDF 直接編寫，包含觀念辨析、情境應用與計算題。

來源固定為 `ckklkkc/SE_Mentor` 的 commit `495f729d41ad11702f05bc1b90aade817a86e2e4`。逐份確認 PDF 的 Git blob SHA 與下載內容一致；本套格式依同一版本的 `DiagnosisQuiz` 與 `/quiz` 抽題邏輯製作。

## 檔案選擇

| 檔案 | 用途 |
|---|---|
| `diagnosis_quiz_all.json` | 全部 179 題，MongoDB Extended JSON 陣列，可供 Compass 匯入 |
| `diagnosis_quiz_all.jsonl` | 相同題庫，每行一個 JSON 文件 |
| `chapters/ch01.json`～`ch11.json` | 分章匯入，各章題數見下表 |
| `Teacher_Review.md` | 完整題幹、四選項、答案、解析、錯誤選項診斷與 PDF 頁碼 |
| `Material_Analysis.md` | 章節涵蓋、教材疑義與命題處理方式 |
| `assessment_catalog.json` | 本套新增的知識點與評量目標對照，供人工查看或未來映射 |
| `import_bank.py` | 預設只驗證；加 `--apply` 才連線匯入，重跑時保留既有同 ID 文件 |
| `validate_bank.py` | 不連線的完整性檢查，只需要 Python 標準函式庫 |
| `bank_manifest.json` | 題數、來源版本、教材雜湊與答案分布 |
| `validation_report.json` | 實際執行的離線相容性測試結果與未測項目 |
| `discord_result_chunking.patch` | 修正目前 bot 答題紀錄可能超過單則訊息長度的分段傳送 |

**不要將 assessment_catalog 或 bank_manifest 匯入 diagnosis_quiz；它們不是題目文件。**

## 章節與只抽本套題目的識別碼

| 章 | 教材 | 題數 | `course_unit_id` |
|---|---|---:|---|
| 01 | 軟體危機與軟體流程 | 18 | `se_manual_20261117_v1:unit:01` |
| 02 | 基礎需求工程 | 15 | `se_manual_20261117_v1:unit:02` |
| 03 | 使用者故事分析 | 15 | `se_manual_20261117_v1:unit:03` |
| 04 | 敏捷開發方法 | 15 | `se_manual_20261117_v1:unit:04` |
| 05 | 基礎專案管理與看板 | 15 | `se_manual_20261117_v1:unit:05` |
| 06 | 版本控制 | 15 | `se_manual_20261117_v1:unit:06` |
| 07 | 軟體設計－系統設計 | 15 | `se_manual_20261117_v1:unit:07` |
| 08 | 軟體設計－模組設計 | 15 | `se_manual_20261117_v1:unit:08` |
| 09 | 軟體測試 | 20 | `se_manual_20261117_v1:unit:09` |
| 10 | 進階軟體測試 | 18 | `se_manual_20261117_v1:unit:10` |
| 11 | DevOps 自動化建置管理 | 18 | `se_manual_20261117_v1:unit:11` |
| 合計 | | **179** | |

現有系統一次 `/quiz` 為 **5～10 題**；各章 15～20 題是可供抽樣的題庫數，不是把單次測驗限制改成 20 題。

在 Discord 的 `/quiz` 參數填入：

```text
course_unit: se_manual_20261117_v1:unit:02
questions: 5
```

這會依照專用 ID 抽本套第二章新題。若輸入 `[02]基礎需求工程` 等原章名，舊題庫中相同章名且為 `approved` 的題目也可能混入。若不指定單元，原有的其他合格單元也可能被選中。

## 匯入：建議使用保留既有資料的 Python 程式

把壓縮檔解開後，整個 `SE_Mentor_QuestionBank_20261117` 資料夾放到你的 SE_Mentor 專案根目錄。先在專案根目錄開 PowerShell，執行本地檢查：

```powershell
uv run python .\SE_Mentor_QuestionBank_20261117\import_bank.py
```

看到 179 題及 11 章統計後，實際匯入：

```powershell
uv run python .\SE_Mentor_QuestionBank_20261117\import_bank.py --apply
```

預設為 `mongodb://localhost:27017/`、資料庫 **TABotAI_quiz**、collection **diagnosis_quiz**，與所檢查版本的 bot 一致。程式透過 BSON `json_util` 將 `$oid` 與 `$date` 轉為真正的 ObjectId 與日期，使用 `$setOnInsert`，不刪除舊題，也不覆寫同 ID 題目後來的人工編修。

第一次完整匯入預期 `inserted: 179`；再次匯入預期 `existing_preserved: 179`、`modified: 0`。若中途失敗，可以重跑補入缺少的 ID；仍應查看輸出確認結果。

需要先存為待審核時可使用：

```powershell
uv run python .\SE_Mentor_QuestionBank_20261117\import_bank.py --apply --status pending
```

`--status` 只影響新插入文件；不會改變已存在文件狀態。自訂 URI 可用 `--uri` 或 `MONGODB_URI` 環境變數；自訂 DB／collection 用 `--database`／`--collection`。不會把連線字串印到執行摘要。

若不在專案環境中執行，先確認目前 Python 已安裝 `pymongo`，再用 `python import_bank.py --apply`。

## 另一種方式：MongoDB Compass

1. 連線 `mongodb://localhost:27017/`。
2. 開啟 `TABotAI_quiz` → `diagnosis_quiz`。
3. 選 **Add Data → Import JSON or CSV file**，選 `diagnosis_quiz_all.json`，格式選 JSON。
4. 確認新增 179 筆，並檢查 `answer` 是數值、`_id` 是 ObjectId、`reviewed_at` 是日期。

整包與分章檔是**同一組題目**，不要重複匯入。Compass 重匯可能遇到 duplicate key；需要可重跑且保留人工修改的操作，請使用上述 Python 匯入程式。

## 上線前的 Discord 分段修補

目前檢查的 `dc_chatbot.py` 已對複習筆記分段，但測驗結束的整份答題紀錄仍一次送出。本套測試中，同章 10 題全錯且附解析的紀錄最長約 2,628 字，會超過一般文字訊息長度限制；錯誤發生在保存作答紀錄之前，可能連帶讓後續學習紀錄未執行。

包內 patch 只把結果文字交給既有 `split_message` 後逐段傳送，不更動題庫、評分、測驗規則或學習筆記。從專案根目錄執行：

```powershell
git apply --check --ignore-space-change .\SE_Mentor_QuestionBank_20261117\discord_result_chunking.patch
git apply --ignore-space-change .\SE_Mentor_QuestionBank_20261117\discord_result_chunking.patch
```

第一行成功後才執行第二行，然後重啟 bot。若目前程式與參考版本不同而檢查失敗，請勿強行套用；對照 patch 將原本一次傳送結果的區塊改為分段傳送即可。已對參考版本完成 patch 套用檢查、語法解析及既有分段函式的完整重組測試；未在真實 Discord 伺服器執行。

## 欄位與整合界線

- `options` 四個字串不包含 A／B／C／D 前綴，由現有 UI 加上字母。
- `answer` 是 **0 起算**的整數：A=0、B=1、C=2、D=3。答案順序已打散，且每章四個位置的題數差不超過 1。
- `analysis` 是正確理由與各選項診斷；`concept` 與 `assessment_target` 是細顆粒的評量重點。
- `option_diagnoses` 每題有四筆，`option_index` 對應打散後順序；正解的 `misconception` 為 null，三個錯誤選項各有診斷。
- `review_status` 預設 `approved`，用來配合現行抽題條件；`reviewed_by` 明記 ChatGPT，**不表示教師人工核准**。可在匯入前或用 `--status pending` 改為待審。
- `_id` 為固定 ObjectId，方便重複匯入不重複新增；`bank_batch` 是 `se_manual_20261117_v1`。
- `source_evidence` 保存教材路徑、PDF 頁序、來源版本與雜湊。`question_code`、`difficulty`、`bank_batch`、`source_evidence` 是新增的輔助欄位；目前 Beanie 模型可能忽略這些額外欄位，但直接存入 MongoDB 後仍可供查核。
- 本套 `knowledge_point_ids`、`assessment_target_id`、`course_unit_id` 為**新增的命名空間**。`source_chunk_ids` 是 `manual_pdf:...:page:...` 的 PDF 頁碼定位，**不是既有向量庫 chunk ID**；`source_analysis_id` 為 null、`section_ids` 為空，不假裝已連結現存分析或 KG。
- 現有抽題與弱點統計可使用這些新 target ID；若要接到既有 KG／教材分析，仍須另做概念映射。本套不會自行新增 Neo4j 節點或 `textbook_analyses` 文件。
- 每個評量目標目前只有一題，重測可能遇到同題；這不是每目標 2～3 題的平行題庫。

## 檢核結果與限制

已檢查 179 題的四選項、單一答案索引、識別碼唯一性、錯誤選項對應、每章題數、答案分布及來源頁碼。以參考模型的實際欄位宣告進行 Pydantic 離線驗證，並以未修改的抽題函式配合記憶體資料介面執行 **155 組案例**：11 章各抽 5～10 題，以 ID 與章名兩種方式查詢、隨機選章及錯誤輸入檢查。各章可抽足 10 個不同評量目標；單題訊息估計最長 353 字。

已確認 Extended JSON 轉 BSON 時的 ObjectId、日期與編碼。這些是格式與離線邏輯檢核，**未實際連線你的 MongoDB、未在 Discord 做端到端實測**，也未取得教師簽核或學生試測數據。題目難度標籤是命題層次，不能當成經學生測得的難度、鑑別度或信效度。
