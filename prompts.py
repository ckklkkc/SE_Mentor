DCCHATBOT_WELCOME_MESSAGE = """
你好，歡迎使用 SE Mentor！
我是一個軟體工程課程虛擬助教，目前提供了這些服務：

1. 課程問答
2. 學習診斷測驗以及個人化筆記
3. 查看學習紀錄與管理學習弱點
4. 上傳開發文件
5. 專案問答

使用前請閱讀使用說明：

### 課程問答

請在文字輸入欄輸入 `/course_qa`，並在 question 欄位中輸入你的問題。

### 學習診斷

請在文字輸入欄輸入 `/quiz`，可選擇題目來源：混合、MongoDB 題庫、即時生成、依過往課程 QA / 學習弱點個人化生成。未指定時使用混合模式。
一共有五題單選題，作答完畢後，如果有答錯的題目，SE Mentor 會生成你的專屬筆記。(如果都答對代表你十分聰明，所以不會生成筆記)
即時或個人化題目會先從課程教材檢索相關內容，再由 LLM 生成，因此需要稍候。

### 學習紀錄與弱點管理

請輸入 `/learning_profile` 查看自己的近期 Quiz、Course QA、目前學習弱點與已處理概念。
若你已透過筆記或 QA 理解某個弱點，可直接在下拉選單將它標記為已克服；該概念會從目前弱點移除，個人化測驗也不會再因舊紀錄持續將它當作弱點焦點。

### 上傳文件

請在文字輸入欄輸入 `/upload_document`，先選擇想上傳的檔案**(pdf或markdown格式)**，再選擇檔案類型 (顯示於文字欄上方)，最後輸入你的所屬組別或代號。
文件處理約需要 2-10 分鐘 (視文件複雜度)，處理完成會發送通知。

### 專案問答

請在文字輸入欄輸入 `/project_qa`，並在 question 欄位中輸入你的問題，在 group 中輸入你的所屬組別或代號。

有任何問題，或 SE Mentor 出現錯誤訊息，請立即通報開發者，感恩的心。
"""

ENTITY_PROMPT_4_TEXTBOOK = """
  	# Role
	你是一位資深的「軟體工程」專家，擅長將非結構化文本轉化為結構化的知識圖譜實體。

	# Task
	請從提供的文章中提取具有「教育價值」的知識點實體。這些實體必須具備與其他概念建立關聯的潛力。

	# Entity Categories (優先提取以下類別)
	- Concept: 核心概念與原理 (如: 耦合度, 內聚力, 微服務, 物件導向)
	- Technology: 技術與工具 (如: Docker, Kubernetes, Git)
	- Methodology: 開發流程與方法論 (如: CI/CD, Scrum, 測試驅動開發)

	# Rules
	1. 去噪點：嚴禁提取「系統」、「文章」、「功能」、「用戶」、「免費」、「工期」、「效能」等過於一般化或描述性的詞彙。
	2. 標準化 (Normalization)：將同義詞映射至教材標準術語。例如：「版控」、「版本控制」統稱為「版本控制」；「寫 Code」統一為「程式編碼」。
	3. 具體性：實體必須是一個獨立的知識點。如果一個實體在脫離本文後無法定義為一個專業術語，請忽略它。
	4. 關係導向：只保留「可以被定義、被解釋、或與其他術語產生邏輯連接」的實體。
	5. 唯一性：確保輸出的 Python list 中沒有重複項。

	# Example 1
	- Input: '在進行敏捷開發時，我們常使用 Git 來進行版本管理，並透過 Jenkins 實作自動化部署。'
	- Output: ['敏捷開發', '版本控制', 'Git', 'Jenkins', '自動化部署']
    
    # Example 2
    - Input: '極限製程是 Kent Beck 於 1999 年提出, 目的是提倡更能「擁抱 改變」的敏捷開發方式'
    - Output: ['極限製程', '敏捷開發'] 

	# Output Format
	請直接輸出一個 Python List，List 中僅包含提取的實體 (string)，不需以註解分類也不需任何額外解釋。
"""

TRIPLE_PROMPT_4_TEXTBOOK = """
  	# Role
	你是一位知識工程師，專精於從軟體工程文本中建構語義網絡。

	# Task
	根據提供的「實體列表」，從「文章內容」中找出這些實體之間的邏輯關係，並以三元組 (Subject, Relation, Object) 的形式輸出。請務必根據文章內容替每個實體與關係加上描述，並分別記錄於實體(Entity)的屬性中(properties)，key 為 'description'，以及關係(Relation)的description中。description 應為單句摘要，不超過 30 個中文字。

	# Relation Schema (請優先使用以下關係語義)
	- 是：分類關係 (如: Git - 是 -> 版本控制工具)
	- 包含於：組成關係，「子」包含於「父」中 (如: 記憶體管理 - 包含於 -> 作業系統)
	- 實作：實作/達成 (如: Scrum - 實作 -> 敏捷開發)
	- 使用 / 依賴：技術依賴 (如: Web App - 使用 -> HTTP 協定)
	- 改善 / 解決：解決特定問題 (如: 索引 - 改善 -> 查詢效率)

	# Rules
	1. 嚴格限制：僅限於提取「實體列表」中出現的術語之間的關係。
	2. 方向性：確保 Subject 是主體，Object 是受體。例如：(索引 - 改善 -> 查詢效率)，語意順序不可顛倒。
	3. 精煉關係：關係詞應為動詞或動詞短語，且盡量使用上述 Relation Schema 中的詞彙。
	4. 顯著性：只提取文章中明確支持的關係，避免主觀臆測。
	5. 連通性：若存在多個由文件明確支持的關係，請全部列出。不得僅為提高圖譜密度而建立關係。
    
    # Example
    - Input: 系統測試測試系統軟體與硬體整體功能是否協調，確認開發人員依照系統需求文件正確無誤開發系統。驗證系統需求文件所描述功能都正確無誤實作，測試環境應與實際環境相似，以驗證系統內部功能。以黑箱測試技術為主。
    - Output:
		[{
		'subject': {
			'name': '系統測試',
			'label': 'Concept',
			'properties': [
				{
					'key': 'description',
					'value': '系統測試的目的為確認開發人員依照系統需求文件正確無誤開發系統。'
				}
			]
		},
		'relation': {
			'name': '使用',
            'description': '系統測試以黑箱測試技術為主。'
        },
		'object': {
			'name': '黑箱測試',
			'label': 'Methodology',
			'properties': 
				{
					'key': 'description',
					'value': '黑箱測試是一種軟體測試技術，須了解軟體產品需求功能後進行測試，且不考慮軟體內部邏輯的結構。'
				}
			]
		}]
"""

# ENTITY_PROMPT_4_SRD = """
#   	# Role
# 	你是一位資深的「軟體工程」專家，擅長將非結構化文本轉化為結構化的知識圖譜實體。

#     # Input Format Recognition (格式識別指南)
# 	文件包含兩種明確的結構化需求格式，你必須能夠區分它們：
	
# 	## Format A: 使用者故事區段
# 	使用者故事按以下結構出現：
# 	- 代號：<ID>（ID 格式：DL-TRACK-*, US-CUS-*, US-ADM-* 等，包含英文字母和數字）
# 	- 故事：「身為...我希望...」、「作為...我想要...」或類似的敘述方式
# 	- 註記、測試方法 等輔助欄位（可選）
	
# 	識別信號：只要同時出現「代號：」和「故事：身為/作為...」，就是一個使用者故事。
	
# 	例子：
# 	```
# 	- 代號：DL-TRACK-08 View order history  
# 	- 故事：身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄。
# 	```
	
# 	## Format B: 功能需求區段
# 	功能需求按以下結構出現：
# 	- 編號：<ID>（ID 格式：FR-* 或 NFR-*，如 FR-DL-04, NFR-SYS-01, SCH06 等，包含英文字母和數字）
# 	- 功能需求：[描述] 或 非功能需求：[描述]
	
# 	識別信號：出現 <編號> <需求描述>。
	
# 	例子：
# 	```
# 	編號：FR-DL-04 
# 	功能需求：外送員可查看歷史配送紀錄
# 	```
    
# 	# Task
# 	請從提供的需求文件中提取名詞實體。這些實體必須具備與其他概念建立關聯的潛力，並能夠跟其他文件 (設計文件、測試文件) 連結。
#   	需求文件需提取每個功能需求與非功能需求，包含操作功能的角色。

# 	# Section Filtering
# 	明確忽略以下區段：
#     - 版次、目錄
# 	- System Description
# 	- 專案介紹 / 特色描述
# 	- 實作方案 / 使用技術
#     - 使用者介面分析
#     **不要忽略**：包含使用者故事或功能需求的任何區段（如「## User Story Map」、「## Functional Requirements」等）

# 	## 優先級 1（按結構信號識別）
# 	UserStory：
# 	- 識別信號：「代號：<ID>」+ 「故事：...」
#     - 範例：身為一名顧客，我希望我可以確認已下單的訂單狀態，讓我追蹤訂單。
	
# 	Requirement（功能與非功能）：
# 	- 識別信號：「編號：<ID>」
#     - 範例：確認訂單狀態
	
# 	## 優先級 2（按語義識別）
# 	Actor：在使用者故事或功能需求中出現的角色（如「外送員」、「顧客」、「管理員」）

#     ## 禁止項
# 	- 禁止提取「系統模組」、「平台特色」類實體
# 	- 禁止把編號當成實體名稱（編號是屬性，不是實體本身）
# 	- 禁止把編號和描述合併成一個名稱（如「FR-DL-04 外送員可查看...」應該分開）
	
# 	# Table Parsing Rules
# 	若文件包含 Markdown 表格：
# 	1. 必須逐「列」處理，每一列視為一個獨立需求
#     2. 編號識別：
# 		- 如果表格有「代號」或「編號」欄位，該欄位的值必須被提取為 us_id 或 req_id
# 		- 編號不要和描述合併，分別作為 properties
# 	3. 不可合併多列內容
# 	4. 不可省略任何一列（即使語意相似）
# 	5. 子編號（如 FR-CUS-RES-01.1）視為獨立需求

# 	# Rules
# 	1. 去噪點：嚴禁提取「系統」、「文章」、「功能」、「用戶」、「免費」、「工期」、「效能」等過於一般化或描述性的詞彙。
#     	- 禁止把編號視為一般詞彙，編號是實體的身份標識，不是噪點
# 	2. 標準化 (Normalization)：將同義詞映射至標準術語。
#     	- 例如：「版控」、「版本控制」統稱為「版本控制」；「寫 Code」統一為「程式編碼」。
# 	3. 具體性：實體必須是一個獨立的知識點。如果一個實體在脫離本文後無法定義為一個專業術語，請忽略它。
# 	4. 關係導向：只保留「可以被定義、被解釋、或與其他實體產生邏輯連接」的實體。
# 	5. 唯一性：確保輸出的 Python list 中沒有重複項。
#   	6. 粒度控制：優先提取「功能單位」而非操作步驟。
#     	- 例如：「購物車管理」優於「修改購物車資訊」。

# 	# Example
# 	- Input: 
# 	```
# 	- **代號**：DL-TRACK-08 View order history  
# 	- **故事**：身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄。
	
# 	編號：FR-DL-04 
# 	功能需求：外送員可查看歷史配送紀錄
# 	```
# 	- Output: ['外送員', '身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄', '查看歷史配送紀錄']
# 	- 注意：編號（DL-TRACK-08 和 FR-DL-04）不在這個列表中，因為它們是屬性，在三元組提取時會被加入

# 	# Output Format
# 	請直接輸出一個 Python List，List 中僅包含提取的實體 (string)，不需以註解分類也不需任何額外解釋。
# """

ENTITY_PROMPT_4_SRD = """
  	# Role
	你是一位資深的「軟體工程」專家，擅長將非結構化文本轉化為結構化的知識圖譜實體。

	# Task
	請從提供的需求文件中提取名詞實體。這些實體必須具備與其他概念建立關聯的潛力，並能夠跟其他文件 (設計文件、測試文件) 連結。請從「需求文件內容」中找出這些實體的標籤及屬性，以及實體之間的邏輯關係，並以三元組 (Subject, Relation, Object) 的形式輸出。請務必根據文件內容替每個實體與關係加上描述，並分別記錄於實體(Entity)的屬性中(properties)，key 為 'description'，以及關係(Relation)的description中。

	# Section Filtering
	明確忽略以下區段：
    - 版次、目錄
	- System Description
	- 專案介紹 / 特色描述
	- 實作方案 / 使用技術
    - 使用者介面分析
    **不要忽略**：包含使用者故事或功能需求的任何區段（如「## User Story Map」、「## Functional Requirements」等）

    # Input Format Recognition (格式識別指南)
	文件包含兩種明確的結構化需求格式，你必須能夠區分它們：
	
	## Format A: 使用者故事區段
	使用者故事按以下結構出現：
	- 代號：<ID>（ID 包含英文字母和數字）
	- 故事：「身為...我希望...」、「作為...我想要...」或類似的敘述方式
	- 註記、測試方法 等輔助欄位（可選）
	
	識別信號：只要同時出現「代號：」和「故事：身為/作為...」，就是一個使用者故事。
	
	例子：
	```
	- 代號：DL-TRACK-08 View order history  
	- 故事：身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄。
	```
	
	## Format B: 功能需求區段
	功能需求按以下結構出現：
	- 編號：<ID>（ID 包含英文字母和數字）
	- 功能需求：[描述] 或 非功能需求：[描述]
    
    或是列點呈現：
    - <ID> [功能需求]
	
	識別信號：出現 <編號> <需求描述>。
	
	例子：
	```
	編號：FR-DL-04 
	功能需求：外送員可查看歷史配送紀錄
	```
    ```
    - SCH01 查詢紀錄
		- 借用者可查詢借用紀錄
    ```
    
	# Entity Schema
	每個實體的資訊，包含標籤 (label) 及屬性 (properties)。對於 properties，請務必使用 [{'key': '...', 'value': '...'}] 的列表格式，請勿使用字典 (dictionary) 格式。
	- label: 
		- Requirement: 功能需求與非功能需求 (識別信號: 同時出現「代號：」和「故事：身為/作為...」)
        - UserStory: 使用者故事 (識別信號: 出現 <編號> <需求描述>)
		- Actor: 操作系統 (功能) 的角色
	- properties: 
		- name: 實體的唯一名稱 (對於 UserStory，使用故事內容或標題；對於 Requirement，使用需求描述；對於 Actor，使用角色名稱)
		- description: 實體的描述 （從文本中提取，必填）
        - us_id: 使用者故事的「代號」 （包含英文字母和數字；如果是 UserStory 必填）
		- req_id: 功能需求和非功能需求的「編號」 (包含英文字母和數字；如果是 Requirement 必填）
		- req_category: 需求類別，僅分為「功能需求」或「非功能需求」（只適用於 Requirement，必填）

    **重要提示**：
	- us_id 和 req_id 是實體的必要身份標識，不能遺漏
	- 如果文本中明確出現了編號、代號，必須提取。請嚴格忠於原始內容，勿自行臆測
	- 不要混淆：編號是屬性，不是實體的 name
        
    ## 禁止項
	- 禁止提取「系統模組」、「平台特色」類實體
	- 禁止把編號當成實體名稱（編號是屬性，不是實體本身）
	- 禁止把編號和描述合併成一個名稱（如「FR-DL-04 外送員可查看...」應該分開）
	
	# Table Parsing Rules
	若文件包含 Markdown 表格：
	1. 必須逐「列」處理，每一列視為一個獨立需求
    2. 編號識別：
		- 如果表格有「代號」或「編號」欄位，該欄位的值必須被提取為 us_id 或 req_id
		- 編號不要和描述合併，分別作為 properties
	3. 不可合併多列內容
	4. 不可省略任何一列（即使語意相似）
	5. 子編號（如 FR-CUS-RES-01.1）視為獨立需求

	# Rules
	1. 去噪點：嚴禁提取「系統」、「文章」、「功能」、「用戶」、「免費」、「工期」、「效能」等過於一般化或描述性的詞彙。
    	- 禁止把編號視為一般詞彙，編號是實體的身份標識，不是噪點
	2. 標準化 (Normalization)：將同義詞映射至標準術語。
    	- 例如：「版控」、「版本控制」統稱為「版本控制」；「寫 Code」統一為「程式編碼」。
	3. 具體性：實體必須是一個獨立的知識點。如果一個實體在脫離本文後無法定義為一個專業術語，請忽略它。
	4. 關係導向：只保留「可以被定義、被解釋、或與其他實體產生邏輯連接」的實體。
	5. 唯一性：確保輸出的 Python list 中沒有重複項。
  	6. 粒度控制：優先提取「功能單位」而非操作步驟。
    	- 例如：「購物車管理」優於「修改購物車資訊」。

    # Example
	- Input: 
		```
		- **代號**：DL-TRACK-08 View order history  
		- **故事**：身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄。
		
		編號：FR-DL-04 
		功能需求：外送員可查看歷史配送紀錄
		```
	- Output:
	[{
			'name': '外送員',
			'label': 'Actor',
			'properties': [
				{
					'key': 'description',
					'value': '負責接單和配送餐點的系統使用者角色'
				}
			]
		},
        {
			'name': '身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄',
			'label': 'UserStory',
			'properties': [
				{
					'key': 'us_id',
					'value': 'DL-TRACK-08'
				},
				{
					'key': 'description',
					'value': '身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄'
				}
			]
		},
		{
			'name': '外送員可查看歷史配送紀錄',
			'label': 'Requirement',
			'properties': [
				{
					'key': 'name',
					'value': '外送員可查看歷史配送紀錄'
				},
				{
					'key': 'req_id',
					'value': 'FR-DL-04'
				},
				{
					'key': 'description',
					'value': '外送員可查看歷史配送紀錄'
				},
				{
					'key': 'req_category',
					'value': '功能需求'
				}
			]
		}
	]
"""

TRIPLE_PROMPT_4_SRD = """
  	# Role
	你是一位知識工程師，專精於從軟體工程文本中建構語義網絡。

	# Task
	根據提供的「實體列表」，從「需求文件內容」中找出這些實體的標籤及屬性，以及實體之間的邏輯關係，並以三元組 (Subject, Relation, Object) 的形式輸出。請務必根據文件內容替每個實體與關係加上描述，並分別記錄於實體(Entity)的屬性中(properties)，key 為 'description'，以及關係(Relation)的description中。

    # Input Format Recognition (格式識別指南)
	文件包含兩種明確的結構化需求格式，你必須能夠區分它們：
	
	## Format A: 使用者故事區段
	使用者故事按以下結構出現：
	- 代號：<ID>（ID 格式：DL-TRACK-*, US-CUS-*, US-ADM-* 等，包含英文字母和數字）
	- 故事：「身為...我希望...」、「作為...我想要...」或類似的敘述方式
	- 註記、測試方法 等輔助欄位（可選）
	
	識別信號：只要同時出現「代號：」和「故事：身為/作為...」，就是一個使用者故事。
	
	例子：
	```
	- 代號：DL-TRACK-08 View order history  
	- 故事：身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄。
	```
	
	## Format B: 功能需求區段
	功能需求按以下結構出現：
	- 編號：<ID>（ID 格式：FR-* 或 NFR-*，如 FR-DL-04, NFR-SYS-01, SCH06 等，包含英文字母和數字）
	- 功能需求：[描述] 或 非功能需求：[描述]
	
	識別信號：出現「編號：」和「功能需求：」或「非功能需求：」。
	
	例子：
	```
	編號：FR-DL-04 
	功能需求：外送員可查看歷史配送紀錄
	```
    
	# Entity Schema
	每個實體的資訊，包含標籤 (label) 及屬性 (properties)。對於 properties，請務必使用 [{'key': '...', 'value': '...'}] 的列表格式，請勿使用字典 (dictionary) 格式。
	- label: 
		- Requirement: 功能需求與非功能需求 (識別信號: 同時出現「代號：」和「故事：身為/作為...」)
        - UserStory: 使用者故事 (識別信號: 出現 <編號> <需求描述>)
		- Actor: 操作系統 (功能) 的角色
	- properties: 
		- name: 實體的唯一名稱 (對於 UserStory，使用故事內容或標題；對於 Requirement，使用需求描述；對於 Actor，使用角色名稱)
		- description: 實體的描述 （從文本中提取，必填）
        - us_id: 使用者故事的「代號」 （格式如 DL-TRACK-08, US-CUS-01 等；如果是 UserStory 必填）
		- req_id: 功能需求和非功能需求的「編號」 (格式如 FR-DL-04, NFR-SYS-01, USR03 等；如果是 Requirement 必填）
		- req_category: 需求類別，僅分為「功能需求」或「非功能需求」（只適用於 Requirement，必填）

	**重要提示**：
	- us_id 和 req_id 是實體的必要身份標識，不能遺漏
	- 如果文本中明確出現了編號，必須提取
	- 不要混淆：編號是屬性，不是實體的 name
        
	# Core Traceability Relations (最高優先)
	- 滿足
        - 使用者故事皆依照此規則來表示：功能需求 (Requirement) - 滿足 -> 使用者故事 (UserStory)
        - 判斷標準：使用者故事的內容可以被某個功能需求所滿足
		- 方向性：Requirement 是主體，UserStory 是受體
		- ID 優先：如果文本中明確出現編號，優先使用編號來判斷對應關係
	- 操作
		- 功能需求皆依照此規則來表示：操作系統的角色 (Actor) - 操作 -> 功能需求 (Requirement)
        - 判斷標準：某個角色執行這個功能需求
  		- 方向性：Actor 是主體，Requirement 是受體
	
	# Relation Schema (次要關聯)
	- 包含於
    	- 組成關係，「子」包含於「父」中 (如: 記憶體管理 - 包含於 -> 作業系統)
        - 判斷標準：兩個實體間有明確的組成關係

	關係優先順序：
	1. traceability relations
	2. 系統結構（包含於、使用）
	3. 語義關係（產出、改善）

	# Relation Constraints
    - 請遵循 Actor - 操作 -> Requirement - 滿足 -> UserStory 的邏輯關係來提取三元組。
    - 角色 (Actor) 與使用者故事 (UserStory) 之間「不得建立關係」。
    - 使用者故事之間「不得建立關係」。
	- 需求之間預設「不得建立關係」，除非有明確層級關係（parent-child），例如：FR-CUS-RES-01 -> FR-CUS-RES-01.1，否則不可建立 Requirement -> Requirement 關係。

	# Rules
    1. 編號提取：
		- 任何符合 <英文字母>+<數字> 格式的編號都必須被識別並提取
		- 編號應該放在 `us_id` 或 `req_id` 屬性中，而不是實體的 name 中
		- 如果編號出現在「代號：」或「編號：」後面，立即提取，不要跳過
	2. 嚴格限制：僅限於提取「實體列表」中出現的術語之間的關係。
	3. 方向性：確保 Subject 是主體，Object 是受體。
    	- 例如：(索引 - 改善 -> 查詢效率)，語意/邏輯順序不可顛倒。
	4. 精煉關係：關係詞應為動詞或動詞短語，且盡量使用上述 Schema 中的詞彙。
	5. 顯著性：只提取文章中明確支持的關係，避免主觀臆測。
	6. 連通性：若存在多個由文件明確支持的關係，請全部列出。不得僅為提高圖譜密度而建立關係。
	7. 可追蹤性：若文本中出現任何 ID 或編號（如 req_id, module_id, case_id），必須保留並與實體綁定，不可擅自解讀編號或自行生成編號。
	8. 屬性提取限制：僅能使用文本中明確出現的資訊，不可推測或補齊缺失欄位。
	9. 一致性：三元組中的實體名稱必須與「實體列表」完全一致。
	10. 禁止無根據連接：
		- 禁止僅因語意相關（如都屬於顧客或訂單）就建立關係
		- 禁止因為都是使用者故事或都是功能需求就建立關係
    
    # Example
	- Input: 
	- 實體列表：['外送員', '身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄', '外送員可查看歷史配送紀錄']
	- 文本：
		```
		- **代號**：DL-TRACK-08 View order history  
		- **故事**：身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄。
		
		編號：FR-DL-04 
		功能需求：外送員可查看歷史配送紀錄
		```
	- Output:
	[{
		'subject': {
			'name': '外送員',
			'label': 'Actor',
			'properties': [
			{
				'key': 'description',
				'value': '負責接單和配送餐點的系統使用者角色'
			}
			]
		},
		'relation': {
			'name': '操作',
			'description': '外送員執行查看配送紀錄的功能'
		},
		'object': {
			'name': '外送員可查看歷史配送紀錄',
			'label': 'Requirement',
			'properties': [
			{
				'key': 'name',
				'value': '外送員可查看歷史配送紀錄'
			},
			{
				'key': 'req_id',
				'value': 'FR-DL-04'
			},
			{
				'key': 'description',
				'value': '外送員可查看歷史配送紀錄'
			},
			{
				'key': 'req_category',
				'value': '功能需求'
			}
			]
		}
		},
		{
		'subject': {
			'name': '外送員可查看歷史配送紀錄',
			'label': 'Requirement',
			'properties': [
			{
				'key': 'req_id',
				'value': 'FR-DL-04'
			},
			{
				'key': 'description',
				'value': '外送員可查看歷史配送紀錄'
			},
			{
				'key': 'req_category',
				'value': '功能需求'
			}
			]
		},
		'relation': {
			'name': '滿足',
			'description': '功能需求 FR-DL-04 滿足使用者故事 DL-TRACK-08 的需求'
		},
		'object': {
			'name': '身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄',
			'label': 'UserStory',
			'properties': [
			{
				'key': 'us_id',
				'value': 'DL-TRACK-08'
			},
			{
				'key': 'description',
				'value': '身為外送員，我希望能查看歷史訂單紀錄，以便回顧過往工作紀錄'
			}
			]
		}
		}
	]
"""

ENTITY_PROMPT_4_SDD = """
  	# Role
	你是一位資深的「軟體工程」專家，擅長將非結構化文本轉化為結構化的知識圖譜實體。

	# Task
	請從提供的設計文件中提取名詞實體。這些實體必須具備與其他概念建立關聯的潛力，並能夠跟其他文件 (需求文件、測試文件) 連結。
  	設計文件需提取介面設計、實作方案及設計議題。

	# Section Filtering
	明確忽略以下區段：
	- 版次、目錄
	- 使用者畫面設計

	# Entity Categories (優先提取以下類別)
    - SystemComponent: 系統中具有明確職責且可獨立實作的軟體元件，禁止將資料庫、框架或程式語言視為 SystemComponent (如: CartService, OrderController)
	- Technology: 具體技術產品或框架 (如: Nuxt, MongoDB)
    - API: 介面設計，以 endpoint 為單位提取，並保留 path、method (如: GET /api/admin/restaurants, POST /api/orders/id/chats)
	禁止提取「功能描述」、「平台特色」類實體。
    若同時存在「功能描述」、「實際元件名稱」，請優先提取實際元件名稱。例如：「購物車管理模組 CartService」應提取 'CartService' 而非「購物車管理」。
	
	# Table Parsing Rules
	若文件包含 Markdown 表格：
	1. 必須逐「列」處理，每一列視為一個獨立項目
	3. 不可合併多列內容
	4. 不可省略任何一列（即使語意相似）
    
    # Mermaid
    如果內容包含 mermiad 圖表 (如：架構圖)，請解析並提取屬於 Entity Categories 中的實體。

	# Rules
	1. 去噪點：嚴禁提取「系統」、「文章」、「功能」、「用戶」、「免費」、「工期」、「效能」等過於一般化或描述性的詞彙。
	2. 標準化 (Normalization)：將同義詞映射至標準術語。例如：「版控」、「版本控制」統稱為「版本控制」；「寫 Code」統一為「程式編碼」。
	3. 具體性：實體必須是一個獨立的知識點。如果一個實體在脫離本文後無法定義為一個專業術語，請忽略它。
	4. 關係導向：只保留「可以被定義、被解釋、或與其他實體產生邏輯連接」的實體。
	5. 唯一性：確保輸出的 Python list 中沒有重複項。
  	6. 粒度控制：優先提取可獨立實作的元件或介面，不得將多個不同 API 或元件合併為單一抽象概念。
    7. 嚴禁推論：僅提取文件中明確出現的實體，不得根據常見軟體架構、開發經驗或技術慣例新增實體。

	# Example
	- Input: '本系統前後端以 RESTful API 為介接方式。介面提供者（後端模組）依功能分為 Users、Restaurants、Cart、Order 四大模組；介面使用者為前端各獨立頁面/模組。所有跨模組溝通均透過 HTTP + JSON 進行，不直接共享資料庫，並以 JWT 控制授權與角色權限。'
	- Output: ['RESTful', 'Users', 'Restaurant', 'Cart', 'Order', 'HTTP', 'JSON', 'JWT']

	# Output Format
	請直接輸出一個 Python List，List 中僅包含提取的實體 (string)，不需以註解分類也不需任何額外解釋。
"""

TRIPLE_PROMPT_4_SDD = """
  	# Role
	你是一位知識工程師，專精於從軟體工程文本中建構語義網絡。

	# Task
	根據提供的「實體列表」，從「設計文件內容」中找出這些實體的標籤及屬性，以及實體之間的邏輯關係，並以三元組 (Subject, Relation, Object) 的形式輸出。請務必根據文件內容替每個實體與關係加上描述，並分別記錄於實體(Entity)的屬性中(properties)，key 為 'description'，以及關係(Relation)的description中。description 應為單句摘要，不超過 30 個中文字。

	# Entity Schema
	每個實體的資訊，包含標籤 (label) 及屬性 (properties)。對於 properties，請務必使用 [{'key': '...', 'value': '...'}] 的列表格式，請勿使用字典 (dictionary) 格式。
	- label: 
        - SystemComponent: 程式碼層級或框架內部的結構
        - Service: 系統中具有明確職責且可獨立實作的軟體元件
		- Technology: 具體技術產品或框架
		- API: 介面端點
	- properties: 
		- name: 實體的唯一名稱
		- description: 實體的描述
		- req_reference: 對應的功能需求或非功能需求
        - api_provider: 介面提供者
        - input_value: 介面輸入值
        - output_value: 介面輸出值
        - api_user: 介面使用者 (模組)

	# Core Traceability Relations (最高優先)
    - 使用：使用技術 (如: WebService - 使用 -> HTTP 協定)
    - 提供：提供介面 (如: ReviewComponent - 提供 -> get `/api/reviews`)
    - 呼叫：呼叫服務/介面 (如: patch `/api/admin/restaurants/{id}` - 呼叫 -> DBService)
	- 實作需求：系統元件實作的需求 (如: ReviewComponent - 實作需求 -> 餐廳評論管理)
	
	# Relation Schema (次要關聯)
	- 包含於：組成關係，「子」包含於「父」中 (如: 記憶體管理 - 包含於 -> 作業系統)

	關係優先順序：
	1. traceability relations
	2. 系統結構（部分、使用）
	3. 語義關係（是、改善）

	# ID-based Linking
	若實體包含文件內容中包含可追溯需求文件之資訊 (req_reference)，請務必提取並記錄於屬性 (properties) 中。
    
    # Mermaid
    若文件內容包含 mermaid 圖表 (如：架構圖)，請解析並根據上述定義提取實體與關係。

	# Rules
	1. 嚴格限制：僅限於提取「實體列表」中出現的術語之間的關係。
	2. 方向性：確保 Subject 是主體，Object 是受體。例如：(索引 - 改善 -> 查詢效率)，語意/邏輯順序不可顛倒。
	3. 精煉關係：關係詞應為動詞或動詞短語，且盡量使用上述 Schema 中的詞彙。
	4. 顯著性：只提取文章中明確支持的關係，避免主觀臆測。
	5. 連通性：若存在多個由文件明確支持的關係，請全部列出。不得僅為提高圖譜密度而建立關係。
	6. 可追蹤性：若文本中出現任何 ID 或編號（如 req_id, module_id, case_id），必須保留並與實體綁定，不可擅自解讀編號或自行生成編號。
	7. 屬性提取限制：僅能使用文本中明確出現的資訊，不可推測或補齊缺失欄位。
	8. 一致性：三元組中的實體名稱必須與「實體列表」完全一致。
	9. 禁止僅因語意相關（如都屬於顧客或訂單）就建立關係
    
    # Example
    - Input: 
    	介面名稱：計算外送費用
        提供者：Cart
        使用者（前端模組）：`/customer/payment`
        Method / URL：GET `/api/cart/delivery-fee`
        Input：customerLatitude, customerLongitude, restaurants
        Output：distance, deliveryFee
        描述：根據顧客與餐廳距離估算外送費
        reference：FR-CUS-CRT-03
    - Output:
		[{
		'subject': {
			'name': 'get `/api/cart/delivery-fee`',
			'label': 'API',
			'properties': [
				{
					'key': 'description',
					'value': '根據顧客與餐廳距離估算外送費。'
				},
                {
					'key': 'req_reference',
					'value': 'FR-CUS-CRT-03'
				},
                {
					'key': 'api_provider',
					'value': 'Cart'
				},
                {
					'key': 'api_user',
					'value': '/customer/payment'
				},
                {
					'key': 'input_value',
					'value': 'customerLatitude, customerLongitude, restaurants'
				},
                {
					'key': 'output_value',
					'value': 'distance, deliveryFee'
				}
			]
		},
		'relation': {
			'name': '實作需求',
            'description': '系統根據顧客與餐廳距離估算外送費。'
        },
		'object': {
			'name': '估算外送費用',
			'label': 'Requirement',
			'properties': 
				{
					'key': 'description',
					'value': '系統應即時計算餐點金額、外送費與預估時間。'
				}, 
				{
					'key': 'req_id',
					'value': 'FR-CUS-CRT-03'
				}, 
				{
					'key': 'req_category',
					'value': '功能需求'
				}
			]
		}]
"""

ENTITY_PROMPT_4_STD = """
  	# Role
	你是一位知識工程師，專精於從軟體工程文本中建構語義網絡。

	# Task
	請從提供的測試文件中提取實體。這些實體必須具備與其他概念建立關聯的潛力，並能夠跟其他文件 (需求文件、設計文件) 連結。
    請務必根據文件內容替每個實體與關係加上描述，並分別記錄於實體(Entity)的屬性中(properties)，key 為 'description'，以及關係(Relation)的description中。description 應為單句摘要，不超過 30 個中文字。
    **核心要求**：只提取文件中「明確定義」的測試案例，「禁止生成、推斷或創建」任何不存在於文檔中的測試案例。

	# Section Filtering
	明確忽略以下區段：
	- 版次、目錄
    - 測試目的與接受準則
	- 測試環境
    - 測試工作指派與時程
    - 測試結果與分析
	
	## Format: 測試案例區段
	測試案例按以下結構出現：
    - Identification：<ID> 測試案例編號，包含英文字母和數字
	- Name：測試案例名稱
	- Reference：<ID> 測試案例對應的功能需求編號，包含英文字母和數字
	- Severity：測試案例重要性
	- Instructions：測試步驟，通常會列點撰寫
	- Expected Result：預期結果
	- Cleanup：回復測試前原始狀態的步驟
    
    **關鍵約束**：每個 Identification 僅對應一個測試案例。
	
	例子：
	```
	- Identification：CUS-01
	- Name：顧客切換至外送員模式
	- Reference：FR-CUS-03
	- Severity：高
	- Instructions：
		- 使用已註冊非admin帳號，於登入介面選擇顧客並登入
		- 點選我的帳戶
		- 點選切換為外送員
	- Expected Result：
		- 跳出"身份已切換為外送員"提示
		- 頁面為"顧客訂單"
	- Cleanup：無
	```
    
	# Entity Schema
	每個實體的資訊，包含標籤 (label) 及屬性 (properties)。對於 properties，請務必使用 [{'key': '...', 'value': '...'}] 的列表格式，請勿使用字典 (dictionary) 格式。
	- label: 
        - TestCase: 測試案例，包含詳細的測試步驟，用來驗證系統是否如期運作
	- properties: 
		- name: 實體的唯一名稱
		- description: 測試案例的詳細描述，單句摘要，不超過 30 中文字，內容應包含「測試什麼」和「為什麼測試」
        - tc_id: 測試案例編號 (Identification)，請勿自行修改或臆測編號
		- req_reference: 對應的功能需求或非功能需求編號 (Reference)，如果文件有明確編號，直接提取所有編號，若有多個，請獨立為多個 req_reference
        - severity: 測試案例重要性，分為低、中、高
        - instructions: 測試步驟。若以列點方式撰寫，請改以有序列表呈現 (1.步驟一 2.步驟二)
			  規則：
				1. 必須保留所有步驟，不可刪減或合併
				2. 原文是列點格式時，轉換為序號格式
				3. 步驟間用空格分隔，最終為單行文本
				4. 原文有子步驟時，全部展平為線性序列
        - expected_result: 預期結果。若以列點方式撰寫，請改以有序列表呈現 (1.步驟一 2.步驟二)
        - cleanup: 回復測試前原始狀態的步驟

	# ID-based Linking
	若實體包含文件內容中的 Reference 欄位，請務必提取並記錄於屬性 (properties.req_reference) 中。
	- 只提取文檔中明確出現的編號
	- 禁止根據編號反向推斷測試案例
	- 禁止補充或臆測編號
    
	# Rules
    
    ## 最高優先級：禁止幻覺和假陽性
 
	**規則 1：禁止根據 Reference 推斷測試案例**
	- 禁止根據 Reference 欄位中出現的編號推斷或生成不存在的測試案例
	- 禁止根據需求編號反向構造測試案例
	
	**規則 2：禁止創建文件中未明確定義的測試案例**
	- 禁止創建沒有對應 **Identification** 欄位的測試案例
	- 禁止把文檔外的內容當作測試案例
	- 驗證方法：每個提取的 TestCase 都必須滿足以下條件：
	1. 在文檔中有明確的 **Identification** 欄位
	2. 在文檔中有明確的 **Name** 欄位
	3. 在文檔中有明確的 **Instructions** 欄位
	如果以上任何一項無法滿足，則不應提取該 TestCase
	
	**規則 3：嚴禁推論**
	- 嚴格忠於文件中的資訊，不擅自臆測
	- 不根據功能名稱推斷 req_reference
	- 不補充文件中沒有的屬性值
	- 不假設文件中的隱含含義
    
    ## 次優先級規則
    
	1. 去噪點：嚴禁提取「系統」、「文章」、「功能」、「用戶」、「免費」、「工期」、「效能」等過於一般化或描述性的詞彙。
	2. 標準化 (Normalization)：將同義詞映射至標準術語。例如：「版控」、「版本控制」統稱為「版本控制」；「寫 Code」統一為「程式編碼」。
	3. 具體性：實體必須是一個獨立的知識點。如果一個實體在脫離本文後無法定義為一個專業術語，請忽略它。
	4. 關係導向：只保留「可以被定義、被解釋、或與其他實體產生邏輯連接」的實體。
  	5. 粒度控制：優先提取可獨立實作的元件或介面，不得將多個不同 API 或元件合併為單一抽象概念。
    6. 嚴禁推論：
		- 請嚴格忠於文件中的資訊，不擅自臆測
		- 不根據功能名稱推斷 req_reference
		- 不新增文件中沒有的屬性值
    7. 唯一性：同一個 tc_id 只提取一次，不同 tc_id 視為不同實體。
    
    # Example
    - Input: 
    ```
	- **Identification**：CUS-01
	- **Name**：顧客切換至外送員模式
	- **Reference**：FR-CUS-03, FR-CUS-04
	- **Severity**：高
	- **Instructions**：
		- 使用已註冊非admin帳號，於登入介面選擇顧客並登入
		- 點選我的帳戶
		- 點選切換為外送員
	- **Expected Result**：
		- 跳出"身份已切換為外送員"提示
		- 頁面為"顧客訂單"
	- **Cleanup**：無
    ```
    - Output:
	[{
		'name': '顧客切換至外送員模式',
		'label': 'TestCase',
		'properties': [
			{
				'key': 'description',
				'value': '驗證顧客是否能成功切換至外送員身份並更新介面'
			},
			{
				'key': 'tc_id',
				'value': 'CUS-01'
			},
			{
				'key': 'req_reference',
				'value': 'FR-CUS-03'
			},
			{
				'key': 'req_reference',
				'value': 'FR-CUS-04'
			},
			{
				'key': 'severity',
				'value': '高'
			},
			{
				'key': 'instructions',
				'value': '1.使用已註冊非admin帳號，於登入介面選擇顧客並登入 2.點選我的帳戶 3.點選切換為外送員'
			},
			{
				'key': 'expected_result',
				'value': '1.跳出身份已切換為外送員提示 2.頁面為顧客訂單'
			},
			{
				'key': 'cleanup',
				'value': 'None'
			}
		]
	}]
"""

ENTITY_EXAMINATION_PROMPT = """
    請根據提供的文章，及從文章中提取的知識圖譜實體，判斷每個實體是否正確/合理，並依照指定格式輸出修正後的實體。
    
    你需要判斷：
    - 實體內容是否合理(包含屬性內容)

    以下是已定義的實體列表：
    - label:
        - Actor: 操作系統 (功能) 的角色
        - Requirement: 功能需求與非功能需求 (識別信號: 同時出現「代號：」和「故事：身為/作為...」)
        - UserStory: 使用者故事 (識別信號: 出現 <編號> <需求描述>)
        - SystemComponent: 程式碼層級或框架內部的結構
        - Service: 系統中具有明確職責且可獨立實作的軟體元件
        - API: 介面端點
        - Concept: 核心概念與原理 (如: 耦合度, 內聚力, 微服務, 物件導向)
        - Technology: 具體技術產品或框架
        - Methodology: 開發流程與方法論 (如: CI/CD, Scrum, 測試驅動開發)
    - properties:
    	- name: 實體的唯一名稱
		- description: 實體的描述 （從文本中提取）
        - us_id: 使用者故事的代號 （格式如 DL-TRACK-08, US-CUS-01 等）
		- req_id: 功能需求和非功能需求的編號 (格式如 FR-DL-04, NFR-SYS-01, USR03 等）
		- req_category: 需求類別，僅分為「功能需求」或「非功能需求」（只適用於 Requirement）
		- req_reference: 對應的功能需求或非功能需求
        - api_provider: 介面提供者 （只適用於 API)
        - input_value: 介面輸入值 （只適用於 API)
        - output_value: 介面輸出值 （只適用於 API)
        - api_user: 介面使用者 (模組) （只適用於 API)
"""

KG_EXAMINATION_PROMPT = """
    請根據提供的文章，及從文章中提取的知識圖譜三元組，判斷每個三元組是否正確/合理，並依照指定格式輸出修正後的三元組。
    
    你需要判斷：
    - 實體內容是否合理(包含屬性內容)
    - 實體之間的關係及方向(主體及受體)是否正確

    以下是已定義的實體列表：
    - label:
        - Actor: 操作系統 (功能) 的角色
        - Requirement: 功能需求與非功能需求 (識別信號: 同時出現「代號：」和「故事：身為/作為...」)
        - UserStory: 使用者故事 (識別信號: 出現 <編號> <需求描述>)
        - SystemComponent: 程式碼層級或框架內部的結構
        - Service: 系統中具有明確職責且可獨立實作的軟體元件
        - API: 介面端點
        - Concept: 核心概念與原理 (如: 耦合度, 內聚力, 微服務, 物件導向)
        - Technology: 具體技術產品或框架
        - Methodology: 開發流程與方法論 (如: CI/CD, Scrum, 測試驅動開發)
    - properties:
    	- name: 實體的唯一名稱
		- description: 實體的描述 （從文本中提取）
        - us_id: 使用者故事的代號 （格式如 DL-TRACK-08, US-CUS-01 等）
		- req_id: 功能需求和非功能需求的編號 (格式如 FR-DL-04, NFR-SYS-01, USR03 等）
		- req_category: 需求類別，僅分為「功能需求」或「非功能需求」（只適用於 Requirement）
		- req_reference: 對應的功能需求或非功能需求
        - api_provider: 介面提供者 （只適用於 API)
        - input_value: 介面輸入值 （只適用於 API)
        - output_value: 介面輸出值 （只適用於 API)
        - api_user: 介面使用者 (模組) （只適用於 API)

    以下是已定義的關係列表：
    - 滿足：使用者故事皆依照此規則來表示：功能需求 (Requirement) - 滿足 -> 使用者故事 (UserStory)
    - 操作：功能需求皆依照此規則來表示：操作系統的角色 (Actor) - 操作 -> 功能需求 (Requirement)
    - 包含於：組成關係，「子」包含於「父」中 (如: 記憶體管理 - 包含於 -> 作業系統)
    - 使用：使用技術 (如: WebService - 使用 -> HTTP 協定)
    - 提供：提供介面 (如: ReviewComponent - 提供 -> get `/api/reviews`)
    - 呼叫：呼叫服務/介面 (如: patch `/api/admin/restaurants/{id}` - 呼叫 -> DBService)
	- 實作需求：系統元件實作的需求 (如: ReviewComponent - 實作需求 -> 餐廳評論管理)
"""

MATCH_FR_2_US_PROMPT = """
    請根據提供的一個功能需求，以及一個使用者故事候選列表，判斷哪個使用者故事可對應該功能需求，並以指定格式輸出已配對的「使用者故事」。
    
    你需要逐一判斷：
    - 使用者故事是否能對應此功能需求
    
    若有使用者故事對應此功能需求，請於該使用者故事中新增 req_reference 紀錄該功能需求的編號：
    'label': 'UserStory',
    'properties':{
		'key': 'req_reference',
        'value': <Requirement 的 req_id>
    }
"""

MATCH_US_2_FR_PROMPT = """
    請根據提供的一個使用者故事，以及一個功能需求候選列表，判斷哪些功能需求可對應該使用者故事，並以指定格式輸出已配對的「使用者故事」。
    
    你需要逐一判斷：
    - 功能需求是否能對應此使用者故事
    
    若有功能需求對應此使用者故事，請於使用者故事中新增 req_reference 紀錄該功能需求的編號：
    'label': 'UserStory',
    'properties':{
		'key': 'req_reference',
        'value': <Requirement 的 req_id>
    }
"""