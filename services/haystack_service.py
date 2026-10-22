import os, asyncio, json, re, sys
from typing import List, Dict, Any
from pathlib import Path

import inspect

import difflib
import anthropic
from opencc import OpenCC
from haystack import Document, Pipeline, component
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.anthropic import AnthropicGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from neo4j_haystack import Neo4jEmbeddingRetriever, Neo4jDocumentStore
from neo4j import GraphDatabase

from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import file_processor
from config import OPENAI_API_KEY, NEO4J_PASSWORD, CLAUDE_API_KEY, GEMINI_API_KEY
from common import TASK_CONFIGS

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ----- 初始化元件 -----
document_store = Neo4jDocumentStore(
    url="neo4j://localhost:7687",
    username="neo4j",
    password=NEO4J_PASSWORD,
    database="neo4j",
    embedding_dim=384,
    index="document-embeddings",
)

# ----- custom components -----
@component
class SourceIdentifier:
    # LLM 判斷跟問題相關的章節，並回傳 filters
    # 目前沒用
    def __init__(self):
        self.client = AnthropicGenerator(api_key=Secret.from_token(CLAUDE_API_KEY), model="claude-haiku-4-5")

    @component.output_types(filters=Dict[str, Any])
    def run(self, user_input: str):
        source_identify_prompt = f"""
        你是一個「軟體工程問題分類器」，請根據使用者的問題內容，判斷該問題可以從哪「兩個」軟體工程課程章節得到答案，並僅輸出章節名稱(包含編號)，中間用頓號(
        、)隔開。

        【章節清單】
        - [01]軟體危機與軟體流程
        - [02]基礎需求工程
        - [03]使用者故事分析
        - [04]敏捷開發方法
        - [05]基礎專案管理與看板
        - [06]版本控制
        - [07]軟體設計-系統設計
        - [08]軟體設計-模組設計
        - [09]軟體測試
        - [10]進階軟體測試
        - [11]DevOps自動化建置管理

        【使用者問題】
        {user_input}
        """
        res = self.client.run(prompt=source_identify_prompt)
        sources = res['replies'][0].split("、")
        print(f"來源資料標題們：{sources}")
        return {
            "operator": "OR",
            "conditions":[
                {
                    "field": "source_file",
                    "operator": "==",
                    "value": f"{source}.pdf"
                } for source in sources
            ]
        }

# 執行 Neo4j 查詢的自訂義 component
# 目前沒在用
@component
class Neo4jExecutor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @component.output_types(query_result=str)
    def run(self, keywords: str):
        kw_list = keywords.split("、")
        print(f"[keywords]: {kw_list}")

        # 使用模糊匹配 (fuzzyMatch) 檢索相關的節點 (Concept)
        cypher = """
        WITH $keywords AS keywords
        UNWIND keywords AS keyword
        MATCH (concept:Concept)
        WHERE apoc.text.fuzzyMatch(concept.name, keyword)
        RETURN DISTINCT concept.name AS concept
        LIMIT 20;
        """

        with self.driver.session() as session:
            try:
                result = session.run(cypher, keywords=kw_list)
                concepts = [record['concept'] for record in result]
                concepts.extend(kw_list)
                # records = [str(record.data()) for record in result]
                if not concepts:
                    return {"query_result": []}
                
                print(f"[debug] received concepts: \n{concepts}")
                return {"query_result": concepts}
            
            except Exception as e:
                return {"query_result": f"Cypher 執行錯誤: {str(e)}"}

@component
class DescriptionBasedReasoning:
    """利用節點和邊的 description 直接推理"""
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @component.output_types(knowledge_base=str, matched_nodes=List[str])
    def run(self, keywords: str, group_id: str = None):  
        kw_list = keywords.split("、")

        print(f"[keywords]: {kw_list}")  

        with self.driver.session() as session:
            try:
                # 跑小組開發文件檢索
                if group_id:
                    cypher = """
                    WITH $keywords AS keywords
                    UNWIND keywords AS keyword
                    MATCH (entity)
                    WHERE apoc.text.fuzzyMatch(entity.name, keyword) and entity.group = $group
                    OPTIONAL MATCH (entity)-[rel]-(neighbor)
                    RETURN 
                        entity as entity,
                        labels(entity)[0] AS entityLabel,
                        type(rel) AS relationType,
                        rel.description AS relationDesc,
                        neighbor.name AS neighborName,
                        neighbor.description AS neighborDesc,
                        labels(neighbor)[0] AS neighborLabel
                    LIMIT 20;
                    """
                    result = session.run(cypher, keywords=kw_list, group=group_id)
                # 跑教材檢索
                else:
                    cypher = """
                    WITH $keywords AS keywords
                    UNWIND keywords AS keyword
                    MATCH (entity)
                    WHERE apoc.text.fuzzyMatch(entity.name, keyword)
                    OPTIONAL MATCH (entity)-[rel]-(neighbor)
                    RETURN 
                        entity as entity,
                        labels(entity)[0] AS entityLabel,
                        type(rel) AS relationType,
                        rel.description AS relationDesc,
                        neighbor.name AS neighborName,
                        neighbor.description AS neighborDesc,
                        labels(neighbor)[0] AS neighborLabel
                    LIMIT 20;
                    """
                    result = list(session.run(cypher, keywords=kw_list))

                knowledge_text = self._format_knowledge(result)

                matched_node_names = list({record["entity"]["name"] for record in result if record["entity"]})
                
                # records = [str(record.data()) for record in result]
                if not knowledge_text:
                    return {"knowledge_base": "查無資料"}
                
                print(f"[debug] knowledge text: \n{knowledge_text}")
                return {
                    "knowledge_base": knowledge_text,
                    "matched_nodes": matched_node_names
                    }
            
            except Exception as e:
                print(f"[error] Cypher 執行錯誤: {str(e)}")
                return {"knowledge_base": "查無資料"}

    def _format_knowledge(self, results):
        """將查詢結果格式化為自然語言"""
        records = list(results)
        
        if not records:
            return None
        
        # 依節點分組
        entity_info = {}
        for record in records:
            entity_dict = dict(record['entity'])
            entity_name = entity_dict['name']
            entity_label = record['entityLabel']

            if entity_name not in entity_info:
                entity_info[entity_name] = {
                    'label': entity_label,
                    'properties': entity_dict,
                    'relations': []
                }
            
            if record['neighborName']:
                entity_info[entity_name]['relations'].append({
                    'type': record['relationType'],
                    'typeDesc': record['relationDesc'],
                    'target': record['neighborName'],
                    'targetLabel': record['neighborLabel'],
                    'targetDesc': record['neighborDesc']
                })
        
        knowledge_text = []
        for entity_name, info in entity_info.items():
            label = info['label']
            props = info['properties']

            # 根據節點類型選擇格式化方法
            if label == 'API':
                formatted = self._format_api_entity(entity_name, props)
            elif label == 'Requirement':
                formatted = self._format_requirement_entity(entity_name, props)
            elif label == 'UserStory':
                formatted = self._format_userstory_entity(entity_name, props)
            elif label == 'SystemComponent':
                formatted = self._format_system_component_entity(entity_name, props)
            else:
                formatted = self._format_default_entity(entity_name, props)

            knowledge_text.append(formatted)
            
            # 相關實體
            if info['relations']:
                knowledge_text.append("\n相關概念：")
                for rel in info['relations']:
                    rel_desc = rel['typeDesc'] or f"({rel['type']})"
                    knowledge_text.append(
                        f"- {entity_name} {rel['type']} {rel['target']}: {rel_desc}"
                    )
                    if rel['targetDesc']:
                        knowledge_text.append(f"    {rel['target']} 是 {rel['targetDesc']}")
            
            knowledge_text.append("")
        
        return "\n".join(knowledge_text)

    def _format_api_entity(self, name: str, props: dict) -> str:
        """API 節點的特殊格式化"""
        lines = [f"【API】{name}"]
        
        if props.get('description'):
            lines.append(f"描述：{props['description']}")
        
        if props.get('api_provider'):
            lines.append(f"提供者：{props['api_provider']}")
        
        if props.get('input_value'):
            lines.append(f"輸入：{props['input_value']}")
        
        if props.get('output_value'):
            lines.append(f"輸出：{props['output_value']}")
        
        if props.get('api_user'):
            lines.append(f"使用者：{props['api_user']}")
        
        if props.get('req_reference'):
            req_ref = props['req_reference']
            if isinstance(req_ref, list):
                req_ref = '、'.join(req_ref)
            lines.append(f"對應需求：{req_ref}")
        
        return "\n".join(lines)

    def _format_requirement_entity(self, name: str, props: dict) -> str:
        """Requirement 節點的特殊格式化"""
        lines = [f"【需求】{name}"]
        
        if props.get('description'):
            lines.append(f"描述：{props['description']}")
        
        if props.get('req_id'):
            lines.append(f"編號：{props['req_id']}")
        
        if props.get('req_category'):
            lines.append(f"類別：{props['req_category']}")
        
        return "\n".join(lines)

    def _format_userstory_entity(self, name: str, props: dict) -> str:
        """UserStory 節點的特殊格式化"""
        lines = [f"【使用者故事】{name}"]
        
        if props.get('description'):
            lines.append(f"描述：{props['description']}")
        
        if props.get('us_id'):
            lines.append(f"編號：{props['us_id']}")

        if props.get('req_reference'):
            req_ref = props['req_reference']
            if isinstance(req_ref, list):
                req_ref = '、'.join(req_ref)
            lines.append(f"對應需求：{req_ref}")
        
        return "\n".join(lines)

    def _format_system_component_entity(self, name: str, props: dict) -> str:
        """SystemComponent 節點的特殊格式化"""
        lines = [f"【系統元件】{name}"]
        
        if props.get('description'):
            lines.append(f"描述：{props['description']}")
        
        # 如果有其他特定的服務屬性，這裡可以加
        
        return "\n".join(lines)

    def _format_default_entity(self, name: str, props: dict) -> str:
        """預設格式化"""
        lines = [f"【{props.get('type', '概念')}】{name}"]
        
        if props.get('description'):
            lines.append(f"描述：{props['description']}")
        
        return "\n".join(lines)

@component
class VectorSearchFilter:
    def __init__(self, embedder, retriever, top_k=5):
        self.embedder = embedder
        self.retriever = retriever
        self.top_k = top_k

    @component.output_types(filtered_documents=List[Document])
    def run(self, question: str, concepts: List[str]):
        if not concepts:
            # 如果沒有概念，直接用向量檢索
            embedding = self.embedder.run(question)['embedding']
            docs = self.retriever.run(
                query_embedding=embedding,
                top_k=self.top_k
            )
            return {"filtered_documents": docs.get('documents', [])}
        
        # 構建過濾條件：文檔必須包含至少一個概念標籤
        filters = {
            "operator": "OR",
            "conditions": [
                {
                    "field": "tags",
                    "operator": "in",
                    "value": concept
                } for concept in concepts
            ]
        }
        
        # 向量檢索 + 過濾
        # run_question = f"關鍵字：{'、'.join(concepts)}\n問題：{question}"
        # print(f"[傳入的概念]: {concepts}")
        embedding = self.embedder.run('、'.join(concepts))['embedding']
        docs = self.retriever.run(
            query_embedding=embedding,
            top_k=self.top_k,
            filters=filters
        )
        
        filtered_docs = docs.get('documents', [])
        print(f"[向量檢索返回]: {len(filtered_docs)} 個文檔")
        print(filtered_docs)
        return {"filtered_documents": filtered_docs}

# ----- 上傳向量檔案 -----
# 目前用的上傳方法
def upload_to_neo4j(documents: list[Document]):
    document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)

    document_store.write_documents(documents_with_embeddings.get("documents"))

    print(document_store.count_documents())

# 替 chunk 加上 metadata 並轉為 Document 物件
def add_metadata(md_documents: list[str], content_tags: list, type: str, **kwargs) -> list[Document]:
    documents = list()
    if type == "doc":
        for doc, content_tag in zip(md_documents, content_tags):
            if doc:
                documents.append(
                    Document(
                        content=f"{content_tag}\n\n{doc}",
                        meta={
                            "doc_type": kwargs["doc_type"],
                            "group": kwargs["group_name"],
                            "uploader": kwargs["uploader"],
                            "store_type": "vector",
                            "tags": content_tag
                        }
                    )
                )
    elif type == "textbook":
        for doc, content_tag in zip(md_documents, content_tags):
            if doc:
                documents.append(
                    Document(
                        content=f"{content_tag}\n\n{doc}",
                        meta={
                            "content_type": "textbook",
                            "source_file": f"{kwargs['textbook_name']}.pdf",
                            "tags": content_tag
                        }
                    )
                )
    return documents

# ----- pipeline builders functions -----
def _build_retriever_pipeline(prompt_builder: ChatPromptBuilder) -> Pipeline:
    ranker = SentenceTransformersSimilarityRanker(top_k=3)

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    pipeline.add_component("retriever", Neo4jEmbeddingRetriever(document_store=document_store, scale_score=False))
    pipeline.add_component("ranker", ranker)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini"))

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "ranker.documents")
    pipeline.connect("ranker.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    return pipeline

def _build_kg_pipeline(kw_prompt_builder: PromptBuilder) -> Pipeline:
    answer_prompt_template = """
    你是一位專業的軟體工程教學助教。請根據以下檢索到的資料，用台灣繁體中文回答學生的問題。

    【資料】
    {{documents}}

    【學生問題】
    {{question}}

    請基於教學資料回答，總字數不超過 1000 字。
    如果資料不足，請說「資料不足無法完整回答」。
    答案：
    """

    pipeline = Pipeline()

    # 關鍵字 + 圖譜中的 description
    pipeline.add_component("kw_prompt", kw_prompt_builder)
    pipeline.add_component("kw_llm", AnthropicGenerator(api_key=Secret.from_token(CLAUDE_API_KEY), model="claude-haiku-4-5"))
    pipeline.add_component("desc_reasoner",  DescriptionBasedReasoning("bolt://localhost:7687", "neo4j", NEO4J_PASSWORD))
    pipeline.add_component("answer_prompt", PromptBuilder(template=answer_prompt_template, required_variables=["question"]))
    pipeline.add_component("answer_llm", OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini"))

    pipeline.connect("kw_prompt.prompt", "kw_llm.prompt")
    pipeline.connect("kw_llm.replies", "desc_reasoner.keywords")
    pipeline.connect("desc_reasoner.knowledge_base", "answer_prompt.documents")
    pipeline.connect("answer_prompt.prompt", "answer_llm.prompt")

    # 舊的 KG 檢索 (純關鍵字)
    # pipeline.add_component("kw_prompt", kw_prompt_builder)
    # pipeline.add_component("cypher_llm", AnthropicGenerator(api_key=Secret.from_token(CLAUDE_API_KEY), model="claude-haiku-4-5"))
    # pipeline.add_component("neo4j_executor",  Neo4jExecutor("bolt://localhost:7687", "neo4j", NEO4J_PASSWORD))
    # pipeline.add_component("answer_prompt", PromptBuilder(template=answer_prompt_template))
    # pipeline.add_component("vector_filter", VectorSearchFilter(
    #     embedder=_embedder,
    #     retriever=_neo4j_retriever,
    #     top_k=5
    # ))
    # pipeline.add_component("answer_llm", OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini"))

    # pipeline.connect("kw_prompt.prompt", "cypher_llm.prompt")
    # pipeline.connect("cypher_llm.replies", "neo4j_executor.keywords")
    # pipeline.connect("neo4j_executor.query_result", "vector_filter.concepts")
    # pipeline.connect("vector_filter.filtered_documents", "answer_prompt.documents")
    # pipeline.connect("answer_prompt.prompt", "answer_llm.prompt")

    return pipeline

def CRIT_check(question: str, bot_response: str) -> float:
    # CRIT 檢查
    CRIT_prompt_template = """
    你現在是一個具備嚴謹批判性思維的「邏輯驗證引擎」，負責執行 CRIT (Critical Reading Inquisitive Template) 演算法。你的任務是檢視【機器人回答】是否正面、有效且具備說服力地回答了【使用者問題】。
    請根據以下四個步驟進行分析：

    【輸入資訊】
    * 使用者問題（結論目標 Ω）：{{question}}
    * 機器人回答（待驗證文本 d）：{{bot_response}}

    【分析步驟】
    1. 擷取與定義 (Definition)：從【機器人回答】中，精煉出其核心結論，並條列出數個獨立的支持理由 (r_n)。
    2. 詰問與驗證 (Elenchus)：針對每一個理由 r ，評估其推導至【使用者問題】(r => Ω) 的「邏輯有效性」與 r 的「來源可信度」。請給予 1 到 10 分的整數評分（10 分為最強）。
    3. 評估邏輯有效性時，除了檢查理由本身是否正確，還必須評估其對【使用者問題】的整體覆蓋率與完整性。如果回答正確但存在嚴重遺漏，必須大幅扣除正方理由的有效性分數（降至 5 分以下）。
    4. 辯證與反方意見 (Dialectic)：請針對上述最弱的論點，或者整體回答的盲點，在軟工領域提出一個具備建設性的反方意見 (r'，例如：遺漏的上下文、適用的極限條件)，評估其推導至【使用者問題】(r' => Ω) 的「邏輯有效性」與 r'的「來源可信度」並同樣給予 1 到 10 分。
    【輸出限制】
    請勿輸出任何解釋性文字或 Markdown 標記，必須嚴格遵守以下的純 JSON 格式直接輸出：

    {
      "core_conclusion": "請在此填入精煉出的核心結論",
      "supporting_reasons": [
        {
          "id": "r1",
          "content": "第一個支持理由的具體內容",
          "validity_score": 邏輯有效性評分 (1-10的整數),
          "credibility_score": 來源可信度評分 (1-10的整數),
          "justification": "給予此分數的簡短原因"
        }
      ],
      "counter_argument": {
        "id": "r_prime",
        "content": "反方意見或整體盲點",
        "validity_score": 邏輯有效性評分 (1-10的整數),
        "credibility_score": 來源可信度評分 (1-10的整數),
        "justification": "給予此反方意見分數的原因"
      }
    }
    """

    pipeline = Pipeline()
    
    pipeline.add_component("crit_prompt", PromptBuilder(template=CRIT_prompt_template, required_variables=["question", "bot_response"]))
    # 這裡使用 gpt-4o-mini 作為快速邏輯驗證引擎
    pipeline.add_component("crit_llm", OpenAIGenerator(
                                            api_key=Secret.from_env_var("OPENAI_API_KEY"), 
                                            model="gpt-4o-mini",
                                            generation_kwargs={"temperature": 0.3 })
                            )
    
    pipeline.connect("crit_prompt.prompt", "crit_llm.prompt")

    result = pipeline.run(
        data={
            "crit_prompt": {
                "question": question, 
                "bot_response": bot_response
            }
        }
    )

    llm_output_str = result["crit_llm"]["replies"][0]

    clean_json = llm_output_str.strip('`').removeprefix('json').strip()


    # try:
    #     parsed_result = json.loads(clean_json)
    #     final_score = float(parsed_result.get("crit_score", 0.0))
    #     print(f"[debug] CRIT 驗證分數: {final_score}")
    #     return final_score
    # except json.JSONDecodeError as e:
    #     print(f"[error] 無法解析 CRIT 輸出為 JSON: {llm_output_str}")
    #     return 0.0

    try:
            parsed_result = json.loads(clean_json)

            validity = []
            credibility = []
            
            print("\n=== CRIT 驗證報告 ===")
            print(f"核心結論: {parsed_result.get('core_conclusion')}")
            
            print("\n[支持理由]")
            for reason in parsed_result.get("supporting_reasons", []):
                print(f"- {reason['id']}: {reason['content']}")
                print(f"  > 有效性: {reason['validity_score']}, 可信度: {reason['credibility_score']}")
                print(f"  > 評分理由: {reason['justification']}")

                validity.append(reason['validity_score'])
                credibility.append(reason['credibility_score'])
                
            counter = parsed_result.get("counter_argument", {})
            if counter:
                print("\n[反方意見/盲點]")
                print(f"- {counter.get('id')}: {counter.get('content')}")
                print(f"  > 有效性: {counter.get('validity_score')}, 可信度: {counter.get('credibility_score')}")
                print(f"  > 評分理由: {counter.get('justification')}")

                validity.append(counter.get('validity_score'))
                credibility.append(counter.get('credibility_score'))

            # 計算最終 CRIT 分數
            temp_score = 0.0
            for v, c in zip(validity, credibility):
                temp_score= v * c + temp_score

            if len(validity) > 0:
                final_score = temp_score / len(validity)

            # final_score = float(parsed_result.get("final_crit_score", 0.0))
            print(f"\n=> 最終 CRIT 總分: {final_score}")
            print("======================\n")
            
            return final_score
            
    except json.JSONDecodeError as e:
        print(f"[error] 無法解析 CRIT 輸出為 JSON: {llm_output_str}")
        return 0.0



# ----- task functions -----
def neo4j_retriever(question: str, chapter: str = None, group: str = None) -> dict[str, Any]:
    # 教材向量檢索
    filters = dict()
    if chapter:
        file_names = ["[01]軟體危機與軟體流程", "[02]基礎需求工程", "[03]使用者故事分析", "[04]敏捷開發方法", "[05]基礎專案管理與看板", "[06]版本控制", "[07]軟體設計-系統設計", "[08]軟體設計-模組設計", "[09]軟體測試", "[10]進階軟體測試", "[11]DevOps自動化建置管理"]
        chapter_name = file_names[int(chapter)-1]
        filters = {
                    "field": "source_file",
                    "operator": "==",
                    "value": f"{chapter_name}.pdf"
                }

    if group:
        filters = {
                    "field": "group",
                    "operator": "==",
                    "value": group
                }

    response_template = [
        ChatMessage.from_user(
            """
            你是一個「軟體工程課程助教」請根據提供的資訊用台灣繁體中文回答問題，並僅輸出你的回答。請勿直接摘要資訊內容，而是針對問題僅採用必要資訊來回答，總字數不超過1000字。
            如果問題與「軟體工程」或是「軟體專案開發」無關，請說「我無法回答與軟體工程無關的問題」。
            如果資料不足，請說「資料不足無法完整回答」。

            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            Question: {{question}}
            Answer:
            """
        )
    ]

    # sourceIdentifier = SourceIdentifier()
    # filters = sourceIdentifier.run(question)

    pipeline = _build_retriever_pipeline(ChatPromptBuilder(template=response_template, required_variables=["question"]))

    result = pipeline.run(
        data={
            # "source_identifier": {"user_input": question},
            "text_embedder": {"text": question}, 
            "prompt_builder": {"question": question},
            "ranker": {"query": question},
            "retriever": {
                "top_k": TASK_CONFIGS["retriever"]["top_k"],
                "filters": filters
            },
            # "llm":{"generation_kwargs":{"max_tokens": TASK_CONFIGS["retriever"]["max_tokens"]}}
        },
        include_outputs_from=["retriever", "llm"]
    )

    return result
    # return result["llm"]["replies"][0]._content[0].text

def neo4j_generate_notes(concept: str) -> str:
    # 生成筆記
    template = [
        ChatMessage.from_user(
            """
            請根據提供的「軟體工程」教材內容，針對指定的概念，生成一份筆記供學生學習，字數限制在2000字以內。
            筆記內容須包含：
            - 關鍵概念解說
            - 相關範例
            - 重點整理
            筆記內容必須清楚易懂，請使用 Markdown 格式，必要時可使用表格、引用等方式呈現。
            僅輸出筆記內容，並使用台灣繁體中文。

            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            Concept: {{concept}}
            Answer:
            """
        )
    ]

    pipeline = _build_retriever_pipeline(ChatPromptBuilder(template=template, required_variables=["concept"]))

    result = pipeline.run(
        data={
            "text_embedder": {"text": concept}, 
            "ranker": {"query": concept},
            "prompt_builder": {"concept": concept},
            "llm": {"generation_kwargs":{"max_tokens": TASK_CONFIGS["note"]["max_tokens"]}}
        },
        include_outputs_from=["retriever", "llm"]
    )

    return result

# ----- user question handler -----
def handle_user_question(question: str):
    user_question_template = """
        請根據以下 Neo4j Schema 替使用者的問題生成 5 個關鍵字，用來檢索知識圖譜中的實體。請僅輸出關鍵字，且關鍵字之間請用頓號(、)隔開。
        
            【Schema 資訊】
            - Labels: Concept, Technology, Methodology
            - Properties: name, source_files, description
            - Relationships: 是, 包含於, 實作, 使用, 操作, 依賴, 改善, 解決
        
            使用者的問題: {{question}}
            生成的關鍵字:
        """
    
    pipeline = Pipeline()
        
    pipeline.add_component("user_question_prompt", PromptBuilder(template=user_question_template, required_variables=["question"]))
        # 這裡使用 gpt-4o-mini 作為快速邏輯驗證引擎
    pipeline.add_component("user_question_llm", OpenAIGenerator(
                                            api_key=Secret.from_env_var("OPENAI_API_KEY"), 
                                            model="gpt-4o-mini",
                                            generation_kwargs={"temperature": 0.3 })
                            )
    pipeline.add_component("desc_reasoner",  DescriptionBasedReasoning("bolt://localhost:7687", "neo4j", NEO4J_PASSWORD))

        
    pipeline.connect("user_question_prompt.prompt", "user_question_llm.prompt")
    pipeline.connect("user_question_llm.replies", "desc_reasoner.keywords")


    #再來是將候選關鍵字與圖譜關鍵字比較
    result = pipeline.run(
        data={
            "user_question_prompt": {"question": question},
        },
        include_outputs_from=[
            "user_question_llm",
            "desc_reasoner"
        ]
    
    )

    # 這裡的 llm_output_keywords 是使用者問題生成的關鍵字列表，desc_keywords 是圖譜中匹配到的節點名稱列表
    llm_output_keywords = [
    k.strip() for k in result["user_question_llm"]["replies"][0].split("、") if k.strip()
    ]
    desc_keywords = result["desc_reasoner"]["matched_nodes"]

    print(f"[debug] LLM 生成關鍵字: {llm_output_keywords}")
    print(f"[debug] 圖譜匹配節點: {desc_keywords}")

    node_similarity_list = []

    for node in desc_keywords:
        # 計算該圖譜節點與所有 LLM 關鍵字的相似度，取最高分者作為該節點的代表分數
        best_score = 0.0
        best_matched_kw = ""
    
        for kw in llm_output_keywords:
            score = difflib.SequenceMatcher(None, kw, node).ratio()
            if score > best_score:
                best_score = score
                best_matched_kw = kw
            
        node_similarity_list.append({
            "node_name": node,
            "matched_llm_kw": best_matched_kw,
            "similarity": round(best_score, 3)
        })

    node_similarity_list = sorted(node_similarity_list, key=lambda x: x["similarity"], reverse=True)
    print(f"[debug] 節點相似度的排序結果: {node_similarity_list}")

    # ==========================================================
    # 根據相似度分佈，執行「前端模糊問題三路分流」
    # ==========================================================    

    # valid_nodes = [item for item in node_similarity_list if item["similarity"] >= 0.6]

    
    
    
    
   

def neo4j_textbook_kg_retriever(question: str) -> dict[str, Any]:
    # 教材知識圖譜檢索
    keyword_prompt_template = """
    請根據以下 Neo4j Schema 替使用者的問題生成 2 至 3 個關鍵字，用來檢索知識圖譜中的實體。請僅輸出關鍵字，且關鍵字之間請用頓號(、)隔開。

    【Schema 資訊】
    - Labels: Concept, Technology, Methodology
    - Properties: name, source_files, description
    - Relationships: 是, 包含於, 實作, 使用, 操作, 依賴, 改善, 解決

    使用者的問題: {{question}}
    生成的關鍵字:
    """

    pipeline = _build_kg_pipeline(PromptBuilder(template=keyword_prompt_template, required_variables=["question"]))

    result = pipeline.run(
        data={
            "kw_prompt": {"question": question},
            "answer_prompt": {"question": question},
            # "vector_filter": {"question": question}
        },
        include_outputs_from=[
            "kw_llm",
            "desc_reasoner",
            "answer_llm"
        ]

    )

    return result

# def neo4j_textbook_kg_retriever_ver2(question: str) -> dict[str, Any]:
#     # 教材知識圖譜檢索
#    router_kw_prompt_template = """
#     你是一位專業的軟體工程知識圖譜路由代理（Routing Agent）。
#     你的任務是分析【學生問題】的真實意圖，並嚴格從提供的【知識圖譜現有節點列表】中，挑選出最相關的 Top-5 候選節點，同時評估每個節點的「意圖匹配信心度（1~10分）」。

#     【知識圖譜現有節點列表】
#     {{node_list}}

#     【學生問題】
#     {{question}}

#     【挑選與評分規則（極重要，請嚴格遵守）】
#     1. 絕對錨定：你挑出的 `node_name` 必須 100% 一字不差地存在於【知識圖譜現有節點列表】中，嚴禁自行發明或修改詞彙。
#     2. 固定數量：請固定輸出 5 個候選節點，並依照 `confidence` 由高至低排序。
#     3. 信心度評分量表 (Rubrics)：
#     - 9~10分（唯一精準命中）：學生問題非常明確地指名該概念，且詢問意圖單一，無其他競爭節點能取代。
#     - 7~8分（高度相關 / 並列競爭）：問題屬於跨概念比較、詢問高階應用，或者問題稍嫌廣泛，導致有 2~3 個子概念節點「同時」符合該問題的潛在意圖（此時請給這些競爭節點相近的 7~8 分，切勿強行替學生決定唯一高分）。
#     - 5~6分（極度廣泛 / 歧義過大）：學生只問了一個涵蓋整章節的籠統大詞（如只問「測試」、「開發」），導致有 4 個以上的節點都勉強沾邊。
#     - 1~4分（代名詞缺失 / 低度相關）：問題嚴重缺乏主詞（如「這要怎麼改？」、「有缺什麼嗎？」），或與軟體工程教材無直接關聯，僅為勉強湊滿 5 個而挑選的次要節點。

#     【參考範例 (Few-Shot Examples)】
#     * 範例一（明確問題 -> 第一名極高分，與第二名差距 >= 3 分）：
#     問題：「請問 Git 發生合併衝突時該怎麼解決？」
#     輸出節點與分數：[{"node_name": "合併衝突", "confidence": 10}, {"node_name": "Git", "confidence": 6}, {"node_name": "分支", "confidence": 5}, {"node_name": "遠端儲存庫", "confidence": 4}, {"node_name": "版本控制", "confidence": 2}]

#     * 範例二（輕微模糊 -> 前 2~3 名同為 7~8 分，差距極小）：
#     問題：「在敏捷開發中，我們應該怎麼規劃測試？」
#     輸出節點與分數：[{"node_name": "敏捷開發", "confidence": 8}, {"node_name": "系統測試", "confidence": 8}, {"node_name": "測試案例", "confidence": 7}, {"node_name": "極限編程", "confidence": 5}, {"node_name": "Scrum", "confidence": 4}]

#     * 範例三（極度模糊 -> 缺乏具體目標，全體 <= 5 分；或大詞導致 4 個以上同為 6~7 分）：
#     問題：「我的系統設計有沒有什麼缺少的？」
#     輸出節點與分數：[{"node_name": "系統設計", "confidence": 4}, {"node_name": "需求文件", "confidence": 4}, {"node_name": "軟體開發", "confidence": 3}, {"node_name": "架構設計", "confidence": 2}, {"node_name": "測試案例", "confidence": 1}]

#     【輸出格式限制】
#     請勿輸出任何解釋性文字或 Markdown 標記（如 ```json ），必須嚴格輸出以下 JSON 格式：
#     {
#         "reasoning": "請用一句話簡述：該問題是目標明確、涵蓋 2~3 個子概念，還是過於籠統缺乏主詞",
#         "candidates": [
#             {"node_name": "節點1", "confidence": 9},
#             {"node_name": "節點2", "confidence": 5},
#             {"node_name": "節點3", "confidence": 4},
#             {"node_name": "節點4", "confidence": 2},
#             {"node_name": "節點5", "confidence": 1}
#         ]
#     }
#     """

#     pipeline = _build_kg_pipeline(PromptBuilder(template=router_kw_prompt_template, required_variables=["question"]))

#     result = pipeline.run(
#         data={
#             "kw_prompt": {"question": question},
#             "answer_prompt": {"question": question},
#             # "vector_filter": {"question": question}
#         },
#         include_outputs_from=[
#             "kw_llm",
#             "desc_reasoner",
#             "answer_llm"
#         ]

#     )

#     return result


def build_entity_index(uri: str, user: str, password: str, group_id: str) -> Dict[str, List[str]]:
    # 取回所有文件節點
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        cypher = """
        MATCH (n)
        WHERE n.group = $group
        RETURN labels(n) AS labels, n.name AS name, n.description AS description
        ORDER BY labels[0], n.name
        """
        
        result = session.run(cypher, group=group_id)
        
        entity_index = {}
        for record in result:
            label = record['labels'][0]  # 取第一個標籤
            name = record['name']
            desc = record['description']
            
            if label not in entity_index:
                entity_index[label] = []
            
            entity_index[label].append({
                'name': name,
                'description': desc
            })
        
        return entity_index

def format_entity_list(entity_index):
    text = list()
    for label, entities in entity_index.items():
        text.append(f"【{label}】")
        for entity in entities:
            text.append(f"- 實體：{entity['name']}")
            text.append(f"    相關描述：{entity['description']}")
    entity_text = "\n".join(text)
    return entity_text

def neo4j_doc_retriever(question: str, group_id: str) -> str:
    # 文件知識圖譜檢索
    keyword_prompt_template = """
    你是軟體工程知識助手。請根據使用者問題，從以下實體清單中選出最相關的 2-3 個實體。

    【可選實體清單】

    (entity_list)

    【使用者問題】
    {{question}}

    任務：
    1. 理解使用者的問題意圖
    2. 從清單中選出最相關的實體名稱
    3. 只輸出實體名稱，名稱之間用頓號(、)隔開

    範例：
    - 使用者問：「管理員可以做什麼？」
    - 你應該回答：系統管理員、系統管理功能、...

    使用者問題：{{question}}
    答案（只輸出實體名稱）：
    """

    entity_index = build_entity_index(uri="bolt://localhost:7687", user="neo4j", password=NEO4J_PASSWORD, group_id=group_id)
    entity_list_text = format_entity_list(entity_index)
    keyword_prompt_template = keyword_prompt_template.replace(
        "(entity_list)", entity_list_text
    )
    pipeline = _build_kg_pipeline(PromptBuilder(template=keyword_prompt_template, required_variables=["question"]))

    result = pipeline.run(
        data={
            "kw_prompt": {"question": question},
            "answer_prompt": {"question": question},
            "desc_reasoner": {"group_id": group_id}
        },
        include_outputs_from=["desc_reasoner", "answer_llm"]
    )

    if "資料不足" in result["answer_llm"]["replies"][0]:
        print(f"> 問題：{question}\n> 由於知識圖譜資料不足以回答問題，啟用備案向量檢索生成")
        vector_res = neo4j_retriever(question=question, group=group_id)
        print(vector_res["retriever"])
        return vector_res["llm"]["replies"][0]._content[0].text
    
    # return result
    return result["answer_llm"]["replies"][0]

async def batch_tags(md_documents, doc_type):
    # 替資料塊加上相關標籤
    chunks = [d.page_content for d in md_documents]
    chunk_dict = json.dumps([{"id": i, "content": c} for i, c in enumerate(chunks)])

    prompt = f"""
    以下是 {len(md_documents)} 個{doc_type}段落，請為每個段落生成 2-3 個關鍵字，並以 JSON 格式回傳，key 是段落編號，value 是關鍵字，以python list呈現。
    
    【輸出格式範例】
    1:[版本控制, git 指令]

    【段落】
    {chunk_dict}
    """

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    res = client.beta.messages.create(
            max_tokens=4000,
            messages=[
                {"role":"user", "content":prompt}
            ],
            model="claude-opus-4-6",
        )
    
    try:
        output = re.sub(r'```', '', res.content[0].text)
        output = re.sub(r'json', '', output)
        result = json.loads(output)
        return result
    except Exception as e:
        print(e)
        print("="*30)
        print(chunk_dict)
        print(output)
    # print(result)

async def upload_textbook_2_vectordb(chapter, textbook_name):
    # 把投影片上傳到資料庫(向量)
    try:
        with open(f"md_files\\textbooks\\ch{chapter}_markdown.md", 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        return

    md_documents = file_processor.md_splitter(md_content, True)
    tags_w_id = await batch_tags(md_documents, "軟體工程教材")
    content_tags = [tags_w_id[str(i)] for i in range(0, len(md_documents))]
    documents = add_metadata(md_documents, content_tags, "textbook", textbook_name=textbook_name)

    upload_to_neo4j(documents)

async def upload_doc_2_vectordb(file_path, doc_type, group_name, uploader):
    # 把開發文件上傳到資料庫(向量)
    try:
        with open(file_path, 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        return

    cleaned_md = file_processor.clean_markdown(md_content)
    md_documents = file_processor.md_splitter(cleaned_md, True)
    print(len(md_documents))
    tags_w_id = await batch_tags(md_documents, "軟體專案開發文件")

    content_tags = [tags_w_id[str(i)] for i in range(0, len(md_documents))]

    document_contents = list()
    for doc in md_documents:
        table_extracted_string = file_processor.replace_tables_in_text(doc.page_content)
        document_contents.append(table_extracted_string)

    documents = add_metadata(document_contents, content_tags, "doc", doc_type=doc_type, group_name=group_name, uploader=uploader)
    upload_to_neo4j(documents)

async def filter_retrieval_test():
    # 測試向量過濾
    embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedder.warm_up()
    result = embedder.run("什麼是凝聚力跟偶合力？")
    neo4j_retriever = Neo4jEmbeddingRetriever(document_store=document_store)
    docs = neo4j_retriever.run(
        query_embedding=result["embedding"],
        top_k=TASK_CONFIGS["filter_test"]["top_k"],
        filters={
            "field": "source_file",
            "operator": "==",
            "value": "[07]軟體設計-系統設計.pdf"
        }
    )
    for d in docs["documents"]:
        print(d.content)

if __name__ == '__main__':
    # 測試 func 用
    question = "什麼是軟體測試？"
    vector_res = neo4j_retriever(question=question, group="測試組")
    for doc in vector_res["retriever"]["documents"]:
        print(doc.content)
        print("-"*30)
    print(vector_res["llm"]["replies"][0]._content[0].text)
