import re, ast, time, random, json, gc, sys, argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import anthropic

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import file_processor
from database.neo4j_importer import Neo4jImporter, TripleList, EntityList, Entity
from common import NEO4J_URI
from config import NEO4J_PASSWORD, CLAUDE_API_KEY
import prompts 

class KGConstructor:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    def entities_extraction(self, prompt: str, user_input: str) -> list:
        """LLM 提取實體 (SDD)"""

        res = self.client.beta.messages.create(
            max_tokens=2500,
            messages=[
                {"role":"assistant", "content":prompt},
                {"role":"user", "content":user_input}
            ],
            model="claude-sonnet-4-6",
        )

        entities = re.sub(r'[\r\n]', '', res.content[0].text)
        # print(entities)

        match = re.search(r"\[.*?\]", entities)
        if match:
            list_str = match.group()
            try:
                result = ast.literal_eval(list_str)  # 轉成真正的 list
                return result
            except Exception as e:
                print(list_str)
            # print(result)
        else:
            print("找不到 list")
            return None

    def relations_extraction(self, prompt: str, user_input: str, entities_list: list) -> TripleList:
        """LLM 提取三元組 (SDD)"""

        doc_content = user_input.page_content if hasattr(user_input, 'page_content') else str(user_input)
        input_data = f"""
        實體列表如下：

        <entity>
        {entities_list}
        </entity>

        文章內容如下：

        <article>
        {doc_content}
        </article>
        """

        res = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=9000,
            messages=[
                {"role":"user", "content":prompt},
                {"role":"user", "content":input_data}
            ],
            output_format=TripleList,
        )

        return res.parsed_output

    def entities_extraction_with_properties(self, prompt: str, user_input: str) -> EntityList:
        """提取包含屬性的實體"""

        doc_content = user_input.page_content if hasattr(user_input, 'page_content') else str(user_input)
        input_data = f"""
        文章內容如下：

        <article>
        {doc_content}
        </article>
        """

        res = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=7000,
            messages=[
                {"role":"user", "content":prompt},
                {"role":"user", "content":input_data}
            ],
            output_format=EntityList,
        )

        return res.parsed_output 

    def match_fr_and_us_llm(self, case: dict, prompt: str, fr2us: bool) -> EntityList:
        """配對功能需求與使用者故事"""

        if fr2us:
            input_data = f"""
            需求功能實體如下：

            <user_story>
            {case['fr']}
            </user_story>

            候選使用者故事實體列表如下：

            <candidates>
            {case['candidates']}
            </candidates>
            """
        else:
            input_data = f"""
            使用者故事實體如下：

            <user_story>
            {case['us']}
            </user_story>

            候選需求功能實體列表如下：

            <candidates>
            {case['candidates']}
            </candidates>
            """

        res = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=5000,
            messages=[
                {"role":"user", "content":prompt},
                {"role":"user", "content":input_data}
            ],
            output_format=EntityList,
        )

        return res.parsed_output

    def convert_props_to_dict(self, raw_list: EntityList) -> list:
        """
        把類別屬性轉為字典格式
        """
        entity_list = list()
        for raw in raw_list.entities:
            # item = dict()
            # item["name"] = raw.name
            item = raw.model_dump()
            item["label"] = str(raw.label.value)
            props = item.get("properties")
            properties_dict = {}
            for prop in props or []:
                key, value = prop["key"], prop["value"]
                if key not in properties_dict:
                    properties_dict[key] = value
                elif isinstance(properties_dict[key], list):
                    properties_dict[key].append(value)
                else:
                    properties_dict[key] = [properties_dict[key], value]
            item["properties_dict"] = properties_dict
            entity_list.append(item)
        return entity_list

    @staticmethod
    def _normalized_identifier(value) -> str:
        return re.sub(r"\s+", "", str(value or "")).upper()

    @staticmethod
    def _reference_values(value) -> list[str]:
        values = value if isinstance(value, list) else [value]
        references = []
        for raw in values:
            references.extend(
                part.strip()
                for part in re.split(r"[,，、;；]", str(raw or ""))
                if part.strip()
            )
        return list(dict.fromkeys(references))

    @staticmethod
    def _actor_values(value) -> list[str]:
        values = value if isinstance(value, list) else [value]
        actors = []
        for raw in values:
            actors.extend(
                part.strip()
                for part in re.split(r"[,，、/／;；]|(?:與|和|及)", str(raw or ""))
                if part.strip()
            )
        return list(dict.fromkeys(actors))

    @staticmethod
    def _deduplicate_entities(entities: list[dict], identity_key: str | None) -> list[dict]:
        deduplicated = []
        seen = set()
        for entity in entities:
            props = entity.get("properties_dict", {})
            identity = props.get(identity_key) if identity_key else entity.get("name")
            marker = str(identity or entity.get("name", "")).strip().casefold()
            if not marker or marker in seen:
                continue
            seen.add(marker)
            deduplicated.append(entity)
        return deduplicated

    def get_specific_entity_by_label(self, entity_list: dict, label: str) -> list:
        """取得指定標籤的所有實體"""
        
        prop_list = list()
        for entity in entity_list:
            if entity["label"] == label:
                prop_list.append(entity)
        return prop_list

    def extract_actor(self, text: str, known_actors: list) -> str:
        """從描述中取得操作角色"""

        for actor in known_actors:
            if actor in text:
                return actor
        return None

    def add_req_reference_to_properties(self, entity_dict: dict, req_ids: list) -> dict:
        """
        更新實體裡的req_reference
        """

        if not entity_dict.get('properties'):
            entity_dict['properties'] = []
        
        # 移除舊的 req_reference（如果有的話）
        entity_dict['properties'] = [
            p for p in entity_dict['properties'] 
            if p.get('key') != 'req_reference'
        ]
        
        # 把每個 req_id 作為獨立的 KeyValPair 加入
        for req_id in req_ids:
            entity_dict['properties'].append({
                'key': 'req_reference',
                'value': req_id
            })
        
        return entity_dict

    def match_us_to_fr_v2(self, raw_entity_list: EntityList) -> tuple[list, list, list]:
        """以文件中的明確 ID 建立可驗證的需求追溯。

        不再用語意相似度猜測 ``req_reference``。若使用者故事已明列 Reference，
        或 us_id 與需求 req_id 完全相同，才建立關係；其他可能關聯留給
        ``/project_qa`` 分析，不寫成知識圖譜事實。
        """

        # 把 EntityList 轉為 dictionary
        entity_list = self.convert_props_to_dict(raw_entity_list)
        print(f"[SRD 追溯] 抽取實體總數：{len(entity_list)}")

        # 取得使用者故事及需求
        us_list = self.get_specific_entity_by_label(entity_list, "UserStory")
        fr_list = self.get_specific_entity_by_label(entity_list, "Requirement")
        actor_list = self.get_specific_entity_by_label(entity_list, "Actor")
        us_list = self._deduplicate_entities(us_list, "us_id")
        fr_list = self._deduplicate_entities(fr_list, "req_id")
        actor_list = self._deduplicate_entities(actor_list, None)
        print("us list: ", len(us_list))
        print("fr list: ", len(fr_list))
        requirement_ids = {
            self._normalized_identifier(fr["properties_dict"].get("req_id")):
            str(fr["properties_dict"].get("req_id")).strip()
            for fr in fr_list
            if fr["properties_dict"].get("req_id")
        }

        match_list = []
        for index, us_obj in enumerate(us_list, start=1):
            props = us_obj["properties_dict"]
            explicit_refs = self._reference_values(props.get("req_reference"))
            valid_refs = [
                requirement_ids[self._normalized_identifier(ref)]
                for ref in explicit_refs
                if self._normalized_identifier(ref) in requirement_ids
            ]
            us_id = self._normalized_identifier(props.get("us_id"))
            if us_id and us_id in requirement_ids:
                # When both tables use the same ID, that is the strongest
                # traceability evidence. Do not retain a different reference
                # that an extraction model may have inferred from nearby text.
                valid_refs = [requirement_ids[us_id]]
            valid_refs = list(dict.fromkeys(valid_refs))
            print(
                f"[{index}/{len(us_list)}] ID 配對："
                f"{props.get('us_id', '?')} -> {valid_refs or '未明列對應需求'}"
            )
            match_list.append({"us": us_obj, "us_req_ids": valid_refs})

        return match_list, fr_list, actor_list

    def create_actor_relationships(self, raw_entity_list: list) -> list:
        """建立操作角色跟功能需求的三元組"""
        entity_list = self.convert_props_to_dict(raw_entity_list)

        # 取得所有操作角色 (標籤為 Actor)
        known_actors = self.get_specific_entity_by_label(entity_list, "Actor")
        known_actors = sorted(
            {actor["name"].strip() for actor in known_actors if actor["name"].strip()},
            key=lambda value: (-len(value), value),
        )
        print("Actors: ", known_actors)

        fr_list = self.get_specific_entity_by_label(entity_list, "Requirement")
        print("fr list: ", len(fr_list))
        
        actor_relations = list()
        seen_relations = set()
        requirements_by_id = {
            self._normalized_identifier(fr["properties_dict"].get("req_id")): fr
            for fr in fr_list
            if fr["properties_dict"].get("req_id")
        }

        def add_relation(actor: str, requirement: dict) -> None:
            actor = actor.strip()
            marker = (actor.casefold(), requirement["name"].casefold())
            if not actor or marker in seen_relations:
                return
            seen_relations.add(marker)
            actor_relations.append({
                "subject": {"name": actor, "label": "Actor"},
                "relation": {
                    "name": "操作",
                    "description": f"{actor}可執行「{requirement['name']}」",
                },
                "object": {"name": requirement["name"], "label": "Requirement"},
            })

        for fr in fr_list:
            props = fr["properties_dict"]
            explicit_actors = self._actor_values(
                props.get("actor") or props.get("actors") or ""
            )
            if explicit_actors:
                for fr_actor in explicit_actors:
                    add_relation(fr_actor, fr)
            else:
                evidence_text = " ".join([
                    fr.get("name", ""),
                    str(props.get("description", "")),
                ])
                for fr_actor in known_actors:
                    if fr_actor in evidence_text:
                        add_relation(fr_actor, fr)

        for us in self.get_specific_entity_by_label(entity_list, "UserStory"):
            props = us["properties_dict"]
            story_text = " ".join([us.get("name", ""), str(props.get("description", ""))])
            actor = self.extract_actor(story_text, known_actors)
            if not actor:
                match = re.search(r"(?:作為|身為)(?:一位)?([^，,。]+)", story_text)
                actor = match.group(1).strip() if match else None
            if not actor:
                continue
            for req_id in self._reference_values(props.get("req_reference")):
                requirement = requirements_by_id.get(
                    self._normalized_identifier(req_id)
                )
                if requirement:
                    add_relation(actor, requirement)
        
        return actor_relations

    def kg_construction_pipeline(self, document_contents: list, entity_prompt: str, triple_prompt: str, group: str) -> TripleList:
        """建立知識圖譜的流程"""
        doc_entity_pairs = list()
        page_entities_list = list()

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_doc = {executor.submit(self.entities_extraction, entity_prompt, content): content for content in document_contents}
            
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    page_entities = future.result()
                    if page_entities:
                        cleaned_page_entities = list(set(e.lower() for e in page_entities))
                        doc_entity_pairs.append((doc, cleaned_page_entities))
                        page_entities_list.extend(cleaned_page_entities)
                except Exception as e:
                    print(f"實體抽取 Exception: {e}")
        
        print("實體抽取完成")
        list_of_TripleList = list()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_relation = {
                executor.submit(self.relations_extraction, triple_prompt, doc, entities): doc 
                for doc, entities in doc_entity_pairs if entities
            }
            
            for future in as_completed(future_to_relation):
                try:
                    triples = future.result()
                    if triples:
                        list_of_TripleList.append(triples)
                except Exception as e:
                    print(f"關係抽取 Exception: {e}")

        data_to_save = [t.model_dump(mode="json") for triples in list_of_TripleList for t in triples.triples]        
        print("關係抽取完成")
        triple_list = TripleList(triples=data_to_save)

        return triple_list

    def entities_extraction_pipeline(self, document_contents: list, doc_type: str, prompt: str, group: str):
        """文件實體抽取流程"""
        output_dir = Path("files") / "doc" / group
        output_dir.mkdir(parents=True, exist_ok=True)
        list_of_EntityList = list()
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_doc = {executor.submit(self.entities_extraction_with_properties, prompt, content): content for content in document_contents}
            
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    entities = future.result()
                    if entities:
                        list_of_EntityList.append(entities)
                except Exception as e:
                    print(f"文件實體抽取 Exception: {e}")

        data_to_save = [t.model_dump(mode="json") for entities in list_of_EntityList for t in entities.entities]
        with (output_dir / f"doc_entities_{doc_type.lower()}.json").open('w', encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        print("文件實體抽取完成")
        entity_list = EntityList(entities=data_to_save)

        return entity_list

    def match_fr_to_us_pipeline(self, entity_list: EntityList, group: str) -> EntityList:
        """配對功能需求跟使用者故事的流程"""
        output_dir = Path("files") / "doc" / group
        output_dir.mkdir(parents=True, exist_ok=True)
        matches, fr_list, actor_list = self.match_us_to_fr_v2(entity_list)
        user_stories = []
        for match in matches:
            us_dict = match['us'].copy()
            self.add_req_reference_to_properties(us_dict, match['us_req_ids'])
            user_stories.append(us_dict)

        actor_entities = [Entity(**item) for item in actor_list]
        user_story_entities = [Entity(**item) for item in user_stories]
        fr_list_entities = [Entity(**item) for item in fr_list]
        data_to_upload = actor_entities + user_story_entities + fr_list_entities

        for i, d in enumerate(data_to_upload):
            print(f"[{i+1}/{len(data_to_upload)}]:\n{d}")
            print("-"*30)

        entity_list = EntityList(entities=data_to_upload)

        # 保存
        data_to_save = [data.model_dump() for data in data_to_upload]
        with (output_dir / "doc_entities_pairs_srd.json").open('w', encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        return entity_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="知識圖譜建立流程")
    parser.add_argument("--textbook", action="store_true", help="建立課程教材的知識圖譜")
    parser.add_argument("file_path", default=None, help="文件路徑")
    parser.add_argument("doc_type", nargs="?", choices=["SRD", "SDD", "STD"], default=None, help="開發文件類型")
    parser.add_argument("group", nargs="?", default="test", help="專案組別")
    parser.add_argument("uploader", nargs="?", default=None, help="上傳者")
    args = parser.parse_args()

    source_file = args.file_path
    doc_type = args.doc_type
    group = args.group
    uploader = args.uploader

    file_path = f"files\\doc\\{group}\\{source_file}"   # 根據自己的目錄修改
    print(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as input_file:
            md_content = input_file.read()
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        sys.exit(0)

    # pdf 轉 markdown
    # suffix = Path(file_path).suffix.lower()
    # if suffix == ".pdf":
    #     md_content = file_processor.pdf2md(file_path)
    #     save_path = file_path.replace(".pdf", ".md")
    #     with open(save_path, "w", encoding="utf-8") as f:
    #         f.write(md_content)
 
    cleaned_md = file_processor.clean_markdown(md_content)
    cleaned_md = file_processor.remove_specific_sections(cleaned_md)
    md_documents = file_processor.md_splitter(cleaned_md)

    document_contents = list()
    for doc in md_documents:
        table_extracted_string = file_processor.replace_tables_in_text(doc.page_content)
        document_contents.append(table_extracted_string)

    constructor = KGConstructor()
    importer = Neo4jImporter(uri=NEO4J_URI, username="neo4j", password=NEO4J_PASSWORD)
    try:
        if importer.connect():
            # 教材
            if args.textbook:
                triple_list = constructor.kg_construction_pipeline(document_contents, prompts.ENTITY_PROMPT_4_TEXTBOOK, prompts.TRIPLE_PROMPT_4_TEXTBOOK, group)
                is_success = importer.upload_doc_triples(triple_list, source_file, doc_type, group, uploader)
                
            # 開發文件
            else:
                # =====建立知識圖譜 (實體+關係)=====
                if doc_type == "SDD":
                    triple_list = constructor.kg_construction_pipeline(document_contents, prompts.ENTITY_PROMPT_4_SDD, prompts.TRIPLE_PROMPT_4_SDD, group)
                    is_success = importer.upload_doc_triples(triple_list, source_file, doc_type, group, uploader)
                    is_success = importer.link_references_to_requirements("API", doc_type, group, "實作需求")

                # =====提取需求文件實體&配對=====
                elif doc_type == "SRD":
                    # 抽實體
                    entity_list = constructor.entities_extraction_pipeline(document_contents, doc_type, prompts.ENTITY_PROMPT_4_SRD, group)
                    # 配對
                    entity_list = constructor.match_fr_to_us_pipeline(entity_list, group)
                    is_success = importer.upload_entities(entity_list, source_file, doc_type, group, uploader)
                    is_success = importer.link_references_to_requirements("UserStory", doc_type, group, "滿足")
                    # 操作角色
                    actor_relations = constructor.create_actor_relationships(entity_list)
                    triple_list = TripleList(triples=actor_relations)
                    is_success = importer.upload_doc_triples(triple_list, source_file, doc_type, group, uploader)
                
                # =====提取測試文件實體&連接需求文件=====    
                elif doc_type == "STD":
                    # 抽實體
                    entity_list = constructor.entities_extraction_pipeline(document_contents, doc_type, prompts.ENTITY_PROMPT_4_STD, group)
                    is_success = importer.upload_entities(entity_list, source_file, doc_type, group, uploader)
                    is_success = importer.link_references_to_requirements("TestCase", doc_type, group, "驗證")
                
            print(f"上傳結果：{is_success}")
    except Exception as e:
        print(f"建立【{doc_type}】知識圖譜時遇到錯誤：{e}")
    finally:
        importer.close()
