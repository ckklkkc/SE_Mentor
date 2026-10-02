# Based on code from [https://github.com/stair-lab/kg-gen]

from typing import Optional, List
from neo4j import GraphDatabase, Driver, Result
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EntityLabel(str, Enum):
    Actor = "Actor"
    Requirement = "Requirement"
    UserStory = "UserStory"
    SystemComponent = "SystemComponent"
    Service = "Service"
    API = "API"
    TestCase = "TestCase"
    Concept = "Concept"
    Technology = "Technology"
    Methodology = "Methodology"

class KeyValPair(BaseModel):
    key: str = Field(..., description="屬性名稱")
    value: str= Field(..., description="屬性值")

class Entity(BaseModel):
    name: str = Field(..., description="實體的唯一名稱") # name 為必填
    label: EntityLabel
    properties: Optional[List[KeyValPair]] = Field(default_factory=list, description="實體的屬性資訊")

class Relation(BaseModel):
    name: str = Field(..., description="實體之間的邏輯關係")
    description: str

class Triple(BaseModel):
    subject: Entity
    relation: Relation
    object: Entity

class TripleList(BaseModel):
    triples: list[Triple]

class EntityList(BaseModel):
    entities: list[Entity]

class Neo4jImporter:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
    
    def connect(self) -> bool:
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
        
    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def upload_textbook_analysis(self, analysis, analysis_id: str) -> bool:
        """上傳細粒度教材分析；新內容一律以 pending 狀態進入圖譜。"""
        if not self.driver:
            logger.error("Neo4j driver is not connected")
            return False

        try:
            chapters = []
            points = []
            relations = []
            targets = []
            point_evidence = []
            target_evidence = []

            for chapter in analysis.chapters:
                chapters.append({
                    "id": chapter.id,
                    "title": chapter.title,
                    "order": chapter.order,
                })
                for point in chapter.knowledge_points:
                    points.append({
                        "id": point.id,
                        "chapter_id": chapter.id,
                        "name": point.name,
                        "description": point.description,
                    })
                    for evidence in point.evidence:
                        point_evidence.append({
                            "point_id": point.id,
                            "chunk_id": evidence.chunk_id,
                            "heading_path": evidence.heading_path,
                            "quote": evidence.quote,
                        })
                for relation in chapter.relations:
                    relations.append(relation.model_dump())
                for target in chapter.assessment_targets:
                    targets.append({
                        "id": target.id,
                        "chapter_id": chapter.id,
                        "title": target.title,
                        "objective": target.objective,
                        "discrimination": target.discrimination,
                        "common_misconceptions": target.common_misconceptions,
                        "knowledge_point_ids": target.knowledge_point_ids,
                    })
                    for chunk_id in target.evidence_chunk_ids:
                        target_evidence.append({
                            "target_id": target.id,
                            "chunk_id": chunk_id,
                        })

            with self.driver.session(database=self.database) as session:
                for label in ["Textbook", "Chapter", "KnowledgePoint", "AssessmentTarget", "SourceChunk"]:
                    session.run(
                        f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                    )

                session.run(
                    """
                    MERGE (book:Textbook {id: $textbook_id})
                    SET book.title = $title,
                        book.source_file = $source_file,
                        book.analysis_id = $analysis_id,
                        book.review_status = 'pending'
                    WITH book
                    UNWIND $chapters AS row
                    MERGE (chapter:Chapter {id: row.id})
                    SET chapter.title = row.title, chapter.order = row.order,
                        chapter.analysis_id = $analysis_id,
                        chapter.review_status = 'pending'
                    MERGE (book)-[:HAS_CHAPTER]->(chapter)
                    """,
                    textbook_id=analysis.textbook_id,
                    title=analysis.title,
                    source_file=analysis.source_file,
                    analysis_id=analysis_id,
                    chapters=chapters,
                )
                session.run(
                    """
                    UNWIND $points AS row
                    MATCH (chapter:Chapter {id: row.chapter_id})
                    MERGE (point:KnowledgePoint {id: row.id})
                    SET point.name = row.name, point.description = row.description,
                        point.analysis_id = $analysis_id,
                        point.review_status = 'pending'
                    MERGE (chapter)-[:HAS_KNOWLEDGE_POINT]->(point)
                    """,
                    points=points,
                    analysis_id=analysis_id,
                )
                session.run(
                    """
                    UNWIND $evidence AS row
                    MATCH (point:KnowledgePoint {id: row.point_id})
                    MERGE (chunk:SourceChunk {id: row.chunk_id})
                    SET chunk.heading_path = row.heading_path,
                        chunk.analysis_id = $analysis_id,
                        chunk.review_status = 'pending'
                    MERGE (point)-[rel:EVIDENCED_BY]->(chunk)
                    SET rel.quote = row.quote
                    """,
                    evidence=point_evidence,
                    analysis_id=analysis_id,
                )
                session.run(
                    """
                    UNWIND $relations AS row
                    MATCH (source:KnowledgePoint {id: row.source_id})
                    MATCH (target:KnowledgePoint {id: row.target_id})
                    MERGE (source)-[rel:KNOWLEDGE_RELATION {
                        relation_type: row.relation_type,
                        analysis_id: $analysis_id
                    }]->(target)
                    SET rel.description = row.description,
                        rel.review_status = 'pending'
                    """,
                    relations=relations,
                    analysis_id=analysis_id,
                )
                session.run(
                    """
                    UNWIND $targets AS row
                    MATCH (chapter:Chapter {id: row.chapter_id})
                    MERGE (assessment:AssessmentTarget {id: row.id})
                    SET assessment.title = row.title,
                        assessment.objective = row.objective,
                        assessment.discrimination = row.discrimination,
                        assessment.common_misconceptions = row.common_misconceptions,
                        assessment.analysis_id = $analysis_id,
                        assessment.review_status = 'pending'
                    MERGE (chapter)-[:HAS_ASSESSMENT_TARGET]->(assessment)
                    WITH assessment, row
                    UNWIND row.knowledge_point_ids AS point_id
                    MATCH (point:KnowledgePoint {id: point_id})
                    MERGE (assessment)-[:ASSESSES]->(point)
                    """,
                    targets=targets,
                    analysis_id=analysis_id,
                )
                session.run(
                    """
                    UNWIND $evidence AS row
                    MATCH (assessment:AssessmentTarget {id: row.target_id})
                    MATCH (chunk:SourceChunk {id: row.chunk_id})
                    MERGE (assessment)-[:EVIDENCED_BY]->(chunk)
                    """,
                    evidence=target_evidence,
                )
            logger.info("Uploaded textbook analysis %s", analysis_id)
            return True
        except Exception as e:
            logger.error("Fail to upload textbook analysis to Neo4j: %s", e)
            return False

    def update_textbook_analysis_status(self, analysis_id: str, status: str) -> bool:
        """將同一分析批次的節點與知識關係同步為教師審核結果。"""
        if status not in {"approved", "rejected"}:
            raise ValueError("status must be approved or rejected")
        if not self.driver:
            logger.error("Neo4j driver is not connected")
            return False
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    """
                    MATCH (node)
                    WHERE node.analysis_id = $analysis_id
                    SET node.review_status = $status
                    WITH count(node) AS nodes
                    MATCH ()-[rel:KNOWLEDGE_RELATION]->()
                    WHERE rel.analysis_id = $analysis_id
                    SET rel.review_status = $status
                    RETURN nodes, count(rel) AS relationships
                    """,
                    analysis_id=analysis_id,
                    status=status,
                ).consume()
            return True
        except Exception as e:
            logger.error("Fail to update textbook review status: %s", e)
            return False

    def upload_textbook_triples(self, triple_list: TripleList, source_file: str) -> bool:
        try:
            with self.driver.session() as session:
                labels = ["Concept", "Technology", "Methodology"]
                for label in labels:
                    session.run(f"""
                    CREATE CONSTRAINT {label.lower()}_group_name_unique IF NOT EXISTS 
                    FOR (n:{label}) 
                    REQUIRE (n.name, n.group) IS UNIQUE
                    """)
                # session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")

                # data_to_upload = [t.model_dump() for t in triple_list.triples]
                data_to_upload = list()
                for t in triple_list.triples:
                    item = t.model_dump()
                    # 強制轉換 Enum 為字串
                    item["subject"]["label"] = str(t.subject.label.value)
                    item["object"]["label"] = str(t.object.label.value)

                    s_kv = item["subject"].get("properties") or []
                    item["subject"]["props_dict"] = {kv["key"]: kv["value"] for kv in s_kv} if s_kv else {}
                    
                    o_kv = item["object"].get("properties") or []
                    item["object"]["props_dict"] = {kv["key"]: kv["value"] for kv in o_kv} if o_kv else {}

                    data_to_upload.append(item)
                
                # print(data_to_upload[0])
                cypher_query = """
                UNWIND $batch AS row
                CALL apoc.merge.node([row.subject.label], {name: row.subject.name}, {}, {}) YIELD node AS sNode
                WITH sNode, row
                SET sNode += row.subject.props_dict
                SET sNode.source_files = apoc.coll.toSet(coalesce(sNode.source_files, []) + $source_file)

                WITH sNode, row
                CALL apoc.merge.node([row.object.label], {name: row.object.name}, {}, {}) YIELD node AS oNode
                WITH sNode, oNode, row
                SET oNode += row.object.props_dict
                SET oNode.source_files = apoc.coll.toSet(coalesce(oNode.source_files, []) + $source_file)

                WITH sNode, oNode, row
                CALL apoc.merge.relationship(sNode, row.relation.name, {}, {}, oNode) YIELD rel
                SET rel.description = row.relation.description
                SET rel.source_files = apoc.coll.toSet(coalesce(rel.source_files, []) + $source_file)
                RETURN count(rel)
                """
                session.run(cypher_query, batch=data_to_upload, source_file=source_file)

                logger.info(f"成功上傳 {len(triple_list.triples)} 條關係。")
                return True
        
        except Exception as e:
            logger.error(f"Fail to upload to Neo4j: {e}")
            return False

    def upload_doc_triples(self, triple_list: TripleList, source_file: str, doc_type: str, group: str, uploader: str) -> bool:
        try:
            with self.driver.session() as session:
                labels = ["Requirement", "Service", "SystemComponent", "API", "TestCase", "Actor", "General", "Technology", "Methodology", "Concept"]
                for label in labels:
                    session.run(f"""
                    CREATE CONSTRAINT {label.lower()}_group_name_unique IF NOT EXISTS 
                    FOR (n:{label}) 
                    REQUIRE (n.name, n.group) IS UNIQUE
                    """)
                
                # data_to_upload = [t.model_dump() for triples in triple_list for t in triples.triples]
                data_to_upload = list()
                for t in triple_list.triples:
                    item = t.model_dump()
                    # 強制轉換 Enum 為字串
                    item["subject"]["label"] = str(t.subject.label.value)
                    item["object"]["label"] = str(t.object.label.value)

                    s_kv = item["subject"].get("properties") or []
                    item["subject"]["props_dict"] = {kv["key"]: kv["value"] for kv in s_kv} if s_kv else {}

                    reference = item["subject"]["props_dict"].get("req_reference")
                    if reference:
                        reference_list = [r.strip() for r in reference.split(',')]
                        item["subject"]["props_dict"]["req_reference"] = reference_list
                    
                    o_kv = item["object"].get("properties") or []
                    item["object"]["props_dict"] = {kv["key"]: kv["value"] for kv in o_kv} if o_kv else {}

                    reference = item["object"]["props_dict"].get("req_reference")
                    if reference:
                        reference_list = [r.strip() for r in reference.split(',')]
                        item["object"]["props_dict"]["req_reference"] = reference_list

                    data_to_upload.append(item)

                cypher_query = """
                UNWIND $batch AS row
                CALL (row) {
                    WITH row
                    CALL apoc.merge.node([row.subject.label], {name: row.subject.name, group: $group, uploader: $uploader}, {}, {}) YIELD node AS sNode
                    SET sNode += row.subject.props_dict
                    SET sNode.source_files = apoc.coll.toSet(coalesce(sNode.source_files, []) + $source_file)
                    SET sNode.doc_type = apoc.coll.toSet(coalesce(sNode.doc_type, []) + $doc_type)

                    WITH sNode, row
                    CALL apoc.merge.node([row.object.label], {name: row.object.name, group: $group, uploader: $uploader}, {}, {}) YIELD node AS oNode
                    SET oNode += row.object.props_dict
                    SET oNode.source_files = apoc.coll.toSet(coalesce(oNode.source_files, []) + $source_file)
                    SET oNode.doc_type = apoc.coll.toSet(coalesce(oNode.doc_type, []) + $doc_type)

                    WITH sNode, oNode, row
                    CALL apoc.merge.relationship(sNode, row.relation.name, {}, {}, oNode) YIELD rel
                    SET rel.description = row.relation.description
                    SET rel.source_files = apoc.coll.toSet(coalesce(rel.source_files, []) + $source_file)
                    RETURN count(rel) AS relCount
                } IN TRANSACTIONS
                RETURN sum(relCount)
                """
                session.run(cypher_query, batch=data_to_upload, source_file=source_file, group=group, uploader=uploader, doc_type=doc_type)

                # print(f"成功上傳 {len(triple_list.triples)} 條關係。")
                return True
        
        except Exception as e:
            logger.error(f"Fail to upload to Neo4j: {e}")
            return False

    def upload_entities(self, entity_list: EntityList, source_file: str, doc_type: str, group: str, uploader: str) -> bool:
        try:
            with self.driver.session() as session:
                labels = ["Requirement", "Service", "SystemComponent", "API", "TestCase", "Actor", "General", "Technology", "Methodology", "Concept"]
                for label in labels:
                    session.run(f"""
                    CREATE CONSTRAINT {label.lower()}_group_name_unique IF NOT EXISTS 
                    FOR (n:{label}) 
                    REQUIRE (n.name, n.group) IS UNIQUE
                    """)
                
                # data_to_upload = [t.model_dump() for entities in entity_list for t in entities.entities]
                data_to_upload = list()
                for t in entity_list.entities:
                    item = t.model_dump()
                    # 強制轉換 Enum 為字串
                    item["label"] = str(t.label.value)

                    item["props_dict"] = self.convert_properties_to_dict(item.get("properties") or [])

                    # reference = item["props_dict"].get("req_reference")
                    # if reference:
                    #     reference_list = [r.strip() for r in reference.split(',')]
                    #     item["props_dict"]["req_reference"] = reference_list

                    data_to_upload.append(item)

                cypher_query = """
                UNWIND $batch AS row
                CALL (row) {
                    WITH row
                    CALL apoc.merge.node([row.label], {name: row.name, group: $group, uploader: $uploader}, {}, {}) YIELD node AS node
                    SET node += row.props_dict
                    SET node.source_files = apoc.coll.toSet(coalesce(node.source_files, []) + $source_file)
                    SET node.doc_type = apoc.coll.toSet(coalesce(node.doc_type, []) + $doc_type)

                    RETURN count(node) AS nCount
                } IN TRANSACTIONS
                RETURN nCount
                """
                session.run(cypher_query, batch=data_to_upload, source_file=source_file, group=group, uploader=uploader, doc_type=doc_type)

                # print(f"成功上傳 {len(entity_list.triples)} 條關係。")
                return True
        
        except Exception as e:
            logger.error(f"Fail to upload to Neo4j: {e}")
            return False
        
    def convert_properties_to_dict(self, properties_list):
        """把 properties 轉成 dict，同 key 多值時變成 list"""
        props_dict = {}
        for kv in properties_list:
            key = kv["key"]
            value = kv["value"]
            
            if key in props_dict:
                # 已存在同 key，轉成 list 或加入 list
                if not isinstance(props_dict[key], list):
                    props_dict[key] = [props_dict[key]]
                props_dict[key].append(value)
            else:
                props_dict[key] = value
        
        # 最後確保 req_reference 一律是 list（即使只有一個值）
        if "req_reference" in props_dict:
            if not isinstance(props_dict["req_reference"], list):
                props_dict["req_reference"] = [props_dict["req_reference"]]
        
        return props_dict

    def link_references_to_requirements(self, label: str, doc_type: str, group: str, rel_type: str) -> bool:
        if doc_type == "SRD":
            cypher = f"""
            MATCH (n:{label})
            WHERE n.group = $group AND $doc_type in n.doc_type AND n.req_reference IS NOT NULL
            UNWIND n.req_reference AS req_id
            MATCH (r:Requirement {{req_id: req_id, group: $group}})
            MERGE (r)-[rel:{rel_type}]->(n)
            RETURN count(rel) AS linked
            """
        else:
            cypher = f"""
                MATCH (n:{label})
                WHERE n.group = $group AND $doc_type in n.doc_type AND n.req_reference IS NOT NULL
                UNWIND n.req_reference AS req_id
                MATCH (r:Requirement {{req_id: req_id, group: $group}})
                MERGE (n)-[rel:{rel_type}]->(r)
                RETURN count(rel) AS linked
            """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, doc_type=doc_type, group = group).single()
                logger.info(f"Linked {result['linked']} references")
                return True
        except Exception as e:
            logger.error(f"Fail to upload to Neo4j: {e}")
            return False

    def run_cypher(self, cypher_query: str) -> Result:
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [record["name"] for record in result]
                return records
        except Exception as e:
                logger.error(f"Fail to query from Neo4j: {e}")
                return False
