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
        """上傳 CourseUnit/Section 細粒度教材分析。"""
        if not self.driver:
            logger.error("Neo4j driver is not connected")
            return False

        try:
            sections = [section.model_dump() for section in analysis.sections]
            points = []
            relations = [relation.model_dump() for relation in analysis.relations]
            targets = []
            point_evidence = []
            target_evidence = []

            for point in analysis.knowledge_points:
                points.append({
                    "id": point.id,
                    "section_ids": point.section_ids,
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
            for target in analysis.assessment_targets:
                targets.append({
                    "id": target.id,
                    "section_ids": target.section_ids,
                    "title": target.title,
                    "objective": target.objective,
                    "discrimination": target.discrimination,
                    "common_misconceptions": target.common_misconceptions,
                    "knowledge_point_ids": target.knowledge_point_ids,
                    "review_status": target.review_status,
                })
                for chunk_id in target.evidence_chunk_ids:
                    target_evidence.append({
                        "target_id": target.id,
                        "chunk_id": chunk_id,
                    })

            with self.driver.session(database=self.database) as session:
                for label in ["CourseUnit", "Section", "KnowledgePoint", "AssessmentTarget", "SourceChunk"]:
                    session.run(
                        f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                    )

                session.run(
                    """
                    MERGE (unit:CourseUnit {id: $course_unit_id})
                    SET unit.title = $title,
                        unit.source_file = $source_file,
                        unit.analysis_id = $analysis_id,
                        unit.review_status = 'pending'
                    WITH unit
                    UNWIND $sections AS row
                    MERGE (section:Section {id: row.id})
                    SET section.title = row.title, section.order = row.order,
                        section.content_type = row.content_type,
                        section.included_in_assessment = row.included_in_assessment,
                        section.exclusion_reason = row.exclusion_reason,
                        section.analysis_id = $analysis_id,
                        section.review_status = 'pending'
                    MERGE (unit)-[:HAS_SECTION]->(section)
                    """,
                    course_unit_id=analysis.course_unit_id,
                    title=analysis.course_unit_title,
                    source_file=analysis.source_file,
                    analysis_id=analysis_id,
                    sections=sections,
                )
                session.run(
                    """
                    UNWIND $points AS row
                    MERGE (point:KnowledgePoint {id: row.id})
                    SET point.name = row.name, point.description = row.description,
                        point.analysis_id = $analysis_id,
                        point.review_status = 'pending'
                    WITH point, row
                    UNWIND row.section_ids AS section_id
                    MATCH (section:Section {id: section_id})
                    MERGE (section)-[:COVERS]->(point)
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
                    MERGE (assessment:AssessmentTarget {id: row.id})
                    SET assessment.title = row.title,
                        assessment.objective = row.objective,
                        assessment.discrimination = row.discrimination,
                        assessment.common_misconceptions = row.common_misconceptions,
                        assessment.analysis_id = $analysis_id,
                        assessment.review_status = row.review_status
                    WITH assessment, row
                    UNWIND row.section_ids AS section_id
                    MATCH (section:Section {id: section_id})
                    MERGE (section)-[:HAS_ASSESSMENT_TARGET]->(assessment)
                    WITH DISTINCT assessment, row
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
                    SET node.review_status = CASE
                        WHEN node:AssessmentTarget AND node.review_status = 'rejected'
                        THEN 'rejected'
                        ELSE $status
                    END
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

    def update_assessment_target_status(self, target_id: str, status: str) -> bool:
        if status not in {"approved", "rejected"}:
            raise ValueError("status must be approved or rejected")
        if not self.driver:
            logger.error("Neo4j driver is not connected")
            return False
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (target:AssessmentTarget {id: $target_id})
                    SET target.review_status = $status
                    RETURN count(target) AS updated
                    """,
                    target_id=target_id,
                    status=status,
                ).single()
            return bool(result and result["updated"])
        except Exception as e:
            logger.error("Fail to update assessment target review status: %s", e)
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
                labels = ["Requirement", "UserStory", "Service", "SystemComponent", "API", "TestCase", "Actor", "General", "Technology", "Methodology", "Concept"]
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
                    CALL apoc.merge.node([row.subject.label], {name: row.subject.name, group: $group}, {}, {}) YIELD node AS sNode
                    SET sNode += row.subject.props_dict
                    SET sNode.uploader = $uploader
                    SET sNode.source_files = apoc.coll.toSet(coalesce(sNode.source_files, []) + $source_file)
                    SET sNode.doc_type = apoc.coll.toSet(coalesce(sNode.doc_type, []) + $doc_type)

                    WITH sNode, row
                    CALL apoc.merge.node([row.object.label], {name: row.object.name, group: $group}, {}, {}) YIELD node AS oNode
                    SET oNode += row.object.props_dict
                    SET oNode.uploader = $uploader
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
                labels = ["Requirement", "UserStory", "Service", "SystemComponent", "API", "TestCase", "Actor", "General", "Technology", "Methodology", "Concept"]
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

                    # Re-importing an SRD must be able to clear an obsolete
                    # traceability value instead of keeping the old property.
                    if item["label"] == "UserStory":
                        item["props_dict"].setdefault("req_reference", [])

                    # reference = item["props_dict"].get("req_reference")
                    # if reference:
                    #     reference_list = [r.strip() for r in reference.split(',')]
                    #     item["props_dict"]["req_reference"] = reference_list

                    data_to_upload.append(item)

                cypher_query = """
                UNWIND $batch AS row
                CALL (row) {
                    WITH row
                    CALL apoc.merge.node([row.label], {name: row.name, group: $group}, {}, {}) YIELD node AS node
                    SET node += row.props_dict
                    SET node.uploader = $uploader
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

    def replace_project_relationships(
        self,
        group: str,
        relation_types: list[str],
    ) -> bool:
        """Delete derived project relations before rebuilding them.

        Nodes and cross-document properties are preserved. This removes stale
        ``滿足``／``操作`` edges created by a previous SRD import so corrected
        traceability does not coexist with old, incorrect relationships.
        """
        allowed = {"滿足", "操作", "實作需求", "驗證"}
        selected = list(dict.fromkeys(
            relation for relation in relation_types if relation in allowed
        ))
        if not selected:
            return True
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (left)-[rel]->(right)
                    WHERE left.group = $group
                      AND right.group = $group
                      AND type(rel) IN $relation_types
                    DELETE rel
                    RETURN count(rel) AS deleted
                    """,
                    group=group,
                    relation_types=selected,
                ).single()
                logger.info(
                    "Removed %s stale project relationships for group %s",
                    result["deleted"] if result else 0,
                    group,
                )
                return True
        except Exception as exc:
            logger.error("Fail to replace project relationships: %s", exc)
            return False

    def project_graph_stats(
        self,
        group: str,
        doc_type: str,
        source_file: str,
    ) -> dict:
        """Verify the nodes written for this source and group relationships."""
        try:
            with self.driver.session(database=self.database) as session:
                node_records = session.run(
                    """
                    MATCH (n)
                    WHERE n.group = $group
                      AND $doc_type IN coalesce(n.doc_type, [])
                      AND $source_file IN coalesce(n.source_files, [])
                    UNWIND labels(n) AS label
                    RETURN label, count(DISTINCT n) AS count
                    ORDER BY label
                    """,
                    group=group,
                    doc_type=doc_type,
                    source_file=source_file,
                )
                labels = {
                    record["label"]: record["count"] for record in node_records
                }
                relation_records = session.run(
                    """
                    MATCH (left)-[rel]->(right)
                    WHERE left.group = $group AND right.group = $group
                    RETURN type(rel) AS relation, count(rel) AS count
                    ORDER BY relation
                    """,
                    group=group,
                )
                relationships = {
                    record["relation"]: record["count"]
                    for record in relation_records
                }
                return {
                    "node_count": sum(labels.values()),
                    "nodes_by_label": labels,
                    "relationship_count": sum(relationships.values()),
                    "relationships_by_type": relationships,
                }
        except Exception as exc:
            logger.error("Fail to verify project graph: %s", exc)
            raise RuntimeError("專案知識圖譜寫入後驗證失敗") from exc
        
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
