"""
JSON2Neo4j.py

將課程教材的 JSON Knowledge Graph 匯入 Neo4j。
Schema 刻意對齊 SE_Mentor/database/neo4j_importer.py 的
upload_textbook_triples()：

Node:
    (:Concept)
    (:Technology)
    (:Methodology)
    以及 JSON 中實際出現的其他合法 label

Node properties:
    name
    原 JSON properties 轉成的 properties
    source_files

Relationship:
    關係名稱直接作為 Neo4j Relationship Type，例如：
    [:包含於]
    [:使用]
    [:解決]

Relationship properties:
    description
    source_files

注意：
    textbook_entities_*.json 只有實體名稱，沒有 label/properties。
    SE_Mentor 原本的 upload_textbook_triples() 只吃 triples，
    因此 entity JSON 不直接建立節點；節點會由 triples 建立。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

# SE_Mentor 原本使用的連線設定
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_DATABASE = "neo4j"

# 與 SE_Mentor/database/neo4j_importer.py 的 textbook constraints 一致
TEXTBOOK_CONSTRAINT_LABELS = ("Concept", "Technology", "Methodology")

# JSON 中常見的 label。原 importer 的 APOC merge.node 實際上可接受
# JSON 提供的其他 label，因此不把這裡當成硬限制。
KNOWN_LABELS = {
    "Actor",
    "Requirement",
    "UserStory",
    "SystemComponent",
    "Service",
    "API",
    "TestCase",
    "Concept",
    "Technology",
    "Methodology",
    "General",
}

# Neo4j label / relationship type 不能含任意字元。
# 這裡只拒絕明顯危險的 Cypher 控制字元；中文、空白、底線等均允許。
_UNSAFE_IDENTIFIER = re.compile(r"[\x00-\x1f\x7f`{};]")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("JSON2Neo4j")


def load_password() -> str:
    """優先讀取 SE_Mentor/config.py 的 NEO4J_PASSWORD。"""
    try:
        import config  # type: ignore

        password = getattr(config, "NEO4J_PASSWORD", "")
        if password:
            return password.strip().strip("<>")
    except Exception:
        pass

    raise RuntimeError(
        "找不到 Neo4j 密碼。請確認 SE_Mentor/config.py 的 "
        "NEO4J_PASSWORD 已設定。"
    )


def validate_identifier(value: Any, kind: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{kind} 不可為空")
    if _UNSAFE_IDENTIFIER.search(value):
        raise ValueError(f"{kind} 含有不允許的 Cypher 字元：{value!r}")
    return value


def properties_to_dict(properties: Any) -> dict[str, Any]:
    """
    與 SE_Mentor/neo4j_importer.py 的 textbook 版本一致：
    {kv["key"]: kv["value"] for kv in properties}
    若同 key 重複，後面的值覆蓋前面的值。
    """
    if not properties:
        return {}

    result: dict[str, Any] = {}
    if not isinstance(properties, list):
        raise ValueError("properties 必須是 list")

    for item in properties:
        if not isinstance(item, dict):
            raise ValueError(f"properties 項目格式錯誤：{item!r}")

        key = str(item.get("key", "")).strip()
        if not key:
            raise ValueError(f"property key 不可為空：{item!r}")

        result[key] = item.get("value")

    return result


def normalize_triples(raw: Any, path: Path) -> list[dict[str, Any]]:
    """
    支援：
      1. [triple, triple, ...]
      2. {"triples": [triple, ...]}
      3. 單一 triple object
    """
    if isinstance(raw, dict):
        if "triples" in raw:
            raw = raw["triples"]
        elif {"subject", "relation", "object"} <= raw.keys():
            raw = [raw]
        else:
            raise ValueError(f"{path.name}: 找不到 triples / subject / relation / object")

    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: triples JSON 必須是 list")

    triples: list[dict[str, Any]] = []

    for index, triple in enumerate(raw, start=1):
        if not isinstance(triple, dict):
            raise ValueError(f"{path.name} 第 {index} 筆不是 object")

        subject = triple.get("subject")
        relation = triple.get("relation")
        obj = triple.get("object")

        if not isinstance(subject, dict):
            raise ValueError(f"{path.name} 第 {index} 筆 subject 格式錯誤")
        if not isinstance(relation, dict):
            raise ValueError(f"{path.name} 第 {index} 筆 relation 格式錯誤")
        if not isinstance(obj, dict):
            raise ValueError(f"{path.name} 第 {index} 筆 object 格式錯誤")

        subject_name = validate_identifier(subject.get("name"), "subject.name")
        object_name = validate_identifier(obj.get("name"), "object.name")
        subject_label = validate_identifier(subject.get("label"), "subject.label")
        object_label = validate_identifier(obj.get("label"), "object.label")
        relation_name = validate_identifier(relation.get("name"), "relation.name")

        # 保留原 JSON properties
        subject_props = properties_to_dict(subject.get("properties"))
        object_props = properties_to_dict(obj.get("properties"))

        # 與原 importer 相同：不額外塞 group/uploader/doc_type。
        triples.append(
            {
                "subject": {
                    "name": subject_name,
                    "label": subject_label,
                    "props_dict": subject_props,
                },
                "relation": {
                    "name": relation_name,
                    "description": relation.get("description", ""),
                },
                "object": {
                    "name": object_name,
                    "label": object_label,
                    "props_dict": object_props,
                },
            }
        )

    return triples


def create_textbook_constraints(driver) -> None:
    """
    完全依照 SE_Mentor/database/neo4j_importer.py
    upload_textbook_triples() 建立的三個 constraints。
    """
    query = """
    CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
    FOR (n:{label})
    REQUIRE (n.name, n.group) IS UNIQUE
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        for label in TEXTBOOK_CONSTRAINT_LABELS:
            # label / constraint_name 都是程式固定值，不來自使用者輸入。
            session.run(
                query.format(
                    constraint_name=f"{label.lower()}_group_name_unique",
                    label=label,
                )
            )


def reset_legacy_entity_graph(driver) -> None:
    """
    刪除先前 JSON2Neo4j.py 建立的舊 Schema：
        (:Entity)-[:RELATED_TO]->(:Entity)

    不刪除其他 label，例如 Haystack 的文件節點。
    """
    query = """
    MATCH (n:Entity)
    DETACH DELETE n
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query).consume()
        logger.info("已刪除舊版 (:Entity) Knowledge Graph。")


def import_triples(driver, triples: list[dict[str, Any]], source_file: str) -> int:
    """
    與原 upload_textbook_triples() 的 Cypher Schema 對齊。

    原始版本：
        apoc.merge.node([row.subject.label], {name: row.subject.name}, ...)
        apoc.merge.node([row.object.label], {name: row.object.name}, ...)
        apoc.merge.relationship(sNode, row.relation.name, ...)
    """
    if not triples:
        return 0

    query = """
    UNWIND $batch AS row

    CALL apoc.merge.node(
        [row.subject.label],
        {name: row.subject.name},
        {},
        {}
    ) YIELD node AS sNode

    WITH sNode, row
    SET sNode += row.subject.props_dict
    SET sNode.source_files =
        apoc.coll.toSet(
            coalesce(sNode.source_files, []) + $source_file
        )

    WITH sNode, row
    CALL apoc.merge.node(
        [row.object.label],
        {name: row.object.name},
        {},
        {}
    ) YIELD node AS oNode

    WITH sNode, oNode, row
    SET oNode += row.object.props_dict
    SET oNode.source_files =
        apoc.coll.toSet(
            coalesce(oNode.source_files, []) + $source_file
        )

    WITH sNode, oNode, row
    CALL apoc.merge.relationship(
        sNode,
        row.relation.name,
        {},
        {},
        oNode
    ) YIELD rel

    SET rel.description = row.relation.description
    SET rel.source_files =
        apoc.coll.toSet(
            coalesce(rel.source_files, []) + $source_file
        )

    RETURN count(rel) AS relationships
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        record = session.run(
            query,
            batch=triples,
            source_file=[source_file],
        ).single()

    return int(record["relationships"])


def inspect_entity_json(folder: Path, triple_files: list[Path]) -> None:
    """
    entity JSON 只是名稱清單；原 importer 的 textbook 流程不會直接
    用它建立 Node。這裡只做統計/提示，避免誤把它全部當 Concept。
    """
    entity_files = sorted(folder.glob("*_entities_*.json"))
    if not entity_files:
        return

    triple_names: set[str] = set()
    for triple_file in triple_files:
        try:
            raw = json.loads(triple_file.read_text(encoding="utf-8-sig"))
            for t in normalize_triples(raw, triple_file):
                triple_names.add(t["subject"]["name"])
                triple_names.add(t["object"]["name"])
        except Exception:
            # triples 的真正錯誤會在主流程再次回報；這裡不重複噴錯。
            continue

    total_names = 0
    not_in_triples = 0

    for path in entity_files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))

            # ch1 曾出現 [["a","b"], ["c","d"]] 的巢狀格式
            def flatten_names(value):
                if isinstance(value, list):
                    for item in value:
                        yield from flatten_names(item)
                elif isinstance(value, str):
                    yield value

            names = list(flatten_names(raw))
            total_names += len(names)
            not_in_triples += sum(1 for name in names if name not in triple_names)
        except Exception:
            pass

    logger.info(
        f"[entity JSON] 偵測到 {len(entity_files)} 個實體清單，"
        f"共 {total_names} 個名稱。"
    )
    if not_in_triples:
        logger.info(
            f"[entity JSON] 有 {not_in_triples} 個名稱沒有出現在 triples JSON；"
            "依 SE_Mentor 原本 textbook importer 的流程，不會單獨建立這些節點。"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="將 textbook triples JSON 依 SE_Mentor neo4j_importer.py Schema 匯入 Neo4j"
    )
    parser.add_argument(
        "folder",
        help="包含 textbook_triples_*.json / textbook_entities_*.json 的資料夾",
    )
    parser.add_argument(
        "--reset-legacy-entity",
        action="store_true",
        help="刪除舊版 JSON2Neo4j 建立的 (:Entity) 圖譜後再匯入；不刪其他 labels",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"找不到資料夾：{folder}")

    triple_files = sorted(folder.glob("*_triples_*.json"))
    if not triple_files:
        raise SystemExit(
            f"在 {folder} 找不到 *_triples_*.json"
        )

    password = load_password()

    logger.info(f"Neo4j: {NEO4J_URI}")
    logger.info(f"資料夾: {folder}")
    logger.info(f"找到 {len(triple_files)} 個 triples JSON")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, password),
    )

    try:
        driver.verify_connectivity()
        logger.info("Neo4j 連線成功。")

        create_textbook_constraints(driver)

        if args.reset_legacy_entity:
            reset_legacy_entity_graph(driver)

        inspect_entity_json(folder, triple_files)

        total_triples = 0
        total_relationships = 0
        failed_files = 0
        labels = set()
        relation_types = set()

        for path in triple_files:
            try:
                raw = json.loads(path.read_text(encoding="utf-8-sig"))
                triples = normalize_triples(raw, path)

                for t in triples:
                    labels.add(t["subject"]["label"])
                    labels.add(t["object"]["label"])
                    relation_types.add(t["relation"]["name"])

                count = import_triples(
                    driver,
                    triples,
                    path.name,
                )

                total_triples += len(triples)
                total_relationships += count

                logger.info(
                    f"[完成] {path.name}: "
                    f"{len(triples)} triples / {count} relationships"
                )

            except Exception as exc:
                failed_files += 1
                logger.error(f"[錯誤] {path}: {exc}")

        logger.info("")
        logger.info("========== 匯入完成 ==========")
        logger.info(f"Triple 數量：{total_triples}")
        logger.info(f"Relationship 建立/更新：{total_relationships}")
        logger.info(f"Node labels：{', '.join(sorted(labels))}")
        logger.info(f"Relationship types：{len(relation_types)} 種")
        logger.info(f"失敗檔案：{failed_files}")

        if failed_files:
            raise SystemExit(1)

    finally:
        driver.close()


if __name__ == "__main__":
    main()