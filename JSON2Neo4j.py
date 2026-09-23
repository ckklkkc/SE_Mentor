import json
import sys
from pathlib import Path

from neo4j import GraphDatabase


# ============================================================
# Neo4j 設定
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "tabotaidb"


# ============================================================
# JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 建立 / 更新 Entity
# ============================================================

def merge_entity(tx, entity):
    """
    entity 格式：

    {
        "name": "軟體危機",
        "label": "Concept",
        "properties": [
            {
                "key": "description",
                "value": "..."
            }
        ]
    }
    """

    name = entity.get("name")

    if not name:
        return

    label = entity.get("label", "Entity")

    description = ""

    for prop in entity.get("properties", []):
        if prop.get("key") == "description":
            description = prop.get("value", "")
            break

    tx.run(
        """
        MERGE (n:Entity {name: $name})
        SET n.label = $label,
            n.description = $description
        """,
        name=name,
        label=label,
        description=description,
    )


# ============================================================
# 建立 Relationship
# ============================================================

def merge_triple(tx, triple):

    subject = triple.get("subject")
    relation = triple.get("relation")
    obj = triple.get("object")

    if not subject or not relation or not obj:
        return

    subject_name = subject.get("name")
    object_name = obj.get("name")

    if not subject_name or not object_name:
        return

    relation_name = relation.get("name", "RELATED_TO")
    relation_description = relation.get("description", "")

    # 建立 / 更新 Subject
    merge_entity(tx, subject)

    # 建立 / 更新 Object
    merge_entity(tx, obj)

    # 建立 Relationship
    tx.run(
        """
        MATCH (s:Entity {name: $subject_name})
        MATCH (o:Entity {name: $object_name})

        MERGE (s)-[r:RELATED_TO {name: $relation_name}]->(o)

        SET r.description = $relation_description
        """,
        subject_name=subject_name,
        object_name=object_name,
        relation_name=relation_name,
        relation_description=relation_description,
    )


# ============================================================
# Entity JSON
# ============================================================

def process_entity_json(session, path):

    data = load_json(path)

    entity_names = set()

    # --------------------------------------------------------
    # 格式 1:
    #
    # [
    #   "需求工程",
    #   "需求確認",
    #   "需求擷取"
    # ]
    # --------------------------------------------------------

    if isinstance(data, list) and all(
        isinstance(x, str) for x in data
    ):

        print(f"[格式] List[str]")

        for name in data:
            if name.strip():
                entity_names.add(name.strip())

    # --------------------------------------------------------
    # 格式 2:
    #
    # [
    #   [
    #     "軟體危機",
    #     "軟體開發"
    #   ],
    #   [
    #     "軟體維護",
    #     "軟體工程"
    #   ]
    # ]
    # --------------------------------------------------------

    elif isinstance(data, list) and all(
        isinstance(x, list) for x in data
    ):

        print(f"[格式] List[List[str]]")

        for group in data:

            for name in group:

                if isinstance(name, str) and name.strip():
                    entity_names.add(name.strip())

    else:

        print(f"[錯誤] Entity JSON 格式無法辨識: {path}")

        return

    # --------------------------------------------------------
    # 寫入 Neo4j
    # --------------------------------------------------------

    with session.begin_transaction() as tx:

        for name in entity_names:

            tx.run(
                """
                MERGE (n:Entity {name: $name})
                """,
                name=name,
            )

        tx.commit()

    print(
        f"[Entity] {path.name}: "
        f"{len(entity_names)} unique entities"
    )


# ============================================================
# Triple JSON
# ============================================================

def process_triples_json(session, path):

    data = load_json(path)

    # 單一 triple object
    if isinstance(data, dict):

        if (
            "subject" in data
            and "relation" in data
            and "object" in data
        ):
            data = [data]

        else:
            print(f"[跳過] Triple 格式錯誤: {path}")
            return

    # Triple array
    if not isinstance(data, list):

        print(f"[跳過] Triple JSON 不是 list: {path}")
        return

    count = 0

    with session.begin_transaction() as tx:

        for triple in data:

            if not isinstance(triple, dict):
                continue

            if not all(
                key in triple
                for key in ["subject", "relation", "object"]
            ):
                continue

            merge_triple(tx, triple)

            count += 1

        tx.commit()

    print(
        f"[Triple] {path.name}: "
        f"{count} triples"
    )


# ============================================================
# 判斷 JSON 類型
# ============================================================

def process_json_file(session, path):

    try:

        data = load_json(path)

    except Exception as e:

        print(f"[錯誤] JSON 讀取失敗: {path}")
        print(f"       {e}")
        return

    # ========================================================
    # List[str]
    # → Entity
    # ========================================================

    if isinstance(data, list):

        if all(isinstance(x, str) for x in data):

            process_entity_json(session, path)
            return

        # ====================================================
        # List[List[str]]
        # → Entity
        # ====================================================

        if all(isinstance(x, list) for x in data):

            # 確認是不是 List[List[str]]
            if all(
                all(isinstance(item, str) for item in group)
                for group in data
            ):

                process_entity_json(session, path)
                return

        # ====================================================
        # List[dict]
        # → Triple
        # ====================================================

        if all(isinstance(x, dict) for x in data):

            if data and all(
                "subject" in x
                and "relation" in x
                and "object" in x
                for x in data
            ):

                process_triples_json(session, path)
                return

    # ========================================================
    # dict
    # → Triple
    # ========================================================

    if isinstance(data, dict):

        if (
            "subject" in data
            and "relation" in data
            and "object" in data
        ):

            process_triples_json(session, path)
            return

    print(
        f"[跳過] 無法辨識格式: {path}"
    )


# ============================================================
# Main
# ============================================================

def process_directory(directory):

    directory = Path(directory)

    if not directory.exists():

        print(f"找不到資料夾: {directory}")
        return

    json_files = sorted(
        directory.rglob("*.json")
    )

    print("=" * 60)
    print(f"資料夾: {directory}")
    print(f"找到 JSON: {len(json_files)} 個")
    print("=" * 60)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            NEO4J_USERNAME,
            NEO4J_PASSWORD
        )
    )

    try:

        with driver.session() as session:

            for path in json_files:

                print()
                print(f"處理: {path}")

                process_json_file(
                    session,
                    path
                )

    finally:

        driver.close()

    print()
    print("=" * 60)
    print("JSON → Neo4j 匯入完成")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "使用方式:"
        )

        print(
            'uv run JSON2Neo4j.py "JSON資料夾"'
        )

        sys.exit(1)

    process_directory(
        sys.argv[1]
    )