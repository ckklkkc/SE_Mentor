import sys
import types
import unittest


# The production config is intentionally not committed. Supply only the values
# needed while importing the pure transformation code under test.
if "config" not in sys.modules:
    config = types.ModuleType("config")
    config.NEO4J_PASSWORD = "test"
    config.CLAUDE_API_KEY = "test"
    sys.modules["config"] = config

# Keep these unit tests focused on deterministic traceability. The runtime
# clients are optional in the lightweight test environment and are never used.
if "neo4j" not in sys.modules:
    neo4j = types.ModuleType("neo4j")
    neo4j.GraphDatabase = object
    neo4j.Driver = object
    neo4j.Result = object
    sys.modules["neo4j"] = neo4j
if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    sys.modules["openai"] = openai
if "anthropic" not in sys.modules:
    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = object
    sys.modules["anthropic"] = anthropic
if "utils.file_processor" not in sys.modules:
    utils = types.ModuleType("utils")
    file_processor = types.ModuleType("utils.file_processor")
    utils.file_processor = file_processor
    sys.modules["utils"] = utils
    sys.modules["utils.file_processor"] = file_processor

from database.neo4j_importer import (  # noqa: E402
    Entity,
    EntityLabel,
    EntityList,
    KeyValPair,
    Neo4jImporter,
)
from services.kg_constructor import KGConstructor  # noqa: E402


def entity(name, label, **properties):
    pairs = []
    for key, value in properties.items():
        values = value if isinstance(value, list) else [value]
        pairs.extend(KeyValPair(key=key, value=item) for item in values)
    return Entity(name=name, label=label, properties=pairs)


class ProjectKnowledgeGraphIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.constructor = KGConstructor.__new__(KGConstructor)

    def test_user_stories_map_only_by_explicit_or_matching_ids(self):
        raw = EntityList(entities=[
            entity("瀏覽菜單故事", EntityLabel.UserStory, us_id="MEN-01",
                   description="作為使用者，我能瀏覽菜單",
                   req_reference="MEN-05"),
            entity("管理購物車故事", EntityLabel.UserStory, us_id="MEN-02",
                   description="作為使用者，我能管理購物車",
                   req_reference="不存在的需求"),
            entity("查詢進度故事", EntityLabel.UserStory, us_id="STA-US-01",
                   description="作為使用者，我能查詢進度",
                   req_reference="STA-01"),
            entity("瀏覽菜單", EntityLabel.Requirement, req_id="MEN-01",
                   description="瀏覽菜單", req_category="功能需求"),
            entity("購物車", EntityLabel.Requirement, req_id="MEN-02",
                   description="管理購物車", req_category="功能需求"),
            entity("菜單評論", EntityLabel.Requirement, req_id="MEN-05",
                   description="檢視評論", req_category="功能需求"),
            entity("訂單狀態", EntityLabel.Requirement, req_id="STA-01",
                   description="顯示訂單狀態", req_category="功能需求"),
        ])

        matches, requirements, actors = self.constructor.match_us_to_fr_v2(raw)

        mapped = {
            item["us"]["properties_dict"]["us_id"]: item["us_req_ids"]
            for item in matches
        }
        self.assertEqual(mapped["MEN-01"], ["MEN-01"])
        self.assertEqual(mapped["MEN-02"], ["MEN-02"])
        self.assertEqual(mapped["STA-US-01"], ["STA-01"])
        self.assertEqual(len(requirements), 4)
        self.assertEqual(actors, [])

    def test_unmatched_story_and_actor_are_preserved(self):
        raw = EntityList(entities=[
            entity("使用者", EntityLabel.Actor, description="一般使用者"),
            entity("未配對故事", EntityLabel.UserStory, us_id="US-X",
                   description="作為使用者，我能提出需求"),
            entity("登入", EntityLabel.Requirement, req_id="USR-01",
                   description="登入", req_category="功能需求"),
        ])

        matches, requirements, actors = self.constructor.match_us_to_fr_v2(raw)

        self.assertEqual(matches[0]["us_req_ids"], [])
        self.assertEqual(requirements[0]["properties_dict"]["req_id"], "USR-01")
        self.assertEqual(actors[0]["name"], "使用者")

    def test_actor_relationship_uses_story_reference(self):
        entities = EntityList(entities=[
            entity("店家老闆", EntityLabel.Actor, description="管理店家"),
            entity("管理訂單故事", EntityLabel.UserStory, us_id="MAN-01",
                   description="作為一位店家老闆，我能管理即時訂單",
                   req_reference="MAN-01"),
            entity("訂單控制", EntityLabel.Requirement, req_id="MAN-01",
                   description="接受或拒絕訂單", req_category="功能需求"),
        ])

        relations = self.constructor.create_actor_relationships(entities)

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["subject"]["name"], "店家老闆")
        self.assertEqual(relations[0]["object"]["name"], "訂單控制")

    def test_composite_requirement_actor_is_split(self):
        entities = EntityList(entities=[
            entity("店家", EntityLabel.Actor, description="店家角色"),
            entity("消費者", EntityLabel.Actor, description="消費者角色"),
            entity("登入驗證", EntityLabel.Requirement, req_id="USR-01",
                   description="登入平台", req_category="功能需求",
                   actor="店家與消費者"),
        ])

        relations = self.constructor.create_actor_relationships(entities)

        self.assertEqual(
            {relation["subject"]["name"] for relation in relations},
            {"店家", "消費者"},
        )

    def test_duplicate_reference_properties_remain_a_list(self):
        importer = Neo4jImporter("bolt://unused", "neo4j", "unused")
        props = importer.convert_properties_to_dict([
            {"key": "req_reference", "value": "MEN-01"},
            {"key": "req_reference", "value": "NF-02"},
        ])
        self.assertEqual(props["req_reference"], ["MEN-01", "NF-02"])


if __name__ == "__main__":
    unittest.main()
