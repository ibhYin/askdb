"""提示词版本测试：版本可切换，且关键约束都在。"""

from __future__ import annotations

from askdb.agent import build_messages
from askdb.prompts import PROMPTS, get_prompt


def test_both_versions_registered():
    assert set(PROMPTS) == {"v1", "v2"}
    assert get_prompt("v1") != get_prompt("v2")


def test_unknown_version_falls_back_to_default():
    assert get_prompt("v9") == get_prompt()
    assert get_prompt(None) == get_prompt("v2")


def test_v2_keeps_v1_hard_rules():
    v2 = get_prompt("v2")
    assert "只输出一条 SELECT" in v2
    assert "```sql" in v2


def test_v2_adds_output_contract_and_evidence_rules():
    v2 = get_prompt("v2")
    assert "问了几样东西，就 SELECT 几列" in v2      # 修多返回列
    assert "evidence" in v2                          # 提醒用参考信息
    assert "【第一步" in v2                          # 先想后写


def test_build_messages_uses_requested_version():
    system = build_messages("q", "schema", prompt_version="v1")[0].content
    assert system.startswith(get_prompt("v1"))
    assert system.endswith("数据库 schema：\nschema")


def test_build_messages_defaults_to_default_version():
    system = build_messages("q", "schema")[0].content
    assert system.startswith(get_prompt())


def test_evidence_is_appended_to_user_message():
    user = build_messages("q", "s", evidence="以 1500 分以上为准")[-1].content
    assert "参考信息：以 1500 分以上为准" in user
