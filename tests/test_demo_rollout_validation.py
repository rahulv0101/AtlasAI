from __future__ import annotations

from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.ui_server import create_ui_app


def _ask(client, query: str) -> dict:
    return client.post("/api/ask-atlas", json={"query": query}).get_json()


def _setup_demo_auto(client) -> None:
    _ask(client, "set execution environment demo")
    _ask(client, "set autonomy mode guarded_execution")
    _ask(client, "enable execution gate")
    _ask(client, "set rollout stage limited_live_pilot approved")
    _ask(client, "configure tiny live pilot XAUUSD continuation 0.01 2 0.6 1")
    _ask(client, "approve tiny live pilot")
    _ask(client, "activate tiny live pilot")
    _ask(client, "create automation policy p1 live continuation XAUUSD test 0.01 0.6 2 limited_live_pilot")
    _ask(client, "approve automation policy p1")
    _ask(client, "activate automation policy p1")
    _ask(client, "activate operator present mode")
    _ask(client, "activate session override")


def test_blocked_trade_reason_is_consistent_across_execution_runtime_and_ask(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags({"execute_trades": False, "read_only_ui": False})
    app = create_ui_app(store)
    client = app.test_client()

    _ask(client, "enable execution gate")
    created = _ask(client, "create execution request paper place_trade dry instrument=XAUUSD setup_family=continuation size=0.01")
    req_id = created["execution_request"]["id"]
    _ask(client, f"approve execution {req_id}")
    _ask(client, "activate kill switch")
    blocked = _ask(client, f"run execution {req_id}")

    assert blocked["execution_result"]["reason"] == "kill_switch_active"
    assert blocked["execution_state"]["kill_switch_active"] is True
    runtime = client.get("/api/runtime").get_json()
    assert runtime["execution"]["kill_switch_active"] is True
    assert int(runtime["execution"].get("pending_live_orders_count", 0) or 0) == 0


def test_allowed_guarded_trade_path_updates_state_and_audit(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags({"execute_trades": False, "read_only_ui": False})
    app = create_ui_app(store)
    client = app.test_client()

    _ask(client, "enable execution gate")
    created = _ask(client, "create execution request paper place_trade dry instrument=XAUUSD setup_family=continuation size=0.01")
    req_id = created["execution_request"]["id"]
    _ask(client, f"approve execution {req_id}")
    out = _ask(client, f"run execution {req_id}")

    assert out["execution_result"]["ok"] is True
    assert out["execution_result"]["dry_run"] is True
    runtime = client.get("/api/runtime").get_json()
    assert runtime["execution"]["last_execution_result"].get("request_id") == req_id
    audit = _ask(client, "show execution audit")
    assert "execution_audit" in audit
    assert len(audit["execution_audit"]) >= 1


def test_generic_queries_remain_read_only_and_side_effect_free(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    app = create_ui_app(store)
    client = app.test_client()

    for q in ("should i buy or sell?", "can you trade now?", "what is the market bias?"):
        _ask(client, q)
    pending = _ask(client, "show execution requests")
    assert "- none" in pending["result"]


def test_only_explicit_command_intent_can_reach_action_routing(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "read_only_ui": False,
            "freshness_runtime": {"status_age_ms": 50, "is_stale": False, "max_freshness_age_ms": 120000},
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    _setup_demo_auto(client)
    generic = _ask(client, "should i buy or sell?")
    assert "automation_result" not in generic
    explicit = _ask(client, "run auto execution demo place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "automation_result" in explicit


def test_stale_freshness_blocks_runtime_auto_execution_and_surfaces_consistently(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "read_only_ui": False,
            "freshness_runtime": {"status_age_ms": 240000, "is_stale": True, "max_freshness_age_ms": 120000},
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    _setup_demo_auto(client)
    out = _ask(client, "run auto execution demo place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert out["automation_result"]["reason"] == "freshness_stale"
    status = _ask(client, "can you auto-trade right now?")
    assert "freshness_stale" in status["result"]
    runtime = client.get("/api/runtime").get_json()
    assert runtime["freshness_runtime"]["is_stale"] is True
    assert runtime["freshness_runtime"]["source"] == "runtime_flags"


def test_cross_surface_execution_state_consistency_for_environment(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    app = create_ui_app(store)
    client = app.test_client()
    ask = _ask(client, "set execution environment demo")
    runtime = client.get("/api/runtime").get_json()
    assert ask["execution_state"]["execution_environment"] == runtime["execution"]["execution_environment"]


def test_readiness_checklist_payload_is_read_only_and_structured(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    app = create_ui_app(store)
    client = app.test_client()

    out = _ask(client, "show guarded demo readiness")
    assert "guarded_demo_readiness" in out
    readiness = out["guarded_demo_readiness"]
    assert readiness["execution_path_source"] == "execution_gate_guarded"
    assert readiness["ask_query_mode"] == "read_only_by_default"
    pending = _ask(client, "show execution requests")
    assert "- none" in pending["result"]
