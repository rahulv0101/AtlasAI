from atlas_ai.learning.online_allocator import OnlineAllocator
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.models import TradeSignal


def test_allocator_prefers_confidence_initially():
    allocator = OnlineAllocator()
    s1 = TradeSignal("a", "buy", 0.9, 100, 150)
    s2 = TradeSignal("b", "buy", 0.2, 100, 150)
    chosen = allocator.choose([s1, s2])
    assert chosen.strategy == "a"


def test_allocator_update_changes_stats():
    allocator = OnlineAllocator()
    allocator.update("ema_crossover", 12.0)
    allocator.update("ema_crossover", -5.0)

    stats = allocator.stats()["ema_crossover"]
    assert stats.wins == 2
    assert stats.losses == 2
    assert stats.pnl == 7.0


def test_allocator_persists_state_with_store(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    allocator = OnlineAllocator(store=store)
    allocator.update("breakout", 10.0)

    reloaded = OnlineAllocator(store=store)
    stats = reloaded.stats()["breakout"]
    assert stats.wins == 2
    assert stats.losses == 1
    assert stats.pnl == 10.0


def test_allocator_choose_respects_score_overrides():
    allocator = OnlineAllocator()
    s1 = TradeSignal("a", "buy", 0.9, 100, 150)
    s2 = TradeSignal("b", "buy", 0.2, 100, 150)
    chosen = allocator.choose([s1, s2], score_overrides={"a": 0.1, "b": 0.9})
    assert chosen.strategy == "b"


def test_allocator_preserves_evolved_rule_signature_for_overrides():
    allocator = OnlineAllocator()
    s1 = TradeSignal("evolved_rule(fast=22,slow=52,rsi_buy=34.0,rsi_sell=57.0)", "flat", 0.0, 100, 150)
    s2 = TradeSignal("momentum_continuation", "buy", 0.3, 100, 150)
    chosen = allocator.choose(
        [s1, s2],
        score_overrides={
            "evolved_rule(fast=22,slow=52,rsi_buy=34.0,rsi_sell=57.0)": 0.0,
            "momentum_continuation": 0.9,
        },
    )
    assert chosen.strategy == "momentum_continuation"
