from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from atlas_ai.decision_framework import DecisionInput, DecisionOutput
from atlas_ai.market_scanner import MarketScanner
from atlas_ai.volatility_intelligence import VolatilityAssessment


class _GateStub:
    def __init__(self):
        self.requests = []

    def create_request(self, **kwargs):  # pragma: no cover - not used in these tests
        self.requests.append(dict(kwargs))
        return {"id": "stub"}


class _SettingsStub:
    def __init__(self, *, storage_dir: str = "storage"):
        self.learning_mode = True
        self.learning_force_entry_per_session = True
        self.learning_graduation_min_trades = 100
        self.learning_force_min_confidence = 0.40
        self.learning_probe_min_edge_gap = 0.07
        self.learning_probe_confidence_bump_elevated = 0.04
        self.learning_probe_confidence_bump_chop = 0.06
        self.learning_probe_confidence_bump_transition = 0.03
        self.demo_mode = True
        self.storage_dir = storage_dir
        self.execute_trades = False
        self.duplicate_signal_window_seconds = 45
        self.scanner_cooldown_seconds = 180
        self.rejected_candidate_debounce_seconds = 5


def _mk_scanner(settings: _SettingsStub | None = None, *, reset_session: bool = True) -> MarketScanner:
    scanner = MarketScanner(execution_gate=_GateStub(), broker_adapter=None, settings=settings or _SettingsStub())
    # Keep tests deterministic: treat current session as having no scanner trades yet.
    if reset_session:
        scanner._session_trades = {}
    return scanner


def test_scanner_session_uses_calendar_not_hour_bucket_on_weekend(monkeypatch, tmp_path) -> None:
    fake_mt5 = SimpleNamespace(
        TIMEFRAME_H1=1,
        TIMEFRAME_M15=2,
        symbol_info_tick=lambda _symbol: None,
        copy_rates_from_pos=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake_mt5)
    scanner = _mk_scanner(_SettingsStub(storage_dir=str(tmp_path)))

    inp = scanner._build_decision_input(datetime(2026, 5, 24, 12, 30, tzinfo=timezone.utc))

    assert inp.session == "none"
    assert inp.session_tradeable is False


def test_adaptive_probe_allows_no_trade_with_safe_minimums() -> None:
    scanner = _mk_scanner()
    out = DecisionOutput(
        trade_decision="NO_TRADE",
        confidence=0.47,
        blockers=[],
        buy_score=0.60,
        sell_score=0.49,
        timestamp=datetime.now(timezone.utc),
    )
    inp = DecisionInput(
        session="london",
        session_tradeable=True,
        in_news_window=False,
        high_impact_news_hard_block=False,
        spread_status="normal",
    )
    vol = VolatilityAssessment(response_rule="normal_analysis")
    assert scanner._should_apply_adaptive_probe(
        settings=_SettingsStub(),
        out=out,
        inp=inp,
        vol=vol,
        closed_trade_count=12,
    )


def test_adaptive_probe_blocks_hard_volatility_or_news() -> None:
    scanner = _mk_scanner()
    out = DecisionOutput(
        trade_decision="NO_TRADE",
        confidence=0.61,
        blockers=[],
        timestamp=datetime.now(timezone.utc),
    )
    inp = DecisionInput(
        session="london",
        session_tradeable=True,
        in_news_window=False,
        high_impact_news_hard_block=False,
        spread_status="normal",
    )
    assert not scanner._should_apply_adaptive_probe(
        settings=_SettingsStub(),
        out=out,
        inp=inp,
        vol=VolatilityAssessment(response_rule="news_hard_block"),
        closed_trade_count=3,
    )
    inp.in_news_window = True
    assert not scanner._should_apply_adaptive_probe(
        settings=_SettingsStub(),
        out=out,
        inp=inp,
        vol=VolatilityAssessment(response_rule="normal_analysis"),
        closed_trade_count=3,
    )


def test_adaptive_probe_allows_candidate_below_threshold_if_safe() -> None:
    scanner = _mk_scanner()
    out = DecisionOutput(
        trade_decision="BUY_CANDIDATE",
        confidence=0.43,
        blockers=[],
        can_create_execution_request=False,
        buy_score=0.63,
        sell_score=0.31,
        timestamp=datetime.now(timezone.utc),
    )
    inp = DecisionInput(
        session="new_york",
        session_tradeable=True,
        in_news_window=False,
        high_impact_news_hard_block=False,
        spread_status="elevated",
    )
    assert scanner._should_apply_adaptive_probe(
        settings=_SettingsStub(),
        out=out,
        inp=inp,
        vol=VolatilityAssessment(response_rule="reduce_risk_require_higher_confluence"),
        closed_trade_count=9,
    )


def test_adaptive_probe_edge_gap_blocks_close_scores() -> None:
    scanner = _mk_scanner()
    out = DecisionOutput(
        trade_decision="NO_TRADE",
        confidence=0.52,
        blockers=[],
        buy_score=0.51,
        sell_score=0.49,
        timestamp=datetime.now(timezone.utc),
    )
    inp = DecisionInput(
        session="london",
        session_tradeable=True,
        in_news_window=False,
        high_impact_news_hard_block=False,
        spread_status="normal",
    )
    assert not scanner._should_apply_adaptive_probe(
        settings=_SettingsStub(),
        out=out,
        inp=inp,
        vol=VolatilityAssessment(response_rule="normal_analysis"),
        closed_trade_count=12,
    )


def test_adaptive_probe_floor_increases_for_chop_volatility() -> None:
    scanner = _mk_scanner()
    base_floor = scanner._effective_probe_confidence_min(
        settings=_SettingsStub(),
        vol=VolatilityAssessment(volatility_class="normal_volatility", response_rule="normal_analysis"),
    )
    chop_floor = scanner._effective_probe_confidence_min(
        settings=_SettingsStub(),
        vol=VolatilityAssessment(volatility_class="chop_volatility", response_rule="prefer_no_trade_reduce_confidence"),
    )
    assert chop_floor > base_floor


def test_probe_state_persists_across_scanner_restart(tmp_path: Path) -> None:
    settings = _SettingsStub(storage_dir=str(tmp_path))
    scanner_one = _mk_scanner(settings=settings, reset_session=False)
    key = scanner_one._session_key()
    scanner_one._record_session_trade()
    assert int(scanner_one._session_trades.get(key, 0)) == 1
    scanner_two = _mk_scanner(settings=settings, reset_session=False)
    assert int(scanner_two._session_trades.get(key, 0)) == 1


class _SequenceFramework:
    def __init__(self, outputs: list[DecisionOutput]):
        self.outputs = list(outputs)

    def evaluate(self, _inp):
        return self.outputs.pop(0)


class _ScannerForCooldownTest(MarketScanner):
    def __init__(self, *args, inputs: list[DecisionInput], **kwargs):
        super().__init__(*args, **kwargs)
        self._inputs = list(inputs)

    def _build_decision_input(self, now):
        inp = self._inputs.pop(0)
        inp.timestamp = now
        return inp

    def _calculate_sl_tp(self, *args, **kwargs):
        return 1.0, 2.0

    def _record_session_trade(self):
        return None

    def _session_key(self) -> str:
        return "test_session"

    def _publish_learning_status(self, settings):
        return None

    def _publish_volatility_state(self, *, status, vol):
        return None


def _tradeable_input() -> DecisionInput:
    return DecisionInput(session="london", session_tradeable=True, spread_pips=1.0)


def _candidate(decision: str, confidence: float, can_create: bool) -> DecisionOutput:
    return DecisionOutput(
        trade_decision=decision,
        confidence=confidence,
        can_create_execution_request=can_create,
        buy_score=confidence if decision == "BUY_CANDIDATE" else 0.0,
        sell_score=confidence if decision == "SELL_CANDIDATE" else 0.0,
        reason=f"{decision} confidence={confidence}",
        session="london",
    )


def test_created_execution_request_still_starts_full_scanner_cooldown(monkeypatch, tmp_path: Path) -> None:
    import atlas_ai.market_scanner as scanner_module

    settings = _SettingsStub(storage_dir=str(tmp_path))
    settings.learning_mode = False
    gate = _GateStub()
    first = datetime(2026, 9, 4, 11, 31, 13, tzinfo=timezone.utc)
    second = datetime(2026, 9, 4, 11, 32, 43, tzinfo=timezone.utc)

    class _FakeDateTime(datetime):
        values = [first, second]

        @classmethod
        def now(cls, tz=None):
            value = cls.values.pop(0)
            return value if tz is None else value.astimezone(tz)

    scanner = _ScannerForCooldownTest(
        execution_gate=gate,
        broker_adapter=None,
        settings=settings,
        inputs=[_tradeable_input(), _tradeable_input()],
    )
    scanner.framework = _SequenceFramework(
        [
            _candidate("BUY_CANDIDATE", 0.65, True),
            _candidate("SELL_CANDIDATE", 0.65, True),
        ]
    )
    scanner.SCAN_LOG_PATH = str(tmp_path / "scan.jsonl")
    scanner.REJECTED_LOG_PATH = str(tmp_path / "rejected.jsonl")
    monkeypatch.setattr(scanner_module, "datetime", _FakeDateTime)
    monkeypatch.setattr(scanner_module, "load_settings", lambda: settings)

    scanner._run_single_scan()
    assert len(gate.requests) == 1
    assert scanner._cooldown_until == first.timestamp() + settings.scanner_cooldown_seconds

    scanner._run_single_scan()
    assert len(gate.requests) == 1


def test_rejected_candidate_debounce_does_not_block_later_threshold_candidate(monkeypatch, tmp_path: Path) -> None:
    import atlas_ai.market_scanner as scanner_module

    settings = _SettingsStub(storage_dir=str(tmp_path))
    settings.learning_mode = False
    gate = _GateStub()
    first = datetime(2026, 8, 26, 2, 58, 20, tzinfo=timezone.utc)
    second = datetime(2026, 8, 26, 3, 0, 20, tzinfo=timezone.utc)

    class _FakeDateTime(datetime):
        values = [first, second]

        @classmethod
        def now(cls, tz=None):
            value = cls.values.pop(0)
            return value if tz is None else value.astimezone(tz)

    scanner = _ScannerForCooldownTest(
        execution_gate=gate,
        broker_adapter=None,
        settings=settings,
        inputs=[_tradeable_input(), _tradeable_input()],
    )
    scanner.framework = _SequenceFramework(
        [
            _candidate("SELL_CANDIDATE", 0.40, False),
            _candidate("SELL_CANDIDATE", 0.53, True),
        ]
    )
    scanner.SCAN_LOG_PATH = str(tmp_path / "scan.jsonl")
    scanner.REJECTED_LOG_PATH = str(tmp_path / "rejected.jsonl")
    monkeypatch.setattr(scanner_module, "datetime", _FakeDateTime)
    monkeypatch.setattr(scanner_module, "load_settings", lambda: settings)

    scanner._run_single_scan()
    assert len(gate.requests) == 0
    assert scanner._cooldown_until is None
    assert scanner._rejected_debounce_until == first.timestamp() + settings.rejected_candidate_debounce_seconds

    scanner._run_single_scan()
    assert len(gate.requests) == 1
