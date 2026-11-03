from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from atlas_ai import main


def test_housekeeping_retries_after_failure_and_records_duration(monkeypatch):
    targets = []
    monkeypatch.setattr(main.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: targets.append(kwargs["target"])))
    sleeps = 0
    def sleep(interval):
        nonlocal sleeps
        assert interval == 15 * 60
        sleeps += 1
        if sleeps == 3:
            raise StopIteration("stop test loop")
    monkeypatch.setattr(main.time, "sleep", sleep)
    store = SimpleNamespace(maintain_events=Mock(side_effect=[PermissionError("busy"), {"kept": 10, "archived": 0}]), log_event=Mock())
    main.start_periodic_housekeeping_background(SimpleNamespace(event_max_lines=10), store)
    with pytest.raises(StopIteration):
        targets[0]()
    assert store.maintain_events.call_count == 2
    failed, succeeded = store.log_event.call_args_list
    assert failed.args[0] == "error"
    assert failed.args[2]["error"] == "busy"
    assert succeeded.args[1] == "Event log maintenance completed."
    assert succeeded.args[2]["kept"] == 10
    assert succeeded.args[2]["duration_ms"] >= 0
