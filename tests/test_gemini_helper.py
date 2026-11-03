from atlas_ai.llm.gemini_helper import GeminiAdvisor


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeLegacyClient:
    def __init__(self, text: str):
        self._text = text

    def generate_content(self, prompt: str):
        return _FakeResponse(self._text)


def test_influence_returns_disabled_when_no_key():
    advisor = GeminiAdvisor(api_key="")
    result = advisor.influence_trade_decision({"chosen": {"side": "buy"}})
    assert result["enabled"] is False
    assert result["applied"] is False
    assert result["veto"] is False


def test_influence_clamps_delta_from_model():
    advisor = GeminiAdvisor(api_key="")
    advisor.enabled = True
    advisor._client = _FakeLegacyClient('{"confidence_delta": 9, "veto": false, "reason": "ok"}')

    result = advisor.influence_trade_decision(
        {"chosen": {"side": "buy"}},
        max_confidence_boost=0.08,
        max_confidence_cut=0.35,
    )
    assert result["enabled"] is True
    assert result["applied"] is True
    assert result["confidence_delta"] == 0.08


def test_quota_error_enters_long_cooldown_and_sanitizes_reason():
    advisor = GeminiAdvisor(api_key="")
    advisor.enabled = True
    advisor._providers = ["openai"]

    advisor._enter_cooldown("openai", RuntimeError("Error code: 429 - insufficient_quota noisy raw payload"))

    health = advisor.health()
    provider = health["providers"][0]
    assert provider["in_cooldown"] is True
    assert provider["cooldown_seconds_left"] > 300
    assert provider["last_error_kind"] == "quota_exhausted"

    result = advisor.influence_trade_decision({"chosen": {"side": "buy"}})
    assert result["applied"] is False
    assert "quota_exhausted" in result["reason"]
    assert "noisy raw payload" not in result["reason"]
