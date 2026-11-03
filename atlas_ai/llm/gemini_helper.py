from __future__ import annotations

import json
import time
from typing import Any


class GeminiAdvisor:
    def __init__(
        self,
        gemini_api_key: str = "",
        openai_api_key: str = "",
        openai_model: str = "gpt-4.1-mini",
        api_key: str = "",
    ) -> None:
        # Backward compatibility: older tests/callers passed api_key for Gemini only.
        if api_key and not gemini_api_key:
            gemini_api_key = api_key
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model or "gpt-4.1-mini"

        self._gemini_client = None
        self._gemini_model = "gemini-2.0-flash"
        self._openai_client = None
        self._providers: list[str] = []
        self._client = None  # legacy compatibility alias used by older tests/callers

        self._cooldown_until: dict[str, float] = {"gemini": 0.0, "openai": 0.0}
        self._last_error: dict[str, str] = {"gemini": "", "openai": ""}
        self._last_error_kind: dict[str, str] = {"gemini": "", "openai": ""}

        if gemini_api_key:
            try:
                import google.genai as genai

                self._gemini_client = genai.Client(api_key=gemini_api_key)
                self._providers.append("gemini")
            except Exception:
                self._gemini_client = None
                try:
                    import google.generativeai as legacy_genai

                    legacy_genai.configure(api_key=gemini_api_key)
                    self._gemini_client = legacy_genai.GenerativeModel("gemini-1.5-flash")
                    self._providers.append("gemini")
                except Exception:
                    self._gemini_client = None

        if openai_api_key:
            try:
                from openai import OpenAI

                self._openai_client = OpenAI(api_key=openai_api_key)
                self._providers.append("openai")
            except Exception:
                self._openai_client = None

        self.enabled = bool(self._providers)
        if self._gemini_client is not None:
            self._client = self._gemini_client

    def _in_cooldown(self, provider: str) -> bool:
        return time.time() < self._cooldown_until.get(provider, 0.0)

    def _enter_cooldown(self, provider: str, exc: Exception) -> None:
        msg = str(exc)
        self._last_error[provider] = msg
        lower = msg.lower()
        # Quota/rate-limit style failures: back off to avoid repeated noisy failures.
        if "insufficient_quota" in lower or "quota" in lower:
            self._last_error_kind[provider] = "quota_exhausted"
            self._cooldown_until[provider] = time.time() + 6 * 60 * 60
        elif "429" in lower or "resource_exhausted" in lower or "rate limit" in lower:
            self._last_error_kind[provider] = "rate_limited"
            self._cooldown_until[provider] = time.time() + 30 * 60
        else:
            self._last_error_kind[provider] = "api_error"
            self._cooldown_until[provider] = time.time() + 60

    def _provider_unavailable_reason(self) -> str:
        reasons = []
        for provider in self._providers:
            err = self._last_error.get(provider, "")
            kind = self._last_error_kind.get(provider, "")
            cooldown = max(0, int(self._cooldown_until.get(provider, 0.0) - time.time()))
            if err:
                label = f"{provider}: {kind or 'api_error'}"
                if cooldown:
                    label += f" cooldown={cooldown}s"
                reasons.append(label)
        return " | ".join(reasons) if reasons else "All providers in cooldown."

    def _summarize_via_gemini(self, prompt: str) -> str:
        if self._gemini_client is None:
            raise RuntimeError("Gemini client unavailable")
        if self._gemini_client.__class__.__module__.startswith("google.genai"):
            response = self._gemini_client.models.generate_content(model=self._gemini_model, contents=prompt)
            text = getattr(response, "text", "") or "No Gemini text response."
            return text.strip()
        response = self._gemini_client.generate_content(prompt)
        text = getattr(response, "text", "") or "No Gemini text response."
        return text.strip()

    def _summarize_via_openai(self, prompt: str) -> str:
        if self._openai_client is None:
            raise RuntimeError("OpenAI client unavailable")
        response = self._openai_client.chat.completions.create(
            model=self.openai_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a concise trading analyst assistant for XAUUSD."},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip() or "No OpenAI text response."

    def _influence_via_gemini(self, prompt: str) -> str:
        if self._gemini_client is None:
            raise RuntimeError("Gemini client unavailable")
        if self._gemini_client.__class__.__module__.startswith("google.genai"):
            response = self._gemini_client.models.generate_content(model=self._gemini_model, contents=prompt)
            return (getattr(response, "text", "") or "").strip()
        response = self._gemini_client.generate_content(prompt)
        return (getattr(response, "text", "") or "").strip()

    def _influence_via_openai(self, prompt: str) -> str:
        if self._openai_client is None:
            raise RuntimeError("OpenAI client unavailable")
        response = self._openai_client.chat.completions.create(
            model=self.openai_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def summarize_market_context(self, context: dict[str, Any]) -> str:
        if not self.enabled:
            return "LLM disabled. Running Atlas AI with local strategy intelligence only."

        prompt = (
            "You are a trading analyst assistant for XAUUSD. "
            "Given this context, provide a concise tactical note in under 120 words: "
            f"{context}"
        )

        for provider in self._providers:
            if self._in_cooldown(provider):
                continue
            try:
                if provider == "gemini":
                    return self._summarize_via_gemini(prompt)
                if provider == "openai":
                    return self._summarize_via_openai(prompt)
            except Exception as exc:
                self._enter_cooldown(provider, exc)
                continue

        reasons = self._provider_unavailable_reason()
        if reasons:
            return "LLM temporarily paused due to API limits/errors. Using local strategy intelligence only. Provider state: " + reasons
        return "LLM temporarily unavailable. Using local strategy intelligence only."

    def influence_trade_decision(
        self,
        context: dict[str, Any],
        max_confidence_boost: float = 0.08,
        max_confidence_cut: float = 0.35,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "applied": False,
                "confidence_delta": 0.0,
                "veto": False,
                "reason": "LLM disabled",
            }

        prompt = (
            "You are a strict risk-aware gold (XAUUSD) trading assistant. "
            "Return JSON only with keys: confidence_delta, veto, reason. "
            f"confidence_delta must be between {-abs(max_confidence_cut):.4f} and {abs(max_confidence_boost):.4f}. "
            "Use negative values to reduce confidence, positive to slightly increase, and veto=true only for high-risk setups. "
            f"Context: {context}"
        )

        # Legacy single-client compatibility path.
        if not self._providers and getattr(self, "_client", None) is not None:
            try:
                response = self._client.generate_content(prompt)
                raw = (getattr(response, "text", "") or "").strip()
                start = raw.find("{")
                end = raw.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise ValueError("No JSON object in legacy response")
                parsed = json.loads(raw[start : end + 1])
                delta = float(parsed.get("confidence_delta", 0.0))
                delta = max(-abs(max_confidence_cut), min(abs(max_confidence_boost), delta))
                veto = bool(parsed.get("veto", False))
                reason = str(parsed.get("reason", "No reason provided")).strip()
                return {
                    "enabled": True,
                    "applied": True,
                    "confidence_delta": delta,
                    "veto": veto,
                    "reason": reason,
                }
            except Exception as exc:
                return {
                    "enabled": True,
                    "applied": False,
                    "confidence_delta": 0.0,
                    "veto": False,
                    "reason": f"Legacy client failed: {exc}",
                }

        for provider in self._providers:
            if self._in_cooldown(provider):
                continue
            try:
                if provider == "gemini":
                    raw = self._influence_via_gemini(prompt)
                elif provider == "openai":
                    raw = self._influence_via_openai(prompt)
                else:
                    continue

                start = raw.find("{")
                end = raw.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise ValueError(f"No JSON object in {provider} response")

                parsed = json.loads(raw[start : end + 1])
                delta = float(parsed.get("confidence_delta", 0.0))
                delta = max(-abs(max_confidence_cut), min(abs(max_confidence_boost), delta))
                veto = bool(parsed.get("veto", False))
                reason = str(parsed.get("reason", "No reason provided")).strip()

                return {
                    "enabled": True,
                    "applied": True,
                    "confidence_delta": delta,
                    "veto": veto,
                    "reason": reason,
                }
            except Exception as exc:
                self._enter_cooldown(provider, exc)
                continue

        return {
            "enabled": True,
            "applied": False,
            "confidence_delta": 0.0,
            "veto": False,
            "reason": "LLM influence unavailable. " + self._provider_unavailable_reason(),
        }

    def analyze_trade_setup(
        self,
        *,
        symbol: str,
        timeframe: str,
        utc_time: str,
        session: str,
        current_price: float,
        recent_candles: list[dict[str, Any]],
        regime: str,
        trend_strength: float,
        volatility_state: str,
        spread_points: float,
        liquidity_state: str,
        higher_tf_bias: str,
        strategy_signals: list[dict[str, Any]],
        open_positions: list[dict[str, Any]],
        equity: float,
        daily_loss_pct: float,
        open_positions_count: int,
        consecutive_losses: int,
    ) -> dict[str, Any]:
        """Send structured market context to the LLM and return parsed trade analysis.

        This is a PROPOSAL layer only. The returned analysis is fed into the trader's
        existing signal pipeline as one input among many. Execution authority remains
        with the ExecutionGate and the trader's 20+ guard layers.
        """
        if not self.enabled:
            return {
                "enabled": False,
                "applied": False,
                "bias": "neutral",
                "confidence": 0,
                "setup": "invalid",
                "decision": "NO_TRADE",
                "reason": "LLM disabled",
                "risks": [],
                "required_conditions": [],
            }

        from pathlib import Path

        prompt_dir = Path(__file__).resolve().parent.parent / "prompts"
        system_prompt = (prompt_dir / "trade_analysis_system.txt").read_text(encoding="utf-8")
        user_template = (prompt_dir / "trade_analysis_user.txt").read_text(encoding="utf-8")

        def _fmt_candles(candles: list[dict[str, Any]]) -> str:
            lines: list[str] = []
            for i, c in enumerate(candles[-5:], start=1):
                lines.append(
                    f"  {i}. O={c.get('open')} H={c.get('high')} L={c.get('low')} C={c.get('close')}"
                )
            return "\n".join(lines) if lines else "  (no recent candle data)"

        def _fmt_signals(signals: list[dict[str, Any]]) -> str:
            if not signals:
                return "  (no active signals)"
            lines: list[str] = []
            for s in signals[:6]:
                lines.append(
                    f"  - {s.get('strategy', 'unknown')}: {s.get('side', 'flat')} "
                    f"(conf={s.get('confidence', 0.0):.3f})"
                )
            return "\n".join(lines)

        def _fmt_positions(positions: list[dict[str, Any]]) -> str:
            if not positions:
                return "  (none)"
            lines: list[str] = []
            for p in positions[:3]:
                lines.append(
                    f"  - {p.get('side', 'flat')} {p.get('volume', 0.0)} lots @ {p.get('entry_price', 0.0)} "
                    f"(PnL={p.get('profit', 0.0):+.2f})"
                )
            return "\n".join(lines)

        user_prompt = user_template.format(
            symbol=symbol,
            timeframe=timeframe,
            utc_time=utc_time,
            session=session,
            current_price=current_price,
            recent_candles=_fmt_candles(recent_candles),
            regime=regime,
            trend_strength=round(float(trend_strength), 4),
            volatility_state=volatility_state,
            spread_points=round(float(spread_points), 2),
            liquidity_state=liquidity_state,
            higher_tf_bias=higher_tf_bias,
            strategy_signals=_fmt_signals(strategy_signals),
            open_positions=_fmt_positions(open_positions),
            equity=round(float(equity), 2),
            daily_loss_pct=round(float(daily_loss_pct), 4),
            open_positions_count=int(open_positions_count),
            consecutive_losses=int(consecutive_losses),
        )

        for provider in self._providers:
            if self._in_cooldown(provider):
                continue
            try:
                if provider == "openai":
                    raw = self._analyze_via_openai(system_prompt, user_prompt)
                elif provider == "gemini":
                    raw = self._analyze_via_gemini(system_prompt, user_prompt)
                else:
                    continue

                parsed = self._parse_analysis_response(raw)
                if parsed:
                    parsed["enabled"] = True
                    parsed["applied"] = True
                    parsed["provider"] = provider
                    return parsed
            except Exception as exc:
                self._enter_cooldown(provider, exc)
                continue

        return {
            "enabled": True,
            "applied": False,
            "bias": "neutral",
            "confidence": 0,
            "setup": "invalid",
            "decision": "NO_TRADE",
            "reason": "LLM analysis unavailable. " + self._provider_unavailable_reason(),
            "risks": [],
            "required_conditions": [],
        }

    def _analyze_via_openai(self, system_prompt: str, user_prompt: str) -> str:
        if self._openai_client is None:
            raise RuntimeError("OpenAI client unavailable")
        response = self._openai_client.chat.completions.create(
            model=self.openai_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def _analyze_via_gemini(self, system_prompt: str, user_prompt: str) -> str:
        if self._gemini_client is None:
            raise RuntimeError("Gemini client unavailable")
        combined = f"{system_prompt}\n\n{user_prompt}"
        if self._gemini_client.__class__.__module__.startswith("google.genai"):
            response = self._gemini_client.models.generate_content(
                model=self._gemini_model, contents=combined
            )
            return (getattr(response, "text", "") or "").strip()
        response = self._gemini_client.generate_content(combined)
        return (getattr(response, "text", "") or "").strip()

    @staticmethod
    def _parse_analysis_response(raw: str) -> dict[str, Any] | None:
        """Parse the structured analysis output from the LLM.

        Expects the mandatory 7-section format. Falls back to regex extraction
        if the response is not perfectly formatted.
        """
        text = str(raw or "").lower()
        if not text:
            return None

        import re

        def _extract_line(pattern: str, default: str = "") -> str:
            m = re.search(pattern, raw, re.IGNORECASE)
            return (m.group(1).strip() if m else default).lower()

        def _extract_int(pattern: str, default: int = 0) -> int:
            m = re.search(pattern, raw, re.IGNORECASE)
            if not m:
                return default
            try:
                return int(m.group(1))
            except Exception:
                return default

        bias = _extract_line(r"bias:\s*(bullish_strong|bullish_weak|neutral|bearish_weak|bearish_strong)", "neutral")
        confidence = _extract_int(r"confidence:\s*(\d+)", 0)
        confidence = max(0, min(100, confidence))
        setup = _extract_line(r"setup:\s*(valid|invalid)", "invalid")
        decision = _extract_line(r"decision:\s*(buy|sell|no_trade)", "no_trade")
        reason = _extract_line(r"reason:\s*(.+?)(?:\n\d+\.\s|\nrisks:|\Z)", "")

        risks: list[str] = []
        risks_match = re.search(r"risks:\s*(.+?)(?:\n\d+\.\srequired|\Z)", raw, re.IGNORECASE | re.DOTALL)
        if risks_match:
            risks_block = risks_match.group(1)
            risks = [r.strip("- ") for r in risks_block.split("\n") if r.strip()][:3]

        required_conditions: list[str] = []
        req_match = re.search(r"required conditions:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        if req_match:
            req_block = req_match.group(1)
            required_conditions = [r.strip("- ") for r in req_block.split("\n") if r.strip()][:3]

        return {
            "bias": bias,
            "confidence": confidence,
            "setup": setup,
            "decision": decision.upper() if decision else "NO_TRADE",
            "reason": reason or "No reason provided by LLM.",
            "risks": risks,
            "required_conditions": required_conditions,
        }

    def health_summary(self) -> dict[str, Any]:
        now = time.time()
        providers: list[dict[str, Any]] = []
        for provider in self._providers:
            cooldown_until = float(self._cooldown_until.get(provider, 0.0))
            providers.append(
                {
                    "name": provider,
                    "in_cooldown": cooldown_until > now,
                    "cooldown_seconds_left": max(0, int(cooldown_until - now)),
                    "last_error": self._last_error.get(provider, ""),
                    "last_error_kind": self._last_error_kind.get(provider, ""),
                }
            )
        return {
            "enabled": self.enabled,
            "providers": providers,
        }

    def health(self) -> dict[str, Any]:
        return self.health_summary()
