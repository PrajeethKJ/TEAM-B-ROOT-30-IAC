"""
BioPrint: Automated Bot, Script and Replay Detector  (patched)

The shipped detector only caught bots that declared themselves: `isTrusted`
and `navigator.webdriver` are both reported by the client, so any script that
lies about them walks straight through, and in practice a Playwright bot with
randomised delays came back as BLOCKED_IMPOSTOR rather than BLOCKED_BOT.

Those two client-declared flags are kept - they are free and they catch lazy
attacks - but they are now demoted to *corroborating* signals. The signals that
carry weight are ones the attacker has to actually defeat:

  * timing distribution shape (CV, skew, kurtosis of dwell and flight)
  * timestamp quantisation (scripted delays land on a grid; humans do not)
  * dwell/flight independence (humans correlate them; sleep loops do not)
  * negative or impossible dwell, missing keyups
  * mouse velocity profile (humans are right-skewed and bursty; interpolation
    is flat), sampling-interval regularity, integer-pixel paths
  * replay: server-issued one-time nonce plus near-duplicate timing detection

ReplayGuard is a separate class so app.py can hold one instance with the same
lifetime as the profile store.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

BOT_THRESHOLD = 0.65


class BotDetector:
    def __init__(self, threshold: float = BOT_THRESHOLD):
        self.threshold = float(threshold)

    def evaluate_telemetry(
        self,
        raw_keystrokes: List[Dict[str, Any]],
        raw_mouse: List[Dict[str, Any]],
        keystroke_features: Dict[str, Any],
        mouse_features: Dict[str, Any],
        client_metadata: Dict[str, Any],
        replay_result: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> Tuple[bool, float, List[str]]:
        """Returns (is_bot, probability in [0,1], human-readable reasons).

        Signals are split into two buckets.

        HARD signals are things a human physically cannot produce: sub-millisecond
        key holds, a dwell coefficient of variation near zero, 25 chars/second, a
        pixel-perfect straight cursor, a spent nonce. These sum without a cap.

        SOFT signals are distribution-shape heuristics: they are informative but
        they are estimated from very few samples on a short password, so they are
        capped below the blocking threshold and can never block on their own.
        They exist to push a partly-suspicious session over the line, and to show
        up in the dashboard. Pass strict=True to let them block by themselves -
        only do that once you have measured the false-positive rate on your own
        teammates' sessions with `evaluate.py --bot-strict`.
        """
        reasons: List[str] = []
        hard = 0.0
        soft = 0.0
        raw_keystrokes = raw_keystrokes or []
        raw_mouse = raw_mouse or []
        client_metadata = client_metadata or {}

        def H(weight: float, reason: str) -> None:
            nonlocal hard
            hard += weight
            reasons.append(reason)

        def S(weight: float, reason: str) -> None:
            nonlocal soft
            soft += weight
            reasons.append("[weak] " + reason)

        # ---- replay (decided server-side, passed in) ----------------------
        if replay_result and replay_result.get("is_replay"):
            H(0.95, f"Replayed Session: {replay_result.get('reason', 'previously seen telemetry')}")

        # ---- self-declared automation (cheap, easily spoofed) -------------
        if any(ev.get("isTrusted") is False for ev in raw_keystrokes + raw_mouse):
            H(0.90, "Synthetic DOM Event: event.isTrusted == false (script-dispatched)")
        if client_metadata.get("webdriver") is True:
            H(0.85, "Automated Browser: navigator.webdriver is set (Selenium/Puppeteer)")

        n_keys = int(keystroke_features.get("keystroke_count", 0) or 0)
        dwells = [d for d in (keystroke_features.get("dwell_sequence") or []) if d is not None]
        flights = [f for f in (keystroke_features.get("flight_dd_sequence") or []) if f is not None]

        # Fallback for callers that only pass aggregate features.
        agg_cv: Dict[str, float] = {}
        if n_keys >= 5:
            for name, mk, sk in (("dwell", "mean_dwell", "std_dwell"),
                                 ("flight", "mean_flight_ud", "std_flight_ud")):
                mu = abs(float(keystroke_features.get(mk, 0.0) or 0.0))
                sd = float(keystroke_features.get(sk, 0.0) or 0.0)
                if mu > 1e-6:
                    agg_cv[name] = sd / mu

        # ---- structural impossibilities -----------------------------------
        if n_keys >= 4:
            unmatched = int(keystroke_features.get("unmatched_downs", 0) or 0)
            if unmatched / max(1, n_keys) > 0.5:
                H(0.70, "Malformed Event Stream: over half the keydowns have no keyup")
            if len(dwells) >= 3 and any(d <= 0.5 for d in dwells):
                H(0.75, "Impossible Key Dwell: sub-millisecond hold times")

        cps = float(keystroke_features.get("cps", 0.0) or 0.0)
        if cps > 22.0:
            H(0.85, f"Inhuman Typing Speed: {round(cps, 1)} chars/second")

        # ---- timing dispersion --------------------------------------------
        # CV is stable at small n; shape statistics are not.
        for name, vals in (("dwell", dwells), ("flight", flights)):
            cv = _shape(vals)["cv"] if len(vals) >= 8 else agg_cv.get(name)
            if cv is None:
                continue
            label = "hold time" if name == "dwell" else "inter-key interval"
            if cv < 0.05:
                H(0.80, f"Synthetic Timing Precision: {name} CV {cv:.3f} (constant {label})")
            elif cv < 0.10:
                S(0.35, f"Near-Constant {name.title()}: CV {cv:.3f}, below human motor variance")

        # Shape statistics need far more samples than a 9-character password
        # provides. They only become trustworthy with the free-text sentence.
        for name, vals in (("dwell", dwells), ("flight", flights)):
            if len(vals) < 14:
                continue
            sh = _shape(vals)
            if sh["cv"] > 0.12 and abs(sh["skew"]) < 0.15 and sh["kurtosis"] < -1.0:
                S(0.30, f"Uniform Random {name.title()} Delays: skew {sh['skew']:.2f}, excess "
                        f"kurtosis {sh['kurtosis']:.2f} (human timing is right-skewed)")

        if len(dwells) >= 20 and len(flights) >= 20:
            n = min(len(dwells), len(flights))
            r = _pearson(dwells[:n], flights[:n])
            if r is not None and abs(r) < 0.03:
                S(0.25, f"Independent Timing Channels: dwell/flight correlation {r:.3f} "
                        f"(human keystroke timing co-varies)")

        # ---- timestamp quantisation ----------------------------------------
        # Soft by default: some browsers clamp performance.now() for privacy
        # (Firefox resistFingerprinting uses a 1 ms clock), which looks identical
        # to a script sleeping in whole milliseconds. Validate before hardening.
        q = _quantisation(raw_keystrokes)
        if q is not None and q["n"] >= 10 and q["grid_ratio"] > 0.90 and q["grid_ms"] >= 1.0:
            S(0.35, f"Quantised Timestamps: {round(q['grid_ratio'] * 100)}% of intervals on a "
                    f"{q['grid_ms']:g} ms grid (programmatic delay or a clamped clock)")

        # ---- mouse kinematics ------------------------------------------------
        n_mouse = int(mouse_features.get("mouse_event_count", 0) or 0)
        if mouse_features.get("move_present") and n_mouse >= 20:
            curvature = float(mouse_features.get("curvature_ratio", 1.1) or 1.1)
            jitter = float(mouse_features.get("jitter_variance", 0.1) or 0.0)
            skew = float(mouse_features.get("velocity_skew", 1.0) or 0.0)
            iv_std = float(mouse_features.get("sample_interval_std", 10.0) or 0.0)
            int_ratio = float(mouse_features.get("integer_coord_ratio", 0.0) or 0.0)

            if curvature <= 1.004 and jitter < 0.0008:
                H(0.75, f"Linear Algorithmic Cursor: curvature {curvature} with near-zero "
                        f"turn variance")
            if n_mouse >= 30 and abs(skew) < 0.08:
                S(0.30, f"Flat Velocity Profile: skew {skew} (no ballistic-then-corrective phase)")
            if n_mouse >= 30 and iv_std < 0.6:
                S(0.30, f"Metronomic Pointer Sampling: interval std {iv_std} ms")
            if n_mouse >= 30 and int_ratio > 0.98:
                S(0.20, "Integer-Only Cursor Path: every sample lands on a whole pixel")

        # Keyboard-only login is a legitimate human pattern, so this is weak.
        if n_mouse < 2 and n_keys > 0:
            S(0.15, "No Pointer Activity: form submitted without any cursor movement")

        soft_cap = 1.0 if strict else (self.threshold - 0.05)
        prob = min(1.0, round(hard + min(soft, soft_cap), 3))
        return prob >= self.threshold, prob, reasons


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #
class ReplayGuard:
    """Server-issued one-time nonce plus near-duplicate timing detection.

    A captured genuine payload replayed verbatim was accepted 5/5 times, because
    nothing in the request was bound to a single login. Two independent defences:

      1. /api/login-challenge issues a nonce. The client echoes it. It is valid
         once, for `ttl_s` seconds. A replayed payload carries a spent nonce.
      2. Even with a fresh nonce, an attacker can splice new-nonce + old timings.
         So the inter-key timing vector of every attempt is fingerprinted and
         compared against recent attempts for that user; human timing never
         repeats to within a few milliseconds.
    """

    def __init__(self, ttl_s: float = 120.0, history: int = 40,
                 duplicate_tolerance: float = 0.012):
        self.ttl_s = float(ttl_s)
        self.history = int(history)
        self.duplicate_tolerance = float(duplicate_tolerance)
        self._issued: Dict[str, float] = {}
        self._spent: Dict[str, float] = {}
        self._recent: Dict[str, Deque[Tuple[float, np.ndarray]]] = {}

    # -- nonce ---------------------------------------------------------- #
    def issue(self) -> Dict[str, Any]:
        self._gc()
        nonce = secrets.token_urlsafe(24)
        self._issued[nonce] = time.time()
        return {"nonce": nonce, "expires_in": self.ttl_s}

    def consume(self, nonce: Optional[str]) -> Tuple[bool, str]:
        self._gc()
        if not nonce:
            return False, "no login nonce supplied"
        if nonce in self._spent:
            return False, "login nonce already used (replayed request)"
        issued = self._issued.pop(nonce, None)
        if issued is None:
            return False, "unknown or expired login nonce"
        if time.time() - issued > self.ttl_s:
            return False, "login nonce expired"
        self._spent[nonce] = time.time()
        return True, "ok"

    # -- timing fingerprint ---------------------------------------------- #
    def check(self, username: str, keystrokes: List[Dict[str, Any]],
              nonce: Optional[str] = None, enforce_nonce: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {"is_replay": False, "reason": "", "similarity": 0.0}

        if enforce_nonce:
            ok, why = self.consume(nonce)
            if not ok:
                result.update(is_replay=True, reason=why)
                return result

        fp = _timing_fingerprint(keystrokes)
        if fp is None:
            return result

        hist = self._recent.setdefault(username, deque(maxlen=self.history))
        now = time.time()
        for _, prev in hist:
            if prev.shape != fp.shape:
                continue
            # Mean relative deviation across the inter-key interval vector.
            denom = np.maximum(np.abs(prev), 1.0)
            sim = float(np.mean(np.abs(fp - prev) / denom))
            if sim < self.duplicate_tolerance:
                result.update(
                    is_replay=True,
                    reason=f"timing vector within {sim * 100:.2f}% of a previous attempt",
                    similarity=round(1.0 - sim, 4),
                )
                hist.append((now, fp))
                return result
        hist.append((now, fp))
        return result

    def digest(self, keystrokes: List[Dict[str, Any]]) -> str:
        fp = _timing_fingerprint(keystrokes)
        if fp is None:
            return ""
        return hashlib.sha256(np.round(fp, 1).tobytes()).hexdigest()[:16]

    def _gc(self) -> None:
        cutoff = time.time() - max(self.ttl_s * 4, 600.0)
        for store in (self._issued, self._spent):
            for k in [k for k, v in store.items() if v < cutoff]:
                store.pop(k, None)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _shape(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    mu = float(a.mean())
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    cv = sd / mu if mu > 1e-9 else 0.0
    if sd > 1e-9:
        z = (a - mu) / sd
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean() - 3.0)
    else:
        skew, kurt = 0.0, 0.0
    return {"mean": mu, "std": sd, "cv": cv, "skew": skew, "kurtosis": kurt}


def _pearson(a: List[float], b: List[float]) -> Optional[float]:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _quantisation(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Do inter-event intervals sit on a regular grid?

    `page.type(delay=100)` and `time.sleep(x)` produce intervals that are a base
    delay plus small event-loop noise, so they cluster on a millisecond grid.
    Human keystrokes produce a continuous spread.
    """
    ts = []
    for ev in events or []:
        try:
            ts.append(float(ev.get("timestamp")))
        except (TypeError, ValueError):
            continue
    if len(ts) < 9:
        return None
    ts.sort()
    deltas = np.diff(np.asarray(ts))
    deltas = deltas[deltas > 0.5]
    if deltas.size < 8:
        return None

    best = {"grid_ms": 0.0, "grid_ratio": 0.0, "n": int(deltas.size)}
    for grid in (1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0):
        rem = np.abs((deltas / grid) - np.round(deltas / grid)) * grid
        ratio = float(np.mean(rem < 0.15))
        if ratio > best["grid_ratio"]:
            best = {"grid_ms": grid, "grid_ratio": round(ratio, 3), "n": int(deltas.size)}
    return best


def _timing_fingerprint(keystrokes: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Inter-event interval vector - invariant to the session's start time."""
    ts = []
    for ev in keystrokes or []:
        try:
            ts.append(float(ev.get("timestamp")))
        except (TypeError, ValueError):
            continue
    if len(ts) < 6:
        return None
    ts.sort()
    return np.diff(np.asarray(ts, dtype=np.float64))
