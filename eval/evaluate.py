#!/usr/bin/env python3
"""
BioPrint evaluation harness.

Runs the real pipeline - the same FeatureExtractor, BiometricModel and
BotDetector the server uses - over captured sessions and reports FRR, FAR and
EER per attack type and per enrolment size. Nothing here is simulated unless
you ask for it with --synthetic.

Usage
-----
  python eval/evaluate.py --data data/sessions --k 3 5 8 --out results/
  python eval/evaluate.py --data data/sessions.jsonl --k 8 --per-user-threshold
  python eval/evaluate.py --self-test            # deterministic fixture checks
  python eval/evaluate.py --synthetic 40         # pipeline smoke test, no real data

Data format
-----------
A directory of .json files or one .jsonl file. One object per captured session:

  {
    "user":   "alice",              # who actually produced the session
    "target": "alice",              # whose account it is aimed at
    "label":  "genuine",            # genuine | mimic | impostor | bot | replay
    "keystrokes": [ {"type":"keydown","key":"a","code":"KeyA",
                     "timestamp":1234.5,"isTrusted":true}, ... ],
    "mouse":      [ {"type":"mousemove","x":10,"y":20,"timestamp":1234.5}, ... ],
    "client_metadata": {}
  }

`target` defaults to `user`. Label "genuine" with target == user is what gets
split into enrolment and test. Everything else is an attack against `target`.

To record sessions, log the exact payload your browser posts to
/api/authenticate - it already contains keystrokes, mouse and client_metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.biometrics.feature_extractor import FeatureExtractor  # noqa: E402
from backend.biometrics.model import BiometricModel                # noqa: E402
from backend.biometrics.bot_detector import BotDetector            # noqa: E402

ATTACK_LABELS = ("mimic", "impostor", "bot", "replay")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_sessions(path: str) -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if not name.endswith((".json", ".jsonl")):
                continue
            sessions += load_sessions(os.path.join(path, name))
        return sessions

    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            for line in fh:
                line = line.strip()
                if line:
                    sessions.append(json.loads(line))
        else:
            blob = json.load(fh)
            sessions += blob if isinstance(blob, list) else [blob]

    for s in sessions:
        s.setdefault("label", "genuine")
        s.setdefault("target", s.get("user"))
        s.setdefault("keystrokes", [])
        s.setdefault("mouse", [])
        s.setdefault("client_metadata", {})
    return sessions


# --------------------------------------------------------------------------- #
# feature stage
# --------------------------------------------------------------------------- #
def featurise(sessions: List[Dict[str, Any]], extractor: FeatureExtractor) -> List[Dict[str, Any]]:
    out = []
    for s in sessions:
        f = extractor.extract_all(s["keystrokes"], s["mouse"])
        f.update(user=s["user"], target=s["target"], label=s["label"],
                 raw=s, keystroke_count=f["keystroke"]["keystroke_count"])
        out.append(f)
    return out


# --------------------------------------------------------------------------- #
# protocol
# --------------------------------------------------------------------------- #
def run_protocol(feats: List[Dict[str, Any]], k: int, seed: int = 0,
                 cross_impostors: bool = True,
                 per_user_threshold: bool = False,
                 target_frr: float = 0.05,
                 bot_gate: bool = True,
                 bot_strict: bool = False) -> Dict[str, Any]:
    """Enrol each user on k genuine sessions, score everything else against them."""
    rng = random.Random(seed)
    model = BiometricModel()
    detector = BotDetector()

    by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in feats:
        if f["label"] == "genuine" and f["user"] == f["target"]:
            by_user[f["user"]].append(f)

    profiles: Dict[str, Dict[str, Any]] = {}
    genuine_test: Dict[str, List[Dict[str, Any]]] = {}
    skipped: List[str] = []

    for user, sessions in by_user.items():
        if len(sessions) < k + 1:
            skipped.append(f"{user} ({len(sessions)} sessions, need {k + 1})")
            continue
        order = sessions[:]
        rng.shuffle(order)
        enrol, test = order[:k], order[k:]
        profiles[user] = model.fit_profile(
            feature_vectors=[e["vector"] for e in enrol],
            digraph_samples=[e["digraphs"] for e in enrol],
            masks=[e["mask"] for e in enrol],
            sequences=[e["sequences"] for e in enrol],
        )
        genuine_test[user] = test

    rows: List[Dict[str, Any]] = []

    def score(profile_user: str, attempt: Dict[str, Any], label: str) -> None:
        profile = profiles[profile_user]
        is_bot, bot_prob, reasons = detector.evaluate_telemetry(
            attempt["raw"]["keystrokes"], attempt["raw"]["mouse"],
            attempt["keystroke"], attempt["mouse"],
            attempt["raw"].get("client_metadata", {}), strict=bot_strict,
        )
        _, conf, metrics = model.evaluate_attempt(
            attempt_vector=attempt["vector"],
            attempt_digraphs=attempt["digraphs"],
            profile_data=profile,
            mask=attempt["mask"],
            sequences=attempt["sequences"],
        )
        effective = 0.0 if (bot_gate and is_bot) else conf
        rows.append({
            "profile_user": profile_user,
            "attempt_user": attempt["user"],
            "label": label,
            "confidence": round(effective, 2),
            "raw_confidence": round(conf, 2),
            "bot_probability": bot_prob,
            "bot_flagged": int(is_bot),
            "bot_reasons": " | ".join(reasons),
            "mahalanobis_norm": metrics["mahalanobis_normalised"],
            "position_divergence": metrics["position_divergence"],
            "digraph_divergence": metrics["digraph_divergence"],
            "dims_scored": metrics["dimensions_scored"],
            "keystrokes": attempt["keystroke_count"],
        })

    for user in profiles:
        for a in genuine_test[user]:
            score(user, a, "genuine")

    for f in feats:
        if f["label"] in ATTACK_LABELS and f["target"] in profiles:
            score(f["target"], f, f["label"])

    if cross_impostors:
        for user in profiles:
            for other, tests in genuine_test.items():
                if other == user:
                    continue
                for a in tests:
                    score(user, a, "impostor")

    thresholds = None
    if per_user_threshold:
        thresholds = {u: user_threshold(profiles[u], target_frr) for u in profiles}
        for r in rows:
            r["threshold"] = thresholds[r["profile_user"]]

    return {
        "rows": rows,
        "profiles": profiles,
        "k": k,
        "n_users": len(profiles),
        "skipped": skipped,
        "per_user_thresholds": thresholds,
    }


def user_threshold(profile: Dict[str, Any], target_frr: float) -> float:
    """Pick this user's operating point from their own leave-one-out distances.

    With only a handful of LOO points this is a coarse estimate; it is still
    better than one global constant, because it scales with how repeatable the
    individual user is.
    """
    cal = profile.get("calibration", {})
    loo = cal.get("loo_maha") or []
    ref = cal.get("d_ref_maha") or 1.0
    if len(loo) < 2:
        return 60.0
    q = float(np.quantile(loo, 1.0 - target_frr))
    ratio = q / max(ref, 1e-6)
    # Mirror the model's fusion sigmoid so the number is on the same scale.
    anomaly = 1.0 / (1.0 + math.exp(-2.20 * (ratio - 1.75)))
    return round(max(25.0, min(85.0, (1.0 - anomaly) * 100.0 - 6.0)), 1)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rates(rows: List[Dict[str, Any]], threshold: float,
          attack_labels: Optional[Tuple[str, ...]] = None) -> Dict[str, float]:
    gen = [r for r in rows if r["label"] == "genuine"]
    att = [r for r in rows if r["label"] != "genuine"
           and (attack_labels is None or r["label"] in attack_labels)]

    def thr(r):
        return r.get("threshold", threshold)

    frr = sum(1 for r in gen if r["confidence"] < thr(r)) / len(gen) if gen else float("nan")
    far = sum(1 for r in att if r["confidence"] >= thr(r)) / len(att) if att else float("nan")
    return {"frr": frr, "far": far, "n_genuine": len(gen), "n_attack": len(att)}


def eer(rows: List[Dict[str, Any]],
        attack_labels: Optional[Tuple[str, ...]] = None) -> Tuple[float, float]:
    gen = [r["confidence"] for r in rows if r["label"] == "genuine"]
    att = [r["confidence"] for r in rows if r["label"] != "genuine"
           and (attack_labels is None or r["label"] in attack_labels)]
    if not gen or not att:
        return float("nan"), float("nan")
    grid = sorted(set(gen + att))
    best, best_t, best_gap = 1.0, 60.0, 9e9
    for t in grid:
        frr = sum(1 for g in gen if g < t) / len(gen)
        far = sum(1 for a in att if a >= t) / len(att)
        gap = abs(frr - far)
        if gap < best_gap:
            best_gap, best, best_t = gap, (frr + far) / 2.0, t
    return best, best_t


def auc(rows: List[Dict[str, Any]],
        attack_labels: Optional[Tuple[str, ...]] = None) -> float:
    gen = np.array([r["confidence"] for r in rows if r["label"] == "genuine"])
    att = np.array([r["confidence"] for r in rows if r["label"] != "genuine"
                    and (attack_labels is None or r["label"] in attack_labels)])
    if gen.size == 0 or att.size == 0:
        return float("nan")
    # Mann-Whitney U, ties counted as half.
    wins = (gen[:, None] > att[None, :]).sum() + 0.5 * (gen[:, None] == att[None, :]).sum()
    return float(wins / (gen.size * att.size))


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def report(results: List[Dict[str, Any]], threshold: float) -> str:
    lines: List[str] = ["# BioPrint evaluation", ""]
    for res in results:
        rows = res["rows"]
        lines.append(f"## K = {res['k']} enrolment samples "
                     f"({res['n_users']} users, {len(rows)} scored attempts)")
        if res["skipped"]:
            lines.append(f"*Skipped for too few sessions:* {', '.join(res['skipped'])}")
        lines.append("")
        overall = rates(rows, threshold)
        e, e_t = eer(rows)
        lines.append(f"- FRR @ {threshold:g}: **{_pct(overall['frr'])}** "
                     f"({overall['n_genuine']} genuine attempts)")
        lines.append(f"- FAR @ {threshold:g}: **{_pct(overall['far'])}** "
                     f"({overall['n_attack']} attack attempts)")
        lines.append(f"- EER: **{_pct(e)}** at confidence {e_t:g}")
        lines.append(f"- AUC: {auc(rows):.3f}")
        if res["per_user_thresholds"]:
            lines.append("- Per-user thresholds in use (from leave-one-out enrolment scores)")
        lines.append("")
        lines.append("| attack type | n | FAR | EER vs genuine |")
        lines.append("|---|---|---|---|")
        for label in ATTACK_LABELS:
            sub = [r for r in rows if r["label"] in ("genuine", label)]
            n = sum(1 for r in rows if r["label"] == label)
            if n == 0:
                continue
            far = rates(sub, threshold, (label,))["far"]
            le, _ = eer(sub, (label,))
            lines.append(f"| {label} | {n} | {_pct(far)} | {_pct(le)} |")
        lines.append("")

        flagged = [r for r in rows if r["bot_flagged"]]
        if flagged:
            bot_rows = [r for r in rows if r["label"] == "bot"]
            fp = [r for r in flagged if r["label"] == "genuine"]
            caught = sum(1 for r in bot_rows if r["bot_flagged"])
            lines.append(f"- Bot detector: {caught}/{len(bot_rows)} bot sessions flagged, "
                         f"{len(fp)} genuine sessions falsely flagged")
            lines.append("")
    return "\n".join(lines)


def _pct(v: float) -> str:
    return "n/a" if v != v else f"{v * 100:.1f}%"


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# self-test: deterministic fixtures, no randomness
# --------------------------------------------------------------------------- #
def self_test() -> int:
    ex = FeatureExtractor(hash_digraphs=False)
    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else ': ' + detail}")
        if not cond:
            failures.append(name)

    print("Feature extraction")

    # 1. Flight time. Four keys, 80 ms dwell, 8.3 ms mean flight.
    ev, t = [], 1000.0
    flights = [10.0, 5.0, 10.0]
    for i, code in enumerate(["KeyA", "KeyB", "KeyC", "KeyD"]):
        ev.append({"type": "keydown", "key": code[-1].lower(), "code": code, "timestamp": t})
        ev.append({"type": "keyup", "key": code[-1].lower(), "code": code, "timestamp": t + 80.0})
        if i < 3:
            t = t + 80.0 + flights[i]
    f = ex.extract_keystroke_features(ev)
    check("mean flight UD is the true 8.3 ms", abs(f["mean_flight_ud"] - 8.3) < 0.2,
          f"got {f['mean_flight_ud']}")
    check("mean dwell is 80 ms", abs(f["mean_dwell"] - 80.0) < 0.1, f"got {f['mean_dwell']}")

    # 1b. Regression fixture for the real-world case: a Shift-modified first key
    #     released early, plus key overlap (rollover). The shipped extractor
    #     returns dwell 86.67 / flight 10.0 with digraphs ['Shift->P','P->a','a->1'];
    #     the true values are dwell 83.33 / flight 0.0 over 3 content keys.
    ev = [
        {"type": "keydown", "key": "Shift", "code": "ShiftLeft", "timestamp": 1000.0},
        {"type": "keydown", "key": "P", "code": "KeyP", "timestamp": 1060.0},
        {"type": "keyup", "key": "Shift", "code": "ShiftLeft", "timestamp": 1100.0},
        {"type": "keyup", "key": "p", "code": "KeyP", "timestamp": 1150.0},
        {"type": "keydown", "key": "a", "code": "KeyA", "timestamp": 1140.0},
        {"type": "keyup", "key": "a", "code": "KeyA", "timestamp": 1230.0},
        {"type": "keydown", "key": "1", "code": "Digit1", "timestamp": 1240.0},
        {"type": "keyup", "key": "1", "code": "Digit1", "timestamp": 1310.0},
    ]
    f = ex.extract_keystroke_features(ev)
    check("shift+rollover dwell is 83.33 ms", abs(f["mean_dwell"] - 83.33) < 0.05,
          f"got {f['mean_dwell']}")
    check("shift+rollover flight is 0.0 ms", abs(f["mean_flight_ud"]) < 0.05,
          f"got {f['mean_flight_ud']}")
    check("negative flight (key overlap) is preserved",
          f["flight_ud_sequence"][0] == -10.0, f"got {f['flight_ud_sequence']}")

    # 2. Shift released early: down 'T'/ShiftLeft, up 't'. Must still pair.
    ev = [
        {"type": "keydown", "key": "Shift", "code": "ShiftLeft", "timestamp": 1000.0},
        {"type": "keydown", "key": "T", "code": "KeyT", "timestamp": 1050.0},
        {"type": "keyup", "key": "Shift", "code": "ShiftLeft", "timestamp": 1090.0},
        {"type": "keyup", "key": "t", "code": "KeyT", "timestamp": 1140.0},
        {"type": "keydown", "key": "a", "code": "KeyA", "timestamp": 1200.0},
        {"type": "keyup", "key": "a", "code": "KeyA", "timestamp": 1290.0},
    ]
    f = ex.extract_keystroke_features(ev)
    check("Shift-modified key keeps its dwell", abs(f["dwell_sequence"][0] - 90.0) < 0.1,
          f"got {f['dwell_sequence']}")
    check("modifier keys excluded from press sequence", f["keystroke_count"] == 2,
          f"got {f['keystroke_count']}")
    check("no Shift digraphs", not any("Shift" in k for k in f["digraph_latencies"]),
          f"got {list(f['digraph_latencies'])}")

    # 3. Repeated letters must not cross-match.
    ev = []
    t = 0.0
    for _ in range(3):
        ev.append({"type": "keydown", "key": "s", "code": "KeyS", "timestamp": t})
        ev.append({"type": "keyup", "key": "s", "code": "KeyS", "timestamp": t + 70.0})
        t += 170.0
    f = ex.extract_keystroke_features(ev)
    check("repeated letters keep 70 ms dwell", abs(f["mean_dwell"] - 70.0) < 0.1,
          f"got {f['mean_dwell']}")

    # 4. Auto-repeat ignored.
    ev = [
        {"type": "keydown", "key": "a", "code": "KeyA", "timestamp": 0.0},
        {"type": "keydown", "key": "a", "code": "KeyA", "timestamp": 30.0, "repeat": True},
        {"type": "keydown", "key": "a", "code": "KeyA", "timestamp": 60.0, "repeat": True},
        {"type": "keyup", "key": "a", "code": "KeyA", "timestamp": 90.0},
    ]
    f = ex.extract_keystroke_features(ev)
    check("auto-repeat suppressed", f["keystroke_count"] == 1 and f["repeat_events"] == 2,
          f"count={f['keystroke_count']} repeats={f['repeat_events']}")

    # 5. Non-numeric timestamp must not raise.
    try:
        ex.extract_keystroke_features([
            {"type": "keydown", "key": "a", "code": "KeyA", "timestamp": "oops"},
            {"type": "keyup", "key": "a", "code": "KeyA", "timestamp": 50.0},
        ])
        ex.extract_mouse_features([{"type": "mousemove", "x": None, "y": 1, "timestamp": "x"}])
        check("non-numeric telemetry is tolerated", True)
    except Exception as exc:  # noqa: BLE001
        check("non-numeric telemetry is tolerated", False, repr(exc))

    # 6. Mask, not zero-fill, when the mouse is absent.
    k = ex.extract_keystroke_features(ev)
    m = ex.extract_mouse_features([])
    mask = ex.build_feature_mask(k, m)
    vec = ex.build_feature_vector(k, m)
    check("mouse dims masked out when absent", not mask[5] and not mask[6] and not mask[7],
          f"mask={mask}")
    check("masked dims are not zero-filled", vec[7] > 0, f"vec={vec}")

    print("\nModel")
    model = BiometricModel()
    rng = np.random.default_rng(7)

    def sample(base, noise=0.06):
        return np.array(base) * (1.0 + rng.normal(0, noise, size=len(base)))

    base = [95.0, 22.0, 140.0, 40.0, 4.2, 1.22, 4.1, 88.0]
    enrol = [sample(base) for _ in range(8)]
    profile = model.fit_profile(feature_vectors=enrol, digraph_samples=[{} for _ in enrol])

    _, conf_gen, _ = model.evaluate_attempt(
        attempt_vector=sample(base), attempt_digraphs={}, profile_data=profile)
    check("genuine attempt clears 60", conf_gen >= 60.0, f"confidence {conf_gen}")

    imp = np.array([170.0, 55.0, 330.0, 95.0, 1.9, 1.05, 1.3, 190.0])
    _, conf_imp, _ = model.evaluate_attempt(
        attempt_vector=imp, attempt_digraphs={}, profile_data=profile)
    check("slow impostor rejected", conf_imp < 45.0, f"confidence {conf_imp}")

    # Missing mouse must not sink a genuine user (the old click_duration=0 bug).
    v = sample(base)
    v[5], v[6], v[7] = 1.15, 4.0, 0.0
    mask_nom = np.array([True] * 5 + [False] * 3)
    _, conf_nomouse, met = model.evaluate_attempt(
        attempt_vector=v, attempt_digraphs={}, profile_data=profile, mask=mask_nom)
    check("keyboard-only genuine login still passes", conf_nomouse >= 60.0,
          f"confidence {conf_nomouse}, dims {met['dimensions_scored']}")

    # K=3 must not reject genuine users outright.
    p3 = model.fit_profile(feature_vectors=[sample(base) for _ in range(3)],
                           digraph_samples=[{}, {}, {}])
    accepted = sum(1 for _ in range(40)
                   if model.evaluate_attempt(attempt_vector=sample(base),
                                             attempt_digraphs={}, profile_data=p3)[1] >= 60.0)
    check("K=3 accepts most genuine attempts", accepted >= 32, f"{accepted}/40 accepted")

    # Legacy v1 profile must still score.
    legacy = {"mean": base, "std": [6.0, 3.0, 10.0, 5.0, 0.4, 0.06, 0.3, 9.0],
              "inv_cov": np.eye(8).tolist(), "avg_digraphs": {}, "sample_count": 3}
    try:
        _, c, _ = model.evaluate_attempt(attempt_vector=sample(base),
                                         attempt_digraphs={}, profile_data=legacy)
        check("legacy v1 profile still scores", c > 0, f"confidence {c}")
    except Exception as exc:  # noqa: BLE001
        check("legacy v1 profile still scores", False, repr(exc))

    print("\nBot detector")
    det = BotDetector()

    # Playwright default `page.type(delay=...)`: no isTrusted lie, no webdriver
    # flag, just a near-constant hold time. This is a HARD signal.
    keys, t = [], 0.0
    r = np.random.default_rng(3)
    for i in range(12):
        hold = 100.0 + float(r.uniform(-2, 2))
        keys.append({"type": "keydown", "key": "a", "code": f"Key{chr(65 + i)}",
                     "timestamp": round(t), "isTrusted": True})
        keys.append({"type": "keyup", "key": "a", "code": f"Key{chr(65 + i)}",
                     "timestamp": round(t + hold), "isTrusted": True})
        t += hold + 100.0 + float(r.uniform(-20, 20))
    kf = ex.extract_keystroke_features(keys)
    mf = ex.extract_mouse_features([])
    is_bot, prob, reasons = det.evaluate_telemetry(keys, [], kf, mf, {})
    check("fixed-delay scripted bot is caught without self-report", is_bot,
          f"probability {prob}, reasons {reasons}")

    # A bot with fully randomised, human-scale delays only trips weak signals,
    # so it must NOT be blocked by the bot detector in default mode - the
    # biometric model is what has to stop it. Asserted so this stays honest.
    keys2, t = [], 0.0
    for i in range(12):
        hold = float(np.exp(r.normal(math.log(95), 0.30)))
        keys2.append({"type": "keydown", "key": "a", "code": f"Key{chr(65 + i)}",
                      "timestamp": t, "isTrusted": True})
        keys2.append({"type": "keyup", "key": "a", "code": f"Key{chr(65 + i)}",
                      "timestamp": t + hold, "isTrusted": True})
        t += hold + float(np.exp(r.normal(math.log(120), 0.45)))
    kf2 = ex.extract_keystroke_features(keys2)
    is_bot2, prob2, _ = det.evaluate_telemetry(keys2, [], kf2, ex.extract_mouse_features([]), {})
    check("weak signals alone cannot block (known coverage gap)", not is_bot2,
          f"probability {prob2}")

    # Human-ish: log-normal timings, no grid.
    keys, t = [], 0.0
    for i in range(12):
        hold = float(np.exp(r.normal(math.log(95), 0.28)))
        keys.append({"type": "keydown", "key": "a", "code": f"Key{chr(65 + i)}",
                     "timestamp": t, "isTrusted": True})
        keys.append({"type": "keyup", "key": "a", "code": f"Key{chr(65 + i)}",
                     "timestamp": t + hold, "isTrusted": True})
        t += hold + float(np.exp(r.normal(math.log(110), 0.42)))
    mouse, mt, x, y = [], 0.0, 100.0, 100.0
    for i in range(40):
        x += float(r.normal(6, 3))
        y += float(r.normal(3, 3))
        mt += float(r.uniform(12, 24))
        mouse.append({"type": "mousemove", "x": x, "y": y, "timestamp": mt, "isTrusted": True})
    mouse.append({"type": "mousedown", "x": x, "y": y, "timestamp": mt + 40, "isTrusted": True})
    mouse.append({"type": "mouseup", "x": x, "y": y, "timestamp": mt + 125, "isTrusted": True})
    kf = ex.extract_keystroke_features(keys)
    mf = ex.extract_mouse_features(mouse)
    is_bot, prob, reasons = det.evaluate_telemetry(keys, mouse, kf, mf, {})
    check("human-like session is not flagged", not is_bot, f"probability {prob}, {reasons}")

    from backend.biometrics.bot_detector import ReplayGuard
    guard = ReplayGuard()
    n = guard.issue()["nonce"]
    first = guard.check("alice", keys, nonce=n)
    replayed_nonce = guard.check("alice", keys, nonce=n)
    n2 = guard.issue()["nonce"]
    spliced = guard.check("alice", keys, nonce=n2)
    check("fresh session passes the replay guard", not first["is_replay"], str(first))
    check("reused nonce is rejected", replayed_nonce["is_replay"], str(replayed_nonce))
    check("fresh nonce + old timings is rejected", spliced["is_replay"], str(spliced))

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# synthetic smoke test
# --------------------------------------------------------------------------- #
def synthetic_sessions(n_users: int, per_user: int, seed: int = 0) -> List[Dict[str, Any]]:
    """Pipeline smoke test only. These numbers are NOT evidence about real users."""
    rng = np.random.default_rng(seed)
    codes = [f"Key{c}" for c in "PASSWORD1"]
    sessions = []

    def emit(profile, jitter, user, target, label):
        keys, t = [], float(rng.uniform(0, 500))
        for i, code in enumerate(codes):
            hold = float(np.exp(rng.normal(math.log(profile["dwell"][i]), jitter)))
            keys.append({"type": "keydown", "key": code[-1].lower(), "code": code,
                         "timestamp": t, "isTrusted": True})
            keys.append({"type": "keyup", "key": code[-1].lower(), "code": code,
                         "timestamp": t + hold, "isTrusted": True})
            if i < len(codes) - 1:
                t += hold + float(np.exp(rng.normal(math.log(profile["flight"][i]), jitter * 1.3)))
        mouse, mt, x, y = [], 0.0, 100.0, 100.0
        for _ in range(30):
            x += float(rng.normal(profile["vx"], 4))
            y += float(rng.normal(2, 4))
            mt += float(rng.uniform(13, 22))
            mouse.append({"type": "mousemove", "x": x, "y": y, "timestamp": mt, "isTrusted": True})
        mouse.append({"type": "mousedown", "x": x, "y": y, "timestamp": mt + 30, "isTrusted": True})
        mouse.append({"type": "mouseup", "x": x, "y": y,
                      "timestamp": mt + 30 + profile["click"], "isTrusted": True})
        sessions.append({"user": user, "target": target, "label": label,
                         "keystrokes": keys, "mouse": mouse, "client_metadata": {}})

    users = []
    for u in range(n_users):
        users.append({
            "name": f"user{u:02d}",
            "dwell": list(np.exp(rng.normal(math.log(95), 0.30, len(codes)))),
            "flight": list(np.exp(rng.normal(math.log(130), 0.40, len(codes) - 1))),
            "vx": float(rng.uniform(4, 9)),
            "click": float(rng.uniform(60, 130)),
        })
    for p in users:
        for _ in range(per_user):
            emit(p, 0.22, p["name"], p["name"], "genuine")
    # Mimic: same overall rate, different per-position structure.
    for p in users:
        other = users[(users.index(p) + 1) % len(users)]
        scale = float(np.mean(p["dwell"])) / float(np.mean(other["dwell"]))
        mimic = dict(other)
        mimic["dwell"] = [d * scale for d in other["dwell"]]
        mimic["flight"] = [f * scale for f in other["flight"]]
        for _ in range(3):
            emit(mimic, 0.22, other["name"], p["name"], "mimic")
    return sessions


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="directory of .json sessions, or a .jsonl file")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 8],
                    help="enrolment sample counts to evaluate")
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--per-user-threshold", action="store_true",
                    help="derive each user's threshold from their leave-one-out scores")
    ap.add_argument("--target-frr", type=float, default=0.05)
    ap.add_argument("--no-cross-impostors", action="store_true",
                    help="only score explicitly labelled attacks, not other users' sessions")
    ap.add_argument("--bot-strict", action="store_true",
                    help="let weak shape heuristics block on their own (measure FP rate first)")
    ap.add_argument("--no-bot-gate", action="store_true",
                    help="score the biometric model alone, ignoring the bot detector")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--synthetic", type=int, metavar="N_USERS",
                    help="run on N synthetic users instead of real data (smoke test only)")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.synthetic:
        sessions = synthetic_sessions(args.synthetic, per_user=12, seed=args.seed)
        print(f"SYNTHETIC RUN: {len(sessions)} generated sessions. "
              f"These numbers describe the pipeline, not real users. "
              f"Do not put them in the report.\n")
    elif args.data:
        sessions = load_sessions(args.data)
        print(f"Loaded {len(sessions)} sessions from {args.data}")
    else:
        ap.error("pass --data, --synthetic or --self-test")

    labels = defaultdict(int)
    for s in sessions:
        labels[s["label"]] += 1
    print("Sessions by label: " + ", ".join(f"{k}={v}" for k, v in sorted(labels.items())))

    extractor = FeatureExtractor()
    feats = featurise(sessions, extractor)

    thin = [f for f in feats if f["keystroke_count"] < 4]
    if thin:
        print(f"Warning: {len(thin)} sessions have fewer than 4 usable keystrokes")

    results = []
    for k in args.k:
        res = run_protocol(
            feats, k=k, seed=args.seed,
            cross_impostors=not args.no_cross_impostors,
            per_user_threshold=args.per_user_threshold,
            target_frr=args.target_frr,
            bot_gate=not args.no_bot_gate,
            bot_strict=args.bot_strict,
        )
        if res["n_users"] == 0:
            print(f"K={k}: no user has {k + 1}+ genuine sessions, skipping")
            continue
        results.append(res)
        write_csv(res["rows"], os.path.join(args.out, f"scores_k{k}.csv"))

    if not results:
        print("Nothing to evaluate. Collect more sessions per user.")
        return 1

    md = report(results, args.threshold)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    print(f"Per-attempt scores and this report written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
