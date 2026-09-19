"""
BioPrint: Behavioral Feature Extractor (patched)
Extracts high-dimensional behavioral biometric vectors from raw browser keystroke and mouse telemetry.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MODIFIER_KEYS = {
    "shift", "shiftleft", "shiftright",
    "control", "controlleft", "controlright", "ctrl",
    "alt", "altleft", "altright", "altgraph",
    "meta", "metaleft", "metaright", "os", "osleft", "osright",
    "capslock"
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _is_modifier(key: str, code: str) -> bool:
    k_low = (key or "").lower()
    c_low = (code or "").lower()
    if k_low in MODIFIER_KEYS or c_low in MODIFIER_KEYS:
        return True
    if key in ("Shift", "Control", "Alt", "Meta", "CapsLock"):
        return True
    if (code or "").startswith(("Shift", "Control", "Alt", "Meta")):
        return True
    return False


class FeatureExtractor:
    def __init__(self, hash_digraphs: bool = False):
        self.hash_digraphs = bool(hash_digraphs)

    def extract_keystroke_features(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts Dwell Time, Flight Time (UD and DD), rhythm cadence, digraph latencies,
        and per-position sequences. Handles modifier pairing, key rollover, and auto-repeats.
        """
        if not events:
            return {
                "mean_dwell": 0.0,
                "std_dwell": 0.0,
                "mean_flight_ud": 0.0,
                "std_flight_ud": 0.0,
                "mean_flight_dd": 0.0,
                "std_flight_dd": 0.0,
                "cps": 0.0,
                "backspace_count": 0,
                "repeat_events": 0,
                "unmatched_downs": 0,
                "dwell_by_key": {},
                "digraph_latencies": {},
                "keystroke_count": 0,
                "dwell_sequence": [],
                "flight_ud_sequence": [],
                "flight_dd_sequence": [],
                "content_keys": [],
            }

        ordered_downs: List[Dict[str, Any]] = []
        pending_by_code: Dict[str, List[Dict[str, Any]]] = {}
        pending_by_key: Dict[str, List[Dict[str, Any]]] = {}

        backspace_count = 0
        repeat_events = 0

        for ev in events:
            if not isinstance(ev, dict):
                continue

            ev_type = str(ev.get("type", ""))
            k = str(ev.get("key", ""))
            code = str(ev.get("code") or k)
            t = _safe_float(ev.get("timestamp"), 0.0)

            if ev.get("repeat") is True:
                repeat_events += 1
                continue

            if k in ("Backspace", "Delete") and ev_type == "keydown":
                backspace_count += 1

            if _is_modifier(k, code):
                continue

            if ev_type == "keydown":
                entry = {
                    "key": k,
                    "code": code,
                    "down_t": t,
                    "up_t": None,
                }
                ordered_downs.append(entry)
                pending_by_code.setdefault(code, []).append(entry)
                pending_by_key.setdefault(k.lower(), []).append(entry)

            elif ev_type == "keyup":
                match: Optional[Dict[str, Any]] = None
                if pending_by_code.get(code):
                    match = pending_by_code[code].pop(0)
                    k_list = pending_by_key.get(match["key"].lower(), [])
                    if match in k_list:
                        k_list.remove(match)
                elif pending_by_key.get(k.lower()):
                    match = pending_by_key[k.lower()].pop(0)
                    c_list = pending_by_code.get(match["code"], [])
                    if match in c_list:
                        c_list.remove(match)

                if match is not None:
                    match["up_t"] = t

        unmatched_downs = sum(len(q) for q in pending_by_code.values())

        dwell_times: List[float] = []
        dwell_by_key: Dict[str, List[float]] = {}
        dwell_sequence: List[Optional[float]] = []

        for entry in ordered_downs:
            if entry["up_t"] is not None:
                dwell = max(0.0, entry["up_t"] - entry["down_t"])
                dwell_times.append(dwell)
                dwell_sequence.append(dwell)
                dwell_by_key.setdefault(entry["key"].lower(), []).append(dwell)
            else:
                dwell_sequence.append(None)

        flight_ud: List[float] = []
        flight_dd: List[float] = []
        flight_ud_sequence: List[Optional[float]] = []
        digraphs: Dict[str, float] = {}

        for i in range(len(ordered_downs) - 1):
            prev = ordered_downs[i]
            curr = ordered_downs[i + 1]

            dd = curr["down_t"] - prev["down_t"]
            flight_dd.append(dd)

            if prev["up_t"] is not None:
                ud = curr["down_t"] - prev["up_t"]
                flight_ud.append(ud)
                flight_ud_sequence.append(ud)
            else:
                flight_ud_sequence.append(None)

            k1 = prev["key"]
            k2 = curr["key"]
            if self.hash_digraphs:
                dk = hashlib.sha256(f"{k1}->{k2}".encode("utf-8")).hexdigest()[:12]
            else:
                dk = f"{k1}->{k2}"
            digraphs[dk] = dd

        # Typing Speed (Chars Per Second)
        total_time = 0.0
        if len(ordered_downs) >= 2:
            total_time = (ordered_downs[-1]["down_t"] - ordered_downs[0]["down_t"]) / 1000.0
        cps = (len(ordered_downs) / total_time) if total_time > 0.05 else 0.0

        mean_dwell = float(np.mean(dwell_times)) if dwell_times else 0.0
        std_dwell = float(np.std(dwell_times, ddof=1)) if len(dwell_times) > 1 else 0.0
        mean_flight_ud = float(np.mean(flight_ud)) if flight_ud else 0.0
        std_flight_ud = float(np.std(flight_ud, ddof=1)) if len(flight_ud) > 1 else 0.0
        mean_flight_dd = float(np.mean(flight_dd)) if flight_dd else 0.0
        std_flight_dd = float(np.std(flight_dd, ddof=1)) if len(flight_dd) > 1 else 0.0

        avg_dwell_by_key = {k: float(np.mean(v)) for k, v in dwell_by_key.items()}

        return {
            "mean_dwell": round(mean_dwell, 2),
            "std_dwell": round(std_dwell, 2),
            "mean_flight_ud": round(mean_flight_ud, 2),
            "std_flight_ud": round(std_flight_ud, 2),
            "mean_flight_dd": round(mean_flight_dd, 2),
            "std_flight_dd": round(std_flight_dd, 2),
            "cps": round(cps, 2),
            "backspace_count": backspace_count,
            "repeat_events": repeat_events,
            "unmatched_downs": unmatched_downs,
            "dwell_by_key": avg_dwell_by_key,
            "digraph_latencies": digraphs,
            "keystroke_count": len(ordered_downs),
            "dwell_sequence": dwell_sequence,
            "flight_ud_sequence": flight_ud_sequence,
            "flight_dd_sequence": flight_dd,
            "content_keys": [d["key"] for d in ordered_downs],
        }

    def extract_mouse_features(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts mouse trajectory curvature, velocity profiles, jitter/tremor variance,
        sample interval regularity, integer coordinates ratio, and click duration.
        """
        if not events:
            return {
                "mouse_event_count": 0,
                "move_present": False,
                "click_present": False,
                "click_count": 0,
                "total_path_length": 0.0,
                "euclidean_distance": 0.0,
                "curvature_ratio": 1.0,
                "mean_velocity": 0.0,
                "velocity_variance": 0.0,
                "velocity_skew": 0.0,
                "jitter_variance": 0.0,
                "sample_interval_std": 0.0,
                "integer_coord_ratio": 0.0,
                "click_duration": 80.0,
                "approach_angle": 0.0,
            }

        points: List[Tuple[float, float, float]] = []
        click_downs: List[float] = []
        click_durations: List[float] = []

        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev_type = str(ev.get("type", ""))

            raw_t = ev.get("timestamp")
            if raw_t is None:
                continue
            t = _safe_float(raw_t)

            if ev_type == "mousemove":
                raw_x = ev.get("x")
                raw_y = ev.get("y")
                if raw_x is not None and raw_y is not None:
                    x = _safe_float(raw_x)
                    y = _safe_float(raw_y)
                    points.append((x, y, t))
            elif ev_type == "mousedown":
                click_downs.append(t)
            elif ev_type == "mouseup" and click_downs:
                click_durations.append(t - click_downs.pop(0))

        click_present = bool(click_durations)
        avg_click = float(np.mean(click_durations)) if click_durations else 80.0

        if len(points) < 2:
            return {
                "mouse_event_count": len(events),
                "move_present": False,
                "click_present": click_present,
                "click_count": len(click_durations),
                "total_path_length": 0.0,
                "euclidean_distance": 0.0,
                "curvature_ratio": 1.0,
                "mean_velocity": 0.0,
                "velocity_variance": 0.0,
                "velocity_skew": 0.0,
                "jitter_variance": 0.0,
                "sample_interval_std": 0.0,
                "integer_coord_ratio": 0.0,
                "click_duration": round(avg_click, 2),
                "approach_angle": 0.0,
            }

        total_path = 0.0
        velocities: List[float] = []
        angles: List[float] = []
        sample_intervals: List[float] = []
        integer_samples = 0

        for pt in points:
            if abs(pt[0] - round(pt[0])) < 1e-4 and abs(pt[1] - round(pt[1])) < 1e-4:
                integer_samples += 1

        for i in range(len(points) - 1):
            x1, y1, t1 = points[i]
            x2, y2, t2 = points[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)
            dt_ms = max(0.1, t2 - t1)
            sample_intervals.append(dt_ms)
            dt_s = dt_ms / 1000.0

            total_path += dist
            velocities.append(dist / dt_s)
            angles.append(math.atan2(dy, dx))

        x_start, y_start, _ = points[0]
        x_end, y_end, _ = points[-1]
        euclidean_dist = math.hypot(x_end - x_start, y_end - y_start)

        curvature_ratio = (total_path / euclidean_dist) if euclidean_dist > 5.0 else 1.0

        angle_diffs: List[float] = []
        for i in range(len(angles) - 1):
            diff = abs(angles[i + 1] - angles[i])
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            angle_diffs.append(diff)

        jitter_variance = float(np.var(angle_diffs)) if len(angle_diffs) > 1 else 0.0
        mean_vel = float(np.mean(velocities)) if velocities else 0.0
        vel_var = float(np.var(velocities)) if len(velocities) > 1 else 0.0

        if len(velocities) >= 3 and np.std(velocities) > 1e-6:
            v_arr = np.asarray(velocities, dtype=np.float64)
            v_sd = float(v_arr.std())
            vel_skew = float(np.mean(((v_arr - mean_vel) / v_sd) ** 3))
        else:
            vel_skew = 0.0

        sample_interval_std = float(np.std(sample_intervals)) if len(sample_intervals) > 1 else 0.0
        integer_coord_ratio = integer_samples / len(points)
        approach_angle = round(float(math.degrees(angles[-1])), 2) if angles else 0.0

        return {
            "mouse_event_count": len(events),
            "move_present": True,
            "click_present": click_present,
            "click_count": len(click_durations),
            "total_path_length": round(total_path, 2),
            "euclidean_distance": round(euclidean_dist, 2),
            "curvature_ratio": round(curvature_ratio, 3),
            "mean_velocity": round(mean_vel, 2),
            "velocity_variance": round(vel_var, 2),
            "velocity_skew": round(vel_skew, 4),
            "jitter_variance": round(jitter_variance, 4),
            "sample_interval_std": round(sample_interval_std, 2),
            "integer_coord_ratio": round(integer_coord_ratio, 4),
            "click_duration": round(avg_click, 2),
            "approach_angle": approach_angle,
        }

    def build_feature_vector(self, keystroke_features: Dict[str, Any], mouse_features: Dict[str, Any]) -> np.ndarray:
        """
        Maps extracted metrics to a normalized 8-dimensional behavioral representation vector.
        Features:
        [0] Mean Dwell Time (ms)
        [1] Std Dwell Time (ms)
        [2] Mean Flight Time UD (ms)
        [3] Std Flight Time UD (ms)
        [4] Characters Per Second (WPM proxy)
        [5] Curvature Ratio
        [6] Velocity Variance (log scaled)
        [7] Click Hold Duration (ms)
        """
        vel_var_log = math.log1p(max(0.0, float(mouse_features.get("velocity_variance", 0.0) or 0.0)))
        vec = [
            float(keystroke_features.get("mean_dwell", 80.0) or 80.0),
            float(keystroke_features.get("std_dwell", 15.0) or 15.0),
            float(keystroke_features.get("mean_flight_ud", 120.0) or 120.0),
            float(keystroke_features.get("std_flight_ud", 30.0) or 30.0),
            float(keystroke_features.get("cps", 4.0) or 4.0),
            float(mouse_features.get("curvature_ratio", 1.15) or 1.15),
            vel_var_log,
            float(mouse_features.get("click_duration", 80.0) or 80.0)
        ]
        return np.array(vec, dtype=np.float64)

    def build_feature_mask(self, keystroke_features: Dict[str, Any], mouse_features: Dict[str, Any]) -> np.ndarray:
        """
        Builds an 8-dimensional boolean mask indicating which feature dimensions were actually
        measured rather than populated with prior fallback values.
        """
        has_k = int(keystroke_features.get("keystroke_count", 0) or 0) >= 2
        has_mouse_move = bool(mouse_features.get("move_present", False) and int(mouse_features.get("mouse_event_count", 0) or 0) >= 2)
        has_mouse_click = bool(mouse_features.get("click_present", False))
        mask = [
            has_k,
            has_k,
            has_k,
            has_k,
            has_k,
            has_mouse_move,
            has_mouse_move,
            has_mouse_click,
        ]
        return np.array(mask, dtype=bool)

    def extract_all(self, raw_keystrokes: List[Dict[str, Any]], raw_mouse: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convenience wrapper extracting full feature vectors, masks, digraphs, and sequences.
        """
        k_feat = self.extract_keystroke_features(raw_keystrokes)
        m_feat = self.extract_mouse_features(raw_mouse)
        vec = self.build_feature_vector(k_feat, m_feat)
        mask = self.build_feature_mask(k_feat, m_feat)
        digraphs = k_feat.get("digraph_latencies", {})
        sequences = {
            "keys": k_feat.get("content_keys", []),
            "dwell": k_feat.get("dwell_sequence", []),
            "flight": k_feat.get("flight_ud_sequence", []),
        }
        return {
            "vector": vec,
            "mask": mask,
            "digraphs": digraphs,
            "sequences": sequences,
            "keystroke": k_feat,
            "mouse": m_feat,
        }
