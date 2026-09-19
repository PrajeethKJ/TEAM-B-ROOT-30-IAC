"""
BioPrint: Behavioral Feature Extractor
Extracts high-dimensional behavioral biometric vectors from raw browser keystroke and mouse telemetry.
"""

import math
import numpy as np
from typing import Dict, List, Any, Tuple


class FeatureExtractor:
    def __init__(self):
        pass

    def extract_keystroke_features(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts Dwell Time, Flight Time (UD and DD), rhythm cadence, and error corrections.
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
                "dwell_by_key": {},
                "digraph_latencies": {},
                "keystroke_count": 0
            }

        # Match keydown with corresponding keyup
        keydowns: Dict[str, List[float]] = {}
        dwell_times: List[float] = []
        dwell_by_key: Dict[str, List[float]] = {}
        ordered_downs: List[Tuple[str, float]] = []
        ordered_ups: List[Tuple[str, float]] = []
        backspace_count = 0

        for ev in events:
            k = ev.get("key", "")
            t = float(ev.get("timestamp", 0))
            ev_type = ev.get("type", "")

            if k in ("Backspace", "Delete") and ev_type == "keydown":
                backspace_count += 1

            if ev_type == "keydown":
                if k not in keydowns:
                    keydowns[k] = []
                keydowns[k].append(t)
                ordered_downs.append((k, t))
            elif ev_type == "keyup":
                ordered_ups.append((k, t))
                if k in keydowns and len(keydowns[k]) > 0:
                    down_t = keydowns[k].pop(0)
                    dwell = max(0.0, t - down_t)
                    dwell_times.append(dwell)
                    if k not in dwell_by_key:
                        dwell_by_key[k] = []
                    dwell_by_key[k].append(dwell)

        # Compute Flight Times (UD: Up to next Down; DD: Down to next Down)
        flight_ud: List[float] = []
        flight_dd: List[float] = []
        digraphs: Dict[str, float] = {}

        for i in range(len(ordered_downs) - 1):
            k1, t1 = ordered_downs[i]
            k2, t2 = ordered_downs[i + 1]
            dd = t2 - t1
            flight_dd.append(dd)

            # Match with closest preceding keyup
            matching_ups = [u for u in ordered_ups if u[0] == k1 and u[1] <= t2]
            if matching_ups:
                up_t = matching_ups[-1][1]
                flight_ud.append(t2 - up_t)

            digraph_key = f"{k1}->{k2}"
            digraphs[digraph_key] = dd

        # Typing Speed (Chars Per Second)
        total_time = 0.0
        if len(ordered_downs) >= 2:
            total_time = (ordered_downs[-1][1] - ordered_downs[0][1]) / 1000.0  # seconds
        cps = (len(ordered_downs) / total_time) if total_time > 0.05 else 0.0

        mean_dwell = float(np.mean(dwell_times)) if dwell_times else 0.0
        std_dwell = float(np.std(dwell_times)) if len(dwell_times) > 1 else 0.0
        mean_flight_ud = float(np.mean(flight_ud)) if flight_ud else 0.0
        std_flight_ud = float(np.std(flight_ud)) if len(flight_ud) > 1 else 0.0
        mean_flight_dd = float(np.mean(flight_dd)) if flight_dd else 0.0
        std_flight_dd = float(np.std(flight_dd)) if len(flight_dd) > 1 else 0.0

        # Mean dwell per unique key
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
            "dwell_by_key": avg_dwell_by_key,
            "digraph_latencies": digraphs,
            "keystroke_count": len(ordered_downs)
        }

    def extract_mouse_features(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts mouse trajectory curvature, velocity profiles, jitter/tremor variance, and click dwell.
        """
        if not events or len(events) < 2:
            return {
                "mouse_event_count": len(events),
                "total_path_length": 0.0,
                "euclidean_distance": 0.0,
                "curvature_ratio": 1.0,
                "mean_velocity": 0.0,
                "velocity_variance": 0.0,
                "jitter_variance": 0.0,
                "click_duration": 0.0,
                "approach_angle": 0.0
            }

        points = []
        click_downs = []
        click_durations = []

        for ev in events:
            x = float(ev.get("x", 0))
            y = float(ev.get("y", 0))
            t = float(ev.get("timestamp", 0))
            ev_type = ev.get("type", "")

            if ev_type == "mousemove":
                points.append((x, y, t))
            elif ev_type == "mousedown":
                click_downs.append(t)
            elif ev_type == "mouseup" and click_downs:
                click_durations.append(t - click_downs.pop(0))

        if len(points) < 2:
            avg_click = float(np.mean(click_durations)) if click_durations else 0.0
            return {
                "mouse_event_count": len(events),
                "total_path_length": 0.0,
                "euclidean_distance": 0.0,
                "curvature_ratio": 1.0,
                "mean_velocity": 0.0,
                "velocity_variance": 0.0,
                "jitter_variance": 0.0,
                "click_duration": round(avg_click, 2),
                "approach_angle": 0.0
            }

        # Trajectory metrics
        total_path = 0.0
        velocities = []
        angles = []

        for i in range(len(points) - 1):
            x1, y1, t1 = points[i]
            x2, y2, t2 = points[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            dt = max(0.001, (t2 - t1) / 1000.0)  # in seconds

            total_path += dist
            vel = dist / dt  # pixels / sec
            velocities.append(vel)

            angle = math.atan2(dy, dx)
            angles.append(angle)

        # Euclidean straight-line distance
        x_start, y_start, _ = points[0]
        x_end, y_end, _ = points[-1]
        euclidean_dist = math.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)

        # Curvature Ratio (Human curves > 1.05; bots or straight lines ~ 1.00)
        curvature_ratio = (total_path / euclidean_dist) if euclidean_dist > 5.0 else 1.0

        # Jitter: variance in angular change between consecutive movement vectors
        angle_diffs = []
        for i in range(len(angles) - 1):
            diff = abs(angles[i + 1] - angles[i])
            # normalize angle diff to [-pi, pi]
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            angle_diffs.append(diff)

        jitter_variance = float(np.var(angle_diffs)) if len(angle_diffs) > 1 else 0.0
        mean_vel = float(np.mean(velocities)) if velocities else 0.0
        vel_var = float(np.var(velocities)) if len(velocities) > 1 else 0.0
        avg_click = float(np.mean(click_durations)) if click_durations else 85.0
        approach_angle = round(float(math.degrees(angles[-1])), 2) if angles else 0.0

        return {
            "mouse_event_count": len(events),
            "total_path_length": round(total_path, 2),
            "euclidean_distance": round(euclidean_dist, 2),
            "curvature_ratio": round(curvature_ratio, 3),
            "mean_velocity": round(mean_vel, 2),
            "velocity_variance": round(vel_var, 2),
            "jitter_variance": round(jitter_variance, 4),
            "click_duration": round(avg_click, 2),
            "approach_angle": approach_angle
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
        vel_var_log = math.log1p(max(0.0, mouse_features.get("velocity_variance", 0.0)))
        vec = [
            keystroke_features.get("mean_dwell", 80.0),
            keystroke_features.get("std_dwell", 15.0),
            keystroke_features.get("mean_flight_ud", 120.0),
            keystroke_features.get("std_flight_ud", 30.0),
            keystroke_features.get("cps", 4.0),
            mouse_features.get("curvature_ratio", 1.15),
            vel_var_log,
            mouse_features.get("click_duration", 80.0)
        ]
        return np.array(vec, dtype=np.float64)

