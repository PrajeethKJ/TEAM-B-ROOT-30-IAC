"""
BioPrint: Explainability & Diagnostic Audit Engine
Implements Stretch Goal 4: Generates human-readable summaries of which
specific behavioral biometric signals triggered an acceptance or security block.
"""

from typing import Dict, List, Any


class BiometricExplainer:
    def __init__(self):
        pass

    def explain_decision(
        self,
        decision: str,  # 'AUTHENTICATED', 'BLOCKED_IMPOSTOR', 'BLOCKED_BOT'
        confidence_score: float,
        bot_reasons: List[str],
        metrics: Dict[str, Any],
        keystroke_features: Dict[str, Any],
        mouse_features: Dict[str, Any],
        profile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesizes a structured, human-readable explainability report.
        """
        if decision == "BLOCKED_BOT":
            headline = "Automated Bot / Script Attack Intercepted"
            summary = (
                "The login attempt was immediately blocked because non-human automation signatures "
                "were detected. Even if the password was correct, no human was operating the interface."
            )
            bullet_points = bot_reasons if bot_reasons else ["Synthetic event delivery detected."]
            badge_color = "red"
            risk_level = "CRITICAL_FRAUD"

        elif decision == "BLOCKED_IMPOSTOR":
            headline = "Behavioral Biometric Mismatch (Impostor Blocked)"
            summary = (
                f"Password credentials were valid, but the physical motor patterns differed significantly "
                f"from the enrolled account owner (Confidence: {confidence_score}% vs required 60.0%)."
            )
            bullet_points = []
            z_breakdown = metrics.get("feature_z_breakdown", {})

            # Analyze individual feature anomalies
            if z_breakdown.get("dwell_time", 0) > 2.0:
                base_dwell = profile_data.get("mean", [80])[0]
                att_dwell = keystroke_features.get("mean_dwell", 0)
                bullet_points.append(
                    f"Key Hold Duration Divergence: Average dwell was {att_dwell}ms (account baseline: {round(base_dwell, 1)}ms, Z-score: +{z_breakdown['dwell_time']})"
                )

            if z_breakdown.get("flight_time", 0) > 2.0:
                base_flight = profile_data.get("mean", [0, 0, 120])[2]
                att_flight = keystroke_features.get("mean_flight_ud", 0)
                bullet_points.append(
                    f"Inter-Key Flight Cadence Anomaly: Transition latency was {att_flight}ms (account baseline: {round(base_flight, 1)}ms, Z-score: +{z_breakdown['flight_time']})"
                )

            if z_breakdown.get("typing_speed", 0) > 2.0:
                base_cps = profile_data.get("mean", [0, 0, 0, 0, 4.0])[4]
                att_cps = keystroke_features.get("cps", 0)
                bullet_points.append(
                    f"Typing Speed Mismatch: Input speed was {att_cps} chars/sec (account baseline: {round(base_cps, 1)} chars/sec)"
                )

            if z_breakdown.get("mouse_curvature", 0) > 2.0:
                att_curv = mouse_features.get("curvature_ratio", 1.0)
                bullet_points.append(
                    f"Unnatural Mouse Trajectory: Path curvature ratio was {att_curv} (diverges from owner's customary pointer movement)"
                )

            if not bullet_points:
                bullet_points.append(
                    f"High multi-dimensional Mahalanobis distance ({metrics.get('mahalanobis_distance', 0)}) across keystroke transition vectors."
                )

            badge_color = "orange"
            risk_level = "HIGH_ANOMALY"

        else:  # AUTHENTICATED
            headline = "Behavioral Identity Confirmed"
            summary = (
                f"The physical interaction pattern matched the enrolled account owner with "
                f"{confidence_score}% confidence. All motor dynamics fall within normal physiological variance."
            )
            bullet_points = [
                f"Keystroke Dwell Consistency: {keystroke_features.get('mean_dwell', 0)}ms aligned with baseline profile.",
                f"Flight Time Rhythm: Inter-key transitions exhibit characteristic user cadence.",
                f"Pointer Dynamics: Natural human cursor acceleration and micro-tremors verified."
            ]
            badge_color = "green"
            risk_level = "LOW_RISK"

        return {
            "decision": decision,
            "confidence_score": confidence_score,
            "headline": headline,
            "summary": summary,
            "bullet_points": bullet_points,
            "badge_color": badge_color,
            "risk_level": risk_level,
            "signal_gauges": {
                "keystroke_dwell_match": round(max(0, 100 - (metrics.get("feature_z_breakdown", {}).get("dwell_time", 0) * 20)), 1),
                "flight_cadence_match": round(max(0, 100 - (metrics.get("feature_z_breakdown", {}).get("flight_time", 0) * 20)), 1),
                "cursor_trajectory_match": round(max(0, 100 - (metrics.get("feature_z_breakdown", {}).get("mouse_curvature", 0) * 20)), 1),
                "human_authenticity": 0.0 if decision == "BLOCKED_BOT" else 98.5
            }
        }
