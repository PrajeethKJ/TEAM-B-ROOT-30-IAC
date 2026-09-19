"""
BioPrint: Automated Bot & Scripted Fraud Detector
Detects non-human traffic, synthetic event injections, programmatic macros, and replay attacks.
"""

from typing import Dict, List, Any, Tuple


class BotDetector:
    def __init__(self):
        pass

    def evaluate_telemetry(
        self,
        raw_keystrokes: List[Dict[str, Any]],
        raw_mouse: List[Dict[str, Any]],
        keystroke_features: Dict[str, Any],
        mouse_features: Dict[str, Any],
        client_metadata: Dict[str, Any]
    ) -> Tuple[bool, float, List[str]]:
        """
        Evaluates incoming telemetry for automation signatures.
        Returns:
            is_bot (bool): True if classified as non-human / automated.
            bot_probability (float): 0.0 to 1.0 confidence that actor is a bot.
            reasons (List[str]): Explanations of detected fraud signals.
        """
        reasons = []
        fraud_score = 0.0

        # 1. Check native browser event trust (isTrusted flag)
        untrusted_keys = [ev for ev in raw_keystrokes if ev.get("isTrusted") is False]
        untrusted_mouse = [ev for ev in raw_mouse if ev.get("isTrusted") is False]

        if len(untrusted_keys) > 0 or len(untrusted_mouse) > 0:
            reasons.append("Synthetic DOM Event: Browser reported event.isTrusted == false (script-dispatched)")
            fraud_score += 0.95

        # 2. Check for automation flags in browser environment
        if client_metadata.get("webdriver", False) is True:
            reasons.append("Automated Browser Environment: navigator.webdriver flag is active (Selenium/Puppeteer)")
            fraud_score += 0.90

        # 3. Check for Inhuman Keystroke Speed (e.g. typing macro or instant payload dump)
        cps = keystroke_features.get("cps", 0.0)
        if cps > 22.0:
            reasons.append(f"Inhuman Typing Speed: {cps} chars/second exceeds human motor limits")
            fraud_score += 0.85

        # 4. Check for Zero-Variance / Synthetic Constant Delays (e.g., page.type(delay=100))
        std_dwell = keystroke_features.get("std_dwell", 10.0)
        std_flight = keystroke_features.get("std_flight_ud", 20.0)
        keystroke_count = keystroke_features.get("keystroke_count", 0)

        if keystroke_count >= 5:
            if std_dwell < 1.5:
                reasons.append(f"Synthetic Timing Precision: Dwell time standard deviation is {std_dwell}ms (near-zero jitter)")
                fraud_score += 0.80

            if std_flight < 2.0 and keystroke_features.get("mean_flight_ud", 0) > 10:
                reasons.append(f"Machine Timing Regularity: Flight time standard deviation is {std_flight}ms (scripted sleep loop)")
                fraud_score += 0.85

        # 5. Check for Linear / Synthesized Mouse Trajectory (Bresenham / Linear Interpolation)
        mouse_count = mouse_features.get("mouse_event_count", 0)
        curvature = mouse_features.get("curvature_ratio", 1.1)
        jitter = mouse_features.get("jitter_variance", 0.1)

        if mouse_count >= 10:
            # Perfectly straight path with zero jitter
            if curvature <= 1.002 and jitter < 0.0005:
                reasons.append(f"Linear Algorithmic Cursor: Curvature {curvature} with 0.0 angle variance indicates programmatic mouse movement")
                fraud_score += 0.80

        # 6. Check for instant teleportation / Instant Form Submit without prior cursor activity
        if mouse_count < 2 and keystroke_count > 0:
            # User submitted form without touching mouse or cursor movement (possible programmatic submit)
            reasons.append("Zero Pre-Submit Cursor Dynamics: Instantaneous form submission without physical pointer approach")
            fraud_score += 0.35

        # Cap bot probability
        bot_probability = min(1.0, round(fraud_score, 3))
        is_bot = bot_probability >= 0.65

        return is_bot, bot_probability, reasons

