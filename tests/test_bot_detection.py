"""
Unit tests for BotDetector: Synthetic events, linear paths, and inhuman metrics
"""

from backend.biometrics.bot_detector import BotDetector


def test_synthetic_untrusted_dom_event():
    detector = BotDetector()

    raw_keys = [
        {"type": "keydown", "key": "a", "timestamp": 100.0, "isTrusted": False},
        {"type": "keyup", "key": "a", "timestamp": 180.0, "isTrusted": False}
    ]
    raw_mouse = []
    k_feat = {"mean_dwell": 80.0, "std_dwell": 12.0, "mean_flight_ud": 90.0, "std_flight_ud": 15.0, "cps": 4.0, "keystroke_count": 2}
    m_feat = {"mouse_event_count": 0, "curvature_ratio": 1.0, "jitter_variance": 0.0}

    is_bot, prob, reasons = detector.evaluate_telemetry(raw_keys, raw_mouse, k_feat, m_feat, {})
    assert is_bot is True
    assert prob >= 0.8
    assert any("Synthetic DOM Event" in r for r in reasons)


def test_zero_variance_scripted_typing():
    detector = BotDetector()

    raw_keys = [{"type": "keydown", "key": "a", "timestamp": 100.0, "isTrusted": True}] * 10
    raw_mouse = [{"type": "mousemove", "x": 10.0, "y": 10.0, "timestamp": 100.0, "isTrusted": True}] * 5
    # Near-zero standard deviations typical of programmed delay
    k_feat = {
        "mean_dwell": 100.0,
        "std_dwell": 0.2,
        "mean_flight_ud": 50.0,
        "std_flight_ud": 0.1,
        "cps": 5.0,
        "keystroke_count": 10
    }
    m_feat = {"mouse_event_count": 5, "curvature_ratio": 1.1, "jitter_variance": 0.05}

    is_bot, prob, reasons = detector.evaluate_telemetry(raw_keys, raw_mouse, k_feat, m_feat, {})
    assert is_bot is True
    assert any("Synthetic Timing Precision" in r for r in reasons)

