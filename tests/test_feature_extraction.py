"""
Unit tests for FeatureExtractor: Keystroke and Mouse Dynamics
"""

import pytest
import numpy as np
from backend.biometrics.feature_extractor import FeatureExtractor


def test_keystroke_feature_extraction():
    extractor = FeatureExtractor()

    # Simulate typing 'abc' with 80ms dwell and 100ms flight
    events = [
        {"type": "keydown", "key": "a", "timestamp": 1000.0, "isTrusted": True},
        {"type": "keyup", "key": "a", "timestamp": 1080.0, "isTrusted": True},
        {"type": "keydown", "key": "b", "timestamp": 1180.0, "isTrusted": True},
        {"type": "keyup", "key": "b", "timestamp": 1260.0, "isTrusted": True},
        {"type": "keydown", "key": "c", "timestamp": 1360.0, "isTrusted": True},
        {"type": "keyup", "key": "c", "timestamp": 1440.0, "isTrusted": True},
    ]

    features = extractor.extract_keystroke_features(events)

    assert features["keystroke_count"] == 3
    assert abs(features["mean_dwell"] - 80.0) < 1.0
    assert abs(features["mean_flight_ud"] - 100.0) < 1.0
    assert features["backspace_count"] == 0
    assert "a->b" in features["digraph_latencies"]


def test_mouse_curvature_extraction():
    extractor = FeatureExtractor()

    # Curved human-like trajectory: (0,0) -> (50, 40) -> (100, 0)
    events = [
        {"type": "mousemove", "x": 0.0, "y": 0.0, "timestamp": 1000.0, "isTrusted": True},
        {"type": "mousemove", "x": 50.0, "y": 40.0, "timestamp": 1050.0, "isTrusted": True},
        {"type": "mousemove", "x": 100.0, "y": 0.0, "timestamp": 1100.0, "isTrusted": True},
        {"type": "mousedown", "x": 100.0, "y": 0.0, "timestamp": 1120.0, "isTrusted": True},
        {"type": "mouseup", "x": 100.0, "y": 0.0, "timestamp": 1200.0, "isTrusted": True},
    ]

    m_features = extractor.extract_mouse_features(events)

    assert m_features["euclidean_distance"] == 100.0
    assert m_features["curvature_ratio"] > 1.15  # Curved, not straight
    assert abs(m_features["click_duration"] - 80.0) < 1.0

