"""
Unit tests for BiometricModel: Genuine User Acceptance vs Impostor Rejection
"""

import numpy as np
from backend.biometrics.model import BiometricModel


def test_genuine_vs_impostor_classification():
    model = BiometricModel()

    # User A baseline profile: Fast typist, low dwell, high flight consistency
    # Vector: [dwell, std_dwell, flight, std_flight, cps, curvature, vel_var_log, click_duration]
    user_a_samples = [
        np.array([75.0, 12.0, 110.0, 20.0, 4.8, 1.25, 4.2, 85.0]),
        np.array([78.0, 14.0, 108.0, 22.0, 4.6, 1.22, 4.0, 80.0]),
        np.array([73.0, 11.0, 115.0, 19.0, 4.9, 1.28, 4.5, 90.0]),
    ]
    digraph_samples = [
        {"p->a": 110.0, "a->s": 95.0, "s->s": 105.0}
    ]

    profile = model.fit_profile(user_a_samples, digraph_samples)

    # 1. Test Genuine User Login (slight natural variance)
    genuine_attempt = np.array([76.0, 13.0, 112.0, 21.0, 4.7, 1.24, 4.1, 82.0])
    genuine_digraphs = {"p->a": 112.0, "a->s": 98.0, "s->s": 102.0}

    is_gen, conf_gen, metrics_gen = model.evaluate_attempt(genuine_attempt, genuine_digraphs, profile)
    assert is_gen is True
    assert conf_gen >= 70.0, f"Genuine user should have high confidence, got {conf_gen}"

    # 2. Test Impostor (e.g. slow, heavy keys, long pauses)
    impostor_attempt = np.array([160.0, 45.0, 310.0, 80.0, 1.8, 1.05, 1.2, 190.0])
    impostor_digraphs = {"p->a": 320.0, "a->s": 290.0, "s->s": 300.0}

    is_imp, conf_imp, metrics_imp = model.evaluate_attempt(impostor_attempt, impostor_digraphs, profile)
    assert is_imp is False
    assert conf_imp < 50.0, f"Impostor should be rejected with low confidence, got {conf_imp}"
