"""
BioPrint: Adaptive Profile Drift Updater
Implements Stretch Goal 1: Gently updates baseline biometric models over time
as genuine users naturally experience circadian, fatigue, or motor drift.
"""

import numpy as np
from typing import Dict, Any


class AdaptiveProfileUpdater:
    def __init__(self, alpha: float = 0.08, min_confidence: float = 75.0):
        self.alpha = alpha  # Smoothing / learning rate
        self.min_confidence = min_confidence

    def update_profile(
        self,
        profile_data: Dict[str, Any],
        confirmed_vector: np.ndarray,
        confirmed_digraphs: Dict[str, float],
        confidence_score: float
    ) -> Dict[str, Any]:
        """
        Applies Exponential Moving Average (EMA) update to the user's mean baseline.
        Only called when login was authenticated with high genuine confidence.
        """
        if confidence_score < self.min_confidence:
            return profile_data  # Do not drift on borderline logins

        old_mean = np.array(profile_data["mean"], dtype=np.float64)
        old_std = np.array(profile_data["std"], dtype=np.float64)
        x = np.array(confirmed_vector, dtype=np.float64)

        # EMA update for mean vector
        new_mean = (1.0 - self.alpha) * old_mean + self.alpha * x

        # Gentle update for variance
        diff = np.abs(x - new_mean)
        new_std = (1.0 - self.alpha) * old_std + self.alpha * diff

        # Ensure minimum bounds
        min_stds = np.array([5.0, 2.0, 8.0, 3.0, 0.3, 0.05, 0.2, 8.0])
        new_std = np.maximum(new_std, min_stds[:len(new_std)])

        # Update digraphs
        avg_digraphs = dict(profile_data.get("avg_digraphs", {}))
        for k, v in confirmed_digraphs.items():
            if k in avg_digraphs:
                avg_digraphs[k] = round((1.0 - self.alpha) * avg_digraphs[k] + self.alpha * v, 2)
            else:
                avg_digraphs[k] = round(v, 2)

        profile_data["mean"] = [round(float(val), 2) for val in new_mean]
        profile_data["std"] = [round(float(val), 2) for val in new_std]
        profile_data["avg_digraphs"] = avg_digraphs
        profile_data["sample_count"] = profile_data.get("sample_count", 0) + 1
        profile_data["last_drift_update"] = confidence_score

        return profile_data
