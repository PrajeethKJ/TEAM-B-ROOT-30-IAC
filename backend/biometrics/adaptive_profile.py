"""
BioPrint: Adaptive Profile Drift Updater (patched)
Implements Stretch Goal 1: Gently updates baseline biometric models over time
as genuine users naturally experience circadian, fatigue, or motor drift.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .model import BiometricModel


class AdaptiveProfileUpdater:
    def __init__(self, alpha: float = 0.08, min_confidence: float = 75.0):
        self.alpha = alpha  # Smoothing / learning rate
        self.min_confidence = min_confidence

    def update_profile(
        self,
        profile_data: Dict[str, Any],
        confirmed_vector: np.ndarray,
        confirmed_digraphs: Dict[str, float],
        confidence_score: float,
        confirmed_mask: Optional[Sequence[bool]] = None,
    ) -> Dict[str, Any]:
        """
        Applies Exponential Moving Average (EMA) update to the user's baseline.
        Only called when login was authenticated with high genuine confidence.
        """
        if confidence_score < self.min_confidence:
            return profile_data  # Do not drift on borderline logins

        old_mean = np.array(profile_data.get("mean", []), dtype=np.float64)
        old_std = np.array(profile_data.get("std", []), dtype=np.float64)
        x = np.array(confirmed_vector, dtype=np.float64)

        if old_mean.size == x.size:
            # EMA update for legacy raw-space mean vector
            new_mean = (1.0 - self.alpha) * old_mean + self.alpha * x

            # Update for variance
            diff = np.abs(x - new_mean)
            new_std = (1.0 - self.alpha) * old_std + self.alpha * diff

            # Ensure minimum bounds
            min_stds = np.array([5.0, 2.0, 8.0, 3.0, 0.3, 0.05, 0.2, 8.0])
            new_std = np.maximum(new_std, min_stds[:len(new_std)])

            profile_data["mean"] = [round(float(val), 2) for val in new_mean]
            profile_data["std"] = [round(float(val), 2) for val in new_std]

        # Update digraphs
        avg_digraphs = dict(profile_data.get("avg_digraphs", {}))
        for k, v in confirmed_digraphs.items():
            if k in avg_digraphs:
                avg_digraphs[k] = round((1.0 - self.alpha) * avg_digraphs[k] + self.alpha * v, 2)
            else:
                avg_digraphs[k] = round(v, 2)

        profile_data["avg_digraphs"] = avg_digraphs
        profile_data["sample_count"] = profile_data.get("sample_count", 0) + 1
        profile_data["last_drift_update"] = confidence_score

        # Update transformed tz space and cov matrix with bounded drift
        profile_data = BiometricModel.drift_update(
            profile_data, confirmed_vector, mask=confirmed_mask, alpha=self.alpha
        )

        return profile_data
