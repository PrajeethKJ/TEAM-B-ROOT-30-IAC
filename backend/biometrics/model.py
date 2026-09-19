"""
BioPrint: Hybrid Behavioral Biometric Matcher
Combines Mahalanobis Distance Metric, Feature Z-Scoring, and One-Class Anomaly Modeling.
"""

import math
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import IsolationForest


class BiometricModel:
    def __init__(self):
        pass

    def fit_profile(self, feature_vectors: List[np.ndarray], digraph_samples: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Fits a user biometric baseline profile from K enrollment samples.
        """
        X = np.array(feature_vectors, dtype=np.float64)  # Shape: (K, D)
        k, d = X.shape

        mean_vec = np.mean(X, axis=0)
        # Regularized standard deviation (avoid division by zero if user is very consistent)
        std_vec = np.std(X, axis=0)
        min_stds = np.array([5.0, 2.0, 8.0, 3.0, 0.3, 0.05, 0.2, 8.0])  # baseline minimum variances
        std_vec = np.maximum(std_vec, min_stds[:d])

        # Compute Covariance Matrix and Regularized Pseudo-Inverse for Mahalanobis Distance
        if k > 1:
            cov = np.cov(X, rowvar=False)
            # Add Ridge regularization to diagonal
            cov_reg = cov + np.eye(d) * (np.diag(cov) * 0.15 + 1e-4)
            inv_cov = np.linalg.pinv(cov_reg)
        else:
            inv_cov = np.diag(1.0 / (std_vec ** 2))

        # Synthesize boundary noise samples to calibrate One-Class Isolation Forest
        rng = np.random.default_rng(42)
        synthetic_genuine = mean_vec + rng.normal(0, std_vec * 0.8, size=(20, d))
        X_train = np.vstack([X, synthetic_genuine])

        iso_forest = IsolationForest(
            n_estimators=50,
            contamination=0.08,
            random_state=42
        )
        iso_forest.fit(X_train)

        # Baseline digraphs average
        all_digraph_keys = set()
        for s in digraph_samples:
            all_digraph_keys.update(s.keys())

        avg_digraphs: Dict[str, float] = {}
        for dk in all_digraph_keys:
            vals = [s[dk] for s in digraph_samples if dk in s]
            avg_digraphs[dk] = float(np.mean(vals))

        return {
            "mean": mean_vec.tolist(),
            "std": std_vec.tolist(),
            "inv_cov": inv_cov.tolist(),
            "avg_digraphs": avg_digraphs,
            "sample_count": k,
            "_iso_model": iso_forest
        }

    def evaluate_attempt(
        self,
        attempt_vector: np.ndarray,
        attempt_digraphs: Dict[str, float],
        profile_data: Dict[str, Any]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluates a live login attempt vector against the enrolled user baseline.
        Returns:
            is_genuine (bool): True if verified as genuine user.
            confidence_score (float): 0.0 to 100.0 confidence.
            metrics (Dict[str, Any]): Detailed sub-scores for explainability and HUD.
        """
        mean_vec = np.array(profile_data["mean"], dtype=np.float64)
        std_vec = np.array(profile_data["std"], dtype=np.float64)
        inv_cov = np.array(profile_data["inv_cov"], dtype=np.float64)
        x = np.array(attempt_vector, dtype=np.float64)

        d = len(x)
        diff = x - mean_vec

        # 1. Z-Score Normalized Manhattan Distance
        z_scores = np.abs(diff) / std_vec
        mean_z = float(np.mean(z_scores))
        z_score_distance = round(mean_z, 2)

        # 2. Mahalanobis Distance
        mahalanobis_sq = float(np.dot(np.dot(diff, inv_cov), diff.T))
        mahalanobis_dist = math.sqrt(max(0.0, mahalanobis_sq))

        # 3. Digraph Latency Matching
        baseline_digraphs = profile_data.get("avg_digraphs", {})
        shared_keys = [k for k in attempt_digraphs if k in baseline_digraphs]
        digraph_divergence = 0.0

        if shared_keys:
            errs = []
            for k in shared_keys:
                base_lat = baseline_digraphs[k]
                att_lat = attempt_digraphs[k]
                errs.append(abs(att_lat - base_lat) / max(30.0, base_lat))
            digraph_divergence = float(np.mean(errs))

        # 4. Isolation Forest Score
        iso_model = profile_data.get("_iso_model")
        iso_anomaly_score = 0.5
        if iso_model is not None:
            # decision_function yields higher values for inliers (around 0.0 to 0.2), negative for outliers
            raw_iso = float(iso_model.decision_function(x.reshape(1, -1))[0])
            # normalize to [0, 1] where 0 is perfect inlier, 1 is total anomaly
            iso_anomaly_score = max(0.0, min(1.0, 0.5 - (raw_iso * 2.0)))

        # 5. Composite Non-Linear Score Fusion
        # Expected baseline mahalanobis for D=8 dimensions is ~ sqrt(8) ~= 2.82
        # Normal human deviation stays below 3.5; impostors usually shoot past 6.0
        norm_maha = 1.0 / (1.0 + math.exp(-0.8 * (mahalanobis_dist - 3.2)))
        norm_z = 1.0 / (1.0 + math.exp(-1.4 * (mean_z - 1.8)))
        norm_di = min(1.0, digraph_divergence * 1.5)

        composite_anomaly = (
            0.40 * norm_maha +
            0.30 * norm_z +
            0.15 * iso_anomaly_score +
            0.15 * norm_di
        )

        confidence_pct = max(0.0, min(100.0, (1.0 - composite_anomaly) * 100.0))
        confidence_pct = round(confidence_pct, 1)

        # Threshold decision (>= 60.0% passes genuine check)
        is_genuine = confidence_pct >= 60.0

        metrics = {
            "confidence_score": confidence_pct,
            "mahalanobis_distance": round(mahalanobis_dist, 2),
            "z_score_distance": z_score_distance,
            "isolation_score": round(iso_anomaly_score, 2),
            "digraph_divergence": round(digraph_divergence, 3),
            "feature_z_breakdown": {
                "dwell_time": round(float(z_scores[0]), 2),
                "dwell_consistency": round(float(z_scores[1]), 2),
                "flight_time": round(float(z_scores[2]), 2),
                "flight_consistency": round(float(z_scores[3]), 2),
                "typing_speed": round(float(z_scores[4]), 2),
                "mouse_curvature": round(float(z_scores[5]), 2),
                "mouse_velocity": round(float(z_scores[6]), 2),
                "click_dwell": round(float(z_scores[7]), 2)
            }
        }

        return is_genuine, confidence_pct, metrics
