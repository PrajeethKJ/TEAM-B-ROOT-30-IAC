"""
BioPrint: Behavioural Biometric Matcher  (patched)

Changes over the shipped version
--------------------------------
1. Isolation Forest removed. It was trained on the 3 real enrolment points plus
   20 synthetic points drawn from those points' own mean and std, so it could
   only rediscover the Gaussian it was handed; and it lived in a process-local
   dict, so after a server restart every user silently scored a constant 0.5 and
   a chunk of borderline decisions flipped.
2. Timing features are modelled in log space. Dwell and flight times are
   right-skewed and roughly log-normal; in raw space a user who is 40 ms slower
   than usual on a 90 ms baseline and one who is 40 ms slower on a 300 ms
   baseline get the same z, which is wrong in both directions.
3. The covariance is shrunk toward a diagonal target with a fixed per-feature
   prior. With K=3 samples in 8 dimensions the sample covariance has rank <= 2,
   so the shipped ridge term was doing essentially all of the work while looking
   like a real covariance. Shrinkage intensity scales with K, so the model
   genuinely improves as the user enrols more samples.
4. Missing modalities are marginalised out, not zero-filled. Scoring restricts
   the Gaussian to the dimensions actually measured and re-inverts that
   submatrix, so a keyboard-only login is scored on keyboard evidence alone.
5. Per-position timing model: the attempt's per-keystroke dwell/flight sequence
   is compared against the enrolled per-position distribution. This is the term
   that separates a speed-matched mimic from the account owner - the mimic can
   match your average rate without matching where your pauses fall.
6. Per-user calibration by leave-one-out. Enrolment distances are recomputed
   with each sample held out; the user's own LOO scale sets the operating point,
   so the threshold means the same thing for a consistent typist and an erratic
   one. The hard-coded sigmoid centre at 3.2 is gone.

Profile dicts stay JSON-serialisable and keep "mean", "std", "inv_cov",
"avg_digraphs" and "sample_count" so explainability.py, adaptive_profile.py and
any already-stored profile keep working.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURE_DIM = 8
PROFILE_VERSION = 2
DEFAULT_THRESHOLD = 60.0

# Transform applied to each dimension before modelling.
# "log"  -> log1p(x)          (positive, right-skewed)
# "slog" -> sign(x)*log1p|x|  (may be negative: key overlap)
# "id"   -> unchanged
_TRANSFORMS = ["log", "log", "slog", "log", "log", "log", "id", "log"]

# Floor on the within-user standard deviation in transformed space. These are
# priors, not fitted values: they stop a user who happened to be very consistent
# across 3 samples from getting an implausibly tight distribution.
_PRIOR_STD = np.array([0.18, 0.40, 0.30, 0.40, 0.15, 0.10, 0.35, 0.25])

# Raw-space std floors, retained so the legacy "std" field stays comparable
# with stored profiles and with explainability.py.
_RAW_MIN_STD = np.array([5.0, 2.0, 8.0, 3.0, 0.3, 0.05, 0.2, 8.0])

_DIST_REF_FLOOR = 0.45
_DIST_REF_CEIL = 3.00
_SIGMOID_SLOPE = 2.20
_SIGMOID_CENTRE = 1.75

_FEATURE_LABELS = [
    "dwell_time", "dwell_consistency", "flight_time", "flight_consistency",
    "typing_speed", "mouse_curvature", "mouse_velocity", "click_dwell",
]


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def transform_vector(v: Sequence[float]) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    out = np.empty_like(x)
    for i, kind in enumerate(_TRANSFORMS[: len(x)]):
        xi = float(x[i])
        if kind == "log":
            out[i] = math.log1p(max(0.0, xi))
        elif kind == "slog":
            out[i] = math.copysign(math.log1p(abs(xi)), xi)
        else:
            out[i] = xi
    return out


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# --------------------------------------------------------------------------- #
# argument shim
# --------------------------------------------------------------------------- #
def _resolve(args: Tuple, kwargs: Dict[str, Any], names: List[str]) -> Dict[str, Any]:
    """Accept both call styles for fit_profile / evaluate_attempt.

    app.py calls these positionally with `username` first; the shipped unit tests
    call them without it. Rather than break one of the two, detect a leading
    string and shift. Delete this shim once both call sites agree.
    """
    args = list(args)
    out: Dict[str, Any] = {n: None for n in names}
    if args and isinstance(args[0], str):
        out["username"] = args.pop(0)
    rest = [n for n in names if n != "username"]
    for name, val in zip(rest, args):
        out[name] = val
    for k, v in kwargs.items():
        if k in out:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class BiometricModel:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = float(threshold)

    # ---------------------------- enrolment --------------------------- #
    def fit_profile(self, *args, **kwargs) -> Dict[str, Any]:
        """Fit a baseline from K enrolment samples.

        fit_profile(feature_vectors, digraph_samples, masks=None, sequences=None)
        fit_profile(username, feature_vectors, digraph_samples, ...)   # also ok
        """
        a = _resolve(args, kwargs,
                     ["username", "feature_vectors", "digraph_samples", "masks", "sequences"])
        vectors = a["feature_vectors"]
        digraph_samples = a["digraph_samples"] or []
        masks = a["masks"]
        sequences = a["sequences"] or []

        if vectors is None or len(vectors) == 0:
            raise ValueError("fit_profile requires at least one feature vector")

        X = np.asarray([np.asarray(v, dtype=np.float64) for v in vectors], dtype=np.float64)
        k, d = X.shape
        if masks is None:
            M = np.ones((k, d), dtype=bool)
        else:
            M = np.asarray([np.asarray(m, dtype=bool) for m in masks], dtype=bool)

        Z = np.vstack([transform_vector(row) for row in X])

        tz_mean, tz_std, usable = self._fit_moments(Z, M)
        tz_cov = self._shrunk_cov(Z, M, tz_mean, tz_std, usable)

        # Legacy raw-space summaries, kept for explainability / adaptive / HUD.
        raw_mean = _masked_mean(X, M, fallback=X.mean(axis=0))
        raw_std = np.maximum(_masked_std(X, M), _RAW_MIN_STD[:d])
        inv_cov = np.diag(1.0 / (raw_std ** 2))

        digraph_mean, digraph_std = self._fit_digraphs(digraph_samples)
        position = self._fit_positions(sequences)

        profile: Dict[str, Any] = {
            "version": PROFILE_VERSION,
            "mean": [round(float(v), 3) for v in raw_mean],
            "std": [round(float(v), 3) for v in raw_std],
            "inv_cov": [[round(float(c), 6) for c in row] for row in inv_cov],
            "avg_digraphs": digraph_mean,
            "digraph_std": digraph_std,
            "sample_count": int(k),
            "tz": {
                "mean": [round(float(v), 6) for v in tz_mean],
                "std": [round(float(v), 6) for v in tz_std],
                "cov": [[round(float(c), 8) for c in row] for row in tz_cov],
                "usable": [bool(u) for u in usable],
            },
            "position": position,
        }

        profile["calibration"] = self._calibrate(
            Z=Z, M=M, digraph_samples=digraph_samples, sequences=sequences, profile=profile
        )
        return profile

    # ---------------------------- verification ------------------------ #
    def evaluate_attempt(self, *args, **kwargs) -> Tuple[bool, float, Dict[str, Any]]:
        """Score a live attempt against an enrolled baseline.

        evaluate_attempt(attempt_vector, attempt_digraphs, profile_data,
                         mask=None, sequences=None, threshold=None)
        evaluate_attempt(username, attempt_vector, attempt_digraphs, profile_data, ...)
        """
        a = _resolve(args, kwargs,
                     ["username", "attempt_vector", "attempt_digraphs", "profile_data",
                      "mask", "sequences", "threshold"])
        x_raw = np.asarray(a["attempt_vector"], dtype=np.float64)
        attempt_digraphs = a["attempt_digraphs"] or {}
        profile = a["profile_data"] or {}
        mask = a["mask"]
        sequences = a["sequences"]

        profile = self._ensure_v2(profile, len(x_raw))
        cal = profile.get("calibration", {})
        threshold = float(a["threshold"] if a["threshold"] is not None
                          else cal.get("threshold", self.threshold))

        d = len(x_raw)
        mask = (np.ones(d, dtype=bool) if mask is None
                else np.asarray(mask, dtype=bool)[:d])

        z = transform_vector(x_raw)
        tz = profile["tz"]
        mu = np.asarray(tz["mean"], dtype=np.float64)
        sd = np.asarray(tz["std"], dtype=np.float64)
        cov = np.asarray(tz["cov"], dtype=np.float64)
        usable = np.asarray(tz["usable"], dtype=bool)

        active = mask & usable
        diff = z - mu

        # --- 1. marginalised Mahalanobis over measured dimensions --------
        maha_norm, maha_raw = _mahalanobis(diff, cov, active)

        # --- 2. per-dimension z (transformed space), for explainability ---
        z_scores = np.abs(diff) / np.maximum(sd, 1e-6)
        z_scores[~active] = 0.0
        mean_z = float(z_scores[active].mean()) if active.any() else 0.0

        # --- 3. per-position timing ---------------------------------------
        pos_norm, pos_n = _position_distance(sequences, profile.get("position"))

        # --- 4. digraph latency -------------------------------------------
        di_norm, di_n = _digraph_distance(attempt_digraphs,
                                          profile.get("avg_digraphs", {}),
                                          profile.get("digraph_std", {}))

        # --- 5. calibrated fusion ------------------------------------------
        terms: List[Tuple[float, float]] = []   # (weight, anomaly in [0,1])
        terms.append((0.45, _anomaly(maha_norm, cal.get("d_ref_maha"))))
        terms.append((0.12, _anomaly(mean_z, cal.get("d_ref_z"))))
        if pos_n >= 3:
            terms.append((0.28, _anomaly(pos_norm, cal.get("d_ref_pos"))))
        if di_n >= 2:
            terms.append((0.15, _anomaly(di_norm, cal.get("d_ref_di"))))

        w_total = sum(w for w, _ in terms)
        composite = sum(w * v for w, v in terms) / w_total

        confidence = round(max(0.0, min(100.0, (1.0 - composite) * 100.0)), 1)
        is_genuine = confidence >= threshold

        metrics = {
            "confidence_score": confidence,
            "threshold": round(threshold, 1),
            "mahalanobis_distance": round(maha_raw, 2),
            "mahalanobis_normalised": round(maha_norm, 3),
            "z_score_distance": round(mean_z, 2),
            "position_divergence": round(pos_norm, 3),
            "position_points": pos_n,
            "digraph_divergence": round(di_norm, 3),
            "digraph_matches": di_n,
            "dimensions_scored": int(active.sum()),
            "dimensions_missing": [_FEATURE_LABELS[i] for i in range(d) if not active[i]],
            "feature_z_breakdown": {
                _FEATURE_LABELS[i]: round(float(z_scores[i]), 2) for i in range(min(d, 8))
            },
        }
        return is_genuine, confidence, metrics

    # Safe subset for the client. The shipped build returned every per-feature
    # z-score and baseline value on a blocked attempt, which tells an attacker
    # exactly which dial to turn.
    @staticmethod
    def public_metrics(metrics: Dict[str, Any], authenticated: bool) -> Dict[str, Any]:
        if authenticated:
            return {"confidence_score": metrics.get("confidence_score")}
        return {"confidence_score": None}

    # ------------------------------------------------------------------ #
    # fitting internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fit_moments(Z: np.ndarray, M: np.ndarray):
        k, d = Z.shape
        mean = np.zeros(d)
        std = np.zeros(d)
        usable = np.zeros(d, dtype=bool)
        for j in range(d):
            col = Z[M[:, j], j]
            if col.size == 0:
                mean[j] = 0.0
                std[j] = _PRIOR_STD[j] if j < len(_PRIOR_STD) else 0.3
                usable[j] = False
                continue
            mean[j] = float(col.mean())
            s = float(col.std(ddof=1)) if col.size > 1 else 0.0
            prior = _PRIOR_STD[j] if j < len(_PRIOR_STD) else 0.3
            # Blend the sample std toward the prior; with few samples the prior
            # dominates, with many the data does.
            n = col.size
            w = n / (n + 4.0)
            std[j] = max(1e-3, w * s + (1.0 - w) * prior)
            usable[j] = col.size >= 2
        return mean, std, usable

    @staticmethod
    def _shrunk_cov(Z: np.ndarray, M: np.ndarray, mean: np.ndarray,
                    std: np.ndarray, usable: np.ndarray) -> np.ndarray:
        k, d = Z.shape
        target = np.diag(std ** 2)
        complete = M.all(axis=1)
        n_complete = int(complete.sum())
        if n_complete >= 3:
            S = np.cov(Z[complete], rowvar=False)
            S = np.atleast_2d(S)
        else:
            S = np.zeros((d, d))
        # Shrinkage intensity: heavy when samples are few relative to dimensions.
        lam = 1.0 / (1.0 + max(0.0, n_complete - 1) / max(1.0, float(d)))
        lam = float(min(0.95, max(0.15, lam)))
        cov = (1.0 - lam) * S + lam * target
        # Blank out unusable dimensions so they never contribute correlation.
        for j in range(d):
            if not usable[j]:
                cov[j, :] = 0.0
                cov[:, j] = 0.0
                cov[j, j] = max(target[j, j], 1e-6)
        cov += np.eye(d) * 1e-8
        return cov

    @staticmethod
    def _fit_digraphs(samples: List[Dict[str, float]]):
        keys = set()
        for s in samples or []:
            keys.update((s or {}).keys())
        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}
        for dk in keys:
            vals = [float(s[dk]) for s in samples if s and dk in s and float(s[dk]) > 0]
            if not vals:
                continue
            logs = np.log([max(1.0, v) for v in vals])
            mean[dk] = round(float(np.exp(logs.mean())), 2)
            s = float(logs.std(ddof=1)) if len(logs) > 1 else 0.0
            n = len(logs)
            w = n / (n + 4.0)
            std[dk] = round(max(0.12, w * s + (1.0 - w) * 0.35), 4)
        return mean, std

    @staticmethod
    def _fit_positions(sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Per-keystroke-position log-timing distribution."""
        seqs = [s for s in (sequences or []) if s and s.get("keys")]
        if len(seqs) < 2:
            return {"keys": [], "dwell_mean": [], "dwell_std": [],
                    "flight_mean": [], "flight_std": [], "n": 0}

        # Only model positions shared by the majority of samples, and require the
        # key identity to agree so we are not averaging across different strings.
        length = int(np.median([len(s["keys"]) for s in seqs]))
        seqs = [s for s in seqs if len(s["keys"]) == length]
        if len(seqs) < 2 or length < 2:
            return {"keys": [], "dwell_mean": [], "dwell_std": [],
                    "flight_mean": [], "flight_std": [], "n": 0}

        keys = list(seqs[0]["keys"])
        for s in seqs:
            if list(s["keys"]) != keys:
                return {"keys": [], "dwell_mean": [], "dwell_std": [],
                        "flight_mean": [], "flight_std": [], "n": 0}

        def moments(field: str, n_pos: int):
            mu, sd = [], []
            for i in range(n_pos):
                vals = []
                for s in seqs:
                    arr = s.get(field) or []
                    if i < len(arr) and arr[i] is not None:
                        vals.append(float(arr[i]))
                if len(vals) < 2:
                    mu.append(None)
                    sd.append(None)
                    continue
                logs = np.array([math.copysign(math.log1p(abs(v)), v) for v in vals])
                n = len(logs)
                w = n / (n + 4.0)
                raw_sd = float(logs.std(ddof=1))
                mu.append(round(float(logs.mean()), 5))
                sd.append(round(max(0.18, w * raw_sd + (1.0 - w) * 0.45), 5))
            return mu, sd

        dm, ds = moments("dwell", length)
        fm, fs = moments("flight", length - 1)
        return {"keys": keys, "dwell_mean": dm, "dwell_std": ds,
                "flight_mean": fm, "flight_std": fs, "n": len(seqs)}

    # ---------------------------- calibration ------------------------- #
    def _calibrate(self, Z, M, digraph_samples, sequences, profile) -> Dict[str, Any]:
        """Leave-one-out enrolment distances set this user's operating scale."""
        k = Z.shape[0]
        maha, zs, pos, di = [], [], [], []

        for j in range(k):
            idx = [i for i in range(k) if i != j]
            if len(idx) < 2:
                continue
            sub_mean, sub_std, sub_usable = self._fit_moments(Z[idx], M[idx])
            sub_cov = self._shrunk_cov(Z[idx], M[idx], sub_mean, sub_std, sub_usable)
            active = M[j] & sub_usable
            diff = Z[j] - sub_mean
            mn, _ = _mahalanobis(diff, sub_cov, active)
            maha.append(mn)
            zz = np.abs(diff) / np.maximum(sub_std, 1e-6)
            zs.append(float(zz[active].mean()) if active.any() else 0.0)

            if digraph_samples and len(digraph_samples) == k:
                sub_dm, sub_ds = self._fit_digraphs([digraph_samples[i] for i in idx])
                dn, dn_n = _digraph_distance(digraph_samples[j], sub_dm, sub_ds)
                if dn_n >= 2:
                    di.append(dn)
            if sequences and len(sequences) == k:
                sub_pos = self._fit_positions([sequences[i] for i in idx])
                pn, pn_n = _position_distance(sequences[j], sub_pos)
                if pn_n >= 3:
                    pos.append(pn)

        def ref(vals, default):
            if not vals:
                return default
            v = float(np.median(vals))
            return round(float(min(_DIST_REF_CEIL, max(_DIST_REF_FLOOR, v))), 4)

        return {
            "d_ref_maha": ref(maha, 1.0),
            "d_ref_z": ref(zs, 1.0),
            "d_ref_pos": ref(pos, 1.0),
            "d_ref_di": ref(di, 1.0),
            "loo_maha": [round(float(v), 4) for v in maha],
            "loo_samples": len(maha),
            "threshold": DEFAULT_THRESHOLD,
        }

    # ---------------------------- compatibility ----------------------- #
    def _ensure_v2(self, profile: Dict[str, Any], d: int) -> Dict[str, Any]:
        """Upgrade a v1 profile (mean/std/inv_cov only) so old data still scores.

        Accuracy on an upgraded profile is worse than on a re-enrolled one, since
        the raw-space moments are all that survive. Re-enrol users when you can.
        """
        if profile.get("tz"):
            return profile
        p = dict(profile)
        raw_mean = np.asarray(p.get("mean", [0.0] * d), dtype=np.float64)[:d]
        raw_std = np.asarray(p.get("std", list(_RAW_MIN_STD[:d])), dtype=np.float64)[:d]
        tz_mean = transform_vector(raw_mean)
        # Delta method: sd_log ~ sd_raw / (1 + mean_raw)
        tz_std = np.maximum(raw_std / np.maximum(1.0 + np.abs(raw_mean), 1e-6),
                            _PRIOR_STD[:d])
        p["tz"] = {
            "mean": [float(v) for v in tz_mean],
            "std": [float(v) for v in tz_std],
            "cov": np.diag(tz_std ** 2).tolist(),
            "usable": [True] * d,
        }
        p.setdefault("position", {"keys": [], "n": 0})
        p.setdefault("digraph_std", {})
        p.setdefault("calibration", {"d_ref_maha": 1.0, "d_ref_z": 1.0,
                                     "d_ref_pos": 1.0, "d_ref_di": 1.0,
                                     "threshold": DEFAULT_THRESHOLD})
        return p

    # Called by the adaptive updater so drift also reaches the scoring space.
    @staticmethod
    def drift_update(profile: Dict[str, Any], vector: Sequence[float],
                     mask: Optional[Sequence[bool]] = None,
                     alpha: float = 0.08, max_drift_sd: float = 2.0) -> Dict[str, Any]:
        """EMA drift in transformed space, capped relative to the enrolment baseline."""
        tz = profile.get("tz")
        if not tz:
            return profile
        mu = np.asarray(tz["mean"], dtype=np.float64)
        sd = np.asarray(tz["std"], dtype=np.float64)
        z = transform_vector(vector)
        m = np.ones_like(mu, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)[:len(mu)]

        anchor = np.asarray(profile.setdefault("tz_anchor", mu.tolist()), dtype=np.float64)
        new_mu = mu.copy()
        new_mu[m] = (1.0 - alpha) * mu[m] + alpha * z[m]
        # Hard cap: never let drift walk more than max_drift_sd from enrolment.
        lo, hi = anchor - max_drift_sd * sd, anchor + max_drift_sd * sd
        new_mu = np.clip(new_mu, lo, hi)

        cov = np.asarray(tz["cov"], dtype=np.float64)
        resid = (z - new_mu)
        for j in np.flatnonzero(m):
            # Update the variance with the squared residual, not |residual|.
            # The shipped adaptive updater fed a mean absolute deviation into a
            # slot that is read as a standard deviation, shrinking variance by
            # about 0.8x on every accepted login.
            cov[j, j] = max((1.0 - alpha) * cov[j, j] + alpha * float(resid[j] ** 2),
                            float(sd[j] ** 2) * 0.25)
        tz["mean"] = [float(v) for v in new_mu]
        tz["cov"] = cov.tolist()
        profile["tz"] = tz
        return profile


# --------------------------------------------------------------------------- #
# distance helpers
# --------------------------------------------------------------------------- #
def _mahalanobis(diff: np.ndarray, cov: np.ndarray, active: np.ndarray) -> Tuple[float, float]:
    """Marginalise to the active dimensions, then invert that submatrix.

    Returns (per-dimension normalised distance, raw distance). Normalising by
    sqrt(n_active) is what makes a 5-dimension keyboard-only login comparable
    with a full 8-dimension one.
    """
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return 0.0, 0.0
    sub = cov[np.ix_(idx, idx)]
    dv = diff[idx]
    try:
        sol = np.linalg.solve(sub, dv)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(sub) @ dv
    d2 = float(dv @ sol)
    d2 = max(0.0, d2)
    return math.sqrt(d2 / idx.size), math.sqrt(d2)


def _position_distance(sequences: Any, position: Optional[Dict[str, Any]]) -> Tuple[float, int]:
    """RMS z of the attempt's per-position log timings against the baseline."""
    if not position or not position.get("keys") or not sequences:
        return 0.0, 0
    seq = sequences[0] if isinstance(sequences, list) and sequences else sequences
    if not isinstance(seq, dict):
        return 0.0, 0
    keys = list(position["keys"])
    if list(seq.get("keys") or []) != keys:
        return 0.0, 0

    zs: List[float] = []
    for field, mu_key, sd_key in (("dwell", "dwell_mean", "dwell_std"),
                                  ("flight", "flight_mean", "flight_std")):
        vals = seq.get(field) or []
        mus = position.get(mu_key) or []
        sds = position.get(sd_key) or []
        for i, v in enumerate(vals):
            if v is None or i >= len(mus) or mus[i] is None or not sds[i]:
                continue
            lv = math.copysign(math.log1p(abs(float(v))), float(v))
            zs.append((lv - float(mus[i])) / float(sds[i]))
    if len(zs) < 3:
        return 0.0, len(zs)
    # Trim the single worst position: one stumble should not fail a genuine login.
    arr = np.sort(np.abs(np.asarray(zs)))[:-1] if len(zs) > 4 else np.abs(np.asarray(zs))
    return float(np.sqrt((arr ** 2).mean())), len(zs)


def _digraph_distance(attempt: Dict[str, float], baseline: Dict[str, float],
                      baseline_std: Dict[str, float]) -> Tuple[float, int]:
    """RMS z of shared digraph latencies, in log space."""
    if not attempt or not baseline:
        return 0.0, 0
    zs = []
    for k, v in attempt.items():
        if k not in baseline:
            continue
        b = float(baseline[k])
        if b <= 0 or float(v) <= 0:
            continue
        s = float(baseline_std.get(k, 0.35)) or 0.35
        zs.append((math.log(max(1.0, float(v))) - math.log(max(1.0, b))) / s)
    if len(zs) < 2:
        return 0.0, len(zs)
    arr = np.abs(np.asarray(zs))
    return float(np.sqrt((arr ** 2).mean())), len(zs)


def _anomaly(dist: float, d_ref: Optional[float]) -> float:
    ref = float(d_ref) if d_ref else 1.0
    ref = min(_DIST_REF_CEIL, max(_DIST_REF_FLOOR, ref))
    return _sigmoid(_SIGMOID_SLOPE * (dist / ref - _SIGMOID_CENTRE))


def _masked_mean(X: np.ndarray, M: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    out = fallback.astype(np.float64).copy()
    for j in range(X.shape[1]):
        col = X[M[:, j], j]
        if col.size:
            out[j] = float(col.mean())
    return out


def _masked_std(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[M[:, j], j]
        out[j] = float(col.std(ddof=1)) if col.size > 1 else 0.0
    return out
