"""
================================================================================
ROUTER v2 - Cost-Aware Conformal Safe-Skip Router (deployment class)
================================================================================

Decision rule:
    s(x) = GBDT.logit(x)     (or LR: (x - mu)/sd . w + b)
    run_grace iff s(x) >= s_floor

s_floor = Q_alpha(s_oof[ori_wrong - ow_gw]), produced by train_router_v2.py.

Features: pure first-token distribution statistics (CORE5), no text-keyword
heuristics; fully dataset-agnostic:
    answer_topp, answer_margin, vocab_full_entropy_norm,
    option_mass, logit_gap_opt_nonopt

Example usage (integrate with Qwen2.5/inference.py):

    from tools.router_v2 import RouterV2
    router = RouterV2.load("router_v2_report.json")

    trigger, info = router.decide(
        answer_topp=0.55,
        answer_margin=0.10,
        vocab_full_entropy_norm=0.12,
        option_mass=0.998,
        logit_gap_opt_nonopt=3.5,
        question_text="Where is the car on the left side?",  # optional, ignored
    )
    # Missing numeric features are filled with the training mean
    # (== 0 in standardized space for LR; GBDT just uses raw mean).

    if trigger:
        # run GRACE
        ...
    else:
        # use ori answer
        ...
"""
from __future__ import annotations
import os
import json
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# RouterV2
# ──────────────────────────────────────────────────────────────────────────────
class RouterV2:
    """
    Cost-aware Conformal Safe-Skip Router (pure-numeric features, audit-friendly).

    Params saved by train_router_v2:
        model_type in {"gbdt", "lr"}
        features: list[str]                (feature names, order-sensitive)
        s_floor: float                     decision threshold in score space
        alpha: float                       recorded conformal miscoverage rate
        [gbdt] gbdt_model_b64 or gbdt_model_path   (pickled GBDT)
        [lr]   weights, bias, feature_mu, feature_sd
    """

    def __init__(self, model_type, features, s_floor, alpha,
                 gbdt_model=None,
                 weights=None, bias=None, feature_mu=None, feature_sd=None,
                 mr_oof_scores_sorted=None):
        self.model_type = model_type
        self.features = list(features)
        self.s_floor = float(s_floor)
        self.alpha = float(alpha)

        self.mr_oof_scores_sorted = (
            list(mr_oof_scores_sorted) if mr_oof_scores_sorted else None
        )

        # Training-set per-feature mean, used as imputation for missing values.
        if feature_mu is None:
            self.feature_mu = np.zeros(len(self.features), dtype=np.float64)
        else:
            self.feature_mu = np.asarray(feature_mu, dtype=np.float64)

        if model_type == "gbdt":
            assert gbdt_model is not None, "gbdt requires gbdt_model"
            self.gbdt = gbdt_model
            self.weights = None; self.bias = None; self.feature_sd = None
        elif model_type == "lr":
            assert weights is not None and bias is not None, "lr requires weights/bias"
            self.weights = np.asarray(weights, dtype=np.float64)
            self.bias = float(bias)
            self.feature_sd = np.asarray(feature_sd, dtype=np.float64) \
                              if feature_sd is not None \
                              else np.ones(len(self.features))
            self.gbdt = None
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def set_alpha(self, new_alpha: float) -> float:
        """
         α s_floor = Q_α(mr_oof_scores)

         router JSON `mr_oof_scores_sorted`train_router_v2.py 
         report RuntimeError

        Args:
            new_alpha: conformal miscoverage rate∈ [0, 1)
                      0 = must-recall 100%
                      0.01 = 1% skip 
                      ...
        Returns:
            new_s_floor: 
        """
        if self.mr_oof_scores_sorted is None:
            raise RuntimeError(
                "mr_oof_scores_sorted not available in this router. "
                "Re-train router with latest train_router_v2.py to enable set_alpha()."
            )
        if new_alpha < 0 or new_alpha >= 1:
            raise ValueError(f"alpha must be in [0, 1), got {new_alpha}")

        scores = self.mr_oof_scores_sorted
        if new_alpha == 0.0:
            new_sf = float(scores[0])
        else:
            new_sf = float(np.quantile(scores, new_alpha))
        self.alpha = float(new_alpha)
        self.s_floor = new_sf
        return new_sf

    # ──────────────────── assemble input vector ────────────────────
    def _build_x(self, kw: dict) -> np.ndarray:
        vec = []
        for i, f in enumerate(self.features):
            v = kw.get(f, None)
            if v is None:
                vec.append(float(self.feature_mu[i]))
            else:
                vec.append(float(v))
        return np.array(vec, dtype=np.float64)

    # ──────────────────── score ────────────────────
    def score(self, **kw) -> float:
        x = self._build_x(kw)
        if self.model_type == "gbdt":
            p = self.gbdt.predict_proba(x.reshape(1, -1))[0, 1]
            return float(np.log(p / (1.0 - p + 1e-12) + 1e-12))
        else:
            x_std = (x - self.feature_mu) / (self.feature_sd + 1e-12)
            return float(x_std @ self.weights + self.bias)

    # ──────────────────── decide ────────────────────
    def decide(self, question_text: str = "", **features):
        """
        Returns (trigger_grace: bool, info: dict).

        Args:
            question_text: kept for backward compatibility; currently unused
                since v2 uses pure-numeric features.
            **features:    numeric features keyed by name as in
                           extract_features_v2.py (answer_topp etc.).
                           Missing keys are imputed by training mean.

        info = {"score": ..., "s_floor": ..., "margin_to_floor": ...,
                "model_type": ...}
        """
        kw = dict(features)
        s = self.score(**kw)
        trigger = s >= self.s_floor
        return bool(trigger), {
            "score": float(s),
            "s_floor": float(self.s_floor),
            "margin_to_floor": float(s - self.s_floor),
            "model_type": self.model_type,
        }

    # ──────────────────── persistence ────────────────────
    @classmethod
    def load(cls, path: str):
        """Load from json saved by train_router_v2.py"""
        with open(path, "r") as f:
            d = json.load(f)
        model_type = d["model_type"]
        features = d["features"]
        s_floor = d["s_floor"]
        alpha = d.get("alpha", 0.0)
        mr_oof = d.get("mr_oof_scores_sorted", None)

        if model_type == "gbdt":
            gbdt = cls._load_gbdt_from_payload(d, base_dir=os.path.dirname(path))
            return cls(model_type="gbdt", features=features,
                       s_floor=s_floor, alpha=alpha, gbdt_model=gbdt,
                       feature_mu=None,
                       mr_oof_scores_sorted=mr_oof)
        else:
            return cls(model_type="lr", features=features,
                       s_floor=s_floor, alpha=alpha,
                       weights=d["weights"], bias=d["bias"],
                       feature_mu=d["feature_mu"], feature_sd=d["feature_sd"],
                       mr_oof_scores_sorted=mr_oof)

    @staticmethod
    def _load_gbdt_from_payload(d: dict, base_dir: str = ""):
        import pickle
        if "gbdt_model_b64" in d and d["gbdt_model_b64"]:
            import base64
            return pickle.loads(base64.b64decode(d["gbdt_model_b64"]))
        if "gbdt_model_path" in d:
            path = d["gbdt_model_path"]
            if not os.path.isabs(path) and base_dir:
                path = os.path.join(base_dir, path)
            with open(path, "rb") as f:
                return pickle.load(f)
        raise RuntimeError("No GBDT model found in payload (need "
                           "'gbdt_model_b64' or 'gbdt_model_path')")


# ──────────────────────────────────────────────────────────────────────────────
# Backward compatibility: keep old Router importable if tools.router exists
# ──────────────────────────────────────────────────────────────────────────────
try:
    from tools.router import Router  # noqa: F401
except Exception:
    pass
