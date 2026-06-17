"""
================================================================================
ROUTER Conformal Safe-Skip Router 
================================================================================


  decision: run_grace ⟺ s(x) ≥ s_floor
   s_floor "" OOF 
  α=0 α>0 α 

 BinaryConservativeRouter 

    from tools.router import Router
    router = Router.load("router_report.json")
    trigger, info = router.decide(
        answer_entropy=0.78,
        answer_topp=0.55,
        answer_margin=0.10,
        question_text="Where is the car on the left side?",
    )
    # info = {"score": ..., "s_floor": ..., "margin_to_floor": ...}
    if trigger: <run GRACE>
    else:       <use ori answer>

 3 entropy + spatial + detail
 features topp/margin 
"""
import os
import json
import re
import numpy as np


SPATIAL_KEYWORDS = {
    "left", "right", "above", "below", "beside", "behind", "front",
    "between", "relation", "position", "side", "under", "over", "top",
    "bottom", "corner", "direction", "orient", "facing", "next to",
}
FINE_DETAIL_KEYWORDS = {
    "text", "sign", "number", "color", "letter", "digit", "write", "read",
    "word", "label", "logo", "sticker", "symbol",
}


def question_priors(question: str):
    q = (question or "").lower()
    return {
        "has_spatial_keyword": int(any(kw in q for kw in SPATIAL_KEYWORDS)),
        "has_fine_detail_keyword": int(any(kw in q for kw in FINE_DETAIL_KEYWORDS)),
    }


class Router:
    """
    Conformal Safe-Skip Router.

    
        s(x) = (x - μ) / σ · w + b logits sigmoid
        run_grace ⟺ s(x) ≥ s_floor

    s_floor α=0 → ori_wrong OOF 
    """

    def __init__(self, weights, bias, feature_mu, feature_sd, s_floor,
                 features=None, alpha=0.0):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.mu = np.asarray(feature_mu, dtype=np.float64)
        self.sd = np.asarray(feature_sd, dtype=np.float64)
        self.s_floor = float(s_floor)
        self.alpha = float(alpha)
        self.features = list(features) if features else [
            "answer_entropy", "answer_topp", "answer_margin",
            "has_spatial_keyword", "has_fine_detail_keyword",
        ]

    def _build_x(self, kw):
        """ self.features x """
        vec = []
        for f in self.features:
            if f in kw and kw[f] is not None:
                vec.append(float(kw[f]))
            else:
                idx = self.features.index(f)
                vec.append(float(self.mu[idx]))
        return np.array(vec, dtype=np.float64)

    def score(self, **kw) -> float:
        """
         self.features 
            answer_entropy, answer_topp, answer_margin,
            has_spatial_keyword, has_fine_detail_keyword
         = 0
         question_text keyword 
        """
        if "question_text" in kw and ("has_spatial_keyword" not in kw or "has_fine_detail_keyword" not in kw):
            pri = question_priors(kw["question_text"])
            kw.setdefault("has_spatial_keyword", pri["has_spatial_keyword"])
            kw.setdefault("has_fine_detail_keyword", pri["has_fine_detail_keyword"])
        x = self._build_x(kw)
        x_std = (x - self.mu) / (self.sd + 1e-12)
        return float(x_std @ self.weights + self.bias)

    def decide(self, answer_entropy: float = None,
               answer_topp: float = None,
               answer_margin: float = None,
               question_text: str = "",
               **extra):
        """
         (trigger_grace: bool, info: dict)
        info = {"score":..., "s_floor":..., "margin_to_floor":...}

         5 
            answer_entropy / answer_topp / answer_margin
            has_spatial_keyword / has_fine_detail_keyword question 
        """
        kw = {
            "answer_entropy": answer_entropy,
            "answer_topp": answer_topp,
            "answer_margin": answer_margin,
            "question_text": question_text,
        }
        kw.update(extra)
        s = self.score(**kw)
        trigger = s >= self.s_floor
        return bool(trigger), {
            "score": float(s),
            "s_floor": float(self.s_floor),
            "margin_to_floor": float(s - self.s_floor),
        }

    def to_dict(self):
        return {
            "model_type": "conformal_safe_skip_router",
            "features": self.features,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "feature_mu": self.mu.tolist(),
            "feature_sd": self.sd.tolist(),
            "s_floor": self.s_floor,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            weights=d["weights"], bias=d["bias"],
            feature_mu=d["feature_mu"], feature_sd=d["feature_sd"],
            s_floor=d["s_floor"], alpha=d.get("alpha", 0.0),
            features=d.get("features"),
        )

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
