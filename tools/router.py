"""
================================================================================
ROUTER — Conformal Safe-Skip Router 部署用类
================================================================================

无硬阈值的非对称代价路由器：
  decision: run_grace ⟺ s(x) ≥ s_floor
  其中 s_floor 不是手工选择，而是从训练数据中"应触发"类别的 OOF 最小分数
  自动计算得到（α=0 严格全召回；α>0 允许 α 比例的漏召回换更大跳过区）。

部署接口与之前 BinaryConservativeRouter 兼容：

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

也兼容旧 3 维路由器（仅 entropy + spatial + detail）：
若加载的 features 中无 topp/margin 字段，调用时这两个参数可省略。
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

    决策规则（无硬阈值）：
        s(x) = (x - μ) / σ · w + b      （logits 空间，未过 sigmoid）
        run_grace ⟺ s(x) ≥ s_floor

    s_floor 由训练数据自动决定：α=0 → 等于 ori_wrong 训练集 OOF 最小分数。
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
        """根据 self.features 列表组装 x 向量"""
        vec = []
        for f in self.features:
            if f in kw and kw[f] is not None:
                vec.append(float(kw[f]))
            else:
                # 缺失时用 mu（标准化后 = 0，对该维度无贡献）
                idx = self.features.index(f)
                vec.append(float(self.mu[idx]))
        return np.array(vec, dtype=np.float64)

    def score(self, **kw) -> float:
        """
        关键字参数应包含 self.features 中需要的字段：
            answer_entropy, answer_topp, answer_margin,
            has_spatial_keyword, has_fine_detail_keyword
        若缺失会用训练时的均值（标准化后 = 0）补齐。
        也接受 question_text 自动算出后两个 keyword 标志。
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
        返回 (trigger_grace: bool, info: dict)
        info = {"score":..., "s_floor":..., "margin_to_floor":...}

        以下 5 个特征中缺失的会自动用训练均值替代：
            answer_entropy / answer_topp / answer_margin
            has_spatial_keyword / has_fine_detail_keyword（自动从 question 提取）
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
