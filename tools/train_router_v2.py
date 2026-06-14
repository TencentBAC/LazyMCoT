"""
================================================================================
TRAIN ROUTER v2 - Cost-Aware Conformal Safe-Skip Router
================================================================================

Design principles:
  - ONLY numeric / statistical features derived from the first-token
    probability distribution P(first_token | image, question).
    No text keyword buckets, no regex heuristics — fully dataset-agnostic.
  - Cost-aware recall: we only force 100% (or 1-alpha) recall on
    "ori_wrong minus ow_gw" because ow_gw fails under GRACE too, so
    skipping them costs nothing.
  - Conformal s_floor: derived from OOF scores, no manual threshold.

Recommended features (verified on features_train_v2.jsonl, α=0.01 GBDT):
  CORE5 (5 pure-numeric features, dataset-agnostic):
    answer_topp               # option top-1 probability   (↓ more confident)
    answer_margin             # top1 - top2                 (↓ sharper)
    vocab_full_entropy_norm   # full-vocab first-token H / log(V)  (↑ harder)
    option_mass               # mass of option letters      (↓ escaping)
    logit_gap_opt_nonopt      # max option logit - max non-option logit

Optional (small gain, possibly less robust across datasets):
  VATT3 (visual-attention distribution stats):
    vatt_entropy_norm, vatt_top10_mass, vatt_focus_area

Discouraged (dataset-biased):
  num_options, num_visual_tokens  - correlate with dataset identity
  question_length_tokens          - correlate with dataset identity
  bucket_* / opt_*                - text-keyword heuristics (NOT used)

Dataset partitions (for training):
  Classifier label : label in {ori_correct, ori_wrong}  ← only supervision
  must-recall mask : (label == ori_wrong) AND (not in ow_gw)
  cgw / ow_gw      : used ONLY to (1) define must-recall, (2) post-hoc report
                     cgw-to-ori rate. They do NOT supervise the classifier.

Optimization:
  Constraint : (ori_wrong - ow_gw) -> GRACE rate >= 1 - alpha
  Secondary  :
    - maximize cgw -> ori rate
    - maximize (oc - cgw) -> ori rate

Output:
  tools/router_v2_report.json   router params (loadable by router_v2.RouterV2)
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import Counter

_THIS = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────────────────────
# AUC helper
# ──────────────────────────────────────────────────────────────────────────────
def auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true > 0.5]
    neg = y_score[y_true < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    ranks = combined.argsort().argsort() + 1
    rs = ranks[: len(pos)].sum()
    return float((rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ──────────────────────────────────────────────────────────────────────────────
# Default feature set  (CORE5: pure-numeric, dataset-agnostic, audit-friendly)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_FEATS = [
    "answer_topp",              # option top-1 probability
    "answer_margin",            # top1 - top2
    "vocab_full_entropy_norm",  # full-vocab first-token entropy / log(V)
    "option_mass",              # sum prob over option letters
    "logit_gap_opt_nonopt",     # max option logit - max non-option logit
]


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=os.path.join(_THIS, "features_train_v2.jsonl"),
                   help="v2 训练特征")
    p.add_argument("--cgw_data",
                   default="/path/to/data/ori_correct_grace_wrong.json",
                   help="ori_correct ∩ grace_wrong（cgw）：事后评估 + 选 α 辅助")
    p.add_argument("--ow_gw_data",
                   default="/path/to/data/ori_wrong_grace_wrong.json",
                   help="ori_wrong ∩ grace_wrong（ow_gw）：定义 must-recall 集")
    p.add_argument("--output", default=os.path.join(_THIS, "router_v2_report.json"))
    p.add_argument("--feats", nargs="+", default=DEFAULT_FEATS,
                   help="features to use; default = CORE5 (5 pure-numeric features)")

    # 决策参数
    p.add_argument("--alpha", type=float, default=0.01,
                   help="must-recall 集的 conformal miscoverage rate；0 = 严格 100% 召回")
    p.add_argument("--model", choices=["gbdt", "lr"], default="gbdt",
                   help="分类器：gbdt（默认，更强）或 lr（更可解释）")

    # GBDT 超参
    p.add_argument("--gbdt_n_estimators", type=int, default=300)
    p.add_argument("--gbdt_max_depth", type=int, default=3)
    p.add_argument("--gbdt_lr", type=float, default=0.05)

    # LR 超参
    p.add_argument("--lr_C", type=float, default=1.0)

    # CV 与随机种子
    p.add_argument("--K", type=int, default=5, help="K-fold OOF")
    p.add_argument("--seed", type=int, default=2077)

    args = p.parse_args()

    # ── 载入数据 ───────────────────────────────────────────────────
    recs_all = [json.loads(l) for l in open(args.features) if l.strip()]
    cnt = Counter(r["label"] for r in recs_all)
    print(f"[load] {len(recs_all)} records, label dist: {dict(cnt)}")

    # (No text-keyword enrichment: v2 uses pure-numeric features only.)

    # 过滤不合格记录
    feats = args.feats
    print(f"[feats] {len(feats)} features: {feats}")

    recs = [r for r in recs_all
            if r["label"] in ("ori_correct", "ori_wrong")
            and all(r.get(f) is not None for f in feats)]
    print(f"[load] training samples (after filter): {len(recs)}")

    X = np.array([[float(r[f]) for f in feats] for r in recs], dtype=np.float64)
    y = np.array([1 if r["label"] == "ori_wrong" else 0 for r in recs],
                 dtype=np.int64)
    print(f"[binary] y=1 (ori_wrong): {int(y.sum())},  "
          f"y=0 (ori_correct): {int((1-y).sum())}")

    # ── 载入 cgw / ow_gw 集合 ───────────────────────────────────────
    cgw_keys = set()
    if os.path.exists(args.cgw_data):
        cgw_keys = {(str(r["id"]), r["image_path"])
                    for r in json.load(open(args.cgw_data))}
    ow_gw_keys = set()
    if os.path.exists(args.ow_gw_data):
        ow_gw_keys = {(str(r["id"]), r["image_path"])
                      for r in json.load(open(args.ow_gw_data))}

    is_cgw = np.array([(str(r["id"]), r["image_path"]) in cgw_keys for r in recs])
    is_owgw = np.array([(str(r["id"]), r["image_path"]) in ow_gw_keys for r in recs])

    # must-recall：ori_wrong \ ow_gw
    must_recall_mask = (y == 1) & (~is_owgw)
    # 子集统计
    n_mr = int(must_recall_mask.sum())
    n_owgw_pos = int(((y == 1) & is_owgw).sum())
    n_cgw = int(((y == 0) & is_cgw).sum())
    n_oc_ngw = int(((y == 0) & (~is_cgw)).sum())
    print(f"[subsets]")
    print(f"  must-recall (ori_wrong \\ ow_gw) : {n_mr:5d}  ← MUST be routed to GRACE")
    print(f"  ow_gw       (in ori_wrong)       : {n_owgw_pos:5d}  ← GRACE can't save, OK to drop")
    print(f"  cgw         (in ori_correct)     : {n_cgw:5d}  ← GRACE would hurt, WANT back to ori")
    print(f"  oc \\ cgw    (in ori_correct)     : {n_oc_ngw:5d}  ← GRACE ok, but skip saves compute")

    # ── 训练模型（K-Fold OOF） ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Training {args.model.upper()} with {args.K}-fold OOF")
    print(f"{'='*70}")

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.K, shuffle=True, random_state=args.seed)

    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs = []

    # 保存每个 fold 的模型 parameters（最终会用 full-fit 模型部署）
    for k, (tr, te) in enumerate(skf.split(X, y)):
        if args.model == "gbdt":
            from sklearn.ensemble import GradientBoostingClassifier
            # GBDT 不需要 scaling；class_weight 通过 sample_weight 实现
            sw = np.where(y[tr] == 1,
                          len(y[tr]) / (2 * max(y[tr].sum(), 1)),
                          len(y[tr]) / (2 * max((1 - y[tr]).sum(), 1)))
            clf = GradientBoostingClassifier(
                n_estimators=args.gbdt_n_estimators,
                max_depth=args.gbdt_max_depth,
                learning_rate=args.gbdt_lr,
                random_state=args.seed,
            )
            clf.fit(X[tr], y[tr], sample_weight=sw)
            p_te = clf.predict_proba(X[te])[:, 1]
            # logit 空间，便于 conformal floor 的数值稳定
            oof[te] = np.log(p_te / (1 - p_te + 1e-12) + 1e-12)
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(class_weight="balanced", C=args.lr_C,
                                     max_iter=2000)
            clf.fit(sc.transform(X[tr]), y[tr])
            oof[te] = sc.transform(X[te]) @ clf.coef_.ravel() + clf.intercept_[0]

        fold_aucs.append(auc(y[te], oof[te]))
        print(f"  [fold {k+1}/{args.K}] AUC = {fold_aucs[-1]:.4f}")
    print(f"  OOF AUC mean = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # ── Conformal Safe Floor（基于 must-recall 集） ────────────────
    mr_scores = oof[must_recall_mask]
    if args.alpha == 0.0:
        s_floor = float(mr_scores.min())
    else:
        # Q_α 下分位数：允许 α 比例的 must-recall 样本被 skip（漏召回）
        s_floor = float(np.quantile(mr_scores, args.alpha))

    print(f"\n[s_floor] α = {args.alpha}")
    print(f"          s_floor = {s_floor:.4f}")
    print(f"          mr_scores: min={mr_scores.min():.3f} "
          f"p5={np.quantile(mr_scores,.05):.3f} "
          f"median={np.median(mr_scores):.3f} "
          f"max={mr_scores.max():.3f}")

    # ── 在 OOF 上评估最终决策 ──────────────────────────────────────
    trigger = oof >= s_floor
    skip = ~trigger

    def rate(mask_sub):
        n = int(mask_sub.sum())
        if n == 0:
            return 0.0, 0, 0
        r = float((trigger & mask_sub).sum()) / n
        return r, int((trigger & mask_sub).sum()), n

    print(f"\n{'='*70}")
    print(f"OOF Routing Evaluation")
    print(f"{'='*70}")

    mr_mask_arr = must_recall_mask
    mr_rec, mr_trig, mr_n = rate(mr_mask_arr)
    owgw_mask = (y == 1) & is_owgw
    ow_gw_rec, ow_gw_trig, ow_gw_n = rate(owgw_mask)
    ow_total_mask = (y == 1)
    ow_tot_rec, ow_tot_trig, ow_tot_n = rate(ow_total_mask)
    cgw_mask = (y == 0) & is_cgw
    cgw_skip_n = int((skip & cgw_mask).sum())
    oc_ngw_mask = (y == 0) & (~is_cgw)
    oc_ngw_skip_n = int((skip & oc_ngw_mask).sum())

    print(f"  ┌─ ori_wrong → GRACE ───────────────────────────")
    print(f"  │ must-recall (ow\\ow_gw): {mr_trig:4d}/{mr_n:4d} = {mr_rec:7.2%}  "
          f"{'✅' if mr_rec >= 1 - args.alpha - 1e-9 else '⚠️'}")
    print(f"  │ ow_gw                 : {ow_gw_trig:4d}/{ow_gw_n:4d} = {ow_gw_rec:7.2%}  "
          f"(GRACE also fails; OK either way)")
    print(f"  │ ow total              : {ow_tot_trig:4d}/{ow_tot_n:4d} = {ow_tot_rec:7.2%}")
    print(f"  ├─ ori_correct → ori (skip) ────────────────────")
    print(f"  │ ★ cgw → ori          : {cgw_skip_n:4d}/{n_cgw:4d} = "
          f"{cgw_skip_n / max(n_cgw,1):7.2%}  ← want as high as possible")
    print(f"  │ oc\\cgw → ori         : {oc_ngw_skip_n:4d}/{n_oc_ngw:4d} = "
          f"{oc_ngw_skip_n / max(n_oc_ngw,1):7.2%}  (compute saved)")
    print(f"  └────────────────────────────────────────────────")

    # ── α 敏感性表 ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"α sensitivity (how choice of α affects the trade-off)")
    print(f"{'='*70}")
    print(f"  {'α':>6s}  {'s_floor':>9s}  {'mr_recall':>10s}  "
          f"{'cgw→ori':>8s}  {'oc\\cgw→ori':>11s}  {'ow_gw→ori':>10s}")
    for a in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10]:
        if a == 0.0:
            sf = mr_scores.min()
        else:
            sf = float(np.quantile(mr_scores, a))
        trig = oof >= sf; sk = ~trig
        mr_r = (trig & mr_mask_arr).sum() / max(mr_n, 1)
        cgw_s = (sk & cgw_mask).sum() / max(n_cgw, 1)
        oc_s  = (sk & oc_ngw_mask).sum() / max(n_oc_ngw, 1)
        ow_gw_s = (sk & owgw_mask).sum() / max(ow_gw_n, 1)
        mark = " ←" if abs(a - args.alpha) < 1e-6 else ""
        print(f"  {a:>6.3f}  {sf:>+9.4f}  {mr_r:>9.2%}   "
              f"{cgw_s:>7.2%}   {oc_s:>10.2%}   {ow_gw_s:>9.2%}{mark}")

    # ── 全量训练部署模型 ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Final model fit (on full training set)")
    print(f"{'='*70}")

    model_payload = {
        "model_type": args.model,
        "features": list(feats),
        "alpha": float(args.alpha),
        "s_floor": s_floor,
    }

    if args.model == "gbdt":
        from sklearn.ensemble import GradientBoostingClassifier
        sw_full = np.where(y == 1,
                           len(y) / (2 * max(y.sum(), 1)),
                           len(y) / (2 * max((1 - y).sum(), 1)))
        clf_full = GradientBoostingClassifier(
            n_estimators=args.gbdt_n_estimators,
            max_depth=args.gbdt_max_depth,
            learning_rate=args.gbdt_lr,
            random_state=args.seed,
        )
        clf_full.fit(X, y, sample_weight=sw_full)
        # 特征重要性
        imp = clf_full.feature_importances_
        print("  [GBDT feature_importances_]")
        for f, v in sorted(zip(feats, imp), key=lambda x: -x[1]):
            print(f"    {f:32s}: {v:.4f}")
        # 序列化：pickle 成 base64，供 router_v2 直接加载；同时存文件
        import pickle, base64
        gbdt_bytes = pickle.dumps(clf_full)
        gbdt_b64 = base64.b64encode(gbdt_bytes).decode("ascii")
        model_payload["gbdt_model_b64"] = gbdt_b64
        # 额外存成独立文件便于调试
        gbdt_path = args.output.replace(".json", "_gbdt.pkl")
        with open(gbdt_path, "wb") as fw:
            fw.write(gbdt_bytes)
        model_payload["gbdt_model_path"] = os.path.abspath(gbdt_path)
        print(f"  [save] GBDT pickle → {gbdt_path} (also embedded as b64 in json)")
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc_full = StandardScaler().fit(X)
        clf_full = LogisticRegression(class_weight="balanced",
                                      C=args.lr_C, max_iter=2000)
        clf_full.fit(sc_full.transform(X), y)
        print("  [LR weights]")
        for f, w in sorted(zip(feats, clf_full.coef_.ravel()),
                           key=lambda x: -abs(x[1])):
            print(f"    {f:32s}: {w:+.4f}")
        print(f"    {'bias':32s}: {clf_full.intercept_[0]:+.4f}")
        model_payload.update({
            "weights": clf_full.coef_.ravel().tolist(),
            "bias": float(clf_full.intercept_[0]),
            "feature_mu": sc_full.mean_.tolist(),
            "feature_sd": sc_full.scale_.tolist(),
        })

    # ── 保存路由器配置 ─────────────────────────────────────────────
    # 为了支持部署时动态调节 α（不用重训），保存 must-recall 集的 OOF 分数。
    # RouterV2.set_alpha(new_alpha) 可以按新 α 在这个分数上取 quantile。
    mr_oof_scores_sorted = sorted(float(s) for s in mr_scores)

    payload = {
        **model_payload,
        "decision_rule": (
            "run_grace iff s(x) >= s_floor  |  "
            "s_floor = Q_alpha(s_oof[ori_wrong - ow_gw])  |  "
            "s(x) = GBDT logit (or LR standardized)"
        ),
        # 保存 must-recall 子集的 OOF 分数（已升序），供部署期动态调 α 用
        "mr_oof_scores_sorted": mr_oof_scores_sorted,
        "oof_evaluation": {
            "must_recall_n": n_mr,
            "must_recall_ratio": mr_rec,
            "ow_total_trigger_ratio": ow_tot_rec,
            "ow_gw_trigger_ratio": ow_gw_rec,
            "cgw_skip_ratio": cgw_skip_n / max(n_cgw, 1),
            "oc_ngw_skip_ratio": oc_ngw_skip_n / max(n_oc_ngw, 1),
            "oof_auc_mean": float(np.mean(fold_aucs)),
            "oof_auc_std": float(np.std(fold_aucs)),
        },
        "training_info": {
            "n_train": int(len(recs)),
            "label_distribution": dict(cnt),
            "must_recall_subset": "ori_wrong - ow_gw",
            "alpha": float(args.alpha),
            "K_fold": int(args.K),
            "seed": int(args.seed),
            "features": list(feats),
        },
        "notes": (
            "Cost-aware Conformal Safe-Skip Router v2. "
            "Training uses only (ori_correct, ori_wrong) labels for the classifier. "
            "cgw / ow_gw sets are used only to (1) define must-recall, (2) report "
            "cgw→ori rate. No hard threshold — s_floor is derived from OOF scores "
            "on the must-recall subset at quantile α."
        ),
    }

    with open(args.output, "w") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)
    print(f"\n[save] router v2 → {args.output}")


if __name__ == "__main__":
    main()
