"""W2 Session B Part 3 — ML pipeline on L1-extracted features."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, roc_curve, auc,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

import shap  # noqa: E402
import xgboost as xgb  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from workspace.scripts.config import PROJECT_ROOT  # type: ignore
else:
    from .config import PROJECT_ROOT


EVAL_DIR = PROJECT_ROOT / "workspace" / "eval"
EVAL_CSV = EVAL_DIR / "eval_dataset_v10.csv"
L1_JSONL = EVAL_DIR / "tz_features_L1.jsonl"
FEATURES_CSV = EVAL_DIR / "ml_features.csv"
RESULTS_MD = EVAL_DIR / "ml_results.md"
RESULTS_JSON = EVAL_DIR / "ml_results.json"
SHAP_PNG = EVAL_DIR / "ml_shap_summary.png"
CM_PNG = EVAL_DIR / "ml_confusion_matrices.png"
ROC_PNG = EVAL_DIR / "ml_roc_curves.png"

SEED = 42
N_SPLITS = 5
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Feature extraction

def extract_ml_features(l1: dict, eval_row: dict) -> dict:
    parsed = l1.get("parsed") or {}
    features = parsed.get("features") or {}
    risk_flags = parsed.get("risk_flags") or []

    brand_mentions = features.get("brand_mentions") or []
    measurement = features.get("measurement_completeness") or {}
    restrictive = features.get("restrictive_language") or []
    ktru = features.get("ktru_alignment") or {}
    general = features.get("general_completeness") or {}

    confidences = [float(f.get("confidence") or 0) for f in risk_flags]
    return {
        "episode_id": l1["episode_id"],
        "notice_id": str(eval_row["notice_id"]),
        "cluster_id": int(eval_row["cluster_id"]),
        "stratum": str(eval_row["stratum"]),
        # Brand
        "has_brand": int(len(brand_mentions) > 0),
        "brand_count": len(brand_mentions),
        "has_equivalent": int(any(bool(b.get("has_equivalent_clause")) for b in brand_mentions)),
        "brands_without_equiv": sum(
            1 for b in brand_mentions if not bool(b.get("has_equivalent_clause"))
        ),
        # Measurement completeness
        "has_units": int(bool(measurement.get("has_units"))),
        "has_ranges": int(bool(measurement.get("has_ranges"))),
        "missing_char_count": len(measurement.get("missing_characteristics") or []),
        # Restrictive
        "restrictive_count": len(restrictive),
        # KTRU
        "ktru_mentioned": int(bool(ktru.get("ktru_code_mentioned"))),
        "beyond_ktru": int(bool(ktru.get("additional_characteristics_beyond_ktru"))),
        # General completeness
        "has_functional": int(bool(general.get("has_functional_requirements"))),
        "has_technical": int(bool(general.get("has_technical_requirements"))),
        "has_quality": int(bool(general.get("has_quality_requirements"))),
        # Risk flags
        "risk_flag_count": len(risk_flags),
        "max_confidence": float(max(confidences) if confidences else 0.0),
        "mean_confidence": float(np.mean(confidences) if confidences else 0.0),
        # Document size
        "tz_char_count": int(eval_row.get("tz_char_count") or 0),
        # Target
        "y": int(eval_row["fas_verdict"] == "violation_established"),
    }


def build_feature_matrix() -> pd.DataFrame:
    eval_df = pd.read_csv(EVAL_CSV, dtype={"notice_id": str})
    eval_by_eid = {r["episode_id"]: r for r in eval_df.to_dict(orient="records")}
    rows: list[dict] = []
    missing = 0
    with L1_JSONL.open(encoding="utf-8") as f:
        for line in f:
            try:
                l1 = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = l1.get("episode_id")
            if eid not in eval_by_eid:
                continue
            if l1.get("status") not in (None, "ok"):
                missing += 1  # we still produce a row with zero features
            rows.append(extract_ml_features(l1, eval_by_eid[eid]))
    df = pd.DataFrame(rows)
    print(f"[ml] feature rows: {len(df)}  (L1 status≠ok: {missing})")
    df.to_csv(FEATURES_CSV, index=False)
    print(f"[ml] wrote {FEATURES_CSV}")
    return df


# CV + GridSearch

FEATURE_COLUMNS = [
    "has_brand", "brand_count", "has_equivalent", "brands_without_equiv",
    "has_units", "has_ranges", "missing_char_count",
    "restrictive_count",
    "ktru_mentioned", "beyond_ktru",
    "has_functional", "has_technical", "has_quality",
    "risk_flag_count", "max_confidence", "mean_confidence",
    "tz_char_count",
]


def build_classifiers() -> dict[str, tuple[Any, dict]]:
    return {
        "LogisticRegression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0],
             "clf__penalty": ["l1", "l2"],
             "clf__solver": ["liblinear"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"n_estimators": [100, 300],
             "max_depth": [5, 10, None],
             "min_samples_leaf": [5, 10]},
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                random_state=SEED, eval_metric="logloss",
                use_label_encoder=False, n_jobs=-1, verbosity=0,
            ),
            {"n_estimators": [100, 300],
             "max_depth": [3, 5, 7],
             "learning_rate": [0.01, 0.1]},
        ),
        "SVM": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(probability=True, random_state=SEED))]),
            {"clf__C": [0.1, 1.0, 10.0],
             "clf__kernel": ["rbf", "linear"]},
        ),
    }


def evaluate_classifier(name: str, estimator: Any, param_grid: dict,
                        X: np.ndarray, y: np.ndarray, groups: np.ndarray
                        ) -> dict[str, Any]:
    """Outer CV with inner GridSearch; collect held-out predictions."""
    outer_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + 1)

    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    all_y_prob: list[float] = []
    best_params_history: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_tr = groups[train_idx]
        gs = GridSearchCV(
            estimator, param_grid, cv=inner_cv.split(X_tr, y_tr, g_tr),
            scoring="f1_macro", n_jobs=-1, refit=True,
        )
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        best_params_history.append(gs.best_params_)
        y_pred = best.predict(X_te)
        if hasattr(best, "predict_proba"):
            y_prob = best.predict_proba(X_te)[:, 1]
        elif hasattr(best, "decision_function"):
            z = best.decision_function(X_te)
            y_prob = 1.0 / (1.0 + np.exp(-z))
        else:
            y_prob = y_pred.astype(float)
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_prob = np.array(all_y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_pos": float(f1_score(y_true, y_pred, average="binary", pos_label=1)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "roc_auc": float(auc(fpr, tpr)),
        "confusion_matrix": cm.tolist(),
        "best_params_per_fold": best_params_history,
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
        "fpr": fpr, "tpr": tpr,
    }


def fit_best_on_all(name: str, estimator: Any, param_grid: dict,
                    X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[Any, dict]:
    """Refit best hyperparams on the full data (for SHAP / inspection)."""
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + 2)
    gs = GridSearchCV(estimator, param_grid, cv=cv.split(X, y, groups),
                      scoring="f1_macro", n_jobs=-1, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


# Plots

def plot_confusion_matrices(results: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, r in zip(axes.flatten(), results):
        cm = np.array(r["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{r['name']}\nacc={r['accuracy']:.3f}  F1={r['f1_macro']:.3f}  κ={r['cohen_kappa']:.3f}")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["not_est", "violation"])
        ax.set_yticklabels(["not_est", "violation"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max()/2 else "black")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=130)
    plt.close()
    print(f"[ml] wrote {CM_PNG}")


def plot_roc_curves(results: list[dict]) -> None:
    plt.figure(figsize=(8, 7))
    for r in results:
        plt.plot(r["fpr"], r["tpr"], label=f"{r['name']} (AUC={r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves — held-out 5-fold StratifiedGroupKFold (by notice_id)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PNG, dpi=130)
    plt.close()
    print(f"[ml] wrote {ROC_PNG}")


def plot_shap_for_best(best_name: str, best_estimator: Any,
                       X: np.ndarray, feature_names: list[str]) -> None:
    """Produce a SHAP summary for the best classifier. Tree models get TreeExplainer;
    others get a sampled KernelExplainer for cost reasons."""
    try:
        is_tree = best_name in ("RandomForest", "XGBoost")
        if is_tree:
            explainer = shap.TreeExplainer(best_estimator)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):  # RF returns list per class
                shap_values = shap_values[1]
        else:
            background = X[np.random.RandomState(SEED).choice(len(X), 50, replace=False)]
            try:
                fn = best_estimator.predict_proba
                explainer = shap.KernelExplainer(lambda x: fn(x)[:, 1], background)
            except Exception:
                explainer = shap.KernelExplainer(best_estimator.predict, background)
            sample = X[np.random.RandomState(SEED + 1).choice(len(X), 100, replace=False)]
            shap_values = explainer.shap_values(sample, nsamples=100)
            X = sample
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=10)
        plt.title(f"SHAP — {best_name} (top 10 features)")
        plt.tight_layout()
        plt.savefig(SHAP_PNG, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[ml] wrote {SHAP_PNG}")
    except Exception as e:
        print(f"[ml] SHAP failed for {best_name}: {type(e).__name__}: {e}")


# Main

def main() -> None:
    df = build_feature_matrix()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    groups = df["notice_id"].to_numpy()
    print(f"[ml] X: {X.shape}  positives: {int(y.sum())}/{len(y)}  groups: {pd.Series(groups).nunique()}")

    results: list[dict] = []
    for name, (estimator, grid) in build_classifiers().items():
        print(f"\n[ml] {name} — CV + GridSearch ...")
        res = evaluate_classifier(name, estimator, grid, X, y, groups)
        print(f"     acc={res['accuracy']:.4f}  F1={res['f1_macro']:.4f}  "
              f"κ={res['cohen_kappa']:.4f}  AUC={res['roc_auc']:.4f}")
        results.append(res)

    plot_confusion_matrices(results)
    plot_roc_curves(results)

    # Best by F1 macro
    best = max(results, key=lambda r: r["f1_macro"])
    print(f"\n[ml] BEST by F1: {best['name']} ({best['f1_macro']:.4f})")

    # Refit best on all data for SHAP + permutation importance
    name = best["name"]
    estimator, grid = build_classifiers()[name]
    best_est, best_params_all = fit_best_on_all(name, estimator, grid, X, y, groups)
    print(f"[ml] best refit hyperparams: {best_params_all}")

    plot_shap_for_best(name, best_est, X, FEATURE_COLUMNS)

    # Permutation importance on the refit best
    perm = permutation_importance(best_est, X, y, n_repeats=10,
                                  random_state=SEED, n_jobs=-1, scoring="f1_macro")
    perm_order = np.argsort(-perm.importances_mean)
    perm_table = [(FEATURE_COLUMNS[i],
                   float(perm.importances_mean[i]),
                   float(perm.importances_std[i])) for i in perm_order]

    # Write outputs
    results_json: dict[str, Any] = {"n_episodes": len(df),
                                    "positives": int(y.sum()),
                                    "feature_columns": FEATURE_COLUMNS,
                                    "classifiers": [],
                                    "best_by_f1": name,
                                    "best_refit_params": best_params_all,
                                    "permutation_importance": [
                                        {"feature": f, "mean": m, "std": s}
                                        for f, m, s in perm_table
                                    ]}
    for r in results:
        results_json["classifiers"].append({
            "name": r["name"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "f1_pos": r["f1_pos"],
            "cohen_kappa": r["cohen_kappa"],
            "roc_auc": r["roc_auc"],
            "confusion_matrix": r["confusion_matrix"],
            "best_params_per_fold": r["best_params_per_fold"],
        })
    RESULTS_JSON.write_text(json.dumps(results_json, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"[ml] wrote {RESULTS_JSON}")

    md = ["# ML pipeline results (L1 features → classical classifiers)",
          "",
          f"- episodes: **{len(df)}** ({int(y.sum())} positive / {len(y) - int(y.sum())} negative)",
          f"- CV: StratifiedGroupKFold (n_splits={N_SPLITS}, groups=notice_id)",
          f"- inner GridSearchCV over 3 folds, scoring=f1_macro",
          f"- best classifier by F1_macro: **{name}**",
          "",
          "## Comparison",
          "",
          "| classifier | accuracy | F1 (macro) | F1 (positive) | Cohen's κ | ROC AUC |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        md.append(f"| {r['name']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} | "
                  f"{r['f1_pos']:.4f} | {r['cohen_kappa']:.4f} | {r['roc_auc']:.4f} |")

    md += ["",
           "## Best refit hyperparameters",
           "",
           f"`{best_params_all}`",
           "",
           "## Permutation importance (top 10, on refit best)",
           "",
           "| feature | mean Δf1 | ±std |",
           "|---|---:|---:|"]
    for f, m, s in perm_table[:10]:
        md.append(f"| `{f}` | {m:+.4f} | {s:.4f} |")

    md += ["",
           "## Confusion matrices (held-out, pooled across folds)",
           ""]
    for r in results:
        cm = r["confusion_matrix"]
        md.append(f"### {r['name']}")
        md.append("")
        md.append("| | pred=not_est | pred=violation |")
        md.append("|---|---:|---:|")
        md.append(f"| true=not_est   | {cm[0][0]} | {cm[0][1]} |")
        md.append(f"| true=violation | {cm[1][0]} | {cm[1][1]} |")
        md.append("")

    md += ["",
           "## Per-fold best params",
           ""]
    for r in results:
        md.append(f"- **{r['name']}**: {r['best_params_per_fold']}")
    md += ["",
           "## Comparison vs LLM baselines (from Wave 1)",
           "",
           "- B2 (zero-shot V3, balanced eval set): 50.7% accuracy",
           "- B3 (taxonomy prompt V3): 50.0% accuracy",
           "- B4 (full 4-detector pipeline V3, OR-aggregation): 54.7% accuracy",
           "- B0 majority class baseline (no API): 62.1% accuracy",
           "",
           "If a classifier above outperforms B2 (50.7%), it means LLM-extracted",
           "features (L1) carry enough signal for a classical classifier to make",
           "comparable or better decisions — the LLM is acting as a feature extractor",
           "with downstream supervised learning, not as the decision-maker.",
           ""]
    RESULTS_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[ml] wrote {RESULTS_MD}")


if __name__ == "__main__":
    main()
