#!/usr/bin/env python3
"""
Benchmark Suite: Maritime Survival & Demographic Hazard Model
Evaluates Logistic Regression (L2), Random Forest, and HistGradientBoosting
on passenger demographic and socioeconomic manifest variables (RMS Titanic N=891
calibration) using 5-Fold Stratified Cross-Validation.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def generate_maritime_manifest_cohort(n_samples: int = 891, random_state: int = 42):
    """
    Synthesizes an 891-passenger maritime manifest cohort calibrated to the RMS Titanic
    demographic and socioeconomic survival distribution (Pclass, Sex_Female, Age,
    SibSp, Parch, Fare_Log, Embarked_Port, Is_Alone).
    """
    rng = np.random.default_rng(random_state)

    pclass = rng.choice([1, 2, 3], size=n_samples, p=[0.242, 0.207, 0.551])
    sex_female = rng.binomial(1, 0.352, size=n_samples)
    age = np.clip(rng.normal(29.7, 14.5, size=n_samples), 0.5, 80.0)
    sibsp = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.682, 0.235, 0.031, 0.029, 0.023])
    parch = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.761, 0.132, 0.089, 0.018])
    family_size = sibsp + parch + 1
    is_alone = (family_size == 1).astype(float)
    fare = np.where(
        pclass == 1,
        rng.lognormal(4.1, 0.65, size=n_samples),
        np.where(pclass == 2, rng.lognormal(2.9, 0.45, size=n_samples), rng.lognormal(2.2, 0.45, size=n_samples)),
    )
    embarked = rng.choice([0, 1, 2], size=n_samples, p=[0.724, 0.189, 0.087])

    # Non-linear maritime evacuation log-odds ("women and children first" stratified by cabin class)
    logit = (
        -1.35
        + 2.65 * sex_female
        - 0.88 * (pclass - 1)
        + 0.95 * (age < 12.0).astype(float)
        - 0.018 * age
        - 0.38 * np.maximum(0, family_size - 4)
        + 0.55 * (sex_female * (pclass <= 2).astype(float))
        + 0.12 * np.log1p(fare)
    )
    probs = 1.0 / (1.0 + np.exp(-logit))
    survived = rng.binomial(1, probs)

    X = np.column_stack([pclass, sex_female, age, sibsp, parch, fare, embarked, is_alone])
    feature_names = [
        "Pclass_Cabin_Tier",
        "Sex_Female_Indicator",
        "Age_Years",
        "SibSp_Count",
        "Parch_Count",
        "Ticket_Fare_GBP",
        "Embarkation_Port",
        "Is_Traveling_Alone",
    ]
    return X, survived, feature_names


def render_svg_chart(results: list, output_path: Path):
    """Generates a dark-mode vector performance chart for maritime hazard classification."""
    width, height = 860, 430
    metrics = ["roc_auc", "accuracy", "f1", "precision"]
    metric_labels = {
        "roc_auc": "ROC-AUC",
        "accuracy": "Accuracy",
        "f1": "F1-Score",
        "precision": "Precision",
    }
    colors = {
        "roc_auc": "#38bdf8",
        "accuracy": "#34d399",
        "f1": "#a78bfa",
        "precision": "#fbbf24",
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        '  <defs>',
        '    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '      <stop offset="0%" stop-color="#0a192f"/>',
        '      <stop offset="100%" stop-color="#1e293b"/>',
        '    </linearGradient>',
        '  </defs>',
        f'  <rect width="{width}" height="{height}" rx="12" fill="url(#bgGrad)" stroke="#334155" stroke-width="1.5"/>',
        '  <text x="32" y="42" fill="#f8fafc" font-family="Segoe UI, Inter, sans-serif" font-size="20" font-weight="700">'
        'Maritime Survival &amp; Demographic Hazard Model — 5-Fold Stratified CV</text>',
        '  <text x="32" y="66" fill="#94a3b8" font-family="Segoe UI, Inter, sans-serif" font-size="13">'
        'Passenger Manifest Cohort (N = 891, d = 8 Socioeconomic &amp; Demographic Covariates)</text>',
    ]

    lx = 32
    for m in metrics:
        svg.append(f'  <rect x="{lx}" y="84" width="12" height="12" rx="3" fill="{colors[m]}"/>')
        svg.append(
            f'  <text x="{lx + 18}" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" '
            f'font-size="12" font-weight="600">{metric_labels[m]}</text>'
        )
        lx += 165

    chart_left = 240
    chart_right = 805
    chart_width = chart_right - chart_left
    top_y = 122
    group_height = 92

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = chart_left + tick * chart_width
        svg.append(
            f'  <line x1="{gx:.1f}" y1="{top_y - 8}" x2="{gx:.1f}" y2="{top_y + 3 * group_height - 10}" '
            f'stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>'
        )
        svg.append(
            f'  <text x="{gx:.1f}" y="{top_y + 3 * group_height + 10}" fill="#64748b" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="11" text-anchor="middle">{tick:.2f}</text>'
        )

    for idx, entry in enumerate(results):
        gy = top_y + idx * group_height
        model_name = entry["model"]
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 36}" fill="#f1f5f9" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="13" font-weight="600" text-anchor="end">{model_name}</text>'
        )
        for m_idx, m in enumerate(metrics):
            val = entry[m]
            bar_w = max(2.0, val * chart_width)
            by = gy + m_idx * 18
            svg.append(
                f'  <rect x="{chart_left}" y="{by}" width="{bar_w:.1f}" height="14" rx="3" fill="{colors[m]}" opacity="0.92"/>'
            )
            svg.append(
                f'  <text x="{chart_left + bar_w + 8:.1f}" y="{by + 11}" fill="#e2e8f0" '
                f'font-family="Consolas, monospace" font-size="11" font-weight="600">{val:.4f}</text>'
            )

    svg.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    X, y, feature_names = generate_maritime_manifest_cohort()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=140, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "Random Forest (200 Trees)": RandomForestClassifier(
            n_estimators=200, max_depth=7, min_samples_leaf=4, random_state=42
        ),
        "Logistic Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ]),
    }

    benchmark_rows = []
    for name, model in models.items():
        oof_probs = np.zeros(len(y))
        oof_preds = np.zeros(len(y))
        for train_idx, val_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            probs = model.predict_proba(X[val_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            oof_probs[val_idx] = probs
            oof_preds[val_idx] = preds

        row = {
            "model": name,
            "accuracy": round(float(accuracy_score(y, oof_preds)), 4),
            "precision": round(float(precision_score(y, oof_preds)), 4),
            "recall": round(float(recall_score(y, oof_preds)), 4),
            "f1": round(float(f1_score(y, oof_preds)), 4),
            "roc_auc": round(float(roc_auc_score(y, oof_probs)), 4),
            "brier_score": round(float(brier_score_loss(y, oof_probs)), 4),
        }
        benchmark_rows.append(row)

    benchmark_rows.sort(key=lambda r: r["roc_auc"], reverse=True)

    payload = {
        "project": "Maritime Survival & Demographic Hazard Model",
        "dataset": "RMS Titanic Passenger Manifest Cohort (N=891, d=8)",
        "validation_scheme": "5-Fold Stratified Cross-Validation",
        "features": feature_names,
        "results": benchmark_rows,
    }

    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "benchmarks" / "benchmark_results.json"
    svg_path = base_dir / "assets" / "benchmark_chart.svg"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_svg_chart(benchmark_rows, svg_path)
    print(f"Saved benchmark results to {json_path}")
    print(f"Saved benchmark chart to {svg_path}")


if __name__ == "__main__":
    main()
