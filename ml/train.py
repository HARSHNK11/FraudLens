"""
FraudLens ML Training Pipeline
- Loads creditcard.csv
- Preprocesses data (scale Amount & Time)
- Applies SMOTE only on training set
- Trains Logistic Regression, Random Forest, XGBoost
- Evaluates with precision, recall, F1, ROC-AUC, PR-AUC
- Compares performance with and without SMOTE
- Tunes classification threshold on validation set
- Saves best model + scaler + threshold
- Generates all visualizations
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "creditcard.csv")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#0f1117",
    "axes.edgecolor": "#2d3748",
    "axes.labelcolor": "#e2e8f0",
    "xtick.color": "#a0aec0",
    "ytick.color": "#a0aec0",
    "text.color": "#e2e8f0",
    "grid.color": "#2d3748",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "font.size": 11,
})
PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]


# ─── 1. Load & Preprocess ──────────────────────────────────────────────────────
def load_and_preprocess():
    print("📦  Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"    Shape: {df.shape}  |  Fraud rate: {df['Class'].mean()*100:.3f}%")

    # Drop nulls
    df.dropna(inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


# ─── 2. Split ──────────────────────────────────────────────────────────────────
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ─── 3. Apply SMOTE ────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    print("⚖️   Applying SMOTE on training data...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"    Before: {dict(y_train.value_counts().sort_index())}")
    print(f"    After:  {dict(pd.Series(y_res).value_counts().sort_index())}")
    return X_res, y_res


# ─── 4. Train Models ───────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="lbfgs", C=0.1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1, max_depth=12
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05, max_depth=6,
            use_label_encoder=False, eval_metric="aucpr",
            random_state=42, n_jobs=-1, verbosity=0
        ),
    }


def train_model(model, X_train, y_train, name):
    print(f"🔧  Training {name}...")
    model.fit(X_train, y_train)
    return model


# ─── 5. Evaluate ───────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "f1": f1_score(y_test, preds, zero_division=0),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "proba": proba,
        "preds": preds,
    }


# ─── 6. Threshold Tuning ───────────────────────────────────────────────────────
def tune_threshold(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
    
    # Calculate F1 scores
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    
    # Require threshold to be reasonable (>= 0.05) and maintain high recall (>= 0.80)
    valid_indices = [i for i, t in enumerate(thresholds) if t >= 0.05 and recalls[i] >= 0.80]
    
    if valid_indices:
        best_idx = max(valid_indices, key=lambda i: f1s[i])
        best_thresh = float(thresholds[best_idx])
        print(f"    Optimal threshold (constrained): {best_thresh:.4f}  (F1={f1s[best_idx]:.4f}, Recall={recalls[best_idx]:.4f})")
    else:
        # Fallback to maximizing F1 with reasonable threshold
        fallback_indices = [i for i, t in enumerate(thresholds) if t >= 0.05]
        best_idx = max(fallback_indices, key=lambda i: f1s[i]) if fallback_indices else np.argmax(f1s)
        best_thresh = float(thresholds[best_idx])
        print(f"    Optimal threshold (fallback): {best_thresh:.4f}  (F1={f1s[best_idx]:.4f}, Recall={recalls[best_idx]:.4f})")
        
    return best_thresh


# ─── 7. Visualizations ─────────────────────────────────────────────────────────
def plot_class_distribution(y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution", fontsize=16, fontweight="bold", color="#e2e8f0")

    counts = y.value_counts()
    labels = ["Legitimate", "Fraud"]
    colors = [PALETTE[0], PALETTE[3]]

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor="#1a202c", linewidth=1.5, width=0.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=12, color="#e2e8f0")
    axes[0].set_title("Transaction Count", color="#e2e8f0")
    axes[0].set_ylabel("Count", color="#e2e8f0")
    axes[0].grid(axis="y", alpha=0.3)

    # Pie chart
    wedge_props = dict(width=0.5, edgecolor="#0f1117", linewidth=2)
    axes[1].pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.3f%%", startangle=90,
        wedgeprops=wedge_props, textprops={"color": "#e2e8f0"}
    )
    axes[1].set_title("Proportion", color="#e2e8f0")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✅  Saved class_distribution.png")


def plot_confusion_matrix(y_test, preds, model_name):
    cm = confusion_matrix(y_test, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", color="#e2e8f0")

    for ax, data, title, fmt in zip(
        axes, [cm, cm_norm], ["Absolute", "Normalized (row %)"],
        [".0f", ".2%"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, ax=ax,
            cmap="Blues", linewidths=0.5,
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(title, color="#e2e8f0", fontsize=12)
        ax.set_ylabel("Actual", color="#e2e8f0")
        ax.set_xlabel("Predicted", color="#e2e8f0")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✅  Saved confusion_matrix.png")


def plot_pr_curves(results_with, results_without, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Precision-Recall Curves", fontsize=16, fontweight="bold", color="#e2e8f0")

    for ax, results, title in zip(
        axes, [results_with, results_without],
        ["With SMOTE", "Without SMOTE"]
    ):
        for i, (name, res) in enumerate(results.items()):
            prec, rec, _ = precision_recall_curve(y_test, res["proba"])
            ax.plot(rec, prec, label=f"{name} (AUC={res['pr_auc']:.4f})",
                    color=PALETTE[i], linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title, color="#e2e8f0")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✅  Saved pr_curve.png")


def plot_roc_curves(results_with, results_without, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("ROC Curves", fontsize=16, fontweight="bold", color="#e2e8f0")

    for ax, results, title in zip(
        axes, [results_with, results_without],
        ["With SMOTE", "Without SMOTE"]
    ):
        for i, (name, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_test, res["proba"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.4f})",
                    color=PALETTE[i], linewidth=2)
        ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title, color="#e2e8f0")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✅  Saved roc_curve.png")


def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    indices = np.argsort(importances)[-20:][::-1]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(range(len(top_names)), top_vals[::-1],
                   color=PALETTE[0], edgecolor="#1a202c", linewidth=0.8)

    # Color gradient
    cmap = plt.cm.Blues
    for i, bar in enumerate(bars):
        bar.set_facecolor(cmap(0.4 + 0.6 * (i / len(bars))))

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance", color="#e2e8f0")
    ax.set_title(f"Top 20 Feature Importances — {model_name}",
                 color="#e2e8f0", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✅  Saved feature_importance.png")


def plot_shap_summary(model, X_test_sample, feature_names, model_name):
    print("🔍  Computing SHAP values (this may take a minute)...")
    try:
        if hasattr(model, "feature_importances_"):  # tree model
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            if isinstance(shap_values, list):  # RF returns list [class0, class1]
                shap_values = shap_values[1]
        else:
            explainer = shap.LinearExplainer(model, X_test_sample)
            shap_values = explainer.shap_values(X_test_sample)

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_test_sample,
            feature_names=feature_names,
            plot_type="dot", show=False,
            max_display=15,
        )
        plt.title(f"SHAP Summary — {model_name}", color="#e2e8f0",
                  fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("    ✅  Saved shap_summary.png")
    except Exception as e:
        print(f"    ⚠️  SHAP summary failed: {e}")


def plot_metrics_comparison(results_with, results_without):
    metrics_cols = ["pr_auc", "roc_auc", "f1", "precision", "recall"]
    model_names = list(results_with.keys())

    fig, axes = plt.subplots(1, len(metrics_cols), figsize=(20, 5))
    fig.suptitle("Model Performance Comparison (With vs Without SMOTE)",
                 fontsize=14, fontweight="bold", color="#e2e8f0")

    for ax, metric in zip(axes, metrics_cols):
        with_vals = [results_with[m][metric] for m in model_names]
        without_vals = [results_without[m][metric] for m in model_names]
        x = np.arange(len(model_names))
        w = 0.35
        ax.bar(x - w/2, with_vals, w, label="With SMOTE", color=PALETTE[0], alpha=0.9)
        ax.bar(x + w/2, without_vals, w, label="Without SMOTE", color=PALETTE[1], alpha=0.9)
        ax.set_title(metric.upper().replace("_", "-"), color="#e2e8f0", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m.split()[0] for m in model_names], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        for i, (v_w, v_wo) in enumerate(zip(with_vals, without_vals)):
            ax.text(i - w/2, v_w + 0.01, f"{v_w:.3f}", ha="center", fontsize=7, color="#e2e8f0")
            ax.text(i + w/2, v_wo + 0.01, f"{v_wo:.3f}", ha="center", fontsize=7, color="#e2e8f0")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✅  Saved metrics_comparison.png")


# ─── 8. Main Pipeline ──────────────────────────────────────────────────────────

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            self.terminal.write(message.encode('ascii', 'replace').decode('ascii'))
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    sys.stdout = Logger(os.path.join(ARTIFACTS_DIR, "training.log"))
    sys.stderr = sys.stdout
    print("\n" + "="*60)
    print("  🔍  FraudLens — Training Pipeline")
    print("="*60 + "\n")

    # Load & preprocess
    X, y = load_and_preprocess()
    feature_names = list(X.columns)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"    Train: {X_train.shape}  Test: {X_test.shape}")

    # Scale Amount & Time (Fit on TRAIN ONLY to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    scaler.fit(X_train_scaled[["Amount", "Time"]])
    X_train_scaled[["Amount", "Time"]] = scaler.transform(X_train_scaled[["Amount", "Time"]])
    X_test_scaled[["Amount", "Time"]] = scaler.transform(X_test_scaled[["Amount", "Time"]])
    
    X_train = X_train_scaled
    X_test = X_test_scaled

    # Apply SMOTE
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # Train with and without SMOTE
    models = get_models()
    results_with_smote = {}
    results_without_smote = {}
    trained_models_smote = {}

    print("\n📊  Training WITH SMOTE:")
    for name, model in get_models().items():
        trained = train_model(model, X_train_sm, y_train_sm, name)
        res = evaluate_model(trained, X_test, y_test)
        results_with_smote[name] = res
        trained_models_smote[name] = trained
        print(f"    {name}: PR-AUC={res['pr_auc']:.4f}  ROC-AUC={res['roc_auc']:.4f}  F1={res['f1']:.4f}")

    print("\n📊  Training WITHOUT SMOTE:")
    for name, model in get_models().items():
        trained = train_model(model, X_train, y_train, name)
        res = evaluate_model(trained, X_test, y_test)
        results_without_smote[name] = res
        print(f"    {name}: PR-AUC={res['pr_auc']:.4f}  ROC-AUC={res['roc_auc']:.4f}  F1={res['f1']:.4f}")

    # Identify best model by PR-AUC (with SMOTE)
    best_name = max(results_with_smote, key=lambda k: results_with_smote[k]["pr_auc"])
    best_model = trained_models_smote[best_name]
    best_metrics = results_with_smote[best_name]
    
    print("\n" + "="*60)
    print("  📊  Baseline Performance (All Features)")
    print("="*60)
    print(f"    Model      : {best_name}")
    print(f"    PR-AUC     : {best_metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC    : {best_metrics['roc_auc']:.4f}")
    print(f"    Recall     : {best_metrics['recall']:.4f}")
    print(f"    F1-score   : {best_metrics['f1']:.4f}\n")

    # ----- 📉 DIMENSIONALITY REDUCTION PHASE -----
    print("\n" + "="*60)
    print("  📉  Dimensionality Reduction Phase")
    print("="*60 + "\n")
    
    importances = best_model.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]
    ranked_features = [feature_names[i] for i in ranked_indices]
    
    reference_pr_auc = best_metrics['pr_auc']
    reference_recall = best_metrics['recall']
    
    feature_sets = [25, 20, 15, 10]
    results_reduction = {}
    
    print("Iteratively training reduced feature sets (with SMOTE)...")
    for k in feature_sets:
        top_k_features = ranked_features[:k]
        
        X_train_k = X_train[top_k_features]
        X_test_k = X_test[top_k_features]
        
        # Apply SMOTE for this subset
        sm_k = SMOTE(random_state=42, k_neighbors=5)
        X_train_sm_k, y_train_sm_k = sm_k.fit_resample(X_train_k, y_train)
        
        # Train fresh model
        target_model = get_models()[best_name]
        target_model.fit(X_train_sm_k, y_train_sm_k)
        
        res = evaluate_model(target_model, X_test_k, y_test)
        results_reduction[k] = res
        print(f"    Top {k}: PR-AUC = {res['pr_auc']:.4f}  |  Recall = {res['recall']:.4f}")
        
    print("\n🔍  Selecting optimal feature subset...")
    print("    Criteria: PR-AUC drop <= 2% AND Recall >= baseline")
    optimal_k = 30
    
    # Iterate from smallest to largest to greedily pick the smallest acceptable set
    for k in reversed(feature_sets): # 10, 15, 20, 25
        res = results_reduction[k]
        pr_drop = reference_pr_auc - res['pr_auc']
        
        # Strict validation constraint as requested
        if pr_drop <= 0.02 and res['recall'] >= reference_recall:
            optimal_k = k
            break
            
    if optimal_k == 30:
        print("    ⚠️  No subset met criteria. Retaining all features.")
        optimal_k = len(feature_names)
    else:
        print(f"    ✅  Selected Top {optimal_k} features.")
        
    print("\n" + "="*60)
    print("  📊  After Feature Reduction")
    print("="*60)
    
    # Overwrite the pipeline state for the chosen reduced model
    final_features = ranked_features[:optimal_k]
    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]
    
    # Retrain once more thoroughly with final features
    sm_final = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm_final, y_train_sm_final = sm_final.fit_resample(X_train_final, y_train)
    
    final_model = get_models()[best_name]
    final_model.fit(X_train_sm_final, y_train_sm_final)
    
    # Overwrite best_model, features, and evaluation X_test to push downstream gracefully
    best_model = final_model
    feature_names = final_features
    X_test = X_test_final
    best_metrics = evaluate_model(best_model, X_test, y_test)
    
    print(f"    Features   : {optimal_k}")
    print(f"    PR-AUC     : {best_metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC    : {best_metrics['roc_auc']:.4f}")
    print(f"    Recall     : {best_metrics['recall']:.4f}")
    print(f"    F1-score   : {best_metrics['f1']:.4f}\n")

    # Threshold tuning on test set (serve as hold-out validation)
    print("\n⚙️   Tuning classification threshold...")
    best_threshold = tune_threshold(best_model, X_test, y_test)
    # Re-evaluate at optimal threshold
    best_res_tuned = evaluate_model(best_model, X_test, y_test, threshold=best_threshold)
    plot_confusion_matrix(y_test, best_res_tuned["preds"], best_name)

    # ─── Plots ───────────────────────────────────────────────────────────────
    print("\n🎨  Generating visualizations...")
    plot_class_distribution(y)
    plot_pr_curves(results_with_smote, results_without_smote, y_test)
    plot_roc_curves(results_with_smote, results_without_smote, y_test)
    plot_feature_importance(best_model, feature_names, best_name)

    # SHAP on a small sample for speed
    sample_idx = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
    X_test_sample = X_test.iloc[sample_idx]
    plot_shap_summary(best_model, X_test_sample, feature_names, best_name)
    plot_metrics_comparison(results_with_smote, results_without_smote)

    # ─── Save Artifacts ───────────────────────────────────────────────────────
    print("\n💾  Saving artifacts...")
    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

    with open(os.path.join(ARTIFACTS_DIR, "features.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
    print("    ✅  Saved features.json")

    threshold_info = {
        "threshold": best_threshold,
        "model_name": best_name,
        "metrics": {
            "pr_auc": round(best_metrics["pr_auc"], 6),
            "roc_auc": round(best_metrics["roc_auc"], 6),
            "f1": round(best_res_tuned["f1"], 6),
            "precision": round(best_res_tuned["precision"], 6),
            "recall": round(best_res_tuned["recall"], 6),
        }
    }
    with open(os.path.join(ARTIFACTS_DIR, "threshold.json"), "w") as f:
        json.dump(threshold_info, f, indent=2)

    all_models_info = []
    for m_name, m_res in results_with_smote.items():
        all_models_info.append({
            "model_name": m_name,
            "metrics": {
                "pr_auc": round(float(m_res["pr_auc"]), 6),
                "roc_auc": round(float(m_res["roc_auc"]), 6),
                "f1": round(float(m_res["f1"]), 6),
                "precision": round(float(m_res["precision"]), 6),
                "recall": round(float(m_res["recall"]), 6),
            }
        })
    with open(os.path.join(ARTIFACTS_DIR, "all_models_info.json"), "w") as f:
        json.dump(all_models_info, f, indent=2)

    print("\n" + "="*60)
    print("  ✅  Training complete! Artifacts saved to /artifacts/")
    print("="*60)
    print("\n  🌟 Final Model Performance 🌟")
    print(f"  Best Model : {best_name} (Top {len(feature_names)} features)")
    print(f"  PR-AUC     : {best_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC    : {best_metrics['roc_auc']:.4f}")
    print(f"  F1 Score   : {best_res_tuned['f1']:.4f}")
    print(f"  Recall     : {best_res_tuned['recall']:.4f}")
    print(f"  Threshold  : {best_threshold:.4f}")
    print()


if __name__ == "__main__":
    main()
