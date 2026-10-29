"""
Model Training and Reproducible Evaluation Script for Phishing URL Detection.

Evaluates the existing baseline model, trains candidate models on stratified
train/test splits from upload.csv, and outputs detailed classification metrics.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def evaluate_model(name, model, X_test, y_test, pos_label=-1):
    """
    pos_label=-1 corresponds to the Phishing class.
    Class 1 corresponds to the Legitimate class.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 0] if -1 in model.classes_ else model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec_phish = precision_score(y_test, y_pred, pos_label=-1)
    rec_phish = recall_score(y_test, y_pred, pos_label=-1)
    f1_phish = f1_score(y_test, y_pred, pos_label=-1)

    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    # cm layout for labels=[-1, 1]:
    # [[TN, FP], [FN, TP]] where -1 is considered "positive" event (Phishing)
    # TN: Pred -1, True -1 (True Phishing)
    # FP: Pred  1, True -1 (Missed Phishing = False Negative in security)
    tn, fp, fn, tp = cm.ravel()

    # In security:
    # False Negative Rate (FNR) = Phishing missed (predicted safe) / Total actual phishing
    fnr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    # False Positive Rate (FPR) = Safe flagged as phishing / Total actual safe
    fpr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_test == -1, y_proba)
    except Exception:
        auc = 0.0

    print(f"\n==================================================")
    print(f"Model Evaluation: {name}")
    print(f"==================================================")
    print(f"Overall Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
    print(f"Phishing Precision:        {prec_phish:.4f} ({prec_phish*100:.2f}%)")
    print(f"Phishing Recall:           {rec_phish:.4f} ({rec_phish*100:.2f}%)")
    print(f"Phishing F1-Score:         {f1_phish:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f} ({fnr*100:.2f}% phishing missed)")
    print(f"False Positive Rate (FPR): {fpr:.4f} ({fpr*100:.2f}% safe flagged)")
    print(f"ROC-AUC:                   {auc:.4f}")
    print(f"\nConfusion Matrix (Labels: [-1=Phishing, 1=Safe]):")
    print(f"                Predicted Phish (-1)   Predicted Safe (1)")
    print(f"Actual Phish (-1):     {tn:6d}                 {fp:6d}")
    print(f"Actual Safe   (1):     {fn:6d}                 {tp:6d}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Phishing (-1)', 'Legitimate (1)']))

    return {
        "accuracy": acc,
        "precision_phish": prec_phish,
        "recall_phish": rec_phish,
        "f1_phish": f1_phish,
        "fnr": fnr,
        "fpr": fpr,
        "auc": auc,
        "confusion_matrix": cm.tolist()
    }


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "upload.csv")
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        return

    print("Loading dataset from", csv_path)
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape} (instances: {len(df)})")

    feature_cols = [c for c in df.columns if c not in ['Id', 'Result']]
    X = df[feature_cols]
    y = df['Result']

    print(f"Class distribution:\n{y.value_counts()}")

    # Stratified Train/Test Split (80% Train, 20% Held-out Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 1. Evaluate existing model.pkl
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            saved_model = pickle.load(f)
        evaluate_model("Existing model.pkl (on 20% holdout test set)", saved_model, X_test, y_test)

    # 2. Train and evaluate a fresh GradientBoostingClassifier
    gbc = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.2, max_depth=4, random_state=42
    )
    gbc.fit(X_train, y_train)
    evaluate_model("Tuned Gradient Boosting (lr=0.2, depth=4)", gbc, X_test, y_test)

    # 3. 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(gbc, X, y, cv=cv, scoring='accuracy')
    print(f"\n5-Fold CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")


if __name__ == "__main__":
    main()

