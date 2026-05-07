"""
AI_stats_lab.py

Lab: Training and Evaluating Classification Models
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Question 1: Confusion Matrix, Metrics, and Threshold Effects

def confusion_matrix_counts(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))

    return TP, FP, FN, TN


def classification_metrics(y_true, y_pred):
    TP, FP, FN, TN = confusion_matrix_counts(y_true, y_pred)

    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fallout   = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy  = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0

    return {
        "recall": recall,
        "fallout": fallout,
        "precision": precision,
        "accuracy": accuracy
    }


def apply_threshold(scores, threshold):
    scores = np.array(scores)
    return (scores >= threshold).astype(int)


def threshold_metrics_analysis(y_true, scores, thresholds):
    results = []
    for t in thresholds:
        y_pred = apply_threshold(scores, t)
        metrics = classification_metrics(y_true, y_pred)
        results.append({
            "threshold": t,
            "recall":    metrics["recall"],
            "fallout":   metrics["fallout"],
            "precision": metrics["precision"],
            "accuracy":  metrics["accuracy"]
        })
    return results


# Question 2: Train Two Classifiers and Evaluate Them

def train_two_classifiers(X_train, y_train):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)

    return {
        "logistic_regression": lr,
        "decision_tree": dt
    }


def evaluate_classifier(model, X_test, y_test, threshold=0.5):
    scores = model.predict_proba(X_test)[:, 1]
    y_pred = apply_threshold(scores, threshold)

    TP, FP, FN, TN = confusion_matrix_counts(y_test, y_pred)
    metrics = classification_metrics(y_test, y_pred)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "recall":    metrics["recall"],
        "fallout":   metrics["fallout"],
        "precision": metrics["precision"],
        "accuracy":  metrics["accuracy"]
    }


def compare_classifiers(X_train, y_train, X_test, y_test, threshold=0.5):
    models = train_two_classifiers(X_train, y_train)

    return {
        "logistic_regression": evaluate_classifier(models["logistic_regression"], X_test, y_test, threshold),
        "decision_tree":       evaluate_classifier(models["decision_tree"],        X_test, y_test, threshold)
    }


if __name__ == "__main__":
    print("All functions implemented.")
