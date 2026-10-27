"""Classification metrics with bootstrap confidence intervals."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def summarize(y_true, y_pred, n_classes, n_boot=1000, seed=0):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.arange(n_classes)
    present = np.unique(y_true)
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)
    boot_acc, boot_f1 = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        boot_acc.append((y_true[i] == y_pred[i]).mean())
        boot_f1.append(f1_score(y_true[i], y_pred[i], labels=present, average="macro", zero_division=0))
    ci = (lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if b else None)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore"):
        recall = np.where(cm.sum(1) > 0, cm.diagonal() / cm.sum(1), np.nan)
    return {
        "n": int(len(y_true)),
        "accuracy": acc,
        "accuracy_ci95": ci(boot_acc),
        "macro_f1": f1,
        "macro_f1_ci95": ci(boot_f1),
        "errors": int((y_true != y_pred).sum()),
        "per_class_recall": [None if np.isnan(r) else float(r) for r in recall],
        "confusion_matrix": cm.tolist(),
    }
