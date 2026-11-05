from sklearn.metrics import average_precision_score, roc_auc_score

def calculate_metrics(y_true, scores) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }
