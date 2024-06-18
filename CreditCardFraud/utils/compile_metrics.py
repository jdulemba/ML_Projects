def create_confusion_matrix(tp, fp, tn, fn):
    import numpy as np
    return np.array([[tn, fp], [fn, tp]])


def metrics2dict(metrics):
    results = {
        "Precision" : metrics["precision_score"],
        "Recall" : metrics["recall_score"],
        "F1" : metrics["f1_score"],
        "Confusion_Matrix" : create_confusion_matrix(metrics["true_positives"], metrics["false_positives"], metrics["true_negatives"], metrics["false_negatives"]),
        "PRCurve_AvgPrec" : metrics["precision_recall_auc"]
    }
    if "Cross_Val" in metrics.keys():
        results.update({"Cross_Val" : metrics["Cross_Val"]})

    return results
