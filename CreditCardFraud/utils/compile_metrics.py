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


def get_roc_auc(fpr, tpr, fpr_thresh=None):
    from sklearn.metrics import auc
    import numpy as np
    def partial_auc(fpr, tpr, max_fpr):
        "Taken from here https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b611bf873bd5836748647221480071a87/sklearn/metrics/_ranking.py#L350-L356"
        # Add a single point at max_fpr by linear interpolation
        stop = np.searchsorted(fpr, max_fpr, "right")
        x_interp = [fpr[stop - 1], fpr[stop]]
        y_interp = [tpr[stop - 1], tpr[stop]]
        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
        fpr = np.append(fpr[:stop], max_fpr)

        return auc(fpr, tpr)

    roc_auc = auc(fpr, tpr)
    if fpr_thresh is not None:
        roc_auc = partial_auc(fpr, tpr, max_fpr=fpr_thresh)

    return roc_auc
