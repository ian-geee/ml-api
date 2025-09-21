import re
import unicodedata
from typing import Iterable, List, Tuple, Dict, Any
import numpy as np

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, jaccard_score, matthews_corrcoef, cohen_kappa_score,
    hamming_loss, zero_one_loss, confusion_matrix, classification_report
)

def binary_classification_metrics(y_test, y_pred) -> Dict[str, Any]:
    """
    Calcule un ensemble compact de métriques binaires
    à partir de y_test et y_pred (labels 0/1).
    Positif attendu: label 1.
    Retourne un dict + inclut un 'classification_report' texte.
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    # Matrice de confusion et composantes
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    # Taux dérivés
    tpr = tp / (tp + fn) if (tp + fn) else 0.0          # True Positive Rate : Sensibilité / Recall+
    tnr = tn / (tn + fp) if (tn + fp) else 0.0          # True Negative Rate : Spécificité / Recall-
    fpr = fp / (fp + tn) if (fp + tn) else 0.0          # False Positive Rate : 
    fnr = fn / (fn + tp) if (fn + tp) else 0.0          # False Negative Rate
    ppv = precision_score(y_test, y_pred, zero_division=0)            # Precision / PPV
    npv = tn / (tn + fn) if (tn + fn) else 0.0                        # NPV (dérivé)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": ppv,
        "recall_sensitivity": recall_score(y_test, y_pred, zero_division=0),
        "specificity": tnr,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "fbeta_2": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        "fbeta_0_5": fbeta_score(y_test, y_pred, beta=0.5, zero_division=0),
        "jaccard": jaccard_score(y_test, y_pred, zero_division=0),
        "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "hamming_loss": hamming_loss(y_test, y_pred),
        "zero_one_loss": zero_one_loss(y_test, y_pred),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "tpr_recall_pos": tpr, "tnr_specificity": tnr, "fpr": fpr, "fnr": fnr,
        "npv": npv,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["neg(0)", "pos(1)"], zero_division=0
        )
    }
    return metrics



def multiclass_classification_metrics():
  """
  Exclusive classes for the target
  In extending a binary metric to multiclass or multilabel problems, the data is treated as a collection of binary problems, one for each class. 
  There are then a number of ways to average binary metric calculations across the set of classes, each of which may be useful in some scenario. 
  Where available, you should select among these using the average parameter.
  - "macro"     : simply calculates the mean of the binary metrics, giving equal weight to each class. 
                  In problems where infrequent classes are nonetheless important, macro-averaging may be a means of highlighting their performance. 
                  On the other hand, the assumption that all classes are equally important is often untrue, such that macro-averaging will over-emphasize the typically low performance on an infrequent class.
  - "weighted"  : accounts for class imbalance by computing the average of binary metrics in which each class’s score is weighted by its presence in the true data sample.
  - "micro"     : gives each sample-class pair an equal contribution to the overall metric (except as a result of sample-weight). 
                  Rather than summing the metric per class, this sums the dividends and divisors that make up the per-class metrics to calculate an overall quotient. 
                  Micro-averaging may be preferred in multilabel settings, including multiclass classification where a majority class is to be ignored.
  - "samples"   : applies only to multilabel problems. 
                  It does not calculate a per-class measure, instead calculating the metric over the true and predicted classes for each sample in the evaluation data, and returning their (sample_weight-weighted) average.

  Selecting average=None will return an array with the score for each class.
  """
  return None

def multilabel_classification_metrics():
  """
  Non-exclusive classes for the target, the target can be multiples classes at the same time.
  In extending a binary metric to multiclass or multilabel problems, the data is treated as a collection of binary problems, one for each class. 
  There are then a number of ways to average binary metric calculations across the set of classes, each of which may be useful in some scenario. 
  Where available, you should select among these using the average parameter.
  - "macro"     : simply calculates the mean of the binary metrics, giving equal weight to each class. 
                  In problems where infrequent classes are nonetheless important, macro-averaging may be a means of highlighting their performance. 
                  On the other hand, the assumption that all classes are equally important is often untrue, such that macro-averaging will over-emphasize the typically low performance on an infrequent class.
  - "weighted"  : accounts for class imbalance by computing the average of binary metrics in which each class’s score is weighted by its presence in the true data sample.
  - "micro"     : gives each sample-class pair an equal contribution to the overall metric (except as a result of sample-weight). 
                  Rather than summing the metric per class, this sums the dividends and divisors that make up the per-class metrics to calculate an overall quotient. 
                  Micro-averaging may be preferred in multilabel settings, including multiclass classification where a majority class is to be ignored.
  - "samples"   : applies only to multilabel problems. 
                  It does not calculate a per-class measure, instead calculating the metric over the true and predicted classes for each sample in the evaluation data, and returning their (sample_weight-weighted) average.

  Selecting average=None will return an array with the score for each class.
  """
  return None