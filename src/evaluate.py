import torch
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from config import THRESHOLD_PERCENTILE

def get_reconstruction_errors(model, X):
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    errors = model.reconstruction_error(X_tensor)
    return errors.detach().numpy()

def find_threshold(errors_train):
    return np.percentile(errors_train, THRESHOLD_PERCENTILE)

def evaluate(model, X_train, y_train, X_test, y_test):
    # Erreurs sur le train (normales uniquement) pour fixer le seuil
    X_train_normal = X_train[y_train == 0]
    errors_train = get_reconstruction_errors(model, X_train_normal)
    threshold = find_threshold(errors_train)


    # Erreurs sur tout le test set
    errors_test = get_reconstruction_errors(model, X_test)
    best_threshold, best_f1 = find_best_threshold(errors_test, y_test)

    # Classification
    predictions = (errors_test > threshold).astype(int)

    precision = precision_score(y_test, predictions)
    recall    = recall_score(y_test, predictions)
    f1        = f1_score(y_test, predictions)
    auc_roc   = roc_auc_score(y_test, errors_test)

    print(f"Seuil               : {threshold:.6f}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"F1                  : {f1:.4f}")
    print(f"AUC-ROC             : {auc_roc:.4f}")
    print(f"Meilleur seuil      : {best_threshold:.6f}")
    print(f"Meilleur F1         : {best_f1:.4f}")
    

    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix :")
    print(cm)

    return predictions, errors_test, threshold

def find_best_threshold(errors, y_true):
    thresholds = np.percentile(errors, np.arange(90, 100, 0.5))
    best_threshold, best_f1 = 0, 0

    for t in thresholds:
        preds = (errors > t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1