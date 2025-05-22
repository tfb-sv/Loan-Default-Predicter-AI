import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import shap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    average_precision_score, precision_score, recall_score
)

from utils.utils_common import trainer_map
from utils.utils_data import split_data

FIG_SIZE = (12, 6.75)  # approximate 16:9 aspect ratio
DPI = 300

# Load trained model
def load_model(model_type, model_path):
    model = joblib.load(f"{model_path}/model_{model_type}.pkl")
    return model

# Evaluates final model
def run_final_evaluation(X_train, y_train, X_test, y_test,
                         model_type, model_path, val_size, random_state):
    # Load model
    model = load_model(model_type, model_path)

    # Evaluate model and print results
    y_pred, y_pred_proba, cm = evaluate_model(model, X_test, y_test)

    # Print sorted feature importances
    if model_type == "LR":
        importance = get_linear_feature_importance(model, X_train)
    elif model_type in ["RF", "XGB"]:
        importance = get_tree_feature_importance(model, X_train)
    print(importance)

    # Plot confusion matrix
    plot_confusion_matrix(cm, model_type, model_path)

    # Plot SHAP summary
    if model_type == "XGB":  # supported only for XGB
        plot_SHAP_summary(model, X_train, model_type, model_path)

    # Plot loss curves
    if model_type == "XGB":  # supported only for XGB
        train_for_loss_curves(
            X_train, y_train, model_type,
            model_path, val_size, random_state
        )

# Evaluate model with optimal threshold and print performance metrics
def evaluate_model(model, X_test, y_test):
    # Predict default probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Find best threshold
    best_thr = find_best_threshold(y_test, y_pred_proba)

    # Predict using the best threshold
    y_pred = (y_pred_proba >= best_thr).astype(int)

    # Calculate standard metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    prc_auc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)  # threshold-dependent

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Print standard metrics
    print("Accuracy:", np.round(acc, 4))
    print("ROC-AUC:", np.round(roc_auc, 4))
    print("PRC-AUC:", np.round(prc_auc, 4))
    print("F1 Score:", np.round(f1, 4))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"TN (True Negative): {tn}")
    print(f"FP (False Positive): {fp}")
    print(f"FN (False Negative): {fn}")
    print(f"TP (True Positive): {tp}\n")

    return y_pred, y_pred_proba, cm

# Feature importance function for linear models
def get_linear_feature_importance(model, X):
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.coef_[0]
    })
    importance['abs_importance'] = np.abs(importance['importance'])
    importance = importance.sort_values('abs_importance', ascending=False)
    return importance[['feature', 'importance']]

# Feature importance for tree-based models
def get_tree_feature_importance(model, X):
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    return importance

# Find best threshold for classification based on F1 score
def find_best_threshold(y_true, y_proba):
    best_threshold = 0.5
    best_f1 = 0
    results = []

    # Loop over thresholds and track F1 scores
    for t in np.arange(0.1, 0.9, 0.1):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds)
        recall = recall_score(y_true, preds)
        precision = precision_score(y_true, preds)
        results.append((t, f1, precision, recall))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # Print best threshold by F1 score
    for t, f1, precision, recall in sorted(results, key=lambda x: x[1], reverse=True)[:1]:
        print(f"Best Threshold: {t:.1f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}\n")

    return best_threshold

# Create heatmap plot for confusion matrix
def plot_confusion_matrix(cm, model_type, model_path):
    # Configure labels
    labels = np.array([
        [f"TN: {cm[0,0]}", f"FP: {cm[0,1]}"],
        [f"FN: {cm[1,0]}", f"TP: {cm[1,1]}"]
    ])

    # Plot confusion matrix
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", cbar=True,
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        annot_kws={"fontsize": 13, "weight": "bold"}
    )

    # Configure plot
    plt.gcf().set_size_inches(FIG_SIZE[1]+1, FIG_SIZE[1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({model_type} Model)")
    plt.tight_layout()

    # Export plot to PNG
    save_path = f"{model_path}/plot_confusion_matrix_{model_type}.png"
    plt.savefig(save_path, dpi=DPI)
    plt.close()

# Create SHAP summary plot for a tree-based model
def plot_SHAP_summary(model, X_train, model_type, model_path):
    # Plot SHAP summary
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, show=False)

    # Configure plot
    plt.gcf().set_size_inches(*FIG_SIZE)
    plt.grid(axis='y', linestyle='-', alpha=1.0)
    plt.tight_layout()

    # Export plot to PNG
    save_path = f"{model_path}/plot_SHAP_summary_{model_type}.png"
    plt.savefig(save_path, dpi=DPI)
    plt.close()

# Train model only for plotting loss curves
def train_for_loss_curves(X_train, y_train, model_type,
                         model_path, n_splits, random_state):
    # Load best hyperparameters
    with open(f"{model_path}/config_{model_type}.json", "r") as f:
        config = json.load(f)

    # Calculate validation set size (for consistency with CV)
    val_size = 1 / n_splits

    # Split data into train and validation sets
    X_tr, X_val, y_tr, y_val = split_data(
        X_train, y_train,
        test_size=val_size,
        random_state=random_state
    )

    # Train model with best parameters
    trainer = trainer_map[model_type]
    model = trainer(X_tr, y_tr, config, X_val, y_val)
    
    # Extract training logs
    results = model.evals_result_
    train_loss = results['validation_0']['logloss']
    val_loss = results['validation_1']['logloss']

    # Plot loss curves
    plt.figure(figsize=FIG_SIZE)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')

    # Configure plot
    plt.xlabel("Iterations")
    plt.ylabel("Logloss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Export plot to PNG
    plot_path = f"{model_path}/plot_loss_curves_{model_type}.png"
    plt.savefig(plot_path, dpi=DPI)
    plt.close()
