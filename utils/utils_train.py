import os
import json
import numpy as np
from functools import partial

import joblib
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from utils.utils_common import (
    search_space_map, const_param_map, trainer_map
)
from utils.utils_eval import evaluate_model

# Run Optuna hyperparameter optimization
def run_optuna_search(objective_func, X_train, y_train,
                      model_type, random_state, n_splits,
                      model_path, n_trials, timeout):
    # Bind constant arguments to the objective function
    objective_fn = partial(
        objective_func,
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
        random_state=random_state,
        n_splits=n_splits
    )

    # Create and run the optimization study to maximize PRC-AUC
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_fn, n_trials=n_trials, timeout=timeout)

    # Merge best hyperparameters with constant parameters
    const_params = const_param_map[model_type](random_state)
    best_params = {**study.best_trial.params, **const_params}

    # Print best PRC-AUC score and parameters
    print("Best PRC-AUC Score:", study.best_value)
    print("Best Parameters:", best_params)

    # Save the best parameters to a JSON file
    os.makedirs(model_path, exist_ok=True)
    with open(f"{model_path}/config_{model_type}.json", "w") as f:
        json.dump(best_params, f, indent=4)

# Objective function for Optuna that returns mean PRC-AUC from cross-validation
def objective_func(trial, X_train, y_train,
                   model_type, random_state, n_splits):
    # Get search space and fixed parameters
    search_space = search_space_map[model_type](trial)
    const_params = const_param_map[model_type](random_state)
    params = {**search_space, **const_params}

    # Stratified K-Fold cross-validation
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    # Loop over CV folds and collect PRC-AUC scores
    scores = []
    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # Train model on each fold
        trainer = trainer_map[model_type]
        model = trainer(X_tr, y_tr, params)

        # Predict and evaluate using PRC-AUC
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        prc_auc = average_precision_score(y_val, y_pred_proba)
        scores.append(prc_auc)

    # Average PRC-AUC scores all folds
    mean_score = np.mean(scores)

    # Log trial performance
    print(f"[Trial {trial.number}] PRC-AUC: {mean_score:.4f}")

    return mean_score

# Trains final model using best parameters and evaluates it on test set
def run_final_training(X_train, y_train,
                       X_test, y_test,
                       model_type, model_path):
    # Load best hyperparameters
    with open(f"{model_path}/config_{model_type}.json", "r") as f:
        config = json.load(f)

    # Train model with best parameters
    trainer = trainer_map[model_type]
    model = trainer(X_train, y_train, config)

    # Save model
    save_model(model, model_type, model_path)

    # Evaluate model and print results
    y_pred, y_pred_proba, cm = evaluate_model(model, X_test, y_test)

# Save trained model
def save_model(model, model_type, model_path):
    joblib.dump(model, f"{model_path}/model_{model_type}.pkl")
