verbose_flag_init = 0  # whether to enable verbose output
log_flag = True  # whether to use logarithmic scale for optimization

# Fixed parameters for Logistic Regression
def get_LR_const_params(random_state):
    const_params = {
        "random_state": random_state,
        "class_weight": "balanced",
        "verbose": verbose_flag_init
    }
    return const_params

# Search space for Logistic Regression
def get_LR_search_space(trial):
    search_space = {
        "max_iter": trial.suggest_int("max_iter", 1500, 3000),
        "C": trial.suggest_float("C", 0.001, 10.0, log=log_flag),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"])
    }
    return search_space

# Fixed parameters for Random Forest
def get_RF_const_params(random_state):
    const_params = {
        "random_state": random_state,
        "class_weight": "balanced",
        "verbose": verbose_flag_init
    }
    return const_params

# Search space for Random Forest
def get_RF_search_space(trial):
    search_space = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
    }
    return search_space

# Fixed parameters for XGBoost
def get_XGB_const_params(random_state):
    const_params = {
        "random_state": random_state,
        "use_label_encoder": False,
        "verbosity": verbose_flag_init
    }
    return const_params

# Search space for XGBoost
def get_XGB_search_space(trial):
    search_space = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=log_flag),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=log_flag),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=log_flag),
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart", "gblinear"]),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=log_flag),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=log_flag)
    }
    return search_space
