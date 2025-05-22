from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

verbose_flag_fit = False  # whether to enable verbose output

# Train Logistic Regression model
def train_LR_model(X_train, y_train, config):
    model = LogisticRegression(**config)
    model.fit(X_train, y_train)
    return model

# Train Random Forest model
def train_RF_model(X_train, y_train, config):
    model = RandomForestClassifier(**config)
    model.fit(X_train, y_train)
    return model

# Train XGBoost model
def train_XGB_model(X_train, y_train, config, X_val=None, y_val=None):
    model = XGBClassifier(**config)

    # Set eval_set to enable loss curve plotting
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=verbose_flag_fit)
    return model
