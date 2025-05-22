from utils.utils_common import (
    DATA_PATH, MODEL_PATH
)
from utils.utils_data import *
from utils.utils_train import *

# Run the full training pipeline
def start_training(
    random_state,
    test_size,
    model_type,
    use_raw_data,
    do_inspect_data,
    run_optimization,
    no_outlier_removal,
    outlier_percentile,
    n_splits,
    n_trials,
    timeout
):
    # Process raw data (optional)
    if use_raw_data:
        # Load raw data
        cust, fin, loan, delinq = read_data(
            DATA_PATH,
            do_inspect_data
        )

        # Preprocess dataset for labels and features
        target = create_target(delinq)
        df = feature_engineering(
            loan, cust, fin, target,
            no_outlier_removal=no_outlier_removal,
            outlier_percentile=outlier_percentile
        )

        # Save preprocessed dataset
        save_dataset(df, DATA_PATH)
    else:
        # Load preprocessed dataset
        df = load_dataset(DATA_PATH)

    # Prepare features and labels
    X, y = prepare_Xy(df)
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Run hyperparameter optimization
    if run_optimization:
        run_optuna_search(
            objective_func, X_train, y_train,
            model_type, random_state, n_splits,
            MODEL_PATH, n_trials, timeout
        )

    # Train final model and evaluate
    run_final_training(
        X_train, y_train,
        X_test, y_test,
        model_type, MODEL_PATH
    )
    
    print("\nTraining is completed!\n")
