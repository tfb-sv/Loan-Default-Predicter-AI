from utils.utils_common import (
    DATA_PATH, MODEL_PATH
)
from utils.utils_data import (
    load_dataset, prepare_Xy, split_data
)
from utils.utils_eval import *

# Run the full evaluation pipeline
def start_evaluation(
    model_type,
    test_size,
    random_state,
    n_splits
):
    # Load preprocessed dataset
    df = load_dataset(DATA_PATH)

    # Prepare features and labels
    X, y = prepare_Xy(df)
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Evaluate final model
    run_final_evaluation(
        X_train, y_train,
        X_test, y_test,
        model_type, MODEL_PATH,
        n_splits, random_state
    )
    
    print("\nEvaluation is completed!\n")
