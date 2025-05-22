import argparse
from training import start_training
from evaluation import start_evaluation

# Entry point to start training or evaluation
def main(args):
    if args.mode == "training":
        start_training(
            random_state=args.random_state,
            test_size=args.test_size,
            model_type=args.model_type,
            use_raw_data=args.use_raw_data,
            do_inspect_data=args.do_inspect_data,
            run_optimization=args.run_optimization,
            no_outlier_removal=args.no_outlier_removal,
            outlier_percentile=args.outlier_percentile,
            n_splits=args.n_splits,
            n_trials=args.n_trials,
            timeout=args.timeout
        )
    elif args.mode == "evaluation":
        start_evaluation(
            model_type=args.model_type,
            test_size=args.test_size,
            random_state=args.random_state,
            n_splits=args.n_splits
        )

# Parse command-line arguments
def load_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate the loan default prediction model."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["training", "evaluation"],
        help="Select mode to run: 'training' or 'evaluation'."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["LR", "RF", "XGB"],
        help="Select model to train: 'LR' (Logistic Regression), 'RF' (Random Forest), or 'XGB' (XGBoost)."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed for reproducibility."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to use for the test set."
    )
    parser.add_argument(
        "--use_raw_data",
        action="store_true",
        help="Process raw data from scratch instead of loading preprocessed dataset."
    )
    parser.add_argument(
        "--do_inspect_data",
        action="store_true",
        help="Print length, duplicate and NaN statistics from raw data."
    )
    parser.add_argument(
        "--run_optimization",
        action="store_true",
        help="Enable hyperparameter optimization using Optuna; otherwise, a config file is needed."
    )
    parser.add_argument(
        "--no_outlier_removal",
        action="store_false",
        help="Disable outlier removal based on percentile threshold."
    )
    parser.add_argument(
        "--outlier_percentile",
        type=float,
        default=5e-3,
        help="Percentile threshold to remove outliers."
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=3,
        help="Number of cross-validation folds."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter optimization."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Maximum time (in seconds) allowed for Optuna hyperparameter optimization."
    )

    args = parser.parse_args()

    return args

# Script entry
if __name__ == "__main__":
    args = load_args()
    main(args)
