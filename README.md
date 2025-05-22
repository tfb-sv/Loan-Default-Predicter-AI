[![License: MIT][mit-shield]](./LICENSE)

# Load Default Predicter AI
This repository contains a loan default prediction model that uses customer, loan, and financial data to optimize loan portfolio profitability by identifying high-risk applicants early.

## Overview
- Build a predictive model for loan default within 12 months

## Dataset Overview
- **loan_data.csv**: Loan-level details including amount, term, and purpose.
- **customer_data.csv**: Demographic and engagement information about the applicants.
- **customer_financials.csv**: Monthly salary and balance info from 2016 to 2018.
- **loan_delinquencies.csv**: Records of late payments within the first 12 installments.

> Note: For confidentiality reasons, only the first 5 rows of each dataset have been retained as examples.

## Key Definitions

- **Default**: A loan is considered defaulted if 3 consecutive monthly installments are missed.
- **Delinquency**: The state of a loan when an installment is overdue.

## Setup
Install required libraries:

> cd <your_directory>\Loan-Default-Predicter-AI

> pip install -r requirements.txt

## Usage

### Training (with best hyperparameters and final dataset):
To train XGBoost model without optimization and data preprocessing.

> python main.py --mode training --model_type XGB

### Training (from scratch):
To train XGBoost model from scratch (takes around 30 minutes).

> python main.py --mode training --model_type XGB --use_raw_data --run_optimization

### Evaluation:
To evaluate the trained XGBoost model.

> python main.py --mode evaluation --model_type XGB

## License
© 2025 [Nural Ozel](https://github.com/tfb-sv).

This work is licensed under the [MIT License](./LICENSE).

[cc-by]: https://creativecommons.org/licenses/by/4.0/
[mit-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
