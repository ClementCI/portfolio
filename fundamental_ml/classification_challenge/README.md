
# Supervised Classification Challenge

## Description

This project addresses a supervised learning task: training a classifier on a labeled dataset and predicting labels on a separate, unlabeled evaluation set. The objective is to produce accurate predictions in a specified format for a challenge submission, where final scoring is based on comparison with hidden ground truth labels.

## Key Results

| Model                              | Validation Accuracy |
|-----------------------------------|---------------------|
| K-Nearest Neighbors                | 66.8%               |
| Support Vector Machine (poly kernel)     | 60.2%               |
| Support Vector Machine (RBF kernel)      | 68.2%               |
| Random Forest                     | 69.1%               |
| AdaBoost + Naive Bayes            | 66.7%               |
| AdaBoost + Decision Tree          | 67.1%               |
| XGBoost                           | 70.2%               |
| 2-layer Feedforward Neural Network            | **70.6%** (selected) |

- The best performance was achieved using a **2-layer Feedforward Neural Network** with Adam optimizer, showing the strongest validation accuracy.
- All models were evaluated using an 80/20 train/validation split from the provided labeled dataset.
- KNN, SVM, and ensemble models provided valuable benchmarks for comparison.

## Features

- End-to-end supervised classification pipeline
- Data preprocessing: missing value handling, encoding, normalization, outlier removal, feature selection, and dimensionality reduction with PCA
- Multiple classifiers: KNN, SVM, ensemble methods (AdaBoost, XGBoost, Random Forest), and feedforward neural networks
- Grid search cross-validation for hyperparameters tuning
- Evaluation via train/validation split with confusion matrix analysis
- Automated CSV export of final predictions for submission

## Dataset

Two datasets are provided:

- `TrainOnMe.csv`: Labeled dataset used for model training and validation  
- `EvaluateOnMe.csv`: Unlabeled dataset used for final prediction and scoring

The input features include both categorical and numerical attributes. The target variable is **multi-class with three distinct labels**, and the format corresponds to standard tabular classification tasks.

## File Structure

- `notebook.ipynb`  
  Contains the full pipeline:
  - Data loading and exploration
  - Preprocessing and transformation
  - Model training and tuning
  - Evaluation and prediction export

-`data/`
  - `TrainOnMe.csv`: Labeled data used for training and validation

  - `EvaluateOnMe.csv`: Unlabeled data used for generating the final submission

## Methodology

- **Preprocessing:**
  - One-hot encoding for categorical features
  - Standard scaling for numerical features
  - Imputation of missing values
  - Outlier detection and removal
  - Feature selection via visualization and LASSO
  - Dimensionality reduction using PCA to retain 95% variance
  - The **pandas** library is used for data handling

- **Modeling:**
  - Evaluated multiple classifiers: KNN, SVM, ensemble methods, and neural networks
  - Most models are implemented using the **scikit-learn** library
  - Neural network experiments, where applicable, are built using the **TensorFlow** framework (via `keras`)
  - Hyperparameters tuned using grid search cross-validation

- **Validation:**
  - Split training data into 80% training and 20% validation sets
  - Model selection based on validation accuracy
  - Confusion matrices used to analyze performance across all three classes

## Output Format

Final predictions are exported as a single-column CSV file without a header, containing only the predicted labels for the evaluation dataset — as specified by the evaluation platform.

## Installation

To set up the environment, install the required packages using:

```bash
pip install -r pandas numpy seaborn matplotlib scikit-learn xgboost lightgbm tensorflow scikeras imblearn