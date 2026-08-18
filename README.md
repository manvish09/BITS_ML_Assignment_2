# Machine Learning Assignment 2 — Classification Model Comparison

## A. Problem Statement
Implement classification models on a public classification dataset, evaluate them using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC), and demonstrate the results through an interactive Streamlit application.

## B. Dataset Description
**Dataset:** Breast Cancer Wisconsin Diagnostic dataset.

The dataset contains **569 instances** and **30 numerical input features**, satisfying the assignment minimum of 500 instances and 12 features. The target is binary classification.

The dataset is commonly used for binary classification of breast cancer cases. In this project, the target encoding follows the bundled dataset used by scikit-learn:
- `0` = malignant
- `1` = benign

A stratified 80/20 train-test split with `random_state=42` is used.

## C. GitHub Repository Link
**To be added after GitHub upload:** `https://github.com/<your-username>/ML_Assignment_2`

## D. Models Used

The assignment lists five models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

> Note: The assignment text refers to "all the 6 ML models" but enumerates five models. This implementation follows the five explicitly listed models.

### Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |\n| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |\n| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |\n| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |\n| Random Forest | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |\n
### Observations

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong linear baseline. Feature scaling is applied before fitting. |
| Decision Tree | Captures nonlinear relationships and is easy to interpret, but can be sensitive to tree depth. |
| kNN | Uses neighborhood similarity and benefits from standardized features. |
| Naive Bayes | Provides a fast probabilistic baseline using the Gaussian assumption for numerical features. |
| Random Forest | Combines many decision trees and generally provides robust nonlinear classification performance. |
| Overall Winner | Select the model with the strongest overall metric profile on the held-out test set. |

## Project Structure

```text
ML_Assignment_2/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── data/
│   └── breast_cancer_dataset.csv
└── model/
    ├── train_models.py
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── evaluation_results.csv
```

## Streamlit App Features
- Test-data CSV upload
- Model selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Prediction preview
- Comparison table for all implemented models

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Push the repository to GitHub and deploy `app.py` using Streamlit Community Cloud.

## Academic Integrity
This project is intended as a learning scaffold. Before submission, review the implementation, understand the model choices and metrics, run the project yourself in the BITS Virtual Lab, and maintain your own Git commit history.
