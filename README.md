# Machine Learning Classification Model Comparison

## Overview

This project compares five classification algorithms on the Breast Cancer Wisconsin Diagnostic dataset. The objective is to evaluate how different classification approaches perform on the same binary classification problem using multiple evaluation metrics.

The project also includes an interactive Streamlit application where a test CSV can be uploaded, a model can be selected, and the corresponding predictions and evaluation results can be viewed.

## Problem Statement

The objective is to build and compare multiple machine learning classification models for predicting the class of observations in the Breast Cancer Wisconsin Diagnostic dataset.

The models are evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The same dataset and train-test split are used across all models so that their performance can be compared consistently.

## Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic (WDBC)** dataset.

The dataset contains:

- **569 instances**
- **30 numerical features**
- **2 target classes**
- No missing values in the dataset used for this project

The target represents the diagnosis:

- `0` — Malignant
- `1` — Benign

The dataset satisfies the assignment requirement of at least 500 instances and 12 features.

For model evaluation, the data is divided into training and test sets using an **80:20 stratified split** with `random_state=42`.

Feature standardization is applied for the models where feature scale affects the learning process, particularly Logistic Regression and k-Nearest Neighbors.

## Models

Five classification models are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

Each model is trained on the same training dataset and evaluated on the same held-out test set.

## Model Performance

The following results were obtained on the held-out test set.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

## Observations

### Logistic Regression

Logistic Regression produced the strongest overall results in this experiment. It achieved **98.25% accuracy**, an **AUC of 0.9954**, and an **MCC of 0.9623**. Its precision, recall and F1 score were also well balanced at 0.9861.

The results show that a linear model performs very well on this dataset when the numerical features are standardized.

### Decision Tree

The Decision Tree achieved **92.11% accuracy**, which was the lowest accuracy among the five models. Its AUC was 0.9163 and MCC was 0.8341.

Although a decision tree can model nonlinear relationships and is easy to interpret, the single tree used in this experiment did not perform as well as the other models on the selected test split.

### k-Nearest Neighbors

kNN achieved **97.37% accuracy** and an **F1 score of 0.9796**. Its recall was **1.0000**, meaning that every observation belonging to the positive class in the test set was correctly identified.

Its AUC of 0.9884 and MCC of 0.9442 also indicate strong classification performance. Standardization is particularly important for kNN because the model relies on distances between observations.

### Gaussian Naive Bayes

Naive Bayes achieved **93.86% accuracy** with an AUC of **0.9878**. The relatively high AUC indicates good class-separation capability, although its accuracy, F1 score and MCC were lower than those of Logistic Regression and kNN.

### Random Forest

Random Forest achieved **94.74% accuracy** and an AUC of **0.9937**. Its high AUC indicates strong separation between the two classes.

However, on this particular test split, its accuracy and MCC were lower than Logistic Regression and kNN.

## Overall Result

Based on the complete set of evaluation metrics, **Logistic Regression was the strongest overall model for this experiment**.

It achieved the highest accuracy, precision, F1 score and MCC, while also recording an AUC of 0.9954.

kNN was a close second, particularly because of its perfect recall on the test set.

## Streamlit Application

The project includes an interactive Streamlit application for evaluating the trained models.

The application provides:

- Test-data CSV upload
- Model selection
- Prediction results
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion matrix
- Classification report
- Model comparison results

The application uses the saved trained models from the `model/` directory.

## Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── data/
│   └── breast_cancer_dataset.csv
│
├── model/
│   ├── train_models.py
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── random_forest.joblib
│   └── evaluation_results.csv
│
└── .streamlit/
    └── config.toml
