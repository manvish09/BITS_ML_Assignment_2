# Machine Learning Classification Model Comparison

## 1. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on the Breast Cancer Wisconsin Diagnostic dataset.

The models are evaluated using the following performance metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The same train-test split is used for all models so that their performance can be compared under the same conditions.

An interactive Streamlit application is also provided to allow users to upload test data, select a model, generate predictions and view the corresponding evaluation results.

## 2. Dataset Description

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

Feature standardization is applied to Logistic Regression and k-Nearest Neighbors, where feature scale can affect model performance.

**Dataset Source:**  
[Add the verified UCI/Kaggle dataset source here.]

## 3. GitHub Repository Link

https://github.com/manvish09/BITS_ML_Assignment_2

The repository contains the application code, dataset, trained models, model-training code, evaluation results, test data and required configuration files.

## 4. Models Used

The following five classification models are implemented and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **k-Nearest Neighbors (kNN)**
4. **Gaussian Naive Bayes**
5. **Random Forest Classifier**

All models are trained using the same training data and evaluated using the same held-out test data.

The trained model files are stored in the `model/` directory.

## 5. Comparison Table

The following results were obtained on the held-out test set.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

### Observations

| ML Model | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieved the **best overall performance**, with **98.25% accuracy**, **0.9954 AUC**, **0.9861 precision**, **0.9861 recall**, **0.9861 F1 score**, and **0.9623 MCC**. The balanced precision and recall indicate consistent classification performance on the test set. |
| **Decision Tree** | Achieved **92.11% accuracy**, **0.9163 AUC**, and **0.8341 MCC**, making it the weakest model among the five on this test split. Although it can capture nonlinear relationships and is easy to interpret, its performance was lower than the other models in this experiment. |
| **kNN** | Achieved **97.37% accuracy**, **0.9884 AUC**, **0.9600 precision**, **1.0000 recall**, **0.9796 F1 score**, and **0.9442 MCC**. Its **perfect recall** means that all positive-class observations in the test set were correctly identified. Standardization is important because kNN relies on distances between observations. |
| **Naive Bayes** | Achieved **93.86% accuracy**, **0.9878 AUC**, **0.9452 precision**, **0.9583 recall**, **0.9517 F1 score**, and **0.8676 MCC**. Its relatively high AUC indicates good class-separation ability, although its overall performance was below Logistic Regression and kNN. |
| **Random Forest** | Achieved **94.74% accuracy**, **0.9937 AUC**, **0.9583 precision**, **0.9583 recall**, **0.9583 F1 score**, and **0.8869 MCC**. Its very high AUC indicates strong class separation, but it did not achieve the same overall performance as Logistic Regression or kNN on this test split. |
| **Overall Winner** | **Logistic Regression** is the strongest overall model in this experiment. It achieved the highest **accuracy, precision, F1 score, MCC, and AUC**, while maintaining balanced precision and recall. **kNN** was the closest competitor, particularly because of its perfect recall. |
