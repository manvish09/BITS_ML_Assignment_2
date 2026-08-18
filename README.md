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

## 6. Observations

### Logistic Regression

Logistic Regression produced the strongest overall performance in this experiment. It achieved **98.25% accuracy**, an **AUC of 0.9954**, and an **MCC of 0.9623**. Its precision, recall and F1 score were also well balanced at 0.9861.

The results indicate that the standardized numerical features work particularly well with a linear classification approach for this dataset.

### Decision Tree

The Decision Tree achieved **92.11% accuracy**, which was the lowest accuracy among the five models. Its AUC was 0.9163 and MCC was 0.8341.

Although a decision tree can capture nonlinear relationships and is relatively easy to interpret, the single tree configuration used in this experiment performed below the other models on the selected test split.

### kNN

kNN achieved **97.37% accuracy** and an **F1 score of 0.9796**. Its recall was **1.0000**, meaning that none of the actual positive-class observations in the test set were missed by the model.

The model also achieved an AUC of 0.9884 and MCC of 0.9442. Feature standardization is important for kNN because the model relies on distances between observations.

### Naive Bayes

Gaussian Naive Bayes achieved **93.86% accuracy** and an **AUC of 0.9878**. The high AUC indicates good class-separation capability, although its accuracy, F1 score and MCC were lower than those of Logistic Regression and kNN.

### Random Forest

Random Forest achieved **94.74% accuracy** and an **AUC of 0.9937**. Its high AUC indicates strong class-separation capability.

However, on this particular test split, its accuracy and MCC were lower than those of Logistic Regression and kNN.

### Overall Winner

Based on the complete set of evaluation metrics, **Logistic Regression was the strongest overall model in this experiment**.

It achieved the highest accuracy, precision, F1 score and MCC, while also recording an AUC of 0.9954.

kNN was a close second and achieved the highest recall of 1.0000.
