# Machine Learning Projects

This repository contains two machine learning classification projects developed using **Python** and **Scikit-learn**:

1. **Titanic Survival Prediction**
2. **Customer Churn Prediction**

Both projects demonstrate the complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and performance improvement.

---

# Project 1: Titanic Survival Prediction

## Description

This project predicts whether a passenger survived the Titanic disaster using **Logistic Regression**. The dataset is preprocessed by handling missing values, encoding categorical features, and tuning the model using **GridSearchCV**.

### Steps

1. Load the Titanic dataset.
2. Handle missing values.
3. Remove unnecessary columns.
4. Encode categorical features using One-Hot Encoding.
5. Split the dataset into training and testing sets.
6. Train a Logistic Regression model.
7. Perform hyperparameter tuning using GridSearchCV.
8. Evaluate the model using multiple metrics.
9. Compare the original and tuned models.

### Key Outputs

- Best Hyperparameters
- Accuracy Score
- Confusion Matrix
- Classification Report
- Original vs Tuned Model Comparison

---

# Project 2: Customer Churn Prediction

## Description

This project predicts whether a customer will leave a company using machine learning models. It includes data preprocessing, feature engineering, Logistic Regression, and Decision Tree Classification.

### Task 1: Logistic Regression & Decision Tree

#### Steps

1. Load the customer churn dataset.
2. Clean the dataset.
3. Encode categorical variables.
4. Train Logistic Regression and Decision Tree models.
5. Evaluate model performance.

#### Key Outputs

- Model Accuracy
- Classification Report
- Confusion Matrix
- Decision Tree Feature Importance

---

### Task 2: Data Preprocessing & Feature Engineering

#### Steps

1. Convert `TotalCharges` to numeric.
2. Handle missing values.
3. Convert the target variable (`Churn`) into binary values.
4. Remove unnecessary columns.
5. Apply One-Hot Encoding to categorical features.
6. Scale numerical features using StandardScaler.
7. Split the dataset into training and testing sets.

#### Key Outputs

- Cleaned Dataset
- Encoded Features
- Scaled Numerical Features
- Train-Test Split

---

# Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```


# Evaluation Metrics

Both projects evaluate model performance using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score

The Titanic project additionally compares the **original Logistic Regression model** with the **hyperparameter-tuned model**, while the Churn Prediction project compares **Logistic Regression** and **Decision Tree Classifier** performance.

> **Note:** Accuracy alone may not always reflect model performance, especially on imbalanced datasets. Therefore, Confusion Matrices, Precision, Recall, and F1-Scores are also used for a more comprehensive evaluation.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Author

**Syed Azan Ahmed**  
Bachelor's in Artificial Intelligence  
Hamdard University
