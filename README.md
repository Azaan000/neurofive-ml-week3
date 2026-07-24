##Titanic Survival Prediction using Logistic Regression with Hyperparameter Tuning TASK1

This project predicts passenger survival on the Titanic dataset using Logistic Regression. The data is preprocessed by handling missing values and encoding categorical features. The model is trained and evaluated before and after hyperparameter tuning using GridSearchCV to compare performance.

Features
Data preprocessing
Handles missing values
One-Hot Encoding for categorical variables
Logistic Regression classifier
Hyperparameter tuning using GridSearchCV
Model evaluation using:
Accuracy
Confusion Matrix
Classification Report
Performance comparison between the original and tuned models
Technologies Used
Python 3
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Dataset

The project uses the Kaggle Titanic dataset.

## Churn Prediction Project TASK2


This project focuses on predicting customer churn using machine learning models. It is divided into two tasks:

1. Logistic Regression and Decision Tree Classifier for churn prediction.
2. Data preprocessing and feature engineering for churn prediction.

---

## Logistic Regression and Decision Tree Classifier

### Description
In this task, we implemented two machine learning models:
- **Logistic Regression**: A statistical model used for binary classification.
- **Decision Tree Classifier**: A tree-based model that splits data based on feature importance.

### Steps
1. **Data Loading**: The dataset is loaded from a CSV file.
2. **Data Cleaning**: Missing values are handled, and categorical variables are encoded.
3. **Model Training**:
   - Logistic Regression is used to predict churn probabilities.
   - Decision Tree Classifier is used to identify important features driving churn.
4. **Evaluation**: Accuracy and feature importance are calculated.

### Key Outputs
- **Accuracy**: The accuracy of both models on the test set.
- **Feature Importance**: Top features driving churn based on the Decision Tree Classifier.

---

## Data Preprocessing and Feature Engineering

### Description
This task focuses on preparing the dataset for machine learning models by:
- Handling missing values.
- Encoding categorical variables.
- Scaling numerical features.

### Steps
1. **Data Loading**: The dataset is loaded from `data/churn.csv`.
2. **Data Cleaning**:
   - The `TotalCharges` column is converted to numeric, and missing values are filled with the median.
   - The `Churn` column is mapped to binary values (`No` → 0, `Yes` → 1).
   - The `customerID` column is dropped as it is not useful for modeling.
3. **Feature Engineering**:
   - Categorical variables are one-hot encoded using `pd.get_dummies()`.
   - Numerical features are scaled using `StandardScaler`.
4. **Train-Test Split**: The dataset is split into training and testing sets.

### Key Outputs
- **Preprocessed Dataset**: A clean and ready-to-use dataset for machine learning models.
- **Train-Test Split**: Separate datasets for training and testing.

---

## Requirements

To run the project, you need the following Python libraries:
- `pandas`
- `scikit-learn`

```bash
pip install pandas scikit-learn
