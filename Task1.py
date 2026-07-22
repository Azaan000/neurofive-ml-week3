import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

test = pd.read_csv('data/test.csv')
train = pd.read_csv('data/train.csv')



test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Cabin'] = test['Cabin'].fillna(test['Cabin'].mode()[0])
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Cabin'] = train['Cabin'].fillna(train['Cabin'].mode()[0])
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

encodersex_train = OneHotEncoder(sparse_output=False, drop='first')
encodersex_test = OneHotEncoder(sparse_output=False, drop='first')

encoderembarked_train = OneHotEncoder(sparse_output=False, drop='first')
encoderembarked_test = OneHotEncoder(sparse_output=False, drop='first')


train_sex_encoded = encodersex_train.fit_transform(train[['Sex']])
test_sex_encoded = encodersex_test.fit_transform(test[['Sex']])


train_embarked_encoded = encoderembarked_train.fit_transform(train[['Embarked']])
test_embarked_encoded = encoderembarked_test.fit_transform(test[['Embarked']])


train_sex_df = pd.DataFrame(train_sex_encoded, columns=encodersex_train.get_feature_names_out(['Sex']))
test_sex_df = pd.DataFrame(test_sex_encoded, columns=encodersex_test.get_feature_names_out(['Sex']))

train_embarked_df = pd.DataFrame(train_embarked_encoded, columns=encoderembarked_train.get_feature_names_out(['Embarked']))
test_embarked_df = pd.DataFrame(test_embarked_encoded, columns=encoderembarked_test.get_feature_names_out(['Embarked']))


train = train.drop(columns=['Sex', 'Embarked', 'Name', 'Ticket', 'Cabin'])
test = test.drop(columns=['Sex', 'Embarked', 'Name', 'Ticket', 'Cabin'])

train = pd.concat([train, train_sex_df, train_embarked_df], axis=1)
test = pd.concat([test, test_sex_df, test_embarked_df], axis=1)

testc = pd.isnull(test).sum()
trainc = pd.isnull(train).sum()


print(testc)
print(trainc)


X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=['Survived']), train['Survived'], test_size=0.2, random_state=42)


param_grid = {
    'C': [0.01, 0.1, 1 , 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

gridsearch = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv= 5, scoring='accuracy',n_jobs=1)

gridsearch.fit(X_train, y_train)
bestmodel = gridsearch.best_estimator_

pred = bestmodel.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print(f"Training Accuracy: {accuracy:.4f}")
cm = confusion_matrix(y_test, pred)
print("Confusion Matrix:\n", cm)
report = classification_report(y_test, pred)
print("Classification Report:\n", report)
print("Best Hyperparameters:", gridsearch.best_params_)

