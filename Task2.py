import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
df = pd.read_csv("data/churn.csv")


print(df.head())

print(df.isnull().sum())


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

df.drop("customerID", axis=1, inplace=True)


X = pd.get_dummies(df.drop("Churn", axis=1), drop_first=True)
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_model = LogisticRegression(max_iter=1000)
tree_model = DecisionTreeClassifier(random_state=42)


log_model.fit(X_train_scaled, y_train)
tree_model.fit(X_train, y_train)


log_pred = log_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test)



print("Accuracy:", accuracy_score(y_test, log_pred))
print("Accuracy:", accuracy_score(y_test, tree_pred))


importance = pd.DataFrame({"Feature": X.columns,"Importance": tree_model.feature_importances_})
importance = importance.sort_values(by="Importance", ascending=False)
print(importance)

joblib.dump(log_model, "logistic_model.pkl")
joblib.dump(tree_model, "tree_model.pkl")

#The churn prediction model was developed to identify customers who are likely to leave the company, allowing the business to take proactive retention measures. 
#The model analyzes customer characteristics such as contract type, payment method, internet service, and monthly charges to predict churn. 
#Feature importance analysis shows which factors have the greatest influence on customer retention, helping management focus on the most impactful business areas. 
#By using these insights, the company can design targeted offers, improve customer satisfaction, and reduce customer churn, ultimately increasing long-term revenue and customer loyalty.

