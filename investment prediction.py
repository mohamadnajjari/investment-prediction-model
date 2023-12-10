import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('investment_data.csv')
data

fig, ax = plt.subplots()
x_values = data['company']
y_values = data['Returns'] 
ax.scatter(x_values, y_values)
ax.set_xlabel('Samples')
ax.set_ylabel('Return values')
ax.set_title('Initial Scatter Plot')
ax.set_ylim(-.2, 1.5)
plt.show()

column_name = 'Returns'
nan_count = data[column_name].isna().sum()
print("Number of missing Return values:", nan_count)

size = data.shape[0]
print(f"{nan_count/size *100} % of our data set Return column is empty")
data_1 = data.copy()

data_1 = data_1.dropna(subset=['Returns'])
nan_count_1 = data_1['Returns'].isna().sum()
print(f"Number of empty Return rows in out new dataset is = {nan_count_1} and now size of it is = {data_1.shape[0]} ")

columns = data_1.columns
columns

numeric_columns = data_1.select_dtypes(include=['int64', 'float64']).columns
numeric_columns = numeric_columns.drop(['Returns'])
numeric_columns

numeric_features = data_1[numeric_columns]
data_2 = data_1.copy()

ss = StandardScaler()
ss.fit(numeric_features)
numeric_features = ss.transform(numeric_features)
data_2[numeric_columns] = numeric_features
data_2

categorical_columns = data_2.select_dtypes(include=['object']).columns
data_3 = data_2.copy()

data_3 = pd.get_dummies(data_3, columns=categorical_columns)
data_3

data_4 = data_3.copy()
data_4 = data_4.fillna(value=0)
data_4

X = data_4.drop('Returns', axis=1)
y = data_4['Returns']

model = RandomForestClassifier()
model.fit(X, y)
importance_scores = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importance_scores})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

threshold = 0.02
selected_features = feature_importances[feature_importances['Importance'] >= threshold]['Feature']
X_selected = X[selected_features]
X_selected

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
model_2 = LogisticRegression()
model_2.fit(X_train, y_train)

y_pred = model_2.predict(X_test)
y_pred

predicted_probability = model_2.predict_proba(X_test)
predicted_probability

importance_scores = model_2.coef_[0]
sorted_indices = np.argsort(importance_scores)[::-1]
sorted_importances = importance_scores[sorted_indices]


plt.figure(figsize=(10, 6))
plt.bar(range(X_selected.shape[1]), sorted_importances)
plt.xticks(range(X_selected.shape[1]), X_selected.columns[sorted_indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Logistic Regression - Feature Importances')
plt.tight_layout()
plt.show()

