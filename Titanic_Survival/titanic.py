# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Data
data = pd.read_csv(r'c:\CodSoft_Yoha\Titanic_Survival\Titanic-Dataset.csv')
print('Data loaded successfully!')
data.head()
# 3. Basic Info
print(data.info())
print(data.describe())
print(data.isnull().sum())
# 4. Visual EDA
plt.figure(figsize=(8, 4))
sns.countplot(x='Survived', data=data, palette='viridis')
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x='Pclass', hue='Survived', data=data, palette='pastel')
plt.title('Survival Rate by Passenger Class')
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(data['Age'].dropna(), bins=30, kde=True, color='coral')
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x='Pclass', y='Age', data=data, palette='Set2')
plt.title('Age vs Pclass')
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x='Sex', hue='Survived', data=data, palette='coolwarm')
plt.title('Survival by Gender')
plt.show()
# 5. Handling Missing Data
# Check nulls again
print(data.isnull().sum())

# Fill Age with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Drop Cabin (too many missing)
if 'Cabin' in data.columns:
    data.drop(['Cabin'], axis=1, inplace=True)

# Fill Embarked with mode
if 'Embarked' in data.columns:
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

print('✅ Missing values handled.')
# 6. Encoding Categorical Features
label_encoders = {}
for col in ['Sex', 'Embarked']:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

print('Categorical columns encoded.')
# 7. Drop Unnecessary Columns
cols_to_drop = ['Name', 'Ticket', 'PassengerId']
for col in cols_to_drop:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

print(data.head())
# 8. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()
# 9. Feature / Target Split
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print('Data split into train/test.')
# 10. Model 1: Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print('Logistic Regression Results:')
print('Accuracy:', accuracy_score(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
# 11. Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print('Random Forest Results:')
print('Accuracy:', accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
# 12. Feature Importance (Random Forest)
importances = rf_model.feature_importances_
feat_names = X.columns

plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=feat_names, palette='viridis')
plt.title('Random Forest Feature Importances')
plt.show()
