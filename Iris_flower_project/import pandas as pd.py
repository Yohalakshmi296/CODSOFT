import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv('IRIS.csv')
print('Data Loaded!')
data.head()
print(data.info())
print(data.describe())
print('\nMissing values:\n', data.isnull().sum())
print('\nSpecies Distribution:\n', data['species'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='species', data=data, palette='Set2')
plt.title('Class Distribution')
plt.show()
sns.pairplot(data, hue='species', palette='husl')
plt.suptitle('Pairplot of Iris Features', y=1.02)
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(data.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])
print('Species column after encoding:')
print(data['species'].head())
X = data.drop('species', axis=1)
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}')
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print('\nLogistic Regression Results')
print('Accuracy:', accuracy_score(y_test, y_pred_log))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_log))
print('\nClassification Report:\n', classification_report(y_test, y_pred_log))
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print('\nRandom Forest Results')
print('Accuracy:', accuracy_score(y_test, y_pred_rf))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_rf))
print('\nClassification Report:\n', classification_report(y_test, y_pred_rf))
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(8,4))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Random Forest Feature Importances')
plt.show()
