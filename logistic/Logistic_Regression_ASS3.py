import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

# Column names
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# Load dataset
pima = pd.read_csv("Logistic _Regression_prime-indian-diabetes.csv", header=None, names=col_names)

# Convert data to numeric and drop missing values
pima = pima.apply(pd.to_numeric, errors='coerce')
pima.dropna(inplace=True)

# Select features and label
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]  
y = pima['label'] 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Create and fit the logistic regression model
logreg = LogisticRegression(random_state=16, max_iter=1000, solver='liblinear', class_weight='balanced')
logreg.fit(X_train, y_train)  # <-- You missed this line

# Predict
y_pred = logreg.predict(X_test)

# Confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names = ['Without Diabetes', 'With Diabetes']

plt.figure(figsize=(6,4))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title("Confusion Matrix")
plt.show()

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
target_names = ['Without Diabetes', 'With Diabetes']
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# ROC curve
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='blue', label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], color='gray', linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
