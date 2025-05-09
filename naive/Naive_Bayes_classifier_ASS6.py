import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

# Load dataset
df = pd.read_csv('Naive_Bayes_classifier.csv')

# Basic data inspection
print(df.info())
print(df.head())

# Visualize target distribution by purpose
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='purpose', hue='not.fully.paid')
plt.title("Loan Purpose vs Default Status")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# One-hot encode categorical column 'purpose'
pre_df = pd.get_dummies(df, columns=['purpose'], drop_first=True)

# Split data
X = pre_df.drop('not.fully.paid', axis=1)
y = pre_df['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)

# Train Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Loan Dataset - Accuracy:", accuracy)
print("Loan Dataset - F1 Score:", f1)
print(classification_report(y_test, y_pred, target_names=["Fully Paid", "Not Fully Paid"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = ["Fully Paid", "Not Fully Paid"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.title("Confusion Matrix - Loan Dataset")
plt.show()