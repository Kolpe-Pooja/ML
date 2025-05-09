# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

def main():
    # Column names
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

    # Load dataset — make sure diabetes.csv is in the same directory as this script
    pima = pd.read_csv("prime-indian-diabetes_decisionTree.csv", names=col_names, skiprows=1)  # Skip header row if present

    # Select feature columns
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
    X = pima[feature_cols]
    y = pima['label']

    # Split dataset into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Create Decision Tree classifier
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    # Train classifier
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Accuracy
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Visualize the decision tree
    plt.figure(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=feature_cols, class_names=['0', '1'])
    plt.show()

if __name__ == "__main__":
    main()
