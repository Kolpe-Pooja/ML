# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  
warnings.filterwarnings('ignore')

# Load the dataset
advertising = pd.read_csv("Salary_linear_regration.csv")  # Change file name if needed
print(advertising.head())  # Display first few rows

# Check column names
print("Column Names:", advertising.columns)

# Scatterplot to visualize YearsExperience vs Salary
sns.pairplot(advertising, x_vars=['YearsExperience'], y_vars='Salary', height=4, kind='scatter')
plt.show()

# Heatmap to check correlation
sns.heatmap(advertising.corr(), cmap='YlGnBu', annot=True)
plt.show()

# Splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split
X = advertising[['YearsExperience']]  # Independent variable (needs to be a DataFrame)
y = advertising['Salary']  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

# Print dataset shapes
print("Training Set:", X_train.shape, y_train.shape)
print("Test Set:", X_test.shape, y_test.shape)

# Performing OLS Regression using statsmodels
import statsmodels.api as sm

# Adding constant to training set
X_train_sm = sm.add_constant(X_train)

# Fit the regression model
lr = sm.OLS(y_train, X_train_sm).fit()

# Print regression summary
print(lr.summary())

# Predict using test data
X_test_sm = sm.add_constant(X_test)
y_pred = lr.predict(X_test_sm)

# Evaluate Model Performance
from sklearn.metrics import mean_squared_error, r2_score

# RMSE (Root Mean Squared Error)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# R-squared Value
print("R-squared:", r2_score(y_test, y_pred))

# Visualizing Regression Line
plt.scatter(X_train, y_train, label="Actual Data", color="blue")
plt.plot(X_train, lr.predict(X_train_sm), label="Regression Line", color="red")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Years of Experience vs Salary")
plt.legend()
plt.show()
