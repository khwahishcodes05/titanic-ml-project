# Import required libraries for data handling, visualization and ML models
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing tools
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Train-test split and model selection
from sklearn.model_selection import train_test_split, cross_val_score

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ignore warnings for clean output
import warnings
warnings.filterwarnings("ignore")

# Load Titanic dataset from seaborn library
df = sns.load_dataset("titanic")

# Display first 5 rows to understand data structure
print(df.head())

# Basic dataset information
print(df.shape)        # rows and columns
print(df.size)         # total elements
print(df.describe())   # statistical summary
print(df.info())       # data types and null values
print(df.isnull().sum())  # missing values count
print(df.duplicated().sum())  # duplicate rows count

# Drop irrelevant columns that are not useful for prediction
df.drop(["class","who","adult_male","alive","embark_town","deck"], axis=1, inplace=True)

# Fill missing values in 'age' with mean
df["age"].fillna(df["age"].mean(), inplace=True)

# Fill missing values in 'embarked' with mode
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

# Convert categorical variables into numerical form
encoder = LabelEncoder()
for col in ["sex", "embarked"]:
    df[col] = encoder.fit_transform(df[col])

# Convert boolean column into integer (0/1)
df["alone"] = df["alone"].astype(int)


# Separate input features and target variable
X = df.drop("survived", axis=1)
y = df["survived"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features for models like Logistic Regression and SVM
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train)
X_test1 = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train1, y_train)

# Predict on test data
y_pred = model.predict(X_test1)

# Evaluate Logistic Regression model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation score
print(cross_val_score(model, X_train1, y_train, cv=5).mean())

# Support Vector Machine (SVM) model training
# SVM tries to find the best hyperplane that separates classes

model_svm = SVC(kernel="linear")

# Train SVM model on scaled training data
model_svm.fit(X_train1, y_train)

# Predict on test data
y_svm = model_svm.predict(X_test1)

# Evaluate SVM performance
print("ACCURACY OF SVM:", accuracy_score(y_test, y_svm))
print("SVM Cross Validation Score:", cross_val_score(model_svm, X_train1, y_train, cv=5).mean())

# Random Forest Classifier model
# Ensemble method that builds multiple decision trees and combines results

model_r = RandomForestClassifier(
    n_estimators=100,   # number of trees
    max_depth=5,        # limit depth of each tree
    random_state=42     # for reproducibility
)

# Train Random Forest on original (unscaled) data
model_r.fit(X_train, y_train)

# Predict on test data
y_r = model_r.predict(X_test)

# Evaluate Random Forest performance
print("Accuracy of Random Forest:", accuracy_score(y_test, y_r))
print("Random Forest Cross Validation Score:", cross_val_score(model_r, X_train, y_train, cv=5).mean())