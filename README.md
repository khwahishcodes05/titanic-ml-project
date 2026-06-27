🚢 Titanic Survival Prediction - Machine Learning Project

📌 Project Overview

This project is an end-to-end Machine Learning implementation on the famous Titanic dataset.
The goal is to predict whether a passenger survived or not based on features like age, gender, class, fare, etc.

Multiple machine learning models are implemented and compared:

Logistic Regression

Support Vector Machine (SVM)

Random Forest Classifier



---

📊 Dataset Information

The dataset is loaded from the seaborn library and contains information about Titanic passengers such as:

Passenger class

Gender

Age

Fare

Embarked location

Survival status (target variable)



---

⚙️ Workflow

1. Data Preprocessing

Handled missing values (Age, Embarked)

Dropped irrelevant columns

Encoded categorical variables using Label Encoding

Converted boolean column into integer


2. Feature Scaling

StandardScaler applied for Logistic Regression and SVM


3. Model Training

Logistic Regression

Support Vector Machine (SVM)

Random Forest Classifier


4. Model Evaluation

Accuracy Score

Classification Report

Confusion Matrix

Cross Validation Score



---

🤖 Models Used

📌 Logistic Regression

Accuracy: ~81%

Good balance between precision and recall


📌 Support Vector Machine (SVM)

Accuracy: ~79%

Performs well with scaled data


📌 Random Forest Classifier

Accuracy: ~78%

Best cross-validation score (~81%)

Most stable model among all



---

📈 Results Summary

Model	Accuracy	Cross Validation

Logistic Regression	81%	79%
SVM	79%	78%
Random Forest	78%	81%



---

🧠 Key Insights

Logistic Regression performed best on test data

Random Forest showed highest stability across folds

SVM performed moderately well

Feature scaling significantly improves linear models



---

🛠️ Libraries Used

pandas

numpy

seaborn

matplotlib

scikit-learn



---

🚀 Future Improvements

Feature engineering to improve accuracy

Hyperparameter tuning (GridSearchCV)

Trying advanced models like XGBoost or LightGBM



---

👩‍💻 Author

Khwahish 


---

📌 Note

This project is part of my Machine Learning learning journey and demonstrates end-to-end workflow including preprocessing, training, and evaluation.
Ye sahi h
