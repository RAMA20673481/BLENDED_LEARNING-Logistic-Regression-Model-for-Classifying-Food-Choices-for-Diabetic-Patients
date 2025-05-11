# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Load Data: Import the dataset and inspect column names.
2.Prepare Data: Separate features (X) and target (y).
3.Split Data: Divide into training (80%) and testing (20%) sets.
4.Scale Features: Standardize the data using StandardScaler.
5.Train Model: Fit a Logistic Regression model on the training data.
6.Make Predictions: Predict on the test set.
7.Evaluate Model: Calculate accuracy, precision, recall, and classification report.
8.Confusion Matrix: Compute and visualize confusion matrix.
```
## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: G.RAMANUJAM
RegisterNumber: 212224240129
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset 
df=pd.read_csv("food_items.csv")
#inspect the dataset
print("Dataset Overview")
print(df.head())
print("\ndatset Info")
print(df.info())

X_raw=df.iloc[:, :-1]
y_raw=df.iloc[:, -1:]
X_raw

scaler=MinMaxScaler()
X=scaler.fit_transform(X_raw)

label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_raw.values.ravel())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

penalty='l2'
multi_class='multnomial'
solver='lbfgs'
max_iter=1000

model = LogisticRegression(max_iter=2000)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Confusion Matrix Plot
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False, 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

## Output:

![Screenshot 2025-05-11 191719](https://github.com/user-attachments/assets/bcf3a353-31ba-4671-920f-c8f835751712)
![Screenshot 2025-05-11 191734](https://github.com/user-attachments/assets/0843d466-a805-4e0e-bae0-1748d5fe549d)
![Screenshot 2025-05-11 191747](https://github.com/user-attachments/assets/87c65bff-6678-4e2e-b027-143ca13d49cf)
![Screenshot 2025-05-11 191758](https://github.com/user-attachments/assets/b082091a-9146-4679-8ced-1a0cf85b5a9a)




## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
