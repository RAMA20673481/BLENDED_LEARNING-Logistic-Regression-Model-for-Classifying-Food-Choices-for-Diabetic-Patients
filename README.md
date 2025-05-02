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
Developed by: G Ramanujam
RegisterNumber:  212224240129
*/
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\admin\Downloads\food_items_binary.csv'  # Ensure the path is corrected
data = pd.read_csv(file_path)

# Print column names
print("Column Names in the Dataset:")
print(data.columns)

# Separate features (X) and target (y)
X = data.drop(columns=['class'])  # Nutritional information as features
y = data['class']  # Target: 1 (suitable), 0 (not suitable)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict the classifications on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
evaluation_report = classification_report(y_test, y_pred)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print results
print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:\n", evaluation_report)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Suitable', 'Suitable'], yticklabels=['Not Suitable', 'Suitable'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show() 
*/
```

## Output:
![Screenshot 2025-05-02 215902](https://github.com/user-attachments/assets/3741f468-7b95-4de8-bd02-5636f534637b)
![Screenshot 2025-05-02 215925](https://github.com/user-attachments/assets/58c1721c-8b6e-47ad-83c9-56eed8a284f0)



## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
