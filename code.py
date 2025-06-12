import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.impute import SimpleImputer 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix 

# Load the dataset 
df = pd.read_csv("/content/Loan-Prediction.csv") 

# Display the first few rows of the dataset 
print(df.head()) 

# Display summary statistics 
print(df.describe()) 

# Display data types and missing values 
print(df.info()) 

# Check for missing values 
print(df.isnull().sum()) 

# Convert 'Dependents' column to string for consistency 
df['Dependents'] = df['Dependents'].astype(str) 

# Define imputers 
numerical_imputer = SimpleImputer(strategy='median') 
categorical_imputer = SimpleImputer(strategy='most_frequent') 

# List of numerical and categorical columns 
numerical_columns = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History'] 
categorical_columns = ['Gender', 'Married', 'Dependents', 'Self_Employed'] 

# Impute missing values 
df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns]) 
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns]) 

# Visualize numerical data 
df.hist(bins=50, figsize=(20, 15)) 
plt.show() 

# Visualize categorical relationships
sns.countplot(x='Credit_History', hue='Gender', data=df)
plt.show()

sns.countplot(x='Credit_History', hue='Married', data=df)
plt.show()

# Encode categorical variables
label_encoder = LabelEncoder() 
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'] 
for column in categorical_columns: 
    df[column] = label_encoder.fit_transform(df[column]) 

# Drop 'Loan_ID' as it is not useful for prediction 
df = df.drop('Loan_ID', axis=1)

# Split features and target
X = df.drop('Credit_History', axis=1) 
y = df['Credit_History'] 

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Initialize and train the model 
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train) 

# Predict on test set 
y_pred_test = model.predict(X_test) 
accuracy_test = accuracy_score(y_test, y_pred_test) 
print(f'Test Accuracy: {accuracy_test}') 

# Predict on training set 
y_pred_train = model.predict(X_train) 
accuracy_train = accuracy_score(y_train, y_pred_train) 
print(f'Training Accuracy: {accuracy_train}')
