 Data Loading & Overview
Loads data from Loan-Prediction.csv.

Displays the first few rows, summary statistics, data types, and counts of missing values.

Data Cleaning & Imputation
Converts the 'Dependents' column to string type for consistent encoding.

Separates columns into numerical (LoanAmount, Loan_Amount_Term, Credit_History) and categorical (Gender, Married, Dependents, Self_Employed).

Applies median imputation for numerical and most frequent imputation for categorical columns.

 Data Visualization
Plots histograms of all numerical features.

Uses seaborn countplot to show distribution of Credit_History by Gender and Married.

Feature Encoding & Selection
Encodes categorical variables (Gender, Married, Dependents, Education, Self_Employed, Property_Area) using LabelEncoder.

Drops the Loan_ID column which is non-informative.

Model Building & Evaluation
Splits the data into training and test sets.

Trains a RandomForestClassifier with 100 trees.

Evaluates the model on both training and testing sets using accuracy scores.

Data Visualization but using Tableau to get the Dashboards.
