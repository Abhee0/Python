# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# Define file paths
file_path = r"C:\Users\abhis\OneDrive\Desktop\My Playground\titanic\train.csv"
save_path_csv = r"C:\Users\abhis\OneDrive\Desktop\My Playground\titanic\titanic_preprocessed.csv"
save_path_excel = r"C:\Users\abhis\OneDrive\Desktop\My Playground\titanic\titanic_preprocessed.xlsx"


# Load the dataset
try:
    df = pd.read_csv(file_path)
    print("\n[INFO] Dataset Loaded Successfully!")
except Exception as e:
    print("\n[ERROR] Failed to Load Dataset:", e)

# Basic Information
print("\n[INFO] Dataset Info:")
df.info()

print(f"\n[INFO] Dataset Shape: {df.shape}")

print("\n[INFO] First 5 Rows:")
print(df.head())

# Summary Statistics
print("\n[INFO] Summary of Numerical Features:")
print(df.describe())

print("\n[INFO] Summary of Categorical Features:")
print(df.describe(include=['O']))

# Missing Values
print("\n[INFO] Missing Values Count:")
print(df.isnull().sum())

# Visualizing missing values
plt.figure(figsize=(10, 5))
msno.bar(df)
plt.title("Missing Values Overview")
plt.show()

# Check for Duplicates
print("\n[INFO] Duplicate Rows:", df.duplicated().sum())

# Data Distribution
# Histogram for numerical features
df.hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numerical Features")
plt.show()

# Countplot for categorical columns (e.g., Gender)
plt.figure(figsize=(5, 3))
sns.countplot(x='Sex', data=df)
plt.title("Count of Passengers by Gender")
plt.show()

# Correlation Analysis
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Outlier Detection
# Boxplot for 'Fare'
plt.figure(figsize=(8, 4))
sns.boxplot(x='Fare', data=df)
plt.title("Boxplot of Fare Prices")
plt.show()

# Handling Missing Values
# 1. Drop Columns with Too Many Missing Values
threshold = 0.6
missing_ratio = df.isnull().sum() / len(df)
columns_to_drop = missing_ratio[missing_ratio > threshold].index
df.drop(columns=columns_to_drop, inplace=True)
print(f"\n[INFO] Dropped Columns: {list(columns_to_drop)}")

# 2. Fill Missing Values for Numerical Columns
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"[INFO] Filled missing values in '{col}' with median ({median_value})")

# 3. Fill Missing Values for Categorical Columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"[INFO] Filled missing values in '{col}' with mode ('{mode_value}')")

# Verify That No Missing Values Remain
print("\n[INFO] Missing Values After Handling:")
print(df.isnull().sum())

# Handling Categorical Data
# Identify Categorical Columns
cat_cols = df.select_dtypes(include=['object']).columns
print("\n[INFO] Categorical Columns:", list(cat_cols))

# Label Encoding for Binary Categorical Features
binary_cols = ['Sex']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
    print(f"[INFO] Applied Label Encoding to '{col}'")

# One-Hot Encoding for Multi-Class Categorical Features
multi_class_cols = ['Embarked']
df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)
print(f"[INFO] Applied One-Hot Encoding to: {multi_class_cols}")

# Feature Scaling
# Selecting Numerical Columns for Scaling
num_features = ['Age', 'Fare']
df_selected = df[num_features]

# Apply Min-Max Scaling (Normalization)
minmax_scaler = MinMaxScaler()
df_minmax_scaled = pd.DataFrame(minmax_scaler.fit_transform(df_selected), columns=df_selected.columns)
print("\n[INFO] Min-Max Scaled Data (0 to 1 Range):")
print(df_minmax_scaled.head())

# Apply Standardization (Z-score Normalization)
standard_scaler = StandardScaler()
df_standardized = pd.DataFrame(standard_scaler.fit_transform(df_selected), columns=df_selected.columns)
print("\n[INFO] Standardized Data (Mean = 0, Std Dev = 1):")
print(df_standardized.head())

# Replace Scaled Values in Original Dataset
df[num_features] = df_minmax_scaled
print("\n[INFO] Feature Scaling Applied Successfully!")

# Outlier Detection & Removal
# Detect Outliers Using Boxplots
plt.figure(figsize=(8, 4))
sns.boxplot(x='Fare', data=df)
plt.title("Boxplot of Fare Prices (Before Outlier Removal)")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x='Age', data=df)
plt.title("Boxplot of Age (Before Outlier Removal)")
plt.show()

# Remove Outliers Using IQR Method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

# Apply to 'Fare' and 'Age' columns
df = remove_outliers_iqr(df, 'Fare')
df = remove_outliers_iqr(df, 'Age')

# Verify Outlier Removal Using Boxplots
plt.figure(figsize=(8, 4))
sns.boxplot(x='Fare', data=df)
plt.title("Boxplot of Fare Prices (After Outlier Removal)")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x='Age', data=df)
plt.title("Boxplot of Age (After Outlier Removal)")
plt.show()

print("\n[INFO] Outlier Detection & Removal Completed!")

# Save the Preprocessed Dataset
# Ensure the directory exists
os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)

try:
    # Save as CSV File
    df.to_csv(save_path_csv, index=False)
    print(f"\n[INFO] Dataset successfully saved as CSV at: {save_path_csv}")

    # Save as Excel File
    df.to_excel(save_path_excel, index=False)
    print(f"[INFO] Dataset successfully saved as Excel at: {save_path_excel}")

except Exception as e:
    print("\n[ERROR] Failed to save dataset:", e)

# Verify the Saved File Exists
if os.path.exists(save_path_csv):
    print("\n[INFO] CSV file saved successfully and verified!")
else:
    print("\n[ERROR] CSV file not found. Check the save path.")

if os.path.exists(save_path_excel):
    print("[INFO] Excel file saved successfully and verified!")
else:
    print("[ERROR] Excel file not found. Check the save path.")
