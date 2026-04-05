# STEP 1 - Data Cleaning

import pandas as pd

df = pd.read_csv("CA2 Murder case of 2001-2012.csv")

# Basic info
print(df.info())

# Missing values
print("Missing Values:\n", df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Fix special characters in column names
df.columns = df.columns.str.replace("&", "and").str.replace("/", "_")

# Convert year safely
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Drop rows where year is invalid
df = df.dropna(subset=['year'])

df['year'] = df['year'].astype(int)

# Fill remaining missing values
df = df.fillna(0)

print("Cleaned Data:")
print(df.head())


# STEP 2 - EDA (Exploratory Data Analysis)

# Total murders per year
yearly_murder = df.groupby('year')['murder'].sum()

# Top states
top_states = df.groupby('state_ut')['murder'].sum().sort_values(ascending=False).head(10)

# Top districts
top_districts = df.groupby('district')['murder'].sum().sort_values(ascending=False).head(10)

print("\nYearly Murder Trend:\n", yearly_murder)
print("\nTop States:\n", top_states)
print("\nTop Districts:\n", top_districts)


# STEP 3 - Visualization

import matplotlib.pyplot as plt

# Graph 1: Murder Trend
plt.figure()
yearly_murder.plot(marker='o')
plt.title("Murder Trend in India (2001-2012)")
plt.xlabel("Year")
plt.ylabel("Total Murders")
plt.grid()
plt.show()

# Graph 2: Top States
plt.figure()
top_states.plot(kind='bar')
plt.title("Top 10 States with Highest Murders")
plt.xlabel("State")
plt.ylabel("Murders")
plt.xticks(rotation=45)
plt.show()

# Graph 3: Crime Distribution
plt.figure()
crime_cols = ['murder','rape','kidnapping_and_abduction','robbery','theft']

df[crime_cols].sum().plot(kind='pie', autopct='%1.1f%%')
plt.title("Crime Distribution")
plt.ylabel("")
plt.show()


# STEP 4 - Statistical Analysis

print("Mean:", df['murder'].mean())
print("Median:", df['murder'].median())
print("Variance:", df['murder'].var())
print("Standard Deviation:", df['murder'].std())


# STEP 5 - Creativity

import seaborn as sns

# Correlation Heatmap
plt.figure()
corr = df[['murder','rape','robbery','theft']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Crime Correlation Matrix")
plt.show()

# Crime Growth Over Time
growth = df.groupby('year')['total_ipc_crimes'].sum()

plt.figure()
growth.plot(marker='o')
plt.title("Total IPC Crime Trend")
plt.xlabel("Year")
plt.ylabel("Total Crimes")
plt.grid()
plt.show()

# Box Plot (Outliers)
plt.figure()
sns.boxplot(data=df[['murder','rape','robbery','theft']])
plt.title("Outlier Detection in Crime Data")
plt.show()


# STEP 6 - Prediction using Linear Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Features
X = df[['rape', 'robbery', 'theft']]

# Target
y = df['murder']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("\nPredicted vs Actual:\n")
for i in range(5):
    print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test.iloc[i]}")

# Evaluation
print("\nR2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Graph (Actual vs Predicted)
plt.figure()
plt.scatter(y_test.values, y_pred)
plt.xlabel("Actual Murder Cases")
plt.ylabel("Predicted Murder Cases")
plt.title("Actual vs Predicted (Linear Regression)")
plt.grid()
plt.show()

# Coefficients with feature names
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Intercept
print("Intercept:", model.intercept_)
