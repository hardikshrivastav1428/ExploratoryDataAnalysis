import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

df = pd.read_csv("data/titanic.csv")  

print("----- INFO -----")
print(df.info())
print("\n----- FIRST 5 ROWS -----")
print(df.head())

print("\n----- SUMMARY STATISTICS -----")
print(df.describe(include="all"))

print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

df.hist(figsize=(12, 8))
plt.suptitle("Histograms of Numerical Columns", y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=np.number))
plt.title("Boxplot for Numerical Features")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

try:
    sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']])
    plt.show()
except:
    print("Pairplot skipped due to missing or incompatible data.")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Survival by Gender")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("Survival by Passenger Class")
plt.show()

print("\n----- KEY INSIGHTS -----")
print("""
1. Age has missing values → needs imputation.
2. Females had significantly higher survival rate.
3. First-class passengers survived more than third class.
4. Fare distribution is right-skewed → has high outliers.
5. Pclass is negatively correlated with Fare.
6. Several numerical features show moderate-to-high skewness.
""")