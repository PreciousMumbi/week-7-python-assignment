import pandas as pd
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target (species)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display first few rows
print(df.head())

# Check data types and missing values
print(df.info())
print(df.isnull().sum())

# No missing values, but if there were:
# df = df.dropna()  # or df.fillna(value)



# Summary statistics
print(df.describe())

# Mean of features grouped by species
grouped = df.groupby("species").mean()
print(grouped)

# Interesting observation:
# Setosa typically has shorter sepals and petals than Virginica and Versicolor.

import matplotlib.pyplot as plt
import seaborn as sns

# Line chart - Fake time series for example purposes
df['index'] = df.index
plt.plot(df['index'], df['sepal length (cm)'], label='Sepal Length')
plt.title("Trend of Sepal Length")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# Bar chart - Average petal length by species
sns.barplot(x="species", y="petal length (cm)", data=df)
plt.title("Average Petal Length by Species")
plt.show()

# Histogram - Petal Width Distribution
sns.histplot(df['petal width (cm)'], kde=True)
plt.title("Distribution of Petal Width")
plt.show()

# Scatter plot - Sepal vs Petal Length
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal vs Petal Length")
plt.show()
