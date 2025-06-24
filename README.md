# Day-2-internship-task2
 EDA helps uncover meaningful patterns, clean the dataset, and prepare for machine learning modeling. Visual tools and statistics are essential to understanding data deeply before building models.

# 🚢 Titanic Dataset - Exploratory Data Analysis (EDA)

This project performs a complete EDA (Exploratory Data Analysis) on the Titanic dataset using Pandas, Seaborn, Matplotlib, and Plotly.

---

## 📁 Dataset

The dataset used is the [Titanic Dataset from Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset).  

---

## ✅ Objectives Covered (Hints/Mini Guide)

1. Generate summary statistics (mean, median, std, etc.)
2. Create histograms and boxplots for numeric features
3. Use pairplot/correlation matrix for feature relationships
4. Identify patterns, trends, or anomalies
5. Make basic feature-level inferences from visuals

---

## 📊 Full EDA Code

```python
# 🚢 Titanic Dataset EDA – All-in-One

# 📦 Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

# ✅ Set styling and suppress warnings
sns.set(style="whitegrid")
%matplotlib inline
warnings.simplefilter(action='ignore', category=FutureWarning)

# 📥 Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# 🧹 Clean up infinite values that trigger seaborn warnings
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 📊 Basic stats and structure
print("Shape:", df.shape)
df.info()
display(df.describe())
print("\nMedian:\n", df[['Age','Fare','SibSp','Parch']].median())
print("\nStandard Deviation:\n", df[['Age','Fare','SibSp','Parch']].std())

# 🧹 Clean missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Survived'] = df['Survived'].astype(str)

# 📈 Histograms
df[['Age', 'Fare', 'SibSp', 'Parch']].hist(figsize=(10, 8), bins=20, color='lightblue', edgecolor='black')
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()

# 📦 Boxplots
plt.figure(figsize=(10, 6))
for i, col in enumerate(['Age', 'Fare', 'SibSp', 'Parch'], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x=col, color='salmon')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# 🔗 Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Survived', 'Age', 'Fare', 'SibSp', 'Parch']].astype(float).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 🔍 Pairplot
sns.pairplot(df[['Survived', 'Age', 'Fare', 'SibSp', 'Parch']], hue='Survived', palette='husl')
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# 📊 Countplots (Patterns/Trends)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.countplot(data=df, x='Sex', hue='Survived', palette='Set2')
plt.title('Survival by Sex')

plt.subplot(1, 3, 2)
sns.countplot(data=df, x='Pclass', hue='Survived', palette='Set1')
plt.title('Survival by Class')

plt.subplot(1, 3, 3)
sns.countplot(data=df, x='Embarked', hue='Survived', palette='Set3')
plt.title('Survival by Embarked Port')

plt.tight_layout()
plt.show()

# ✨ Interactive Plotly Chart
fig = px.histogram(df, x='Age', color='Survived',
                   nbins=30, title="Age Distribution by Survival",
                   labels={'Survived': 'Survived'},
                   color_discrete_map={'0': 'red', '1': 'green'})
fig.show()




🧠 Observations
Sex: Females had significantly higher survival rates.

Pclass: First-class passengers had better survival odds.

Age: Children were more likely to survive.

Fare: Higher fare correlated with higher survival.

Embarked: Port 'C' had a slight survival advantage.

👨‍💻 Author
Nayan Nakum
B.Sc (CS/IT) | AI/ML Enthusiast
📫 nayan.nakum2005@gmail.com
📍 Gota, Ahmedabad
🔗 

🔗 References
Kaggle Titanic Dataset

Matplotlib

Seaborn

Plotly
