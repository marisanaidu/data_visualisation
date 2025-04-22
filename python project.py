import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
df = pd.read_csv("C:/Users/naidu/OneDrive/Desktop/electric veihcle pollution.csv")
df = pd.DataFrame(df)
print(df)
print("Missing values per column:")
print(df.isna().sum())
df = df.dropna()
for col in df.select_dtypes(include='object').columns:
   df[col] = df[col].str.strip()
df = df.drop_duplicates()
df.reset_index(drop=True, inplace=True)
print("Cleaned data shape:", df.shape)
print(df.head())
print(df.isna().sum())

# Line plot
df.groupby("Model Year")["Electric Range"].mean().plot(kind='line')
plt.title("Average Electric Range by Model Year")
plt.ylabel("Electric Range")
plt.xlabel("Model Year")
plt.grid(True)
plt.show()

#Bar plot
sns.countplot(data=df, x="Electric Vehicle Type")
plt.title("Count of EV Types")

plt.show()

#histogram
sns.histplot(df["Electric Range"], bins=20, kde=True)
plt.title("Distribution of Electric Range")
plt.xlabel("Electric Range")
plt.show()

# Scatter plot
sns.scatterplot(data=df, x="Base MSRP", y="Electric Range", hue="Electric Vehicle Type")
plt.title("Electric Range vs Base MSRP")
plt.show()

# Heat map
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Box Plot
sns.boxplot(data=df, x="Electric Vehicle Type", y="Electric Range")
plt.title("Electric Range Distribution by EV Type")
plt.show()

#Pie Chart
ev_counts = df["Electric Vehicle Type"].value_counts()
plt.pie(ev_counts, labels=ev_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("EV Type Distribution")
plt.axis('equal')
plt.show()

#Count plot
df_cleaned = df.dropna()
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x="Model Year", order=sorted(df_cleaned["Model Year"].unique()))
plt.title("EV Count by Model Year")
plt.xlabel("Model Year")
plt.ylabel("Count")
plt.show()

#Scatter plot with linear regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
df_filtered = df[(df['Electric Range'] > 0) & df['Model Year'].notna()]
X = df_filtered[['Model Year']]
y = df_filtered['Electric Range']
model = LinearRegression()
model.fit(X, y)
x_vals = np.linspace(X.min(), X.max(), 100)
y_pred = model.predict(x_vals)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Model Year', y='Electric Range', data=df_filtered, alpha=0.5, label='Data Points')
plt.plot(x_vals, y_pred, color='red', linewidth=2, label='Linear Regression Line')
plt.title('Electric Range vs Model Year')
plt.xlabel('Model Year')
plt.ylabel('Electric Range (miles)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

##Outlier Detection with IQR (Interquartile Range)
import pandas as pd
df_filtered = df[df['Electric Range'] > 0]
Q1 = df_filtered['Electric Range'].quantile(0.25)
Q3 = df_filtered['Electric Range'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df_filtered[(df_filtered['Electric Range'] < lower_bound) | 
                       (df_filtered['Electric Range'] > upper_bound)]
print("Number of outliers detected:", len(outliers))
print(outliers[['Model Year', 'Make', 'Model', 'Electric Range']])


# outlier detection
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file


# Filter out zero or missing Electric Range values
df_filtered = df[df['Electric Range'] > 0]

# Calculate Q1, Q3, and IQR
Q1 = df_filtered['Electric Range'].quantile(0.25)
Q3 = df_filtered['Electric Range'].quantile(0.75)
IQR = Q3 - Q1

# Determine bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df_filtered[(df_filtered['Electric Range'] < lower_bound) |
                       (df_filtered['Electric Range'] > upper_bound)]

# Print the number of outliers
print("Number of outliers detected:", len(outliers))

# Plot boxplot to visualize outliers
plt.figure(figsize=(10, 4))
sns.boxplot(x='Electric Range', data=df_filtered)
plt.title('Boxplot of Electric Range with Outliers')
plt.xlabel('Electric Range (miles)')
plt.grid(True)
plt.tight_layout()
plt.show()

#  Linear Regression Model Code
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df_filtered = df[(df['Electric Range'] > 0) & df['Model Year'].notna()]
X = df_filtered[['Model Year']]
y = df_filtered['Electric Range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel("Model Year")
plt.ylabel("Electric Range")
plt.title("Linear Regression: Model Year vs Electric Range")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()