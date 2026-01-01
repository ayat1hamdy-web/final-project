#House Price Prediction Using Linear Regression
# 1. Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# 2. Load the Dataset
# Load dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

# Display first rows
df.head()
# 3. Data Preprocessing
df.isnull().sum()
# Encode Categorical Variables
# Categorical columns:
# Location
# Condition
# Garage
label_encoder = LabelEncoder()

categorical_cols = ['Location', 'Condition', 'Garage']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])
# 4. Data Visualization
# Data Visualization
plt.figure()
plt.scatter(df['Area'], df['Price'])
plt.xlabel("Area (sq ft)")
plt.ylabel("House Price")
plt.title("Area vs House Price")
plt.show()
# Average Price by Number of Bedrooms
df.groupby('Bedrooms')['Price'].mean().plot(kind='bar')
plt.xlabel("Bedrooms")
plt.ylabel("Average Price")
plt.title("Average House Price by Bedrooms")
plt.show()
# 5. Split the Data (Train–Test)
X = df.drop(['Price', 'Id'], axis=1)  # Features
y = df['Price']                      # Target

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 6. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
# 7. Model Evaluation
# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-Squared (R²):", r2)
# 8.Actual vs Predicted Prices Visualization
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
