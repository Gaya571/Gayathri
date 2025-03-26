import numpy as np
import pandas as pd

# Load the dataset
file_path = r"C:\Users\kamal\Downloads\data.csv"
df = pd.read_csv(file_path)

# Select relevant numerical features
features = ["bedrooms", "bathrooms", "sqft_living", "floors", "waterfront", "view", "condition", "sqft_above", "sqft_basement", "yr_built"]
X = df[features].values
y = df["price"].values

# Normalize features for better gradient descent performance
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y = (y - np.mean(y)) / np.std(y)

# Add bias term (column of ones)
X = np.c_[np.ones(X.shape[0]), X]

# Initialize parameters
theta = np.zeros(X.shape[1])
learning_rate = 0.01
iterations = 1000

# Gradient Descent
m = len(y)
for _ in range(iterations):
    predictions = X.dot(theta)
    errors = predictions - y
    gradient = (1/m) * X.T.dot(errors)
    theta -= learning_rate * gradient

# Predict function
def predict(features):
    features = np.array(features)
    features = (features - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
    features = np.insert(features, 0, 1)  # Add bias term
    return features.dot(theta)

# Example prediction
example_house = [3, 2, 1500, 1, 0, 0, 3, 1000, 500, 1990]
normalized_price = predict(example_house)
print("Predicted normalized price:", normalized_price)



