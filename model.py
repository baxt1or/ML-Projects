import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example data
data = {
    'tmax': [50, 60, 70, 80, 90, 100, 110],
    'tmax_tomorrow': [55, 65, 75, 85, 95, 105, 115]
}

# Load the data into a DataFrame
df = pd.DataFrame(data)

# Define the feature and target
X = df[['tmax']]
y = df['tmax_tomorrow']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test_scaled, y_test)
print(f'Mean Squared Error on test set: {loss:.2f}')

# Make predictions
y_pred = model.predict(X_test_scaled)

# Visualize the results
plt.figure(figsize=(10, 6))

# Scatter plot of the original data
plt.scatter(X, y, color='blue', label='Actual data')

# Plot the neural network predictions
plt.scatter(X_test, y_pred, color='red', label='NN predictions')

# Plot the perfect correlation line (45-degree line)
plt.plot([30, 120], [30, 120], "green", label='Perfect correlation line')

plt.xlabel('Tmax Today')
plt.ylabel('Tmax Tomorrow')
plt.title('Neural Network: Tmax Today vs. Tmax Tomorrow')
plt.legend()
plt.grid(True)
plt.show()
