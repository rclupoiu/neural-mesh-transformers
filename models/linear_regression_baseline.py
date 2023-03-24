import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# # Load the data. Flatten X to be 2D.
# X = np.load('/workspace/data_gen/X.npy')
# X = X.reshape(X.shape[0], -1)[:1000]
# y = np.load('/workspace/data_gen/y.npy')[:1000]

# #Randomly shuffle the data
# np.random.seed(42)
# np.random.shuffle(X)
# np.random.seed(42)
# np.random.shuffle(y)

supernodes = np.load('/workspace/models/supernode_indices_allcoords.npy')

X = np.load('/workspace/data_gen/X.npy')[:, supernodes, :]
X = X.reshape(X.shape[0], -1)
y = np.load('/workspace/data_gen/y.npy')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression object
reg = LinearRegression()

print("Training model...")

# Train the model using the training sets
reg.fit(X_train, y_train)

print("Evaluating model...""")

# Make predictions on the testing set
y_pred = reg.predict(X_test)

# Evaluate the performance of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("R-squared: ", r2)
