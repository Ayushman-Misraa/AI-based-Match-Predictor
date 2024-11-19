import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def sigmoid_func(X, theta):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]

def loss(pred_y1, y):
    return (-y * np.log(pred_y1) - (1 - y) * np.log(1 - pred_y1)).mean()

def update_theta(theta, alpha, gradient):
    return theta - alpha * gradient

def train(itr, l_rate, theta, x_train, y_train):
    optimized_thetas = []
    for rate in l_rate:
        for _ in range(itr):
            y1 = sigmoid_func(x_train, theta)
            gradient = gradient_descent(x_train, y1, y_train)
            theta = update_theta(theta, rate, gradient)
        optimized_thetas.append(theta)
    return optimized_thetas

def predict(itr, l_rate, thetas, x_test, y_test):
    results = []
    for index, theta in enumerate(thetas):
        pred_y = sigmoid_func(x_test, theta)
        pred_classes = (pred_y >= 0.5).astype(int)
        accuracy = (pred_classes.flatten() == y_test).mean() * 100
        results.append({"learning_rate": l_rate[index], "accuracy": accuracy})
    return results

# Load dataset
dataFrame = pd.read_csv("Dataset/encoded_dataset.csv")

# Mapping
dataFrame["teampoint_diff"] = (dataFrame["teampoint_diff"] >= 0).astype(int)

# Feature and target variables
feature_vars = [
    'team1', 'powerplay_runs_team1', 'powerplay_wickets_team1', 
    'team2', 'powerplay_runs_team2', 'powerplay_wickets_team2', 
    'homeground_advantage', 'team1_toss_win', 'teampoint_diff'
]
target_var = 'team1_win'

X_features = dataFrame[feature_vars]
y_target = dataFrame[target_var]

learning_rate = [0.001, 0.01, 0.1]
iterations = 5000

intercept = np.ones((X_features.shape[0], 1))
X_features = np.concatenate((intercept, X_features), axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X_features, y_target, test_size=1/3, random_state=0)
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

theta = np.zeros(X_train.shape[1])

# Train the model
model = train(iterations, learning_rate, theta, X_train, Y_train)

# Predict and display results
predictions = predict(iterations, learning_rate, model, X_test, Y_test)
for result in predictions:
    print(f"Learning Rate: {result['learning_rate']}, Accuracy: {result['accuracy']:.2f}%")
