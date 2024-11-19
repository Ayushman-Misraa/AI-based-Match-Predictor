from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from main import train, predict, sigmoid_func
import webbrowser
import threading

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_ipl():
    file = request.files["file"]
    if not file:
        return "No file uploaded!", 400
    df = pd.read_csv(file)

    iterations = int(request.form["iterations"])
    learning_rate = list(map(float, request.form["learning_rate"].split(",")))

    feature_vars = [
        'team1', 'powerplay_runs_team1', 'powerplay_wickets_team1',
        'team2', 'powerplay_runs_team2', 'powerplay_wickets_team2',
        'homeground_advantage', 'team1_toss_win', 'teampoint_diff'
    ]
    target_var = 'team1_win'

    X_features = df[feature_vars]
    y_target = df[target_var]

    intercept = np.ones((X_features.shape[0], 1))
    X_features = np.concatenate((intercept, X_features), axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X_features, y_target, test_size=1/3, random_state=0)
    Y_train = Y_train.to_numpy()
    Y_test = Y_test.to_numpy()

    theta = np.zeros(X_train.shape[1])

    model = train(iterations, learning_rate, theta, X_train, Y_train)
    predictions = predict(iterations, learning_rate, model, X_test, Y_test)

    return render_template("results.html", results=predictions)

# Function to open the browser
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Run the browser in a separate thread to avoid blocking the Flask app
    threading.Timer(1, open_browser).start()
    app.run(debug=True)
