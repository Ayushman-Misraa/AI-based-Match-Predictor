
# 🏏 AI-based Match Predictor  

Predict the outcome of cricket matches using machine learning! This project implements Logistic Regression from scratch to predict match winners based on historical match data. It includes a user-friendly web application powered by Flask for easy interaction.

---

## 🚀 Features  

- **Custom Logistic Regression**: Built from scratch for better understanding of ML fundamentals.  
- **Accurate Predictions**: Predicts match winners based on team performance, toss decisions, and other key metrics.  
- **Interactive Web App**: Upload your dataset and get instant predictions.  
- **Dynamic Learning Rates**: Choose learning rates for optimized accuracy.  

---

## 📋 Requirements  

Before running the project, make sure you have the following:

### 1. **Software**
- **Python**: Version 3.7 or above.

### 2. **Python Libraries**
Install the required libraries using the following command:  
```bash
pip install pandas numpy scikit-learn matplotlib flask
```

#### Detailed Library List:
- **pandas**: For data manipulation and analysis.  
- **numpy**: For numerical operations.  
- **scikit-learn**: For splitting the dataset into training and testing sets.  
- **matplotlib**: For plotting training performance (optional).  
- **flask**: To create the web application for predictions.  

### 3. **Dataset**
A CSV file containing the following columns:  

| Column                  | Description                                              |
|-------------------------|----------------------------------------------------------|
| `team1`                 | Team 1 ID or name (encoded).                             |
| `powerplay_runs_team1`  | Runs scored in powerplay by Team 1.                      |
| `powerplay_wickets_team1` | Wickets lost in powerplay by Team 1.                    |
| `team2`                 | Team 2 ID or name (encoded).                             |
| `powerplay_runs_team2`  | Runs scored in powerplay by Team 2.                      |
| `powerplay_wickets_team2` | Wickets lost in powerplay by Team 2.                    |
| `homeground_advantage`  | Whether Team 1 has home advantage (1 = Yes, 0 = No).     |
| `team1_toss_win`        | Whether Team 1 won the toss (1 = Yes, 0 = No).           |
| `teampoint_diff`        | Difference in team points (encoded as binary).           |
| `team1_win`             | Target variable: 1 if Team 1 wins, 0 otherwise.          |

---

## 🛠️ Installation  

1. Clone this repository:  
   ```bash
   git clone <repository-url>
   cd AI-based-Match-Predictor-main
   ```

2. Run the Flask application:  
   ```bash
   python app.py
   ```

3. Open your browser (it will auto-launch) or go to:  
   `http://127.0.0.1:5000`

---

## 📂 File Structure  

- **`main.py`**: Core ML logic, including training and prediction.  
- **`app.py`**: Flask web application for user interaction.  
- **`templates/`**: HTML templates for the web interface:  
  - `index.html`: For file uploads and input parameters.  
  - `results.html`: Displays prediction results.  
- **`Dataset/`**: Placeholder for your dataset file.  

---

## 📊 How It Works  

### 1. Prepare the Dataset  

Ensure your dataset is saved as a CSV file and meets the structure outlined in the **Requirements** section.

### 2. Upload the Dataset  

1. Navigate to `http://127.0.0.1:5000`.  
2. Upload your dataset CSV file.  
3. Specify:  
   - **Number of Iterations**  
   - **Learning Rates** (comma-separated, e.g., `0.001, 0.01, 0.1`).  

### 3. View Results  

- **Predictions**: Get the predicted winner (Team 1 or Team 2).  
- **Accuracy**: See the model’s accuracy for different learning rates.  

---

## 🌟 Example  

### Input CSV:  

```csv
team1,powerplay_runs_team1,powerplay_wickets_team1,team2,powerplay_runs_team2,powerplay_wickets_team2,homeground_advantage,team1_toss_win,teampoint_diff,team1_win
1,40,1,0,30,2,1,1,1,1
0,25,3,1,20,1,0,0,0,0
...
```

### Output:  

- **Predicted Winner**: Team 1 or Team 2.  
- **Accuracy**: Percentage accuracy for different learning rates.  

---

## 📈 Visualization  

The model includes optional plots to analyze the training process:  

- **Iterations vs Loss (2D)**: Shows how the cost reduces over iterations.  
- **Iterations vs Loss (3D)**: A dynamic visualization of learning across iterations.  

Uncomment the relevant sections in `main.py` to enable these visualizations.

---

## 🚀 Future Scope  

- Integrate advanced machine learning models for better predictions.  
- Improve the UI with live graphs and better interactivity.  
- Expand support for more sports and datasets.  

---

## 🤝 Contributing  

Contributions are welcome! Feel free to raise issues or submit pull requests to enhance the project.
Contact mail - ayushmanmisra036@gmail.com

---

## 💡 License  

This project is licensed under the **MIT License**.  

--- 
