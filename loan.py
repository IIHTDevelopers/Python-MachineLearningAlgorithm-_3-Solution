import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
import joblib

# Function 1: Load, clean, and prepare real loan dataset
def load_and_prepare_data(path="loan_dataset.csv"):
    df = pd.read_csv(path)

    # Show real values before scaling
    print("\n📊 Loan Amount - Mean: {:.2f}, Max: {:.2f}".format(df['loan_amount'].mean(), df['loan_amount'].max()))

    # Encode categorical columns
    for col in ['term', 'home_ownership']:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])

    # Scale features
    scaler = StandardScaler()
    df[df.columns.difference(['defaulted'])] = scaler.fit_transform(df[df.columns.difference(['defaulted'])])

    print("✅ Real dataset loaded and preprocessed.")
    return df

# Function 2: EDA for 'loan_amount'
def explore_data(df):
    pass  # Removed scaled loan amount display

# Function 3: Sigmoid activation demo
def sigmoid_demo():
    z = 1.5
    sigmoid = 1 / (1 + np.exp(-z))
    print(f"\n🧠 Sigmoid(1.5) = {sigmoid:.4f}")

# Function 4: Custom log loss cost function
def cost_function(y_true, y_pred_prob):
    epsilon = 1e-15
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))

# Function 5: Train and evaluate model
def train_and_evaluate(X_train, y_train, X_test, y_test, path="loan_model.pkl"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    print(f"\n✅ Model trained and saved to '{path}'")

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    cost = cost_function(y_test.values, y_pred_prob)

    print(f"\n🎯 Log Loss (Custom Cost): {cost:.4f}")
    print("🔍 Sample Predictions:", y_pred[:10])

# --------- Main Logic ---------
if __name__ == "__main__":
    df = load_and_prepare_data("loan_dataset.csv")

    explore_data(df)
    sigmoid_demo()

    X = df.drop(columns=['defaulted'])
    y = df['defaulted']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_evaluate(X_train, y_train, X_test, y_test)
