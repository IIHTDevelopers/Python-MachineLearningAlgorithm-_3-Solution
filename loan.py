import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib


# Function 1: Load and inspect data
def load_data(path="loan_dataset.csv"):
    df = pd.read_csv(path)
    print("\n📊 Loan Amount - Mean: {:.2f}, Max: {:.2f}".format(
        df['loan_amount'].mean(), df['loan_amount'].max()
    ))
    return df

# Function 2: Basic EDA — count people living in rented homes
def explore_home_ownership(df):
    if 'home_ownership' not in df.columns:
        print("❌ 'home_ownership' column not found.")
        return

    rent_count = (df['home_ownership'] == 'RENT').sum()
    print(f"\n🏠 Number of people living in rented homes: {rent_count}")
    return rent_count

# Function 3: Encode and scale features
def prepare_data(df):
    for col in ['term', 'home_ownership']:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])
    scaler = StandardScaler()
    features_to_scale = df.columns.difference(['defaulted'])
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    print("✅ Data encoded and scaled.")
    return df

# Function 4: Sigmoid function demo
def sigmoid_demo():
    z = 1.5
    sigmoid = 1 / (1 + np.exp(-z))
    print(f"\n🧠 Sigmoid(1.5) = {sigmoid:.4f}")
    return sigmoid

# Function 5: Train and evaluate logistic regression model
def train_and_evaluate(X_train, y_train, X_test, y_test, path="loan_model.pkl"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, path)
    print(f"\n✅ Model trained and saved to '{path}'")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    print("🔍 Sample Predictions:", y_pred[:10])

    # Return the trained model, predictions, and probabilities
    return {
        "model": model,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob
    }


# --------- Main Logic ---------
if __name__ == "__main__":
    # Step 1: Load data
    df = load_data("loan_dataset.csv")

    # Step 2: Run EDA before encoding
    explore_home_ownership(df)

    # Step 3: Encode & scale
    df = prepare_data(df)

    # Step 4: Demo sigmoid
    sigmoid_demo()

    # Step 5: Prepare features/target
    X = df.drop(columns=['defaulted'])
    y = df['defaulted']

    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 7: Train and evaluate
    train_and_evaluate(X_train, y_train, X_test, y_test)
