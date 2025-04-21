import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# Function 1: Load and preprocess the dataset
def load_and_preprocess(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    df = df.dropna()
    print("✅ Data loaded and cleaned.")
    return df

# Function 2: Show standard deviation of price and max number of rooms
def show_key_stats(df):
    price_std = df['price'].std()
    rooms_max = df['rooms'].max()
    print(f"\n📊 Standard Deviation of Price: ${price_std:,.2f}")
    print(f"🛏️  Maximum Number of Rooms: {rooms_max}")

# Function 3: Prepare data for training
def prepare_data(df, features, target):
    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("\n📊 Data prepared and split.")
    return X_train, X_test, y_train, y_test, scaler

# Function 4: Train and save model
def train_and_save_model(X_train, y_train, model_path="house_price_model.pkl"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"\n✅ Model trained and saved to '{model_path}'")
    return model

# Function 5: Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\n📉 Mean Squared Error: {mse:.2f}")
    print("🔍 Sample Predictions:", y_pred[:10])

# ---- MAIN SCRIPT ----
if __name__ == "__main__":
    features = ['rooms', 'area', 'bathrooms', 'floors', 'age']
    target = 'price'

    df = load_and_preprocess("Housing.csv")
    show_key_stats(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, features, target)
    model = train_and_save_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
