import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. Load synthetic fish disease dataset
def load_fish_disease_data():
    print("📥 Loading dataset...")
    url = "fish_disease_data.csv"  # Replace with your actual file
    df = pd.read_csv(url)
    df = df.head(1000)
    print(f"✅ Loaded {len(df)} records.\n")
    return df

# 2. EDA Function to count fish with age > 1 year
def perform_eda_on_age(df):
    print("📊 Performing EDA on Age column...")
    if 'Age' not in df.columns:
        print("❌ 'Age' column not found.\n")
        return

    count_over_1 = df[df['Age'] > 1].shape[0]
    print(f"🐟 Number of fish with age > 1 year: {count_over_1}\n")
    return count_over_1

# 3. Preprocess data
def preprocess_fish_data(df):
    print("🛠️ Preprocessing data...")
    df = pd.get_dummies(df, drop_first=True)

    if "Disease_Status_Healthy" not in df.columns:
        raise ValueError("❌ 'Disease_Status_Healthy' column missing after encoding!")

    X = df.drop("Disease_Status_Healthy", axis=1)
    y = df["Disease_Status_Healthy"]
    print("✅ Features and target separated.\n")
    return X, y, df

# 4. Split the data
def split_fish_data(X, y, test_size=0.2):
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test

# 5. Create and train Decision Tree model
def create_and_train_model(X_train, y_train):
    print("🔧 Creating Decision Tree model...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model trained.\n")
    return model

# 6. Calculate entropy
def calculate_entropy(y):
    print("📊 Calculating entropy...")
    value_counts = y.value_counts(normalize=True)
    entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
    print(f"🧮 Entropy of target (Disease_Status_Healthy): {entropy:.4f}\n")
    return entropy

# 7. Predict new fish data from JSON
def check_new_data_from_json(model, df_encoded, json_file="fish_data.json"):
    import json

    print(f"📄 Checking new fish data from {json_file}...")
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        fish = data['fish']

        original_df = load_fish_disease_data()

        temp_df = pd.DataFrame([{
            'Age': fish['Age'],
            'Species': fish['Species'],
            'Water_Temperature': fish['Water_Temperature'],
            'Feeding_Behavior': fish['Feeding_Behavior'],
            'Coloration': fish['Coloration'],
            'Swimming_Behavior': fish['Swimming_Behavior'],
            'Disease_Status': 'Healthy'  # Dummy value
        }])

        combined_df = pd.concat([original_df, temp_df], ignore_index=True)

        combined_encoded = pd.get_dummies(combined_df, drop_first=True)

        new_fish_features = combined_encoded.iloc[[-1]].drop("Disease_Status_Healthy", axis=1)

        prediction = model.predict(new_fish_features)[0]

        print("🧠 New Fish Data:")
        for key, value in fish.items():
            print(f"{key}: {value}")

        result = "Healthy" if prediction == 1 else "Diseased"
        print(f"\n🔮 Model Prediction: {result}")

        print("\n📋 FINAL FISH DISEASE PREDICTION RESULT:")
        print(f"🔍 Fish is healthy: {'YES' if prediction == 1 else 'NO'}\n")

    except Exception as e:
        print(f"❌ Error checking new data: {e}\n")

# --- Pipeline Execution ---
df = load_fish_disease_data()
perform_eda_on_age(df)
X, y, df_encoded = preprocess_fish_data(df)
X_train, X_test, y_train, y_test = split_fish_data(X, y)
model = create_and_train_model(X_train, y_train)

# Save model
joblib.dump(model, 'decision_tree_fish_disease_model.pkl')
print("💾 Model saved as 'decision_tree_fish_disease_model.pkl'")

calculate_entropy(y)

check_new_data_from_json(model, df_encoded)
