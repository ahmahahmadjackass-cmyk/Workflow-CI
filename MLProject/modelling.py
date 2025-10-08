import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Automatically log model, parameters, metrics, and artifacts to MLflow
mlflow.autolog()

# === Load preprocessed dataset ===
# Adjust path if needed based on how 'mlflow run' sets working directory
data = pd.read_csv('winequality_white_preprocessing.csv')

# === Prepare features and labels ===
X = data.drop('quality_encoded', axis=1)
y = data['quality_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Start MLflow run ===
with mlflow.start_run(run_name="Basic_RandomForest_Autolog") as run:
    # Print Run ID for GitHub Actions to parse
    print(f"RUN_ID::{run.info.run_id}")

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log accuracy manually (optional, autolog already handles this)
    print(f"Accuracy: {accuracy}")
