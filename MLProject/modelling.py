import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow.sklearn 
import os # <-- CRITICAL: Needed to read the environment variable

mlflow.autolog()

# Data loading and splitting
data = pd.read_csv('winequality_white_preprocessing.csv') 
X = data.drop('quality_encoded', axis=1)
y = data['quality_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# === CRITICAL FIX: Get Run ID directly from environment variable ===
# This is the most stable way to get the ID when running via 'mlflow run' CLI.
RUN_ID = os.environ.get("MLFLOW_RUN_ID")

if RUN_ID:
    # Print the ID using the unique marker for the CI job to capture
    print(f"RUN_ID::{RUN_ID}") 
else:
    print("Warning: MLFLOW_RUN_ID environment variable not found.")
    
# The model training and logging proceed normally.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
    
# === Explicitly log the model artifact (Guarantees the location for CI) ===
mlflow.sklearn.log_model(
    sk_model=model, 
    artifact_path="model"
)