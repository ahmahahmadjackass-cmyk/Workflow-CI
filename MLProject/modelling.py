import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow.sklearn 

mlflow.autolog()

# NOTE: Path assumed to be correct based on previous context.
data = pd.read_csv('winequality_white_preprocessing.csv') 

X = data.drop('quality_encoded', axis=1)
y = data['quality_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === CRITICAL CHANGE: Get the existing run started by 'mlflow run' CLI ===
# The 'mlflow run' command handles the context, we just attach to it.
run = mlflow.active_run()

# The rest of the code is now outside the 'with' block, but still executes within the run context.

# === ADDED LINE: Print the Run ID with the unique marker ===
# We use run.info.run_id from the active run.
print(f"RUN_ID::{run.info.run_id}") 
    
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
    
# === CRITICAL FIX: Explicitly log the model artifact ===
mlflow.sklearn.log_model(
    sk_model=model, 
    artifact_path="model"
)
# =========================================================

# The script finishes, and the CLI (mlflow run) ends the run.