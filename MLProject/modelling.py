import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow.sklearn 

mlflow.autolog()

# ... (Data loading and splitting remains the same)
data = pd.read_csv('winequality_white_preprocessing.csv') 
X = data.drop('quality_encoded', axis=1)
y = data['quality_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# === SIMPLIFIED LOGIC: Use last_active_run() which is usually available after autolog() ===

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model fitting triggers autologging, which ensures an active run is present.
run = mlflow.last_active_run() 

# Now we should have the run object safely.
print(f"RUN_ID::{run.info.run_id}") 
    
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
    
# === Explicitly log the model artifact ===
mlflow.sklearn.log_model(
    sk_model=model, 
    artifact_path="model"
)