import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.autolog()

# NOTE: If your CSV file is at the root of the repo, and your script is in MLProject/, 
# you might need to adjust this path to '../winequality_white_preprocessing.csv' or 
# 'winequality_white_preprocessing.csv' depending on how mlflow run handles the CWD. 
# We'll assume this path works based on previous context.
data = pd.read_csv('winequality_white_preprocessing.csv') 

X = data.drop('quality_encoded', axis=1)
y = data['quality_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === CRITICAL CHANGE: Capture the run object using 'as run' ===
with mlflow.start_run(run_name="Basic_RandomForest_Autolog") as run:

    # === ADDED LINE: Print the Run ID with the unique marker ===
    print(f"RUN_ID::{run.info.run_id}") 
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")