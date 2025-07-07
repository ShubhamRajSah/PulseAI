from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import joblib

def train_model(X, y, selected_features, save_path="rf_model.pkl"):
    # Use only selected features from feature selection
    X_selected = X[selected_features]

    # Stratified Train-test split to preserve label balance
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=42
    )

    # Initialize and train Random Forest with optional tuning
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("\n📈 Model Evaluation:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test,y_pred))
    

    # Save the trained model
    joblib.dump(model, save_path)
    print(f"✅ Model saved to: {save_path}")

    # Return metrics for UI or logging use
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }