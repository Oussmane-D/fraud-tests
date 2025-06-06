import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Essayer d’importer MLflow si disponible
try:
    import mlflow
    import mlflow.sklearn
    mlflow_available = True
except ModuleNotFoundError:
    mlflow_available = False
    print("MLflow n'est pas installé localement, on passera outre le logging.")

# --- Chargement des données simulées (ou réelles si tu remplaces) ---
df = pd.DataFrame({
    'transaction_id': np.arange(1, 5001),
    'amount': np.round(np.random.exponential(scale=50, size=5000), 2),
    'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer'], size=5000, p=[0.7, 0.2, 0.1]),
    'country': np.random.choice(['FR', 'US', 'DE', 'ES', 'IT'], size=5000),
    'is_fraud': np.random.choice([0, 1], size=5000, p=[0.98, 0.02])
})

# --- Préprocessing ---
df_encoded = pd.get_dummies(df[['amount', 'transaction_type', 'country']], drop_first=True)
X = df_encoded
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Connexion à MLflow (Hugging Face) si disponible ---
if mlflow_available:
    # Si tu as réglé MLFLOW_TRACKING_URI via export, pas besoin de le répéter ici :
    # mlflow.set_tracking_uri("https://<ton-serveur-mlflow>.hf.space")
    mlflow.set_experiment("fraud_detection")  # nom de l'expérience
    run = mlflow.start_run(run_name="random_forest_baseline")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

# --- Entraînement du modèle ---
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

# --- Prédictions et calcul métriques ---
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# --- Logging sur l’instance MLflow HF ---
if mlflow_available:
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc_auc)
    # Enregistre le modèle dans le Model Registry (sous le dossier "model")
    mlflow.sklearn.log_model(rf, "model")
    mlflow.end_run()

# --- Affichage des résultats en local ---
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Classification Report:\n", report_df)
print(f"\nAccuracy: {acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
