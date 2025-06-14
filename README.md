# fraud-tests
# Pipeline de Détection de Fraude

Une démonstration **end-to-end** d’un workflow de détection de fraude, construite avec :

- **Apache Airflow** (DAG pour validation des données & entraînement quotidien du modèle)  
- **MLflow** (suivi d’expériences & registry sur S3)  
- **FastAPI** + **Uvicorn** (API REST servant le dernier modèle)  
- **PostgreSQL** (stores des métadonnées Airflow)  
- **Docker Compose** pour l’orchestration locale  
- **GitHub Actions** CI (lint, tests, build)

---

## 🚀 Démarrage rapide

1. **Cloner le dépôt**  
   ```bash
   git clone https://github.com/Oussmane-D/fraud-tests.git
   cd fraud-detection

Configurer votre fichier .env
Copiez config/.env.example → config/.env et complétez vos secrets :

# MLflow
MLFLOW_TRACKING_URI=https://votre-mlflow.example.com

# AWS (pour le stockage S3 des artefacts)
AWS_ACCESS_KEY_ID=VOTRE_AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=VOTRE_AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION=eu-west-3

# (optionnel) Données locales pour les tests
DATA_CSV_PATH=/opt/airflow/data/transactions.csv

Lancer la stack
docker-compose up -d
    Airflow UI → http://localhost:8080 (login: admin / admin)
    Health API → http://localhost:8000/health
    Swagger → http://localhost:8000/docs

Déclencher le DAG
    Dans l’UI Airflow, triggerlez train_and_log_fraud_model.

    Validation de DATA_CSV_PATH

    Entraînement d’un RandomForest toy

    Logging dans MLflow (expérience fraud_detection_dag)

Interroger l’API
curl http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount":123.4,"merchant_id":17,"country":"FR","card_age_months":9}'

🧪 Exécuter les tests localement
docker-compose run --rm \
  -e AIRFLOW__CORE__EXECUTOR=SequentialExecutor \
  -e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////tmp/airflow_test.db \
  airflow \
  bash -c "airflow db init && pytest /opt/airflow/tests -q"

🔧 CI / GitHub Actions
Le workflow (.github/workflows/ci.yml) :

Build de l’image Airflow

Exécution de pytest sur une base SQLite

(optionnel) Lint de votre code

Ajoutez ces secrets dans les paramètres GitHub :

MLFLOW_TRACKING_URI

AWS_ACCESS_KEY_ID

AWS_SECRET_ACCESS_KEY

AWS_DEFAULT_REGION

📂 Structure du projet

.
├── api/                   # FastAPI + loader de modèle
│   ├── main.py
│   └── model_loader.py
├── dags/                  # DAGs Airflow
│   └── train_and_log_fraud_model.py
├── tests/                 # pytest
│   ├── test_dag_loads.py
│   └── test_validate_data.py
├── config/
│   └── .env.example       # template pour vos secrets
├── docker-compose.yml
├── Dockerfile             # image Airflow + requirements.txt
├── requirements.txt
├── .github/
│   └── workflows/ci.yml
└── README.md
