# File: Dockerfile

FROM apache/airflow:2.5.1

USER root
# Copier le requirements.txt dans le conteneur
COPY requirements.txt ${AIRFLOW_HOME}/requirements.txt
#COPY dags/ /opt/airflow/dags
# Passer en root pour installer les dépendances globalement
USER airflow
# empêcher Airflow de downgrader les paquets
ENV _PIP_ADDITIONAL_REQUIREMENTS=""
# Installer les dépendances Python dans l'environnement système
RUN pip install --no-cache-dir -r ${AIRFLOW_HOME}/requirements.txt

# Revenir à l'utilisateur airflow
#USER airflow
