import pandas as pd
import pytest
import sys
sys.path.append("/opt/airflow/dags")
# chemin absolu interne au conteneur
MODULE = "dags.train_and_log_fraud_model"

def _make_valid_df():
    return pd.DataFrame(
        {
            "amount": [10, 20],
            "transaction_type": ["purchase", "withdrawal"],
            "country": ["FR", "US"],
            "is_fraud": [0, 1],
        }
    )

def test_validate_ok(monkeypatch):
    from importlib import import_module
    m = import_module(MODULE)

    # on force pd.read_csv à retourner un DataFrame conforme
    monkeypatch.setattr(m.pd, "read_csv", lambda _: _make_valid_df())

    # ne doit pas lever d’erreur
    m.validate_data_callable()

@pytest.mark.parametrize("bad_df", [
    pd.DataFrame({"x": [1]}),                     # schéma faux
    _make_valid_df().assign(is_fraud=[0, 2]),     # cible non binaire
    _make_valid_df().assign(amount=[None, 20]),   # valeurs manquantes
])
def test_validate_ko(monkeypatch, bad_df):
    from importlib import import_module
    m = import_module(MODULE)
    monkeypatch.setattr(m.pd, "read_csv", lambda _: bad_df)

    with pytest.raises(ValueError):
        m.validate_data_callable()
