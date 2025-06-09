import requests

base = "http://localhost:8000"
print(requests.get(f"{base}/health").json())

payload = {"amount": 50.0, "transaction_type": "transfer", "country": "US"}
print(requests.post(f"{base}/predict", json=payload).json())
