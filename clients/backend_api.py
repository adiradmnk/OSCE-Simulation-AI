import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8080")

def get_case_data(case_id: str):
    response = requests.get(f"{BACKEND_URL}/cases/{case_id}")
    if response.status_code == 200:
        return response.json()
    raise Exception("Gagal ambil data kasus dari backend")

