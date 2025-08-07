from google.cloud import storage
import json
import os
from datetime import datetime
from project.config import Config
import redis

# Redis-Verbindung initialisieren
r = redis.Redis.from_url(Config.CELERY_BROKER_URL, decode_responses=True)

class GCSStorage:
    def __init__(self, project_id, bucket_name):
        try:
            self.client = storage.Client(project=project_id)
            self.bucket = self.client.bucket(bucket_name)
            if not self.bucket.exists():
                self.bucket.create()
            print("GCS Storage initialisiert")
        except Exception as e:
            print(f"GCS Initialisierungsfehler: {str(e)}")
            self.bucket = None

    def save_ticker_data(self, ticker, data_dict):
        if not self.bucket:
            print("GCS nicht verfügbar - verwende lokalen Speicher")
            return False
        try:
            blob = self.bucket.blob(f"data/{ticker}/{datetime.now().strftime('%Y-%m-%d')}.json")
            blob.upload_from_string(json.dumps(data_dict, indent=2), content_type='application/json')
            return True
        except Exception as e:
            print(f"Fehler beim Speichern der Ticker-Daten in GCS: {str(e)}")
            return False

def update_model_registry(ticker, model_type, model_path, hparams=None):
    """Speichert Modellpfad und Hyperparameter in Redis Registry"""
    registry_key = f"model_registry:{ticker}"
    entry = {
        'model_type': model_type,
        'path': model_path,
        'hparams': json.dumps(hparams) if hparams else None,
        'timestamp': datetime.now().isoformat()
    }
    r.hset(registry_key, model_type, json.dumps(entry))
    return True

def get_model_registry(ticker):
    """Lädt Registry-Einträge aus Redis"""
    registry_key = f"model_registry:{ticker}"
    entries = r.hgetall(registry_key)
    result = {}
    for model_type, data in entries.items():
        result[model_type] = json.loads(data)
        if result[model_type].get('hparams'):
            result[model_type]['hparams'] = json.loads(result[model_type]['hparams'])
    return result

def test_function():
    return True