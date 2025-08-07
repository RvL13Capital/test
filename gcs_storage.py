"""
Google Cloud Storage Integration für das Ignition-Wert Framework
Persistente Datenspeicherung für CSV-Uploads
"""

import json
import os
from datetime import datetime
from google.cloud import storage
from google.auth import default
import traceback

class GCSStorage:
    def __init__(self):
        self.project_id = "ignition-ki-csv-storage"
        self.bucket_name = "ignition-ki-csv-data-2025-user123"
        self.region = "us-central1"
        
        # Setze Credentials-Pfad
        credentials_path = os.path.join(os.path.dirname(__file__), 'google-credentials.json')
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        try:
            self.client = storage.Client(project=self.project_id)
            self.bucket = self.client.bucket(self.bucket_name)
            print(f"✅ GCS Storage initialisiert: {self.bucket_name}")
        except Exception as e:
            print(f"❌ GCS Storage Fehler: {str(e)}")
            self.client = None
            self.bucket = None
    
    def save_ticker_data(self, ticker, data):
        """Speichere Ticker-Daten in GCS"""
        if not self.bucket:
            print("❌ GCS nicht verfügbar")
            return False
        
        try:
            # Erstelle JSON-Datei für den Ticker
            file_data = {
                'ticker': ticker,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(data)
            }
            
            # Speichere in GCS
            blob_name = f"tickers/{ticker}.json"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(file_data, indent=2),
                content_type='application/json'
            )
            
            print(f"✅ Ticker {ticker} in GCS gespeichert: {len(data)} Datenpunkte")
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern von {ticker}: {str(e)}")
            traceback.print_exc()
            return False
    
    def load_ticker_data(self, ticker):
        """Lade Ticker-Daten aus GCS"""
        if not self.bucket:
            return None
        
        try:
            blob_name = f"tickers/{ticker}.json"
            blob = self.bucket.blob(blob_name)
            
            if not blob.exists():
                return None
            
            content = blob.download_as_text()
            file_data = json.loads(content)
            
            print(f"✅ Ticker {ticker} aus GCS geladen: {file_data.get('data_points', 0)} Datenpunkte")
            return file_data['data']
            
        except Exception as e:
            print(f"❌ Fehler beim Laden von {ticker}: {str(e)}")
            return None
    
    def load_all_tickers(self):
        """Lade alle Ticker-Daten aus GCS"""
        if not self.bucket:
            return {}
        
        try:
            all_data = {}
            blobs = self.bucket.list_blobs(prefix="tickers/")
            
            for blob in blobs:
                if blob.name.endswith('.json'):
                    ticker = blob.name.replace('tickers/', '').replace('.json', '')
                    data = self.load_ticker_data(ticker)
                    if data:
                        all_data[ticker] = data
            
            print(f"✅ {len(all_data)} Ticker aus GCS geladen")
            return all_data
            
        except Exception as e:
            print(f"❌ Fehler beim Laden aller Ticker: {str(e)}")
            return {}
    
    def list_available_tickers(self):
        """Liste alle verfügbaren Ticker in GCS"""
        if not self.bucket:
            return []
        
        try:
            tickers = []
            blobs = self.bucket.list_blobs(prefix="tickers/")
            
            for blob in blobs:
                if blob.name.endswith('.json'):
                    ticker = blob.name.replace('tickers/', '').replace('.json', '')
                    tickers.append(ticker)
            
            return sorted(tickers)
            
        except Exception as e:
            print(f"❌ Fehler beim Listen der Ticker: {str(e)}")
            return []
    
    def get_storage_stats(self):
        """Hole Speicher-Statistiken"""
        if not self.bucket:
            return {
                'available_tickers': [],
                'total_tickers': 0,
                'total_data_points': 0,
                'storage_status': 'offline'
            }
        
        try:
            tickers = self.list_available_tickers()
            total_data_points = 0
            
            # Zähle Datenpunkte für jeden Ticker
            for ticker in tickers:
                data = self.load_ticker_data(ticker)
                if data:
                    total_data_points += len(data)
            
            return {
                'available_tickers': tickers,
                'total_tickers': len(tickers),
                'total_data_points': total_data_points,
                'storage_status': 'online'
            }
            
        except Exception as e:
            print(f"❌ Fehler beim Abrufen der Statistiken: {str(e)}")
            return {
                'available_tickers': [],
                'total_tickers': 0,
                'total_data_points': 0,
                'storage_status': 'error'
            }

