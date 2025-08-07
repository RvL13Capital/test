import os
import sys

# Füge das aktuelle Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importiere die Flask-App aus dem src-Modul
from src.main import app

if __name__ == '__main__':
    # Starte die App
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))

