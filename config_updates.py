"""
Configuration updates needed for GCS integration

Add these configurations to your project.config.Config class:
"""

class Config:
    # ... existing configurations ...
    
    # GCS Configuration - ADD THESE
    GCS_BUCKET_NAME = "your-ml-models-bucket"  # Replace with your actual bucket name
    GCS_PROJECT_ID = "your-gcp-project-id"    # Replace with your GCP project ID (optional)
    
    # GCS Path Prefixes
    GCS_LSTM_MODEL_PREFIX = "models/lstm"
    GCS_XGBOOST_MODEL_PREFIX = "models/xgboost" 
    GCS_SCALER_PREFIX = "models/scalers"
    
    # Model Retention Policy (optional)
    MAX_MODELS_PER_TICKER = 5  # Keep only the 5 most recent models per ticker
    MODEL_CLEANUP_ENABLED = True  # Enable automatic cleanup of old models
    
    # ... rest of existing configurations ...

"""
Environment Variables Setup:

You'll also need to set up authentication for Google Cloud Storage.
Choose one of these methods:

Method 1: Service Account Key File
- Set environment variable: GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

Method 2: Application Default Credentials (recommended for production)
- Use Google Cloud SDK: gcloud auth application-default login
- Or use IAM roles if running on GCP (Compute Engine, Cloud Run, etc.)

Method 3: Explicit credentials in code (not recommended for production)
- Pass credentials directly to the GCS client
"""

# Example Docker environment variables
DOCKER_ENV_EXAMPLE = """
# Add to your Dockerfile or docker-compose.yml:
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json
ENV GCS_BUCKET_NAME=your-ml-models-bucket
ENV GCS_PROJECT_ID=your-gcp-project-id
"""

# Example Kubernetes ConfigMap/Secret
KUBERNETES_CONFIG_EXAMPLE = """
# ConfigMap for GCS settings
apiVersion: v1
kind: ConfigMap
metadata:
  name: gcs-config
data:
  GCS_BUCKET_NAME: "your-ml-models-bucket"
  GCS_PROJECT_ID: "your-gcp-project-id"

---
# Secret for service account key
apiVersion: v1
kind: Secret
metadata:
  name: gcs-credentials
type: Opaque
data:
  service-account-key.json: <base64-encoded-service-account-key>
"""

# GCS Bucket Setup Commands
GCS_SETUP_COMMANDS = """
# Create GCS bucket for model storage
gsutil mb gs://your-ml-models-bucket

# Set up bucket structure
gsutil -m cp /dev/null gs://your-ml-models-bucket/models/lstm/.keep
gsutil -m cp /dev/null gs://your-ml-models-bucket/models/xgboost/.keep
gsutil -m cp /dev/null gs://your-ml-models-bucket/models/scalers/.keep

# Set appropriate permissions (replace with your service account email)
gsutil iam ch serviceAccount:your-service-account@your-project.iam.gserviceaccount.com:objectAdmin gs://your-ml-models-bucket
"""

# Required Python packages
REQUIRED_PACKAGES = """
# Add to requirements.txt:
google-cloud-storage>=2.10.0
google-auth>=2.17.0
google-auth-oauthlib>=1.0.0
google-auth-httplib2>=0.1.0
"""

# Application initialization example
INITIALIZATION_EXAMPLE = """
# Add to your application startup (e.g., in __init__.py or main.py):

from project.gcs_storage import init_gcs_storage
from project.config import Config

def initialize_app():
    # Initialize GCS storage
    init_gcs_storage(
        bucket_name=Config.GCS_BUCKET_NAME,
        project_id=Config.GCS_PROJECT_ID
    )
    
    # Other initialization code...
"""