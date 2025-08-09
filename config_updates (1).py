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
  name: gcs-