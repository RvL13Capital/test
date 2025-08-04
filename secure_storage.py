# project/storage.py
import io
import joblib
import torch
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from google.cloud import storage
from google.cloud.exceptions import NotFound
from cryptography.fernet import Fernet

# Import from other project modules
from .config import Config

logger = logging.getLogger(__name__)

class SecureGCSStorage:
    """2025 Enhanced GCS storage with encryption, integrity checks, and audit logging"""
    
    def __init__(self, bucket_name: str, project_id: Optional[str] = None):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self._setup_clients()
        self._setup_encryption()
    
    def _setup_clients(self):
        """Initialize GCS client with error handling"""
        try:
            self.client = storage.Client(project=self.project_id)
            self.bucket = self.client.get_bucket(self.bucket_name)
            logger.info(f"Successfully connected to GCS bucket '{self.bucket_name}'")
        except Exception as e:
            logger.error(f"Failed to connect to GCS bucket '{self.bucket_name}': {e}")
            self.client = None
            self.bucket = None
    
    def _setup_encryption(self):
        """Setup client-side encryption if key is provided"""
        encryption_key = Config.MODEL_ENCRYPTION_KEY
        if encryption_key:
            try:
                self.cipher = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
                logger.info("Client-side encryption enabled for models")
            except Exception as e:
                logger.error(f"Failed to setup encryption: {e}")
                self.cipher = None
        else:
            self.cipher = None
            logger.info("Client-side encryption disabled (no key provided)")
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum for data integrity"""
        return hashlib.sha256(data).hexdigest()
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data if cipher is available"""
        if self.cipher:
            return self.cipher.encrypt(data)
        return data
    
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data if cipher is available"""
        if self.cipher:
            return self.cipher.decrypt(encrypted_data)
        return encrypted_data
    
    def _create_audit_metadata(self, operation: str, user_id: Optional[str] = None, 
                              additional_info: Optional[Dict] = None) -> Dict[str, str]:
        """Create audit metadata for GCS objects"""
        metadata = {
            'operation': operation,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id or 'system',
            'encrypted': str(self.cipher is not None),
            'system_version': '2025.1.0'
        }
        
        if additional_info:
            for key, value in additional_info.items():
                metadata[f'custom_{key}'] = str(value)
        
        return metadata
    
    def _log_audit_event(self, event_type: str, gcs_path: str, user_id: Optional[str] = None, 
                        success: bool = True, error: Optional[str] = None):
        """Log audit events for compliance"""
        if Config.AUDIT_LOG_ENABLED:
            audit_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type,
                'gcs_path': f"gs://{self.bucket_name}/{gcs_path}",
                'user_id': user_id or 'system',
                'success': success,
                'error': error,
                'bucket': self.bucket_name
            }
            
            # In production, send to proper audit logging system
            logger.info(f"AUDIT: {json.dumps(audit_data)}")
    
    def _upload_to_gcs(self, data: bytes, gcs_path: str, content_type: str, 
                      user_id: Optional[str] = None, additional_metadata: Optional[Dict] = None) -> str:
        """Internal method to upload data to GCS with security features"""
        if not self.bucket:
            raise ConnectionError("GCS bucket is not initialized")
        
        try:
            # Calculate checksum before encryption
            original_checksum = self._calculate_checksum(data)
            
            # Encrypt data
            encrypted_data = self._encrypt_data(data)
            
            # Create metadata
            metadata = self._create_audit_metadata('upload', user_id, {
                'original_checksum': original_checksum,
                'size_bytes': len(data),
                'encrypted_size_bytes': len(encrypted_data)
            })
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Upload to GCS
            blob = self.bucket.blob(gcs_path)
            blob.metadata = metadata
            blob.upload_from_string(encrypted_data, content_type=content_type)
            
            # Audit log
            self._log_audit_event('model_upload', gcs_path, user_id, success=True)
            
            logger.info(f"Successfully uploaded to gs://{self.bucket_name}/{gcs_path}")
            return f"gs://{self.bucket_name}/{gcs_path}"
            
        except Exception as e:
            self._log_audit_event('model_upload', gcs_path, user_id, success=False, error=str(e))
            logger.error(f"Failed to upload to {gcs_path}: {e}")
            raise
    
    def _download_from_gcs(self, gcs_path: str, user_id: Optional[str] = None) -> bytes:
        """Internal method to download and decrypt data from GCS"""
        if not self.bucket:
            raise ConnectionError("GCS bucket is not initialized")
        
        try:
            blob = self.bucket.blob(gcs_path)
            
            if not blob.exists():
                raise NotFound(f"GCS object not found: {gcs_path}")
            
            # Download encrypted data
            encrypted_data = blob.download_as_bytes()
            
            # Decrypt data
            data = self._decrypt_data(encrypted_data)
            
            # Verify integrity if checksum is available
            if blob.metadata and 'original_checksum' in blob.metadata:
                expected_checksum = blob.metadata['original_checksum']
                actual_checksum = self._calculate_checksum(data)
                
                if expected_checksum != actual_checksum:
                    error_msg = f"Integrity check failed for {gcs_path}"
                    self._log_audit_event('model_download', gcs_path, user_id, success=False, error=error_msg)
                    raise ValueError(error_msg)
            
            # Audit log
            self._log_audit_event('model_download', gcs_path, user_id, success=True)
            
            logger.info(f"Successfully downloaded from gs://{self.bucket_name}/{gcs_path}")
            return data
            
        except Exception as e:
            self._log_audit_event('model_download', gcs_path, user_id, success=False, error=str(e))
            logger.error(f"Failed to download from {gcs_path}: {e}")
            raise
    
    def upload_joblib(self, model_object: Any, gcs_path: str, user_id: Optional[str] = None) -> str:
        """Upload scikit-learn/XGBoost model with security features"""
        try:
            buffer = io.BytesIO()
            joblib.dump(model_object, buffer)
            data = buffer.getvalue()
            
            metadata = {
                'model_type': 'joblib',
                'model_class': model_object.__class__.__name__
            }
            
            return self._upload_to_gcs(data, gcs_path, 'application/octet-stream', user_id, metadata)
            
        except Exception as e:
            logger.error(f"Failed to upload joblib model: {e}")
            raise
    
    def download_joblib(self, gcs_path: str, user_id: Optional[str] = None) -> Any:
        """Download and load scikit-learn/XGBoost model"""
        try:
            data = self._download_from_gcs(gcs_path, user_id)
            buffer = io.BytesIO(data)
            return joblib.load(buffer)
            
        except Exception as e:
            logger.error(f"Failed to download joblib model: {e}")
            raise
    
    def upload_pytorch_model(self, model_state_dict: Dict, gcs_path: str, 
                           user_id: Optional[str] = None, model_info: Optional[Dict] = None) -> str:
        """Upload PyTorch model state dict with security features"""
        try:
            buffer = io.BytesIO()
            torch.save(model_state_dict, buffer)
            data = buffer.getvalue()
            
            metadata = {
                'model_type': 'pytorch',
                'state_dict_keys': len(model_state_dict)
            }
            
            if model_info:
                metadata.update({f'model_{k}': str(v) for k, v in model_info.items()})
            
            return self._upload_to_gcs(data, gcs_path, 'application/octet-stream', user_id, metadata)
            
        except Exception as e:
            logger.error(f"Failed to upload PyTorch model: {e}")
            raise
    
    def download_pytorch_model(self, gcs_path: str, device: str = 'cpu', 
                              user_id: Optional[str] = None) -> Dict:
        """Download and load PyTorch model state dict"""
        try:
            data = self._download_from_gcs(gcs_path, user_id)
            buffer = io.BytesIO(data)
            return torch.load(buffer, map_location=device)
            
        except Exception as e:
            logger.error(f"Failed to download PyTorch model: {e}")
            raise
    
    def list_models(self, prefix: str = "models/", user_id: Optional[str] = None) -> list:
        """List available models with metadata"""
        if not self.bucket:
            raise ConnectionError("GCS bucket is not initialized")
        
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            models = []
            
            for blob in blobs:
                model_info = {
                    'name': blob.name,
                    'size': blob.size,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'updated': blob.updated.isoformat() if blob.updated else None,
                    'metadata': blob.metadata or {}
                }
                models.append(model_info)
            
            self._log_audit_event('model_list', prefix, user_id, success=True)
            return models
            
        except Exception as e:
            self._log_audit_event('model_list', prefix, user_id, success=False, error=str(e))
            logger.error(f"Failed to list models: {e}")
            raise
    
    def delete_model(self, gcs_path: str, user_id: Optional[str] = None) -> bool:
        """Securely delete a model with audit logging"""
        if not self.bucket:
            raise ConnectionError("GCS bucket is not initialized")
        
        try:
            blob = self.bucket.blob(gcs_path)
            
            if not blob.exists():
                raise NotFound(f"Model not found: {gcs_path}")
            
            # Store metadata before deletion for audit
            metadata = blob.metadata or {}
            
            blob.delete()
            
            self._log_audit_event('model_delete', gcs_path, user_id, success=True)
            logger.info(f"Successfully deleted model: gs://{self.bucket_name}/{gcs_path}")
            
            return True
            
        except Exception as e:
            self._log_audit_event('model_delete', gcs_path, user_id, success=False, error=str(e))
            logger.error(f"Failed to delete model {gcs_path}: {e}")
            raise
    
    def get_model_metadata(self, gcs_path: str, user_id: Optional[str] = None) -> Dict:
        """Get model metadata without downloading the model"""
        if not self.bucket:
            raise ConnectionError("GCS bucket is not initialized")
        
        try:
            blob = self.bucket.blob(gcs_path)
            
            if not blob.exists():
                raise NotFound(f"Model not found: {gcs_path}")
            
            metadata = {
                'name': blob.name,
                'size': blob.size,
                'created': blob.time_created.isoformat() if blob.time_created else None,
                'updated': blob.updated.isoformat() if blob.updated else None,
                'content_type': blob.content_type,
                'metadata': blob.metadata or {}
            }
            
            self._log_audit_event('model_metadata', gcs_path, user_id, success=True)
            return metadata
            
        except Exception as e:
            self._log_audit_event('model_metadata', gcs_path, user_id, success=False, error=str(e))
            logger.error(f"Failed to get metadata for {gcs_path}: {e}")
            raise

# Singleton pattern for GCS storage
_gcs_storage_client = None

def get_gcs_storage() -> SecureGCSStorage:
    """Get or create secure GCS storage client"""
    global _gcs_storage_client
    if _gcs_storage_client is None:
        _gcs_storage_client = SecureGCSStorage(
            Config.GCS_BUCKET_NAME, 
            Config.GCS_PROJECT_ID
        )
    return _gcs_storage_client

def cleanup_old_models(days_old: int = 30, user_id: Optional[str] = None) -> int:
    """Cleanup old models for storage management"""
    storage = get_gcs_storage()
    if not storage.bucket:
        logger.warning("Cannot cleanup models: GCS not available")
        return 0
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        deleted_count = 0
        
        for blob in storage.bucket.list_blobs(prefix="models/"):
            if blob.time_created and blob.time_created < cutoff_date:
                try:
                    storage.delete_model(blob.name, user_id)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete old model {blob.name}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old models (older than {days_old} days)")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Model cleanup failed: {e}")
        return 0