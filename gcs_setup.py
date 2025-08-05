# setup_gcs.py
"""
Google Cloud Storage Setup and Validation
Run this after setting up your GCS credentials
"""

import os
import json
from google.cloud import storage
from google.api_core import exceptions
from datetime import datetime

def setup_gcs_bucket():
    """Create and configure GCS bucket for ML models"""
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    project_id = os.getenv('GCS_PROJECT_ID')
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not all([bucket_name, project_id, credentials_path]):
        print("❌ Missing GCS configuration in .env file")
        print("Required: GCS_BUCKET_NAME, GCS_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS")
        return False
    
    if not os.path.exists(credentials_path):
        print(f"❌ Credentials file not found: {credentials_path}")
        print("\nTo create service account credentials:")
        print("1. Go to Google Cloud Console")
        print("2. Create a new service account")
        print("3. Download JSON key file")
        print("4. Place it at:", credentials_path)
        return False
    
    try:
        # Initialize client
        client = storage.Client(project=project_id)
        print(f"✅ GCS client initialized for project: {project_id}")
        
        # Check if bucket exists
        try:
            bucket = client.get_bucket(bucket_name)
            print(f"✅ Bucket '{bucket_name}' already exists")
        except exceptions.NotFound:
            # Create bucket
            print(f"Creating bucket: {bucket_name}")
            bucket = client.create_bucket(bucket_name)
            print(f"✅ Bucket '{bucket_name}' created")
        
        # Create directory structure
        directories = [
            'models/',
            'models/breakout_predictor/',
            'models/lstm/',
            'models/xgboost/',
            'screening_results/',
            'optimization_history/',
            'alerts/',
            'backtest_results/'
        ]
        
        for directory in directories:
            # Create placeholder file to establish directory
            blob = bucket.blob(f"{directory}.gitkeep")
            blob.upload_from_string("# Directory placeholder")
        
        print("✅ Directory structure created")
        
        # Test upload/download
        test_data = {
            "test_timestamp": datetime.now().isoformat(),
            "setup_status": "success"
        }
        
        test_blob = bucket.blob("test/setup_validation.json")
        test_blob.upload_from_string(json.dumps(test_data))
        
        # Download to verify
        downloaded = json.loads(test_blob.download_as_text())
        
        if downloaded["setup_status"] == "success":
            print("✅ Upload/download test successful")
            
            # Clean up test file
            test_blob.delete()
            return True
        else:
            print("❌ Upload/download test failed")
            return False
            
    except Exception as e:
        print(f"❌ GCS setup failed: {e}")
        return False

def validate_data_providers():
    """Validate data provider API keys"""
    from dotenv import load_dotenv
    load_dotenv()
    
    providers = {
        'POLYGON_API_KEY': 'Polygon.io',
        'ALPHA_VANTAGE_API_KEY': 'Alpha Vantage',
        'IEX_CLOUD_API_KEY': 'IEX Cloud',
        'TIINGO_API_KEY': 'Tiingo'
    }
    
    print("\n📊 Validating Data Provider APIs...")
    
    valid_providers = 0
    
    for env_var, provider_name in providers.items():
        api_key = os.getenv(env_var)
        if api_key and api_key != 'your_api_key_here':
            print(f"✅ {provider_name}: API key configured")
            valid_providers += 1
        else:
            print(f"⚠️  {provider_name}: No API key configured")
    
    if valid_providers == 0:
        print("❌ No data provider APIs configured!")
        print("\nTo get API keys:")
        print("• Polygon.io: https://polygon.io/")
        print("• Alpha Vantage: https://www.alphavantage.co/")
        print("• IEX Cloud: https://iexcloud.io/")
        print("• Tiingo: https://www.tiingo.com/")
        return False
    else:
        print(f"✅ {valid_providers} data provider(s) configured")
        return True

def validate_security_setup():
    """Validate security configuration"""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n🔐 Validating Security Setup...")
    
    required_secrets = [
        'SECRET_KEY',
        'JWT_SECRET_KEY',
        'ENCRYPTION_KEY'
    ]
    
    missing_secrets = []
    
    for secret in required_secrets:
        value = os.getenv(secret)
        if not value or value in ['your_secret_key_here', 'your_jwt_secret_here', 'your_fernet_key_here']:
            missing_secrets.append(secret)
        else:
            print(f"✅ {secret}: Configured")
    
    if missing_secrets:
        print(f"❌ Missing security configuration: {missing_secrets}")
        print("\nGenerate secure secrets:")
        print("SECRET_KEY: openssl rand -hex 32")
        print("JWT_SECRET_KEY: openssl rand -hex 32")
        print("ENCRYPTION_KEY: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'")
        return False
    
    print("✅ Security configuration complete")
    return True

def main():
    """Run complete setup validation"""
    print("🚀 ML Trading System Setup Validation")
    print("=" * 50)
    
    checks = [
        ("Security Setup", validate_security_setup),
        ("Data Providers", validate_data_providers),
        ("Google Cloud Storage", setup_gcs_bucket)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}")
        print("-" * 30)
        
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All setup checks passed!")
        print("\nYou can now run:")
        print("python run_system.py")
    else:
        print("❌ Some setup checks failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()
