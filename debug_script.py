# debug_system.py
"""
Quick debugging script to identify common issues
"""

import os
import sys
import importlib
import traceback
from dotenv import load_dotenv

def debug_imports():
    """Test all critical imports"""
    print("🔍 Testing Python imports...")
    
    critical_modules = [
        'torch',
        'pandas', 
        'numpy',
        'flask',
        'redis',
        'google.cloud.storage',
        'sklearn',
        'xgboost',
        'optuna',
        'yfinance',
        'aiohttp',
        'celery'
    ]
    
    failed_imports = []
    
    for module in critical_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n🚨 Failed imports: {failed_imports}")
        print("Run: pip install -r requirements_complete.txt")
        return False
    
    print("✅ All imports successful")
    return True

def debug_environment():
    """Check environment variables"""
    print("\n🔍 Testing environment variables...")
    
    load_dotenv()
    
    required_vars = {
        'SECRET_KEY': 'Flask secret key',
        'JWT_SECRET_KEY': 'JWT secret key',
        'GCS_BUCKET_NAME': 'Google Cloud Storage bucket',
        'CELERY_BROKER_URL': 'Redis/Celery broker'
    }
    
    missing_vars = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            print(f"❌ {var}: Missing ({description})")
            missing_vars.append(var)
        elif value in ['your_secret_key_here', 'your_jwt_secret_here']:
            print(f"⚠️  {var}: Default value detected")
            missing_vars.append(var)
        else:
            print(f"✅ {var}: Configured")
    
    # Check data providers
    data_providers = ['POLYGON_API_KEY', 'ALPHA_VANTAGE_API_KEY', 'IEX_CLOUD_API_KEY']
    has_provider = any(os.getenv(var) for var in data_providers)
    
    if not has_provider:
        print("❌ No data provider API keys configured")
        missing_vars.append('DATA_PROVIDER_API_KEY')
    else:
        print("✅ At least one data provider configured")
    
    return len(missing_vars) == 0

def debug_gcs():
    """Test Google Cloud Storage connection"""
    print("\n🔍 Testing Google Cloud Storage...")
    
    try:
        from google.cloud import storage
        from google.api_core import exceptions
        
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        if not bucket_name:
            print("❌ GCS_BUCKET_NAME not configured")
            return False
        
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        
        # Test upload
        test_blob = bucket.blob("debug/test.txt")
        test_blob.upload_from_string("debug test")
        
        # Test download
        downloaded = test_blob.download_as_text()
        
        # Cleanup
        test_blob.delete()
        
        print("✅ GCS connection and permissions working")
        return True
        
    except exceptions.NotFound:
        print(f"❌ GCS bucket '{bucket_name}' not found")
        return False
    except Exception as e:
        print(f"❌ GCS error: {e}")
        return False

def debug_redis():
    """Test Redis connection"""
    print("\n🔍 Testing Redis connection...")
    
    try:
        import redis
        
        redis_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        
        # Test connection
        r.ping()
        
        # Test set/get
        r.set('debug_test', 'working')
        value = r.get('debug_test')
        r.delete('debug_test')
        
        if value == b'working':
            print("✅ Redis connection working")
            return True
        else:
            print("❌ Redis set/get test failed")
            return False
            
    except Exception as e:
        print(f"❌ Redis error: {e}")
        print("Make sure Redis is running: docker run -d --name redis -p 6379:6379 redis:7-alpine")
        return False

def debug_market_data():
    """Test market data API"""
    print("\n🔍 Testing market data APIs...")
    
    import requests
    
    # Test Alpha Vantage
    av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if av_key and av_key != 'your_alpha_vantage_key':
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={av_key}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data:
                print("✅ Alpha Vantage API working")
                return True
            elif 'Error Message' in data:
                print(f"❌ Alpha Vantage error: {data['Error Message']}")
            elif 'Note' in data:
                print(f"⚠️  Alpha Vantage rate limited: {data['Note']}")
            else:
                print(f"❌ Alpha Vantage unexpected response: {data}")
                
        except Exception as e:
            print(f"❌ Alpha Vantage connection error: {e}")
    
    # Test Polygon
    polygon_key = os.getenv('POLYGON_API_KEY')
    if polygon_key and polygon_key != 'your_polygon_api_key':
        try:
            url = f"https://api.polygon.io/v2/last/nbbo/AAPL?apiKey={polygon_key}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('status') == 'OK':
                print("✅ Polygon API working")
                return True
            else:
                print(f"❌ Polygon API error: {data}")
                
        except Exception as e:
            print(f"❌ Polygon connection error: {e}")
    
    print("⚠️  No working market data APIs found")
    return False

def debug_project_imports():
    """Test project-specific imports"""
    print("\n🔍 Testing project imports...")
    
    # Add current directory to path
    sys.path.insert(0, '.')
    
    project_modules = [
        'project.config',
        'project.storage', 
        'project.market_data',
        'project.consolidation_network',
        'project.breakout_strategy',
        'project.auto_optimizer_working'
    ]
    
    failed_imports = []
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
        except Exception as e:
            print(f"⚠️  {module}: {e}")
    
    return len(failed_imports) == 0

def run_minimal_test():
    """Run a minimal system test"""
    print("\n🔍 Running minimal system test...")
    
    try:
        # Add to path
        sys.path.insert(0, '.')
        
        # Test config loading
        from project.config import Config
        print("✅ Config loaded")
        
        # Test storage initialization 
        from project.storage import get_gcs_storage
        storage = get_gcs_storage()
        print("✅ Storage initialized")
        
        # Test basic model import
        from project.consolidation_network import NetworkConsolidationAnalyzer
        analyzer = NetworkConsolidationAnalyzer()
        print("✅ Analyzer created")
        
        print("✅ Minimal system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all debugging tests"""
    print("🐛 ML Trading System Debug")
    print("=" * 50)
    
    tests = [
        ("Python Imports", debug_imports),
        ("Environment Variables", debug_environment), 
        ("Google Cloud Storage", debug_gcs),
        ("Redis Connection", debug_redis),
        ("Market Data APIs", debug_market_data),
        ("Project Imports", debug_project_imports),
        ("Minimal System Test", run_minimal_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "="*50)
    print(f"🎯 Debug Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System should work.")
        print("\nRun: python run_system.py --simple")
    else:
        print("🚨 Some tests failed. Fix issues above before starting system.")
        
        if passed >= total - 2:
            print("💡 You're close! Try running anyway: python run_system.py --simple")

if __name__ == "__main__":
    main()
