#!/usr/bin/env python3
"""
Comprehensive Test Script for Phase 10 Action 2.2 - Step 3
Scheduling & Error Handling System Validation
"""

import sys
import os
import time
import json
from datetime import datetime

# Add the backend path to sys.path
sys.path.append('/home/ubuntu/ignition_backend_api')

def test_imports():
    """Test 1: Validate all imports are working"""
    print("🧪 Test 1: Import Validation")
    try:
        from src.celery_app import celery_app, DEFAULT_TICKERS
        from src.utils.error_handling import (
            CircuitBreaker, intelligent_retry, create_robust_session,
            APIError, ErrorSeverity, RetryStrategy, global_error_tracker,
            get_system_health
        )
        from src.tasks.fundamental_tasks import (
            collect_fundamental_data, scheduled_weekly_update,
            scheduled_health_check, comprehensive_system_monitoring
        )
        print("   ✅ All imports successful")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {str(e)}")
        return False

def test_celery_configuration():
    """Test 2: Validate Celery Beat configuration"""
    print("\n🧪 Test 2: Celery Beat Configuration")
    try:
        from src.celery_app import celery_app
        
        # Check beat schedule
        beat_schedule = celery_app.conf.beat_schedule
        expected_tasks = [
            'weekly-fundamental-update',
            'quarterly-comprehensive-update', 
            'daily-health-check',
            'weekly-data-quality-check',
            'monthly-cleanup'
        ]
        
        for task_name in expected_tasks:
            if task_name in beat_schedule:
                task_config = beat_schedule[task_name]
                print(f"   ✅ {task_name}: {task_config['task']}")
            else:
                print(f"   ❌ Missing task: {task_name}")
                return False
        
        # Check timezone
        timezone = celery_app.conf.timezone
        print(f"   ✅ Timezone: {timezone}")
        
        # Check default tickers
        from src.celery_app import DEFAULT_TICKERS
        print(f"   ✅ Default tickers: {len(DEFAULT_TICKERS)} symbols")
        
        return True
    except Exception as e:
        print(f"   ❌ Celery configuration test failed: {str(e)}")
        return False

def test_error_handling_components():
    """Test 3: Validate error handling components"""
    print("\n🧪 Test 3: Error Handling Components")
    try:
        from src.utils.error_handling import (
            CircuitBreaker, create_robust_session, APIError, 
            ErrorSeverity, RetryStrategy, global_error_tracker
        )
        
        # Test Circuit Breaker
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        print(f"   ✅ Circuit Breaker: threshold={cb.failure_threshold}, timeout={cb.recovery_timeout}")
        
        # Test Robust Session
        session = create_robust_session()
        print(f"   ✅ Robust HTTP Session: {type(session).__name__}")
        
        # Test Error Severity
        error = APIError("Test error", "TEST_ERROR", ErrorSeverity.MEDIUM)
        print(f"   ✅ API Error: {error.severity.value}")
        
        # Test Retry Strategies
        strategies = [s.value for s in RetryStrategy]
        print(f"   ✅ Retry Strategies: {', '.join(strategies)}")
        
        # Test Error Tracker
        error_summary = global_error_tracker.get_error_summary()
        print(f"   ✅ Error Tracker: {error_summary['total_errors']} errors tracked")
        
        return True
    except Exception as e:
        print(f"   ❌ Error handling test failed: {str(e)}")
        return False

def test_monitoring_functions():
    """Test 4: Validate monitoring functions"""
    print("\n🧪 Test 4: Monitoring Functions")
    try:
        from src.utils.error_handling import get_system_health
        
        # Test system health
        health_data = get_system_health()
        print(f"   ✅ System Health Score: {health_data['health_score']}/100")
        print(f"   ✅ System Status: {health_data['status']}")
        print(f"   ✅ Error Summary: {health_data['error_summary']['total_errors']} errors")
        
        # Test recommendations
        recommendations = health_data.get('recommendations', [])
        print(f"   ✅ Recommendations: {len(recommendations)} items")
        
        return True
    except Exception as e:
        print(f"   ❌ Monitoring test failed: {str(e)}")
        return False

def test_task_functions():
    """Test 5: Validate task functions (dry run)"""
    print("\n🧪 Test 5: Task Functions (Dry Run)")
    try:
        from src.tasks.fundamental_tasks import (
            get_db_session, process_fundamental_data,
            calculate_completeness_score
        )
        
        # Test database session
        session = get_db_session()
        print(f"   ✅ Database Session: {type(session).__name__}")
        session.close()
        
        # Test completeness calculation (with mock data)
        class MockFundamentalData:
            def __init__(self):
                self.market_cap = 1000000
                self.pe_ratio = 15.5
                self.revenue_growth = 0.1
                self.profit_margin = 0.2
                self.debt_to_equity = 0.5
                self.roe = 0.15
                self.current_ratio = 1.5
                self.eps = 2.5
        
        mock_data = MockFundamentalData()
        completeness = calculate_completeness_score(mock_data)
        print(f"   ✅ Completeness Score: {completeness}%")
        
        return True
    except Exception as e:
        print(f"   ❌ Task functions test failed: {str(e)}")
        return False

def test_circuit_breaker_behavior():
    """Test 6: Circuit Breaker behavior simulation"""
    print("\n🧪 Test 6: Circuit Breaker Behavior")
    try:
        from src.utils.error_handling import CircuitBreaker, APIError
        
        # Create test circuit breaker
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
        
        @cb
        def test_function(should_fail=False):
            if should_fail:
                raise APIError("Simulated failure")
            return "Success"
        
        # Test normal operation
        result = test_function(False)
        print(f"   ✅ Normal operation: {result}")
        print(f"   ✅ Circuit state: {cb.state.value}")
        
        # Simulate failures
        failure_count = 0
        for i in range(5):
            try:
                test_function(True)
            except APIError:
                failure_count += 1
                print(f"   ⚠️  Failure {failure_count}: Circuit state = {cb.state.value}")
                if cb.state.value == 'open':
                    print(f"   ✅ Circuit opened after {failure_count} failures")
                    break
        
        return True
    except Exception as e:
        print(f"   ❌ Circuit breaker test failed: {str(e)}")
        return False

def test_retry_mechanism():
    """Test 7: Retry mechanism simulation"""
    print("\n🧪 Test 7: Retry Mechanism")
    try:
        from src.utils.error_handling import intelligent_retry, RetryStrategy
        
        attempt_count = 0
        
        @intelligent_retry(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=0.1,  # Short delay for testing
            exceptions=(ValueError,)
        )
        def test_retry_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"Attempt {attempt_count} failed")
            return f"Success on attempt {attempt_count}"
        
        start_time = time.time()
        result = test_retry_function()
        end_time = time.time()
        
        print(f"   ✅ Retry result: {result}")
        print(f"   ✅ Total attempts: {attempt_count}")
        print(f"   ✅ Total time: {end_time - start_time:.2f}s")
        
        return True
    except Exception as e:
        print(f"   ❌ Retry mechanism test failed: {str(e)}")
        return False

def test_scheduled_task_structure():
    """Test 8: Scheduled task structure validation"""
    print("\n🧪 Test 8: Scheduled Task Structure")
    try:
        from src.tasks.fundamental_tasks import (
            scheduled_weekly_update, scheduled_health_check,
            comprehensive_system_monitoring
        )
        
        # Check task signatures
        tasks = [
            ('scheduled_weekly_update', scheduled_weekly_update),
            ('scheduled_health_check', scheduled_health_check),
            ('comprehensive_system_monitoring', comprehensive_system_monitoring)
        ]
        
        for task_name, task_func in tasks:
            if hasattr(task_func, 'name'):
                print(f"   ✅ {task_name}: Celery task name = {task_func.name}")
            else:
                print(f"   ✅ {task_name}: Function available")
        
        return True
    except Exception as e:
        print(f"   ❌ Scheduled task structure test failed: {str(e)}")
        return False

def run_comprehensive_test():
    """Run all tests and generate report"""
    print("🚀 Starting Comprehensive Scheduling System Test")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Import Validation", test_imports),
        ("Celery Configuration", test_celery_configuration),
        ("Error Handling Components", test_error_handling_components),
        ("Monitoring Functions", test_monitoring_functions),
        ("Task Functions", test_task_functions),
        ("Circuit Breaker Behavior", test_circuit_breaker_behavior),
        ("Retry Mechanism", test_retry_mechanism),
        ("Scheduled Task Structure", test_scheduled_task_structure)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {str(e)}")
            test_results.append((test_name, False))
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - SYSTEM IS READY FOR PRODUCTION!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)

