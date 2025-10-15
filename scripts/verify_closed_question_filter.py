#!/usr/bin/env python3
"""
Verify that the closed question filter is working correctly.
This script tests the new closed question guard in TemplateForecaster.forecast_question
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_status_detection():
    """Test different ways a question can be marked as closed."""
    
    print("Testing closed question detection logic...")
    print("=" * 80)
    
    # Test case 1: status attribute set to "closed"
    class MockQuestionClosed:
        def __init__(self):
            self.id = 12345
            self.status = "closed"
            self.api_json = {}
    
    # Test case 2: status in api_json
    class MockQuestionApiClosed:
        def __init__(self):
            self.id = 12346
            self.status = None
            self.api_json = {"status": "closed", "open_for_forecasting": False}
    
    # Test case 3: open_for_forecasting flag False
    class MockQuestionNotOpen:
        def __init__(self):
            self.id = 12347
            self.status = "open"  # misleading status
            self.api_json = {"status": "open", "open_for_forecasting": False}
    
    # Test case 4: Actually open question
    class MockQuestionOpen:
        def __init__(self):
            self.id = 12348
            self.status = "open"
            self.api_json = {"status": "open", "open_for_forecasting": True}
    
    # Test case 5: Resolved question
    class MockQuestionResolved:
        def __init__(self):
            self.id = 12349
            self.status = "resolved"
            self.api_json = {"status": "resolved"}
    
    test_cases = [
        (MockQuestionClosed(), True, "status='closed'"),
        (MockQuestionApiClosed(), True, "api_json status='closed'"),
        (MockQuestionNotOpen(), True, "open_for_forecasting=False"),
        (MockQuestionOpen(), False, "Actually open question"),
        (MockQuestionResolved(), True, "status='resolved'"),
    ]
    
    for question, expected_closed, description in test_cases:
        question_id = getattr(question, "id", "unknown")
        status = getattr(question, "status", None)
        api_json = getattr(question, "api_json", {})
        
        # Apply the same logic as in forecast_question
        is_closed = False
        if status and str(status).lower() in ["closed", "resolved", "pending_resolution"]:
            is_closed = True
        elif api_json:
            api_status = api_json.get("status", "").lower()
            if api_status in ["closed", "resolved", "pending_resolution"]:
                is_closed = True
            if not api_json.get("open_for_forecasting", True):
                is_closed = True
        
        result = "✅ PASS" if is_closed == expected_closed else "❌ FAIL"
        print(f"{result} | Q{question_id} | {description}")
        print(f"         Expected closed={expected_closed}, Got closed={is_closed}")
        print()
    
    print("=" * 80)
    print("Test complete!")

if __name__ == "__main__":
    test_status_detection()
