#!/usr/bin/env python3
"""
Test script for table_processor.py to verify the fix for integer column names
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from table_processor import TableProcessor

def test_integer_columns():
    """Test that tables with integer column names are processed correctly"""
    processor = TableProcessor()

    # Test data with integer columns (this would cause the original error)
    table_data = {
        'data': [
            {0: 'Company A', 1: 1000, 2: 500},
            {0: 'Company B', 1: 2000, 2: 750}
        ],
        'columns': [0, 1, 2],  # Integer columns
        'page_number': 1,
        'context_before': 'Financial summary table',
        'context_after': 'End of table'
    }

    try:
        result = processor.process_table_for_embedding(table_data)
        print("✅ SUCCESS: Table with integer columns processed without error")
        print(f"Result length: {len(result)} characters")
        print(f"First 200 chars: {result[:200]}...")

        # Verify the result contains expected elements
        assert "Column Headers: 0, 1, 2" in result
        assert "Row 1:" in result
        assert "Row 2:" in result
        print("✅ All assertions passed")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_string_columns():
    """Test that tables with string columns still work"""
    processor = TableProcessor()

    table_data = {
        'data': [
            {'Company': 'Company A', 'Revenue': 1000, 'Profit': 500},
            {'Company': 'Company B', 'Revenue': 2000, 'Profit': 750}
        ],
        'columns': ['Company', 'Revenue', 'Profit'],
        'page_number': 2,
        'context_before': 'Profit table',
        'context_after': ''
    }

    try:
        result = processor.process_table_for_embedding(table_data)
        print("✅ SUCCESS: Table with string columns processed correctly")
        assert "Column Headers: Company, Revenue, Profit" in result
        print("✅ String columns test passed")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Testing table_processor.py fixes...\n")

    test1_passed = test_integer_columns()
    print()
    test2_passed = test_string_columns()

    print("\n" + "="*50)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED - The fix is working correctly!")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED - Check the implementation")
        sys.exit(1)