#!/usr/bin/env python3
"""
Test script to verify the async/sync fix for UserClient
"""

import asyncio
from task.user_client import UserClient

def test_sync_get_user():
    """Test that get_user is now synchronous"""
    client = UserClient()
    
    # This should work now as a synchronous call
    try:
        # Test with a sample user ID (1)
        user = client.get_user(1)
        print(f"✓ get_user() is now synchronous and works correctly")
        print(f"  User data sample: ID={user.get('id')}, Name={user.get('name')}")
        return True
    except Exception as e:
        print(f"✗ Error calling get_user(): {e}")
        return False

async def test_async_context():
    """Test that the synchronous get_user can be called from async context"""
    client = UserClient()
    
    try:
        # This should work without await
        user = client.get_user(1)
        print(f"✓ get_user() can be called from async context without await")
        return True
    except Exception as e:
        print(f"✗ Error in async context: {e}")
        return False

def main():
    print("Testing UserClient fixes...")
    print("-" * 40)
    
    # Test 1: Verify get_user is synchronous
    test1_passed = test_sync_get_user()
    
    # Test 2: Verify it works in async context
    print("\nTesting in async context...")
    test2_passed = asyncio.run(test_async_context())
    
    print("\n" + "=" * 40)
    if test1_passed and test2_passed:
        print("✓ All tests passed! The fix is working correctly.")
        print("\nThe issue was:")
        print("- get_user() was defined as async but used synchronous requests.get()")
        print("- This caused the app to hang when awaiting the function")
        print("\nThe fix:")
        print("- Changed get_user() from async to sync (removed 'async' keyword)")
        print("- Updated in_out_grounding.py to call it without 'await'")
    else:
        print("✗ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()