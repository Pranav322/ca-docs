import requests
import time
import sys

BASE_URL = "http://localhost:8001"

def wait_for_server():
    print("Waiting for server to start...")
    for _ in range(30):
        try:
            requests.get(f"{BASE_URL}/api/health")
            print("Server is up!")
            return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    print("Server failed to start.")
    return False

def test_rate_limit():
    if not wait_for_server():
        sys.exit(1)

    print("Testing rate limit (5/day)...")
    success_count = 0
    blocked = False
    
    # Try 7 requests (limit is 5)
    for i in range(1, 8):
        try:
            # Simple question request
            response = requests.post(
                f"{BASE_URL}/api/questions/ask",
                json={
                    "question": "What is the syllabus?",
                    "level": "Foundation",
                    "paper": "Paper-1",
                    "include_tables": False
                }
            )
            
            print(f"Request {i}: Status {response.status_code}")
            
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                print("✅ Rate limit hit as expected!")
                try:
                    detail = response.json().get("detail", "")
                    print(f"Message: {detail}")
                    if "We are having limited server capacity" in detail:
                        print("✅ Custom error message verified!")
                    else:
                        print("❌ Custom error message missing!")
                except:
                    print("❌ Could not parse error response")
                blocked = True
                break
            else:
                print(f"❌ Unexpected status code: {response.status_code}")
                # print(response.text)
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            
    if success_count <= 5 and blocked:
        print("\n✅ Rate limiting passed verification!")
    else:
        print(f"\n❌ Rate limiting verification failed! Successes: {success_count}, Blocked: {blocked}")

if __name__ == "__main__":
    test_rate_limit()
