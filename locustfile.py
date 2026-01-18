"""
Locust load test for CA RAG Assistant API
Run with: uv run locust -f locustfile.py --host=http://localhost:8000
"""
from locust import HttpUser, task, between
import random


class FrontendUser(HttpUser):
    """Simulates a user using the React frontend"""
    
    # Wait 2-5 seconds between tasks (realistic user behavior)
    wait_time = between(2, 5)
    
    questions = [
        "What is accounting?",
        "Explain the concept of depreciation",
        "What are the types of shares?",
        "Define trial balance",
        "What is the difference between revenue and capital expenditure?",
        "What is a balance sheet?",
        "Explain GST input tax credit",
        "What are contingent liabilities?",
    ]
    
    @task(2)  # Weight 2 - common on page load
    def get_curriculum(self):
        """Fetch curriculum hierarchy (called on page load)"""
        self.client.get("/api/curriculum")
    
    @task(5)  # Weight 5 - main action users take
    def ask_question(self):
        """Ask a question - the main RAG functionality"""
        question = random.choice(self.questions)
        
        with self.client.post(
            "/api/questions/ask",
            json={
                "question": question,
                "level": None,
                "paper": None,
                "include_tables": True
            },
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

