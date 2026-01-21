import asyncio
import json
import logging
from classifier import ContentClassifier
from database import VectorDatabase

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pipeline():
    print("\n🚀 Starting End-to-End Pipeline Test...\n")

    # 1. Initialize Components
    try:
        classifier = ContentClassifier(use_llm_fallback=True) # Use LLM!
        print("✅ Classifier Initialized")
    except Exception as e:
        print(f"❌ Classifier Init Failed: {e}")
        return

    # 2. Test Content (Simulated extracted text)
    sample_theory_text = """
    Ind AS 116 Leases:
    A lessee applies a single recognition and measurement approach for all leases, 
    except for short-term leases and leases of low-value assets. 
    The lessee recognizes a lease liability to make lease payments and a right-of-use asset represents the right to use the underlying asset.
    """

    sample_quiz_text = """
    Question 4:
    ABC Ltd has a lease contract for 10 years. The annual lease payment is Rs 1,00,000. 
    What is the discount rate if the PV is Rs 6,50,000?
    
    Answer:
    Using the annuity formula, the implicit rate is approx 8.5%.
    """

    # 3. Test Enrichment (Theory)
    print("\n--- Testing Theory Enrichment ---")
    context_theory = "Level: Final, Paper: Financial Reporting, Chapter: Ind AS 116. Context from Chapter Weights: {'Ind AS 116': 'A'}"
    enrichment_theory = classifier.enrich_content(sample_theory_text, context=context_theory)
    print(json.dumps(enrichment_theory, indent=2))

    # 4. Test Enrichment (Quiz Question)
    print("\n--- Testing Quiz Enrichment (Q&A Separation) ---")
    context_quiz = "Level: Final, Paper: Financial Management, Chapter: Leases."
    enrichment_quiz = classifier.enrich_content(sample_quiz_text, context=context_quiz)
    print(json.dumps(enrichment_quiz, indent=2))

    # 5. Validation
    if enrichment_quiz.get("question_text") and enrichment_quiz.get("answer_text"):
        print("\n✅ Q&A Separation SUCCESS!")
    else:
        print("\n❌ Q&A Separation FAILED!")

    if enrichment_theory.get("importance"):
        print(f"✅ Importance Tagging SUCCESS: {enrichment_theory.get('importance')}")
    else:
        print("\n❌ Importance Tagging FAILED!")

    print("\n✨ Test Complete.")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
cd /home/pranawww/Work/ca/ca-docs-backend && uv run python -c "from database import VectorDatabase; db = VectorDatabase(); conn = db.get_connection(); cur = conn.cursor(); cur.execute('TRUNCATE TABLE documents, tables, file_metadata, question_logs RESTART IDENTITY CASCADE;'); conn.commit(); print('🔥 Database Wiped Clean')"