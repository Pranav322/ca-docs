"""
Content Classifier for CA RAG System
Uses a hybrid approach: regex heuristics first, then Azure GPT-4 fallback
"""

import re
import logging
from typing import Optional, Tuple
from enum import Enum
from openai import AzureOpenAI

from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_VERSION,
    AZURE_LLM_DEPLOYMENT  # Use existing deployment
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type classification for CA chunks"""
    THEORY = "theory"
    FORMULA = "formula"
    DEFINITION = "definition"
    EXAMPLE = "example"
    TABLE = "table"
    UNKNOWN = "unknown"


class ContentClassifier:
    """
    Smart regex-based content classifier for CA study materials.
    Optimized for speed - no LLM fallback needed with comprehensive patterns.
    """

    # FORMULA PATTERNS - Mathematical, financial, and accounting formulas
    FORMULA_PATTERNS = [
        # LaTeX patterns
        r'\\frac\{',           # LaTeX fractions
        r'\\sum',              # Summation
        r'\\int',              # Integration
        r'\\sqrt',             # Square root
        r'\$\$.+?\$\$',        # Display math
        r'\$.+?\$',            # Inline math
        
        # Mathematical equations
        r'=\s*[^=]+[+\-*/^×÷]',  # Equations with operators
        r'\([^)]+\)\s*[×x÷/]\s*\d',  # Bracketed calculations
        r'\d+\s*[×x÷/+\-]\s*\d+\s*=',  # Simple arithmetic
        
        # Financial formulas (CA-specific)
        r'(?:NPV|IRR|EMI|PV|FV|ROI|ROE|ROA|EPS|P/E|WACC)\s*=',
        r'(?:Gross Profit|Net Profit|EBITDA|EBIT)\s*=',
        r'(?:Current Ratio|Quick Ratio|Debt Equity)\s*=',
        r'(?:Rate of Return|Rate of Interest|Discount Rate)\s*=',
        r'(?:Depreciation|Amortization)\s*=\s*',
        r'(?:Taxable Income|Tax Payable|TDS)\s*=',
        r'(?:Contribution|Margin|BEP|Break.?Even)\s*=',
        r'(?:Variance|Standard Deviation|Std\.? Dev\.?)\s*=',
        
        # Percentage calculations
        r'\d+\.?\d*\s*%\s*(?:of|×|x)',
        r'=\s*\d+\.?\d*\s*%',
        
        # Ratio expressions
        r'\d+\s*:\s*\d+',  # Ratios like 3:2
    ]

    # DEFINITION PATTERNS - Formal definitions and concepts
    DEFINITION_PATTERNS = [
        # Explicit definition markers
        r'(?:^|\n)\s*(?:Definition|Meaning|Concept|Term)[:\s]+',
        r'(?:is defined as|means|refers to|can be defined as)',
        r'(?:shall mean|would mean|includes|excludes)',
        
        # CA/Accounting standards references
        r'As per (?:AS|Ind AS|IAS|IFRS|SA|SQC)\s*\d*',
        r'(?:According to|Under|In terms of)\s+(?:AS|Ind AS|Section|Clause)',
        r'Section\s+\d+[A-Z]?\s+(?:of|defines|states)',
        
        # Quote-based definitions
        r'(?:^|\n)\s*[""][A-Z][^""]+[""]\s+(?:means|refers|is)',
        
        # Key term introductions
        r'(?:The term|The concept of|The expression)\s+[""\'"][^""\']+[""\'"]',
        r'(?:^|\n)\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+–\s+',  # Term dash definition
    ]

    # EXAMPLE/ILLUSTRATION PATTERNS
    EXAMPLE_PATTERNS = [
        # Explicit markers
        r'(?:^|\n)\s*(?:Example|Illustration|Case Study|Practical Problem)[:\s\-\d]*',
        r'(?:^|\n)\s*(?:Problem|Solution|Working|Computation)[:\s\-\d]*',
        r'(?:^|\n)\s*(?:Q\.|Ques\.|Question)\s*\d+',
        
        # Contextual markers
        r'(?:For (?:example|instance)|e\.g\.|viz\.|i\.e\.|Let us consider)',
        r'(?:Consider the following|Suppose that|Assume that|Given that)',
        r'(?:Mr\.|Mrs\.|Ms\.|M/s|ABC Ltd|XYZ Ltd|PQR Ltd)',  # Company/person names
        
        # Numerical examples
        r'(?:^|\n)\s*₹\s*[\d,]+',  # Rupee amounts at start
        r'(?:purchased|sold|acquired|invested)\s+.*\s+(?:for|at)\s+₹',
    ]

    # PROCEDURE/STEPS PATTERNS (classify as THEORY with procedure subtype)
    PROCEDURE_PATTERNS = [
        r'(?:^|\n)\s*(?:Step\s*\d+|Procedure|Process|Method)[:\s\-]',
        r'(?:^|\n)\s*(?:First|Second|Third|Finally|Lastly)[,:\s]',
        r'(?:following steps|step.?by.?step|procedure is)',
    ]

    # THEORY INDICATORS (helps distinguish from unknown)
    THEORY_PATTERNS = [
        r'(?:^|\n)\s*(?:Introduction|Objective|Scope|Applicability)[:\s]',
        r'(?:^|\n)\s*(?:Features|Characteristics|Types|Classification)[:\s]',
        r'(?:^|\n)\s*(?:Advantages|Disadvantages|Merits|Demerits)[:\s]',
        r'(?:^|\n)\s*(?:Provisions|Requirements|Conditions)[:\s]',
        r'(?:It is important to|It should be noted|It may be observed)',
        r'(?:The (?:main|key|important|significant) (?:point|aspect|feature))',
    ]

    def __init__(self, use_llm_fallback: bool = True):
        """
        Initialize the classifier
        
        Args:
            use_llm_fallback: Whether to use Azure GPT-4o-mini for ambiguous cases
        """
        self.use_llm_fallback = use_llm_fallback
        self._client = None

    @property
    def client(self) -> AzureOpenAI:
        """Lazy initialization of Azure OpenAI client"""
        if self._client is None:
            self._client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_KEY,
                api_version=AZURE_OPENAI_VERSION
            )
        return self._client

    def _check_formula(self, text: str) -> bool:
        """Check if text contains formula patterns"""
        for pattern in self.FORMULA_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_definition(self, text: str) -> bool:
        """Check if text is a definition"""
        for pattern in self.DEFINITION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_example(self, text: str) -> bool:
        """Check if text is an example/illustration"""
        for pattern in self.EXAMPLE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_procedure(self, text: str) -> bool:
        """Check if text describes a procedure/steps"""
        for pattern in self.PROCEDURE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_theory(self, text: str) -> bool:
        """Check if text is theory content"""
        for pattern in self.THEORY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _classify_with_regex(self, text: str) -> Tuple[Optional[ContentType], float]:
        """
        Smart regex classification - no LLM needed
        
        Returns:
            Tuple of (ContentType, confidence score 0-1)
        """
        # Check for examples first (highest specificity - case studies, illustrations)
        if self._check_example(text):
            return ContentType.EXAMPLE, 0.90

        # Check for formulas (mathematical, financial)
        if self._check_formula(text):
            return ContentType.FORMULA, 0.92

        # Check for definitions (AS/Ind AS, formal definitions)
        if self._check_definition(text):
            return ContentType.DEFINITION, 0.88

        # Check for theory/procedure content
        if self._check_theory(text) or self._check_procedure(text):
            return ContentType.THEORY, 0.85

        # Default to THEORY for unmatched content (most CA material is theory)
        # This is better than UNKNOWN and avoids LLM calls
        return ContentType.THEORY, 0.70

    def _classify_with_llm(self, text: str) -> ContentType:
        """
        Use Azure GPT-4o-mini for classification (cheap fallback)
        
        Returns:
            ContentType classification
        """
        try:
            # Truncate text to control token usage
            truncated = text[:1000] if len(text) > 1000 else text

            response = self.client.chat.completions.create(
                model=AZURE_LLM_DEPLOYMENT,  # Use existing gpt-4.1 deployment
                messages=[
                    {
                        "role": "system",
                        "content": """You are a content classifier for Chartered Accountancy study materials.
Classify the given text into exactly one of these categories:
- THEORY: Conceptual explanations, principles, standards
- FORMULA: Mathematical formulas, equations, calculations
- DEFINITION: Formal definitions of terms or concepts
- EXAMPLE: Practical examples, illustrations, case studies

Respond with ONLY the category name (THEORY, FORMULA, DEFINITION, or EXAMPLE)."""
                    },
                    {
                        "role": "user",
                        "content": f"Classify this text:\n\n{truncated}"
                    }
                ],
                max_tokens=10,
                temperature=0
            )

            result = response.choices[0].message.content.strip().upper()

            # Map response to ContentType
            mapping = {
                "THEORY": ContentType.THEORY,
                "FORMULA": ContentType.FORMULA,
                "DEFINITION": ContentType.DEFINITION,
                "EXAMPLE": ContentType.EXAMPLE
            }

            return mapping.get(result, ContentType.THEORY)

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}. Defaulting to THEORY.")
            return ContentType.THEORY

    def classify(self, text: str, is_table: bool = False) -> ContentType:
        """
        Classify content type using hybrid approach
        
        Args:
            text: The text content to classify
            is_table: Whether this content is a table (pre-classified)
            
        Returns:
            ContentType enum value
        """
        if not text or not text.strip():
            return ContentType.UNKNOWN

        # Pre-classified as table
        if is_table:
            return ContentType.TABLE

        # Try regex classification first (free)
        content_type, confidence = self._classify_with_regex(text)

        if content_type and confidence >= 0.8:
            logger.debug(f"Regex classified as {content_type.value} (confidence: {confidence})")
            return content_type

        # Use LLM fallback for ambiguous cases
        if self.use_llm_fallback:
            logger.debug("Using LLM fallback for classification")
            return self._classify_with_llm(text)

        # Default to theory if no LLM fallback
        return ContentType.THEORY

    def classify_batch(self, texts: list, is_table_flags: list = None) -> list:
        """
        Classify multiple texts
        
        Args:
            texts: List of text strings to classify
            is_table_flags: Optional list of boolean flags indicating if each text is a table
            
        Returns:
            List of ContentType values
        """
        if is_table_flags is None:
            is_table_flags = [False] * len(texts)

        return [
            self.classify(text, is_table)
            for text, is_table in zip(texts, is_table_flags)
        ]


def generate_node_id(file_path: str) -> str:
    """
    Generate a stable node ID from the file path
    
    Converts: ca/final/fr/module1/chapter3/unit1/file.pdf
    To: final_fr_m1_c3_u1
    
    Args:
        file_path: Relative path to the PDF file
        
    Returns:
        Stable node ID string
    """
    import os

    # Normalize path
    path = file_path.lower().replace('\\', '/')
    parts = path.split('/')

    # Skip 'ca' folder if present
    if parts and parts[0] == 'ca':
        parts = parts[1:]

    # Extract components
    level = ""
    paper = ""
    module = ""
    chapter = ""
    unit = ""

    for part in parts:
        clean_part = part.lower().strip()

        if any(l in clean_part for l in ['foundation', 'intermediate', 'final']):
            level = 'found' if 'foundation' in clean_part else \
                    'inter' if 'intermediate' in clean_part else 'final'

        elif 'paper' in clean_part:
            # Extract paper number (e.g., "Paper-1" -> "p1")
            match = re.search(r'paper[^\d]*(\d+[a-z]?)', clean_part)
            if match:
                paper = f"p{match.group(1)}"

        elif 'module' in clean_part:
            match = re.search(r'module\s*(\d+)', clean_part)
            if match:
                module = f"m{match.group(1)}"

        elif 'chapter' in clean_part:
            match = re.search(r'chapter\s*(\d+)', clean_part)
            if match:
                chapter = f"c{match.group(1)}"

        elif 'unit' in clean_part:
            match = re.search(r'unit\s*(\d+)', clean_part)
            if match:
                unit = f"u{match.group(1)}"

    # Build node ID
    components = [c for c in [level, paper, module, chapter, unit] if c]
    return '_'.join(components) if components else 'unknown'


# Default instance for convenience
_default_classifier = None


def get_classifier(use_llm_fallback: bool = True) -> ContentClassifier:
    """Get or create the default classifier instance"""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = ContentClassifier(use_llm_fallback=use_llm_fallback)
    return _default_classifier
