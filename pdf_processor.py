import asyncio
import concurrent.futures
import pymupdf as fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from typing import List, Dict, Tuple, Any
import logging
import io
import base64
import tempfile
import os
import pytesseract
import cv2
import numpy as np
from PIL import Image
import time
import hashlib
import re
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# EXTRACTION METHOD QUALITY SCORES (Higher = Better, More Reliable)
# =====================================================================
METHOD_QUALITY = {
    "camelot_lattice": 100,  # Best for bordered tables
    "camelot_stream": 85,   # Good for borderless tables
    "pymupdf_tables": 80,   # Fast, stable, no external deps
    "tabula": 75,           # Java-based, reliable
    "pdfplumber": 60,       # Pure Python fallback
}

# =====================================================================
# BOS QUESTION PATTERNS - For parsing CA/BOS exam structure
# =====================================================================
# Main question patterns: Q1, Question 1, Q.1
QUESTION_PATTERNS = [
    r'(?:^|\n)\s*(?:Q|Question|Ques)[\.\s]*(\d+)\s*[:\.\-]?\s*(.{0,100})?',
]

# Subquestion patterns: (a), (b)(i), (i), (ii)
SUBQUESTION_PATTERNS = [
    r'(?:^|\n)\s*\(([a-z])\)\s*(.{0,100})?',         # (a), (b)
    r'(?:^|\n)\s*\(([ivxlc]+)\)\s*(.{0,100})?',      # (i), (ii), (iii)
    r'(?:^|\n)\s*\(([a-z])\)\s*\(([ivxlc]+)\)',     # (a)(i), (b)(ii)
]

# Special section patterns
SPECIAL_PATTERNS = [
    r'(?:^|\n)\s*(Case Study|Practical Problem|Illustration)[\s\-:]*(\d+)?\s*[:\.\-]?\s*(.{0,100})?',
    r'(?:^|\n)\s*(Solution|Answer|Working|Computation)\s*[:\.\-]',
]

# MCQ patterns
MCQ_PATTERNS = [
    # Pattern 1: Inline (a)...(b)...(c)...(d)
    r'\(([a-d]|[A-D])\)\s*(.+?)(?=\s*\([a-d]|[A-D]\)|$)',
    
    # Pattern 2: Stacked/Labeled Option A., (A), [A]
    r'(?:^|\n)\s*(?:Option|Choice)?\s*[\(\[]?([A-D]|[a-d])[\)\]\.]\s+(.+)',
]

# Try importing optional dependencies
try:
    import camelot.io as camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning(
        "Camelot not available - install with: pip install camelot-py[cv] ghostscript"
    )

try:
    import tabula as tabula_py
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    logger.warning("Tabula not available - table extraction will skip tabula method")


class OptimizedPDFProcessor:
    """
    Robust PDF processor with prioritized multi-method table extraction.
    
    Extraction Priority (Best → Fallback):
    1. Camelot Lattice - Best for bordered/structured tables
    2. Camelot Stream - Good for borderless tables  
    3. PyMuPDF find_tables - Fast, stable, no external deps
    4. Tabula - Java-based, reliable for structured data
    5. pdfplumber - Pure Python fallback
    
    Tables are deduplicated using content-aware signatures.
    """
    
    def __init__(self, max_workers: int = 1):
        # NOTE: max_workers is set to 1 by default because we are now running
        # inside a ProcessPoolExecutor. We don't want to spawn threads inside processes.
        self.supported_formats = [".pdf"]
        self.max_workers = max_workers
        self.page_cache = {}

    def extract_text_and_tables(self, pdf_path: str) -> Dict[str, Any]:
        """
        Synchronous entry point for processing.
        This is now designed to run in a dedicated process.
        """
        start_time = time.time()
        try:
            # 1. Get Basic Info
            basic_info = self._get_basic_pdf_info_sync(pdf_path)

            results = {"text_chunks": [], "tables": []}

            # 2. Extract Text (PyMuPDF - always reliable)
            text_result = self._process_with_pymupdf(pdf_path)
            results["text_chunks"].extend(text_result["text_chunks"])

            # 3. Extract Tables - PRIORITIZED APPROACH (Best First)
            # Each method adds tables with quality scores
            # Deduplication happens at the end
            
            all_tables = []
            
            # Method 1: Camelot Lattice (BEST for bordered tables)
            if CAMELOT_AVAILABLE:
                try:
                    camelot_lattice = self._process_with_camelot(pdf_path, flavor="lattice")
                    if camelot_lattice["tables"]:
                        logger.info(f"Camelot Lattice found {len(camelot_lattice['tables'])} tables")
                        all_tables.extend(camelot_lattice["tables"])
                except Exception as e:
                    logger.warning(f"Camelot Lattice failed (continuing): {e}")

            # Method 2: Camelot Stream (Good for borderless tables)
            if CAMELOT_AVAILABLE:
                try:
                    camelot_stream = self._process_with_camelot(pdf_path, flavor="stream")
                    if camelot_stream["tables"]:
                        logger.info(f"Camelot Stream found {len(camelot_stream['tables'])} tables")
                        all_tables.extend(camelot_stream["tables"])
                except Exception as e:
                    logger.warning(f"Camelot Stream failed (continuing): {e}")

            # Method 3: PyMuPDF Tables (Fast, Stable)
            try:
                pymupdf_tables = self._process_with_pymupdf_tables(pdf_path)
                if pymupdf_tables["tables"]:
                    logger.info(f"PyMuPDF found {len(pymupdf_tables['tables'])} tables")
                    all_tables.extend(pymupdf_tables["tables"])
            except Exception as e:
                logger.warning(f"PyMuPDF tables failed (continuing): {e}")

            # Method 4: Tabula (Java-based, page-by-page for proper page numbers)
            if TABULA_AVAILABLE:
                try:
                    tabula_tables = self._process_with_tabula(pdf_path, basic_info["total_pages"])
                    if tabula_tables["tables"]:
                        logger.info(f"Tabula found {len(tabula_tables['tables'])} tables")
                        all_tables.extend(tabula_tables["tables"])
                except Exception as e:
                    logger.warning(f"Tabula failed (continuing): {e}")

            # Method 5: pdfplumber (Fallback)
            try:
                pdfplumber_tables = self._process_with_pdfplumber(pdf_path)
                if pdfplumber_tables["tables"]:
                    logger.info(f"pdfplumber found {len(pdfplumber_tables['tables'])} tables")
                    all_tables.extend(pdfplumber_tables["tables"])
            except Exception as e:
                logger.warning(f"pdfplumber failed (continuing): {e}")

            # Store all tables (will be deduplicated in post-processing)
            results["tables"] = all_tables

            # 4. OCR (if needed for text-sparse pages)
            if basic_info.get("needs_ocr", False):
                ocr_result = self._process_with_ocr(pdf_path)
                results["text_chunks"].extend(ocr_result["text_chunks"])

            # 5. Combine & Post-process (includes smart deduplication)
            combined_results = self._combine_extraction_results(
                {
                    "text": {"text_chunks": results["text_chunks"]},
                    "tables_all": {"tables": results["tables"]},
                },
                basic_info,
            )
            final_results = self._post_process_results(combined_results)

            processing_time = time.time() - start_time
            logger.info(
                f"PDF processing completed in {processing_time:.2f}s - "
                f"Found {len(final_results['text_chunks'])} text chunks and "
                f"{len(final_results['tables'])} unique tables"
            )
            return final_results

        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise

    def _get_basic_pdf_info_sync(self, pdf_path: str) -> Dict[str, Any]:
        """Sync version of get info"""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            needs_ocr = False
            has_images = False

            check_pages = min(3, total_pages)
            for page_num in range(check_pages):
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                if len(text) < 50:
                    needs_ocr = True
                if page.get_images():
                    has_images = True
            doc.close()
            return {
                "total_pages": total_pages,
                "needs_ocr": needs_ocr,
                "has_images": has_images,
            }
        except Exception:
            return {"total_pages": 0, "needs_ocr": False, "has_images": False}

    def _process_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text_chunks = []
            has_images = False
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_chunks.append(
                        {
                            "content": text.strip(),
                            "page_number": page_num + 1,
                            "extraction_method": "PyMuPDF",
                            "content_type": "text",
                        }
                    )
                if page.get_images():
                    has_images = True
            total_pages = len(doc)
            doc.close()
            return {
                "text_chunks": text_chunks,
                "total_pages": total_pages,
                "has_images": has_images,
            }
        except Exception as e:
            logger.error(f"PyMuPDF text failed: {e}")
            return {"text_chunks": [], "total_pages": 0, "has_images": False}

    def _process_with_camelot(self, pdf_path: str, flavor: str = "lattice") -> Dict[str, Any]:
        """
        Extract tables using Camelot (BEST quality for structured tables).
        
        Args:
            pdf_path: Path to PDF file
            flavor: 'lattice' for bordered tables, 'stream' for borderless
            
        Returns:
            Dict with 'tables' list
        """
        if not CAMELOT_AVAILABLE:
            return {"tables": []}
        
        try:
            tables = []
            method_name = f"camelot_{flavor}"
            
            # Extract tables with Camelot
            camelot_tables = camelot.read_pdf(
                pdf_path,
                pages="all",
                flavor=flavor,
                suppress_stdout=True,
            )
            
            for i, table in enumerate(camelot_tables):
                try:
                    df = table.df
                    
                    # Skip empty or single-cell tables
                    if df.empty or (len(df) == 1 and len(df.columns) == 1):
                        continue
                    
                    # Use first row as header if it looks like a header
                    if len(df) > 1:
                        # Check if first row looks like headers (no numeric values)
                        first_row = df.iloc[0]
                        if all(isinstance(v, str) or pd.isna(v) for v in first_row):
                            new_header = df.iloc[0]
                            df = df[1:]
                            df.columns = new_header
                    
                    # Get accuracy from Camelot
                    accuracy = getattr(table, 'accuracy', None)
                    if accuracy:
                        accuracy = float(accuracy)
                    
                    table_data = self._process_dataframe_table(
                        df,
                        page_number=table.page,
                        table_index=i,
                        method=method_name,
                        accuracy=accuracy,
                    )
                    if table_data:
                        tables.append(table_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to process Camelot table {i}: {e}")
                    continue
                    
            return {"tables": tables}
            
        except Exception as e:
            logger.warning(f"Camelot ({flavor}) extraction failed: {e}")
            return {"tables": []}

    def _process_with_pymupdf_tables(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables using PyMuPDF's built-in find_tables (Stable & Fast)"""
        try:
            doc = fitz.open(pdf_path)
            tables = []

            for page_num, page in enumerate(doc):
                # Find tables
                found_tables = page.find_tables()

                for i, table in enumerate(found_tables):
                    try:
                        # Extract table content as list of lists
                        table_content = table.extract()
                        # Allow single-row tables (might be headers/summaries)
                        if not table_content or len(table_content) < 1:
                            continue

                        # Convert to DataFrame
                        df = pd.DataFrame(table_content)

                        # Use first row as header if we have more than 1 row
                        if len(df) > 1:
                            new_header = df.iloc[0]
                            df = df[1:]
                            df.columns = new_header

                        # Process
                        table_data = self._process_dataframe_table(
                            df, page_num + 1, i, "pymupdf_tables"
                        )
                        if table_data:
                            tables.append(table_data)

                    except Exception as e:
                        logger.warning(
                            f"Failed to process PyMuPDF table {i} on p{page_num + 1}: {e}"
                        )

            doc.close()
            return {"tables": tables}

        except Exception as e:
            logger.error(f"PyMuPDF table extraction failed: {e}")
            return {"tables": []}

    def _process_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables using pdfplumber"""
        try:
            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        # Allow tables with at least 1 row
                        if table and len(table) >= 1:
                            try:
                                # Handle both with and without headers
                                if len(table) > 1:
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                else:
                                    df = pd.DataFrame([table[0]])
                                    
                                table_data = self._process_dataframe_table(
                                    df, page_num + 1, table_idx, "pdfplumber"
                                )
                                if table_data:
                                    tables.append(table_data)
                            except Exception:
                                continue
            return {"tables": tables}
        except Exception as e:
            logger.warning(f"pdfplumber failed (skipping): {e}")
            return {"tables": []}

    def _process_with_tabula(self, pdf_path: str, total_pages: int) -> Dict[str, Any]:
        """Extract tables using tabula with proper page tracking"""
        if not TABULA_AVAILABLE:
            return {"tables": []}
        try:
            tables = []
            
            # Process page by page to get accurate page numbers
            for page_num in range(1, total_pages + 1):
                try:
                    dfs = tabula_py.read_pdf(
                        pdf_path, 
                        pages=str(page_num), 
                        multiple_tables=True,
                        silent=True
                    )
                    
                    for i, df in enumerate(dfs):
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            table_data = self._process_dataframe_table(
                                df, page_num, i, "tabula"
                            )
                            if table_data:
                                tables.append(table_data)
                except Exception as e:
                    logger.debug(f"Tabula failed on page {page_num}: {e}")
                    continue
                    
            return {"tables": tables}
        except Exception as e:
            logger.warning(f"Tabula failed (skipping): {e}")
            return {"tables": []}

    def _process_with_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """OCR Fallback"""
        try:
            text_chunks = []
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text().strip()
                if len(text) < 50:
                    ocr_text = self._extract_text_with_ocr(page, page_num + 1)
                    if ocr_text:
                        text_chunks.append(
                            {
                                "content": ocr_text,
                                "page_number": page_num + 1,
                                "extraction_method": "OCR",
                                "content_type": "text_ocr",
                            }
                        )
            doc.close()
            return {"text_chunks": text_chunks}
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return {"text_chunks": []}

    def _extract_text_with_ocr(self, page, page_number: int) -> str:
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            # Fast Preprocess
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            processed_image = Image.fromarray(thresh)

            return pytesseract.image_to_string(
                processed_image, lang="eng", config="--psm 6"
            ).strip()
        except Exception:
            return None

    def _process_dataframe_table(
        self,
        df: pd.DataFrame,
        page_number: int,
        table_index: int,
        method: str,
        accuracy: float = None,
    ) -> Dict[str, Any]:
        """Process a DataFrame into our standard table format with quality scoring"""
        try:
            # Clean columns - handle duplicates
            if df.columns.duplicated().any():
                new_cols = []
                counts = {}
                for col in df.columns:
                    c = str(col)
                    counts[c] = counts.get(c, 0) + 1
                    new_cols.append(f"{c}_{counts[c]}" if counts[c] > 1 else c)
                df.columns = new_cols

            df = df.fillna("")
            
            # Calculate quality score
            base_quality = METHOD_QUALITY.get(method, 50)
            
            # Adjust quality based on table characteristics
            quality_score = base_quality
            
            # Bonus for accuracy if available (Camelot provides this)
            if accuracy is not None:
                quality_score = int(base_quality * (accuracy / 100))
            
            # Penalty for empty cells
            total_cells = df.size
            empty_cells = (df == "").sum().sum() + df.isna().sum().sum()
            if total_cells > 0:
                fill_ratio = 1 - (empty_cells / total_cells)
                quality_score = int(quality_score * (0.5 + 0.5 * fill_ratio))
            
            # Generate content hash for deduplication
            content_str = df.to_string()
            content_hash = hashlib.md5(content_str.encode()).hexdigest()[:12]
            
            return {
                "data": df.to_dict("records"),
                "columns": list(df.columns),
                "html": df.to_html(index=False, classes="table table-striped"),
                "page_number": page_number,
                "table_index": table_index,
                "extraction_method": method,
                "rows": len(df),
                "cols": len(df.columns),
                "context_before": "",
                "context_after": "",
                "accuracy": accuracy,
                "quality_score": quality_score,
                "content_hash": content_hash,
            }
        except Exception:
            return None

    def _combine_extraction_results(
        self, results: Dict[str, Any], basic_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        combined = {
            "text_chunks": results.get("text", {}).get("text_chunks", []),
            "tables": [],
            "metadata": {
                "total_pages": basic_info["total_pages"],
                "has_images": basic_info["has_images"],
                "has_tables": False,
                "processing_methods": [],
            },
        }

        # Merge all table lists
        for key, val in results.items():
            if "tables" in key and val.get("tables"):
                combined["tables"].extend(val["tables"])
                combined["metadata"]["processing_methods"].append(key)

        if combined["tables"]:
            combined["metadata"]["has_tables"] = True

        return combined

    def _post_process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process results with smart deduplication.
        
        Uses content hash + page number to identify duplicates,
        then keeps the highest quality version.
        """
        # Group tables by content hash + page (same table from different methods)
        table_groups = {}
        for table in results["tables"]:
            # Key: content hash + page number (same content on same page = duplicate)
            key = f"{table.get('content_hash', '')}_{table['page_number']}"
            
            if key not in table_groups:
                table_groups[key] = []
            table_groups[key].append(table)
        
        # Keep the highest quality version from each group
        unique_tables = []
        for key, group in table_groups.items():
            # Sort by quality score (descending), then by method priority
            sorted_group = sorted(
                group,
                key=lambda t: (
                    t.get("quality_score", 0),
                    METHOD_QUALITY.get(t.get("extraction_method", ""), 0)
                ),
                reverse=True
            )
            best_table = sorted_group[0]
            
            # Log if we're deduplicating
            if len(group) > 1:
                methods = [t.get("extraction_method", "?") for t in group]
                logger.debug(
                    f"Deduped table on p{best_table['page_number']}: "
                    f"kept {best_table.get('extraction_method')} from {methods}"
                )
            
            unique_tables.append(best_table)
        
        # Sort by page number, then table index
        unique_tables.sort(key=lambda t: (t["page_number"], t["table_index"]))
        results["tables"] = unique_tables

        # Add context (linking tables to text)
        for table in results["tables"]:
            pg = table["page_number"]
            for chunk in results["text_chunks"]:
                if abs(chunk["page_number"] - pg) <= 1:
                    if "table_references" not in chunk:
                        chunk["table_references"] = []
                    chunk["table_references"].append(
                        {
                            "table_id": f"table_{pg}_{table['table_index']}",
                            "page_number": pg,
                        }
                    )
        
        # Store counts before merge for logging
        total_raw_extractions = sum(len(g) for g in table_groups.values())
        unique_before_merge = len(unique_tables)
        
        # CRITICAL: Merge tables that span multiple pages
        results["tables"] = self._merge_cross_page_tables(results["tables"])
        
        # CRITICAL: Extract and attach question context (Q1, (a)(i), Case Study, etc.)
        page_context_map = self._extract_question_context(results["text_chunks"])
        if page_context_map:
            results["text_chunks"], results["tables"] = self._attach_question_context(
                results["text_chunks"], 
                results["tables"],
                page_context_map
            )
            logger.info(f"Attached question context to {len(page_context_map)} pages")
        
        # MCQs: Detect and parse options
        results["text_chunks"] = self._detect_and_parse_mcq(results["text_chunks"])

        logger.info(
            f"Post-processing: {len(results['tables'])} final tables "
            f"(from {total_raw_extractions} raw extractions → {unique_before_merge} after dedup → {len(results['tables'])} after merge)"
        )
        
        return results

    def _merge_cross_page_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge tables that span across consecutive pages.
        
        Detection criteria:
        1. Tables are on consecutive pages
        2. Column structure is similar (fuzzy match for OCR variations)
        3. Second table has no clear header row OR header matches first table
        
        Returns:
            List of merged tables
        """
        if len(tables) <= 1:
            return tables
        
        # Sort tables by page number, then table index
        sorted_tables = sorted(tables, key=lambda t: (t["page_number"], t["table_index"]))
        
        merged_tables = []
        i = 0
        
        while i < len(sorted_tables):
            current_table = sorted_tables[i]
            merged_data = list(current_table.get("data", []))
            merged_columns = list(current_table.get("columns", []))
            merged_pages = [current_table["page_number"]]
            merge_count = 0
            
            # Look ahead for continuation tables
            j = i + 1
            while j < len(sorted_tables):
                next_table = sorted_tables[j]
                
                # Check if this could be a continuation
                if self._is_table_continuation(current_table, next_table, merged_pages[-1]):
                    # Merge the data (skip header row if it matches)
                    next_data = next_table.get("data", [])
                    
                    if next_data:
                        # Check if first row of next table looks like a repeated header
                        if self._is_repeated_header(merged_columns, next_data[0]):
                            # Skip the header row
                            next_data = next_data[1:] if len(next_data) > 1 else []
                        
                        merged_data.extend(next_data)
                        merged_pages.append(next_table["page_number"])
                        merge_count += 1
                    
                    j += 1
                else:
                    break
            
            # Create merged table entry
            if merge_count > 0:
                logger.info(
                    f"Merged {merge_count + 1} table fragments across pages {merged_pages} "
                    f"into single table ({len(merged_data)} rows)"
                )
                
                # Rebuild the merged table
                merged_table = self._create_merged_table(
                    current_table, merged_data, merged_columns, merged_pages
                )
                merged_tables.append(merged_table)
            else:
                merged_tables.append(current_table)
            
            i = j
        
        return merged_tables

    def _is_table_continuation(
        self, 
        first_table: Dict[str, Any], 
        second_table: Dict[str, Any],
        last_merged_page: int
    ) -> bool:
        """
        Determine if second_table is a continuation of first_table.
        
        Criteria:
        1. On consecutive pages (or same page for different fragments)
        2. Same or very similar column structure
        3. Similar column count
        """
        page_diff = second_table["page_number"] - last_merged_page
        
        # Must be on consecutive pages (or same page)
        if page_diff > 1 or page_diff < 0:
            return False
        
        first_cols = first_table.get("columns", [])
        second_cols = second_table.get("columns", [])
        
        # Must have same number of columns (with tolerance of 1)
        if abs(len(first_cols) - len(second_cols)) > 1:
            return False
        
        # Check column similarity using fuzzy matching
        similarity = self._column_similarity(first_cols, second_cols)
        
        # Threshold: 70% similarity means likely continuation
        return similarity >= 0.70

    def _column_similarity(self, cols1: List[str], cols2: List[str]) -> float:
        """
        Calculate similarity between two column lists.
        Uses multiple strategies:
        1. SequenceMatcher for fuzzy string matching (handles OCR variations)
        2. Jaccard similarity on word tokens
        3. Column subset detection (for multi-row headers)
        """
        if not cols1 or not cols2:
            return 0.0
        
        # Normalize column names
        def normalize(col):
            return str(col).lower().strip().replace("_", " ")
        
        normalized1 = [normalize(c) for c in cols1]
        normalized2 = [normalize(c) for c in cols2]
        
        # Strategy 1: SequenceMatcher fuzzy matching
        total_similarity = 0.0
        matched_indices = set()
        
        for c1 in normalized1:
            best_match = 0.0
            best_idx = -1
            for idx, c2 in enumerate(normalized2):
                if idx in matched_indices:
                    continue
                # Use SequenceMatcher for fuzzy comparison
                ratio = SequenceMatcher(None, c1, c2).ratio()
                if ratio > best_match:
                    best_match = ratio
                    best_idx = idx
            
            if best_match >= 0.6:  # At least 60% similar
                total_similarity += best_match
                if best_idx >= 0:
                    matched_indices.add(best_idx)
        
        max_cols = max(len(normalized1), len(normalized2))
        sequence_score = total_similarity / max_cols if max_cols > 0 else 0.0
        
        # Strategy 2: Jaccard similarity on word tokens (for restructured headers)
        words1 = set()
        words2 = set()
        for c in normalized1:
            words1.update(c.split())
        for c in normalized2:
            words2.update(c.split())
        
        if words1 and words2:
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            jaccard_score = intersection / union if union > 0 else 0.0
        else:
            jaccard_score = 0.0
        
        # Strategy 3: Check if one is subset of other (multi-row header flattening)
        subset_score = 0.0
        if len(normalized1) != len(normalized2):
            smaller = set(normalized1) if len(normalized1) < len(normalized2) else set(normalized2)
            larger = set(normalized1) if len(normalized1) >= len(normalized2) else set(normalized2)
            if smaller.issubset(larger) or len(smaller & larger) >= len(smaller) * 0.8:
                subset_score = 0.85  # High score for subset match
        
        # Combined score: weighted average
        final_score = max(
            sequence_score * 0.6 + jaccard_score * 0.4,  # Combined
            sequence_score,  # Pure sequence match
            subset_score  # Subset match
        )
        
        return final_score

    def _is_repeated_header(self, original_columns: List[str], first_row: Dict) -> bool:
        """
        Check if a row looks like a repeated header (common in continued tables).
        """
        if not first_row or not original_columns:
            return False
        
        # Get values from the first row
        row_values = list(first_row.values()) if isinstance(first_row, dict) else list(first_row)
        
        # Normalize for comparison
        def normalize(val):
            return str(val).lower().strip().replace(" ", "").replace("_", "")
        
        norm_cols = [normalize(c) for c in original_columns]
        norm_vals = [normalize(v) for v in row_values]
        
        # If most values match column names, it's a repeated header
        matches = sum(1 for v in norm_vals if v in norm_cols or any(v in c or c in v for c in norm_cols if len(v) > 2))
        
        # If more than 60% match, consider it a header
        return matches / max(len(norm_vals), 1) >= 0.6

    def _create_merged_table(
        self,
        base_table: Dict[str, Any],
        merged_data: List[Dict],
        merged_columns: List[str],
        merged_pages: List[int]
    ) -> Dict[str, Any]:
        """
        Create a new merged table entry from fragments.
        """
        # Rebuild DataFrame for HTML generation
        df = pd.DataFrame(merged_data)
        if not df.empty:
            df = df.fillna("")
        
        # Recalculate content hash
        content_str = df.to_string() if not df.empty else ""
        content_hash = hashlib.md5(content_str.encode()).hexdigest()[:12]
        
        return {
            "data": merged_data,
            "columns": merged_columns,
            "html": df.to_html(index=False, classes="table table-striped") if not df.empty else "",
            "page_number": merged_pages[0],  # Start page
            "page_end": merged_pages[-1],    # End page (NEW FIELD)
            "pages_spanned": merged_pages,   # All pages (NEW FIELD)
            "table_index": base_table.get("table_index", 0),
            "extraction_method": base_table.get("extraction_method", "merged"),
            "rows": len(merged_data),
            "cols": len(merged_columns),
            "context_before": base_table.get("context_before", ""),
            "context_after": base_table.get("context_after", ""),
            "accuracy": base_table.get("accuracy"),
            "quality_score": base_table.get("quality_score", 50),
            "content_hash": content_hash,
            "is_merged": True,  # Flag for downstream processing
            "merge_info": {
                "fragment_count": len(merged_pages),
                "original_extraction": base_table.get("extraction_method", "unknown"),
            }
        }

    # =========================================================================
    # QUESTION CONTEXT EXTRACTION - For BOS/CA exam materials
    # =========================================================================
    
    def _extract_question_context(self, text_chunks: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Build a page-to-question-context mapping from text chunks.
        
        Parses BOS patterns like:
        - Q1, Question 1, Q.1
        - (a), (b), (i), (ii)
        - (a)(i), (b)(ii)
        - Case Study, Practical Problem, Illustration
        
        Returns:
            Dict mapping page_number -> {
                "question_id": "Q4",
                "subquestion": "(b)(i)",
                "full_id": "Q4 (b)(i)",
                "context_text": "Q4 (b)(i) – Compute depreciation...",
                "section_type": "question" | "case_study" | "illustration"
            }
        """
        page_context_map = {}
        current_question = None
        current_subquestion = None
        current_section_type = "question"
        current_context_text = ""
        
        for chunk in sorted(text_chunks, key=lambda c: c.get("page_number", 0)):
            page_num = chunk.get("page_number", 0)
            content = chunk.get("content", "")
            
            if not content:
                continue
            
            # Check for main question patterns: Q1, Question 1
            for pattern in QUESTION_PATTERNS:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    current_question = f"Q{match.group(1)}"
                    current_subquestion = None  # Reset subquestion when new question starts
                    current_section_type = "question"
                    # Extract title/description (first 100 chars of match)
                    desc = match.group(2).strip() if match.group(2) else ""
                    current_context_text = f"{current_question} – {desc}" if desc else current_question
                    break
            
            # Check for special sections: Case Study, Illustration
            for pattern in SPECIAL_PATTERNS:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    section_name = match.group(1)
                    section_num = match.group(2) if len(match.groups()) > 1 and match.group(2) else ""
                    desc = match.group(3).strip() if len(match.groups()) > 2 and match.group(3) else ""
                    
                    if "case study" in section_name.lower():
                        current_section_type = "case_study"
                        current_question = f"Case Study {section_num}".strip()
                    elif "illustration" in section_name.lower():
                        current_section_type = "illustration"
                        current_question = f"Illustration {section_num}".strip()
                    elif "practical problem" in section_name.lower():
                        current_section_type = "practical_problem"
                        current_question = f"Practical Problem {section_num}".strip()
                    else:
                        # Solution/Answer/Working - keep current question
                        pass
                    
                    current_subquestion = None
                    current_context_text = f"{current_question} – {desc}" if desc else current_question
                    break
            
            # Check for subquestion patterns: (a), (b)(i), (i)
            for pattern in SUBQUESTION_PATTERNS:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    if len(match.groups()) >= 2 and match.group(2) and match.group(2).startswith("("):
                        # Pattern like (a)(i)
                        current_subquestion = f"({match.group(1)})({match.group(2)})"
                    elif len(match.groups()) >= 1:
                        # Simple pattern like (a) or (i)
                        sub = match.group(1)
                        # Determine if this is a sub-sub question
                        if current_subquestion and len(sub) <= 3 and sub in "ivxlc":
                            # Roman numeral under existing letter
                            base = current_subquestion.split(")")[0] + ")" if ")" in current_subquestion else ""
                            current_subquestion = f"{base}({sub})"
                        else:
                            current_subquestion = f"({sub})"
                    
                    # Update context text with description
                    desc = match.group(2).strip() if len(match.groups()) >= 2 and match.group(2) and not match.group(2).startswith("(") else ""
                    break
            
            # Build full context for this page
            if current_question:
                full_id = current_question
                if current_subquestion:
                    full_id = f"{current_question} {current_subquestion}"
                
                page_context_map[page_num] = {
                    "question_id": current_question,
                    "subquestion": current_subquestion or "",
                    "full_id": full_id,
                    "context_text": current_context_text,
                    "section_type": current_section_type,
                }
        
        return page_context_map

    def _attach_question_context(
        self, 
        text_chunks: List[Dict[str, Any]], 
        tables: List[Dict[str, Any]],
        page_context_map: Dict[int, Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Attach question context to text chunks and tables based on page mapping.
        
        Each chunk/table gets:
        - question_id: "Q4"
        - subquestion: "(b)(i)"
        - question_context: "Q4 (b)(i) – Compute depreciation..."
        - section_type: "question" | "case_study" | "illustration"
        """
        # Attach to text chunks
        for chunk in text_chunks:
            page_num = chunk.get("page_number", 0)
            
            # Try exact page, then nearby pages
            context = (
                page_context_map.get(page_num) or 
                page_context_map.get(page_num - 1) or 
                page_context_map.get(page_num + 1)
            )
            
            if context:
                chunk["question_id"] = context["question_id"]
                chunk["subquestion"] = context["subquestion"]
                chunk["question_context"] = context["context_text"]
                chunk["section_type"] = context["section_type"]
            else:
                chunk["question_id"] = ""
                chunk["subquestion"] = ""
                chunk["question_context"] = ""
                chunk["section_type"] = "content"
        
        # Attach to tables
        for table in tables:
            page_num = table.get("page_number", 0)
            pages_spanned = table.get("pages_spanned", [page_num])
            
            # For merged tables, use the first page's context
            context = None
            for pg in pages_spanned:
                context = page_context_map.get(pg)
                if context:
                    break
            
            # Fallback: try nearby pages
            if not context:
                context = (
                    page_context_map.get(page_num - 1) or 
                    page_context_map.get(page_num + 1)
                )
            
            if context:
                table["question_id"] = context["question_id"]
                table["subquestion"] = context["subquestion"]
                table["question_context"] = context["context_text"]
                table["section_type"] = context["section_type"]
            else:
                table["question_id"] = ""
                table["subquestion"] = ""
                table["question_context"] = ""
                table["section_type"] = "content"
        
        return text_chunks, tables

    def _detect_and_parse_mcq(self, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect if chunks contain MCQs and extract options.
        
        Adds fields:
        - is_mcq: bool
        - mcq_options: Dict[str, str] e.g. {"a": "Option A", "b": "Option B"}
        """
        for chunk in text_chunks:
            content = chunk.get("content", "")
            if not content:
                continue
                
            options = {}
            
            # Check for patterns
            # Pattern 1: Inline (a)...(b)...
            matches1 = list(re.finditer(MCQ_PATTERNS[0], content, re.MULTILINE))
            if len(matches1) >= 2:  # Need at least 2 options to be an MCQ
                for m in matches1:
                    key = m.group(1).lower()
                    val = m.group(2).strip()
                    options[key] = val
            
            # Pattern 2: Stacked A. ... B. ...
            if not options:
                matches2 = list(re.finditer(MCQ_PATTERNS[1], content, re.MULTILINE))
                if len(matches2) >= 2:
                    for m in matches2:
                        key = m.group(1).lower()
                        val = m.group(2).strip()
                        options[key] = val
            
            # If we found at least 2 options (e.g. A and B)
            if len(options) >= 2:
                chunk["is_mcq"] = True
                chunk["mcq_options"] = options
                # Tweak content type if currently just 'text'
                if chunk.get("content_type") == "text":
                    chunk["content_type"] = "question_mcq"
            else:
                chunk["is_mcq"] = False
                chunk["mcq_options"] = {}
                
        return text_chunks


# Expose
PDFProcessor = OptimizedPDFProcessor
