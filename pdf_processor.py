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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import camelot.io as camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning("Camelot not available - table extraction will use pdfplumber and tabula only")
    
try:
    import tabula as tabula_py
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    logger.warning("Tabula not available - table extraction will use pdfplumber only")

class OptimizedPDFProcessor:
    def __init__(self, max_workers: int = 4):
        self.supported_formats = ['.pdf']
        self.max_workers = max_workers
        # Cache for processed pages to avoid reprocessing
        self.page_cache = {}
        
    async def extract_text_and_tables_async(self, pdf_path: str) -> Dict[str, Any]:
        """
        Optimized PDF processing with parallel extraction methods
        """
        start_time = time.time()
        
        try:
            # First, get basic info and cache pages
            basic_info = await self._get_basic_pdf_info(pdf_path)
            
            # Run extraction methods in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all extraction tasks
                tasks = []
                
                # Text extraction task
                text_task = executor.submit(self._process_with_pymupdf, pdf_path)
                tasks.append(('text', text_task))
                
                # Table extraction tasks (run in parallel)
                # NOTE: pdfplumber disabled due to libpdfium.so segfaults on Linux
                # Camelot + Tabula provide sufficient table extraction
                # table_task = executor.submit(self._process_with_pdfplumber, pdf_path)
                # tasks.append(('tables_pdfplumber', table_task))
                
                if CAMELOT_AVAILABLE:
                    camelot_task = executor.submit(self._process_with_camelot, pdf_path)
                    tasks.append(('tables_camelot', camelot_task))
                
                if TABULA_AVAILABLE:
                    tabula_task = executor.submit(self._process_with_tabula, pdf_path)
                    tasks.append(('tables_tabula', tabula_task))
                
                # OCR task (only if needed - check if pages have little text)
                if basic_info.get('needs_ocr', False):
                    ocr_task = executor.submit(self._process_with_ocr, pdf_path)
                    tasks.append(('ocr', ocr_task))
                
                # Wait for all tasks to complete
                results = {}
                for task_name, task in tasks:
                    try:
                        results[task_name] = task.result(timeout=60)  # 60 second timeout per task
                    except Exception as e:
                        logger.warning(f"Task {task_name} failed: {e}")
                        results[task_name] = {'text_chunks': [], 'tables': []}
            
            # Combine results
            combined_results = self._combine_extraction_results(results, basic_info)
            
            # Post-process (deduplicate, add context)
            final_results = self._post_process_results(combined_results)
            
            processing_time = time.time() - start_time
            logger.info(f"PDF processing completed in {processing_time:.2f}s - Found {len(final_results['text_chunks'])} text chunks and {len(final_results['tables'])} tables")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Optimized PDF processing failed: {e}")
            raise
    
    def extract_text_and_tables(self, pdf_path: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for async processing
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.extract_text_and_tables_async(pdf_path))
        finally:
            loop.close()
    
    async def _get_basic_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """Quick scan to get basic PDF info and determine if OCR is needed"""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            needs_ocr = False
            has_images = False
            
            # Quick check first few pages to determine if OCR is needed
            check_pages = min(3, total_pages)
            for page_num in range(check_pages):
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                
                if len(text) < 50:  # Very little text, might need OCR
                    needs_ocr = True
                
                # Check for images
                image_list = page.get_images()
                if image_list:
                    has_images = True
            
            doc.close()
            
            return {
                'total_pages': total_pages,
                'needs_ocr': needs_ocr,
                'has_images': has_images
            }
            
        except Exception as e:
            logger.error(f"Failed to get basic PDF info: {e}")
            return {'total_pages': 0, 'needs_ocr': False, 'has_images': False}
    
    def _process_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF - optimized version"""
        try:
            doc = fitz.open(pdf_path)
            text_chunks = []
            has_images = False
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    text_chunks.append({
                        'content': text.strip(),
                        'page_number': page_num + 1,
                        'extraction_method': 'PyMuPDF',
                        'content_type': 'text'
                    })
                
                # Check for images
                image_list = page.get_images()
                if image_list:
                    has_images = True
            
            total_pages = len(doc)
            doc.close()
            
            return {
                'text_chunks': text_chunks,
                'total_pages': total_pages,
                'has_images': has_images
            }
            
        except Exception as e:
            logger.error(f"PyMuPDF processing failed: {e}")
            return {'text_chunks': [], 'total_pages': 0, 'has_images': False}
    
    def _process_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables using pdfplumber - optimized version"""
        try:
            tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract tables
                    page_tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1:
                            try:
                                # Optimized table processing
                                table_data = self._process_table_data(table, page_num + 1, table_idx, 'pdfplumber')
                                if table_data:
                                    tables.append(table_data)
                            except Exception as e:
                                logger.warning(f"Failed to process table {table_idx} on page {page_num + 1}: {e}")
                                continue
            
            return {'tables': tables}
            
        except Exception as e:
            logger.error(f"pdfplumber processing failed: {e}")
            return {'tables': []}
    
    def _process_with_camelot(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables using camelot - optimized version"""
        if not CAMELOT_AVAILABLE:
            return {'tables': []}
        try:
            tables = []
            
            # Use both methods but with timeout
            for method in ['lattice', 'stream']:
                try:
                    camelot_tables = camelot.read_pdf(pdf_path, flavor=method, pages='all')
                    
                    for table_idx, table in enumerate(camelot_tables):
                        if table.accuracy > 50:  # Only use high accuracy tables
                            try:
                                df = table.df
                                df = df.dropna(how='all').dropna(axis=1, how='all')
                                
                                if isinstance(df, pd.DataFrame) and not df.empty and len(df) > 1:
                                    table_data = self._process_dataframe_table(df, table.page, table_idx, f'camelot-{method}', table.accuracy)
                                    if table_data:
                                        tables.append(table_data)
                            except Exception as e:
                                logger.warning(f"Failed to process camelot table {table_idx}: {e}")
                                continue
                                
                except Exception as method_error:
                    logger.warning(f"Camelot {method} method failed: {method_error}")
                    continue
            
            return {'tables': tables}
            
        except Exception as e:
            logger.error(f"Camelot processing failed: {e}")
            return {'tables': []}
    
    def _process_with_tabula(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables using tabula - optimized version"""
        if not TABULA_AVAILABLE:
            return {'tables': []}
        try:
            tables = []
            
            tabula_tables = tabula_py.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            for table_idx, df in enumerate(tabula_tables):
                if isinstance(df, pd.DataFrame) and not df.empty and len(df) > 1:
                    try:
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        
                        if not df.empty:
                            table_data = self._process_dataframe_table(df, 0, table_idx, 'tabula')
                            if table_data:
                                tables.append(table_data)
                    except Exception as e:
                        logger.warning(f"Failed to process tabula table {table_idx}: {e}")
                        continue
            
            return {'tables': tables}
            
        except Exception as e:
            logger.error(f"Tabula processing failed: {e}")
            return {'tables': []}
    
    def _process_with_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """Process scanned pages with OCR - optimized version"""
        try:
            text_chunks = []
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Check if page needs OCR (has little extractable text)
                text = page.get_text().strip()
                if len(text) < 50:  # Threshold for OCR
                    try:
                        # Optimized OCR processing
                        ocr_text = self._extract_text_with_ocr(page, page_num + 1)
                        if ocr_text:
                            text_chunks.append({
                                'content': ocr_text,
                                'page_number': page_num + 1,
                                'extraction_method': 'OCR',
                                'content_type': 'text_ocr'
                            })
                    except Exception as ocr_error:
                        logger.warning(f"OCR failed for page {page_num + 1}: {ocr_error}")
                        continue
            
            doc.close()
            return {'text_chunks': text_chunks}
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return {'text_chunks': []}
    
    def _extract_text_with_ocr(self, page, page_number: int) -> str:
        """Optimized OCR extraction for a single page"""
        try:
            # Convert page to image with optimized settings
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # Reduced resolution for speed
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Quick preprocessing
            image = self._preprocess_image_for_ocr_fast(image)
            
            # Perform OCR with optimized settings
            ocr_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
            
            return ocr_text.strip() if ocr_text.strip() else None
            
        except Exception as e:
            logger.warning(f"OCR extraction failed for page {page_number}: {e}")
            return None
    
    def _preprocess_image_for_ocr_fast(self, image: Image.Image) -> Image.Image:
        """Fast image preprocessing for OCR"""
        try:
            # Convert PIL Image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply simple threshold (faster than OTSU)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(thresh)
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Fast image preprocessing failed: {e}")
            return image
    
    def _process_table_data(self, table: List, page_number: int, table_index: int, method: str) -> Dict[str, Any]:
        """Process table data from pdfplumber"""
        try:
            if not table or len(table) <= 1:
                return None
            
            # Clean column names and data - handle duplicates
            raw_columns = [str(col) if col is not None else f'col_{i}' for i, col in enumerate(table[0])]
            columns = []
            col_counts = {}
            for col in raw_columns:
                if col in col_counts:
                    col_counts[col] += 1
                    columns.append(f"{col}_{col_counts[col]}")
                else:
                    col_counts[col] = 0
                    columns.append(col)
            
            df = pd.DataFrame(table[1:], columns=columns)
            df = df.fillna('')
            
            return {
                'data': df.to_dict('records'),
                'columns': list(df.columns),
                'html': df.to_html(index=False, classes='table table-striped'),
                'page_number': page_number,
                'table_index': table_index,
                'extraction_method': method,
                'rows': len(df),
                'cols': len(df.columns),
                'context_before': '',
                'context_after': ''
            }
            
        except Exception as e:
            logger.warning(f"Failed to process table data: {e}")
            return None
    
    def _process_dataframe_table(self, df: pd.DataFrame, page_number: int, table_index: int, method: str, accuracy: float = None) -> Dict[str, Any]:
        """Process table data from DataFrame (camelot/tabula)"""
        try:
            # Fix duplicate column names
            if df.columns.duplicated().any():
                new_columns = []
                col_counts = {}
                for col in df.columns:
                    col_str = str(col)
                    if col_str in col_counts:
                        col_counts[col_str] += 1
                        new_columns.append(f"{col_str}_{col_counts[col_str]}")
                    else:
                        col_counts[col_str] = 0
                        new_columns.append(col_str)
                df.columns = new_columns
            
            # Replace NaN values with empty strings
            df = df.fillna('')
            
            table_data = {
                'data': df.to_dict('records'),
                'columns': list(df.columns),
                'html': df.to_html(index=False, classes='table table-striped'),
                'page_number': page_number,
                'table_index': table_index,
                'extraction_method': method,
                'rows': len(df),
                'cols': len(df.columns),
                'context_before': '',
                'context_after': ''
            }
            
            if accuracy is not None:
                table_data['accuracy'] = accuracy
            
            return table_data
            
        except Exception as e:
            logger.warning(f"Failed to process DataFrame table: {e}")
            return None
    
    def _combine_extraction_results(self, results: Dict[str, Any], basic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from different extraction methods"""
        combined = {
            'text_chunks': [],
            'tables': [],
            'metadata': {
                'total_pages': basic_info['total_pages'],
                'has_images': basic_info['has_images'],
                'has_tables': False,
                'processing_methods': []
            }
        }
        
        # Combine text chunks
        for result_key in ['text', 'ocr']:
            if result_key in results and 'text_chunks' in results[result_key]:
                combined['text_chunks'].extend(results[result_key]['text_chunks'])
                if results[result_key]['text_chunks']:
                    combined['metadata']['processing_methods'].append(result_key.title())
        
        # Combine tables
        table_methods = []
        for result_key in ['tables_pdfplumber', 'tables_camelot', 'tables_tabula']:
            if result_key in results and 'tables' in results[result_key]:
                combined['tables'].extend(results[result_key]['tables'])
                if results[result_key]['tables']:
                    method_name = result_key.replace('tables_', '')
                    table_methods.append(method_name)
        
        if combined['tables']:
            combined['metadata']['has_tables'] = True
            combined['metadata']['processing_methods'].extend(table_methods)
        
        return combined
    
    def _post_process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process results: deduplicate and add context"""
        try:
            # Deduplicate tables
            results['tables'] = self._deduplicate_tables(results['tables'])
            
            # Add table context to text chunks
            results = self._add_table_context(results)
            
            return results
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return results
    
    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables from different extraction methods - optimized"""
        if not tables:
            return []
        
        unique_tables = []
        seen_signatures = set()
        
        for table in tables:
            # Create a fast signature based on table dimensions and first few cell values
            signature = f"{table['rows']}x{table['cols']}"
            
            if 'data' in table and table['data'] and len(table['data']) > 0:
                # Add first few cell values to signature (limited for speed)
                first_row = table['data'][0] if table['data'] else {}
                values = [str(v) for v in first_row.values() if v is not None][:3]  # Limit to first 3 values
                if values:
                    signature += str(sorted(values))
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_tables.append(table)
        
        logger.info(f"Deduplicated {len(tables)} tables to {len(unique_tables)} unique tables")
        return unique_tables
    
    def _add_table_context(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add table references to relevant text chunks - optimized"""
        try:
            # For each table, try to associate it with nearby text chunks
            for table in results['tables']:
                table_page = table['page_number']
                
                # Find text chunks from the same page or adjacent pages
                relevant_chunks = [
                    chunk for chunk in results['text_chunks']
                    if abs(chunk['page_number'] - table_page) <= 1
                ]
                
                # Add table reference to metadata of relevant chunks
                for chunk in relevant_chunks:
                    if 'table_references' not in chunk:
                        chunk['table_references'] = []
                    
                    chunk['table_references'].append({
                        'table_id': f"table_{table_page}_{table['table_index']}",
                        'page_number': table_page,
                        'extraction_method': table['extraction_method'],
                        'rows': table['rows'],
                        'cols': table['cols']
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to add table context: {e}")
            return results

# Alias for backward compatibility
PDFProcessor = OptimizedPDFProcessor



