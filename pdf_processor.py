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

class PDFProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_and_tables(self, pdf_path: str) -> Dict[str, Any]:
        """
        Comprehensive PDF processing to extract text, tables, and metadata
        Returns structured data with text chunks and table data
        """
        try:
            results = {
                'text_chunks': [],
                'tables': [],
                'metadata': {
                    'total_pages': 0,
                    'has_images': False,
                    'has_tables': False,
                    'processing_methods': []
                }
            }
            
            # Process with PyMuPDF for text and basic structure
            pymupdf_results = self._process_with_pymupdf(pdf_path)
            results['text_chunks'].extend(pymupdf_results['text_chunks'])
            results['metadata']['total_pages'] = pymupdf_results['total_pages']
            results['metadata']['has_images'] = pymupdf_results['has_images']
            results['metadata']['processing_methods'].append('PyMuPDF')
            
            # Process with pdfplumber for enhanced table detection
            pdfplumber_results = self._process_with_pdfplumber(pdf_path)
            results['tables'].extend(pdfplumber_results['tables'])
            if pdfplumber_results['tables']:
                results['metadata']['has_tables'] = True
                results['metadata']['processing_methods'].append('pdfplumber')
            
            # Process with camelot for robust table extraction (if available)
            if CAMELOT_AVAILABLE:
                camelot_results = self._process_with_camelot(pdf_path)
                results['tables'].extend(camelot_results['tables'])
                if camelot_results['tables']:
                    results['metadata']['has_tables'] = True
                    results['metadata']['processing_methods'].append('camelot')
            
            # Process with tabula for additional table extraction (if available)
            if TABULA_AVAILABLE:
                tabula_results = self._process_with_tabula(pdf_path)
                results['tables'].extend(tabula_results['tables'])
                if tabula_results['tables']:
                    results['metadata']['has_tables'] = True
                    results['metadata']['processing_methods'].append('tabula')
            
            # Process scanned pages with OCR if needed
            ocr_results = self._process_with_ocr(pdf_path)
            if ocr_results['text_chunks']:
                results['text_chunks'].extend(ocr_results['text_chunks'])
                results['metadata']['processing_methods'].append('OCR')
            
            # Deduplicate and merge tables
            results['tables'] = self._deduplicate_tables(results['tables'])
            
            # Add table context to text chunks
            results = self._add_table_context(results)
            
            logger.info(f"PDF processing completed. Found {len(results['text_chunks'])} text chunks and {len(results['tables'])} tables")
            return results
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise
    
    def _process_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and basic information using PyMuPDF"""
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
        """Extract tables using pdfplumber"""
        try:
            tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract tables
                    page_tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Ensure table has content
                            # Convert to DataFrame
                            # Clean column names and data - handle duplicates
                            raw_columns = [str(col) if col is not None else f'col_{i}' for i, col in enumerate(table[0])]
                            # Make column names unique
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
                            
                            # Get context around table
                            context_before, context_after = self._get_table_context(
                                page, table, page_num + 1
                            )
                            
                            # Replace NaN values with empty strings for JSON compatibility
                            df = df.fillna('')
                            
                            table_data = {
                                'data': df.to_dict('records'),
                                'columns': list(df.columns),
                                'html': df.to_html(index=False, classes='table table-striped'),
                                'page_number': page_num + 1,
                                'table_index': table_idx,
                                'context_before': context_before,
                                'context_after': context_after,
                                'extraction_method': 'pdfplumber',
                                'rows': len(df),
                                'cols': len(df.columns)
                            }
                            
                            tables.append(table_data)
            
            return {'tables': tables}
            
        except Exception as e:
            logger.error(f"pdfplumber processing failed: {e}")
            return {'tables': []}
    
    def _process_with_camelot(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables using camelot"""
        if not CAMELOT_AVAILABLE:
            return {'tables': []}
        try:
            tables = []
            
            # Use both lattice and stream methods
            for method in ['lattice', 'stream']:
                try:
                    camelot_tables = camelot.read_pdf(pdf_path, flavor=method, pages='all')
                    
                    for table_idx, table in enumerate(camelot_tables):
                        if table.accuracy > 50:  # Only use tables with good accuracy
                            df = table.df
                            
                            # Clean the dataframe
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            if isinstance(df, pd.DataFrame) and not df.empty and len(df) > 1:
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
                                
                                # Replace NaN values with empty strings for JSON compatibility
                                df = df.fillna('')
                                
                                table_data = {
                                    'data': df.to_dict('records'),
                                    'columns': list(df.columns),
                                    'html': df.to_html(index=False, classes='table table-striped'),
                                    'page_number': table.page,
                                    'table_index': table_idx,
                                    'extraction_method': f'camelot-{method}',
                                    'accuracy': table.accuracy,
                                    'rows': len(df),
                                    'cols': len(df.columns),
                                    'context_before': '',
                                    'context_after': ''
                                }
                                
                                tables.append(table_data)
                                
                except Exception as method_error:
                    logger.warning(f"Camelot {method} method failed: {method_error}")
                    continue
            
            return {'tables': tables}
            
        except Exception as e:
            logger.error(f"Camelot processing failed: {e}")
            return {'tables': []}
    
    def _process_with_tabula(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables using tabula"""
        if not TABULA_AVAILABLE:
            return {'tables': []}
        try:
            tables = []
            
            # Read all tables from PDF
            tabula_tables = tabula_py.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            for table_idx, df in enumerate(tabula_tables):
                if isinstance(df, pd.DataFrame) and not df.empty and len(df) > 1:
                    # Clean the dataframe
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    
                    if not df.empty:
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
                        
                        # Replace NaN values with empty strings for JSON compatibility
                        df = df.fillna('')
                        
                        table_data = {
                            'data': df.to_dict('records'),
                            'columns': list(df.columns),
                            'html': df.to_html(index=False, classes='table table-striped'),
                            'page_number': 0,  # tabula doesn't provide page info easily
                            'table_index': table_idx,
                            'extraction_method': 'tabula',
                            'rows': len(df),
                            'cols': len(df.columns),
                            'context_before': '',
                            'context_after': ''
                        }
                        
                        tables.append(table_data)
            
            return {'tables': tables}
            
        except Exception as e:
            logger.error(f"Tabula processing failed: {e}")
            return {'tables': []}
    
    def _process_with_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """Process scanned pages with OCR"""
        try:
            text_chunks = []
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Check if page has extractable text
                text = page.get_text().strip()
                
                # If no text or very little text, try OCR
                if len(text) < 50:
                    try:
                        # Convert page to image
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                        img_data = pix.tobytes("png")
                        
                        # Convert to PIL Image
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Preprocess image for better OCR
                        image = self._preprocess_image_for_ocr(image)
                        
                        # Perform OCR
                        ocr_text = pytesseract.image_to_string(image, lang='eng')
                        
                        if ocr_text.strip():
                            text_chunks.append({
                                'content': ocr_text.strip(),
                                'page_number': page_num + 1,
                                'extraction_method': 'OCR',
                                'content_type': 'text_ocr'
                            })
                        
                        # Try to extract tables from OCR text
                        ocr_tables = self._extract_tables_from_ocr(image, page_num + 1)
                        if ocr_tables:
                            text_chunks.extend(ocr_tables)
                            
                    except Exception as ocr_error:
                        logger.warning(f"OCR failed for page {page_num + 1}: {ocr_error}")
                        continue
            
            doc.close()
            return {'text_chunks': text_chunks}
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return {'text_chunks': []}
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            # Convert PIL Image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)
            
            # Apply threshold to get black and white image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(thresh)
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def _extract_tables_from_ocr(self, image: Image.Image, page_number: int) -> List[Dict]:
        """Extract table structure from OCR using image processing"""
        try:
            # This is a simplified table detection - in production, you might want
            # to use more sophisticated methods like detecting table lines
            
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Group text by lines (simplified table detection)
            lines = {}
            for i, text in enumerate(ocr_data['text']):
                if text.strip():
                    # Ensure coordinates are numeric
                    try:
                        y_coord = float(ocr_data['top'][i]) if ocr_data['top'][i] != '' else 0.0
                        x_coord = float(ocr_data['left'][i]) if ocr_data['left'][i] != '' else 0.0
                        confidence = float(ocr_data['conf'][i]) if ocr_data['conf'][i] != '' else 0.0
                    except (ValueError, TypeError):
                        continue  # Skip invalid coordinate data
                        
                    line_key = int(y_coord // 20)  # Group by approximate line
                    
                    if line_key not in lines:
                        lines[line_key] = []
                    
                    lines[line_key].append({
                        'text': text.strip(),
                        'x': x_coord,
                        'confidence': confidence
                    })
            
            # If we have multiple lines with similar structure, it might be a table
            if len(lines) > 2:
                # Sort lines by y-coordinate (line_key is guaranteed to be int)
                sorted_lines = sorted(lines.items(), key=lambda x: x[0])
                
                # Check if lines have similar number of elements (table-like structure)
                line_lengths = [len(line[1]) for line in sorted_lines]
                if line_lengths:  # Avoid division by zero
                    avg_length = sum(line_lengths) / len(line_lengths)
                    
                    # If most lines have similar length, treat as table
                    similar_length_lines = sum(1 for length in line_lengths if abs(length - avg_length) <= 2)
                    
                    if similar_length_lines / len(line_lengths) > 0.6:  # 60% of lines have similar length
                        table_rows = []
                        for _, line_data in sorted_lines:
                            # Sort by x-coordinate (x is guaranteed to be float)
                            sorted_cells = sorted(line_data, key=lambda cell: cell['x'])
                            row = [cell['text'] for cell in sorted_cells if cell['text'].strip()]
                            if row:  # Only add non-empty rows
                                table_rows.append(row)
                        
                        if len(table_rows) > 1:
                            return [{
                                'content': f"Table extracted from OCR:\n{pd.DataFrame(table_rows[1:], columns=table_rows[0]).to_string()}",
                                'page_number': page_number,
                                'extraction_method': 'OCR_table',
                                'content_type': 'table_ocr'
                            }]
            
            return []
            
        except Exception as e:
            logger.warning(f"OCR table extraction failed: {e}")
            return []
    
    def _get_table_context(self, page, table, page_number: int) -> Tuple[str, str]:
        """Get text context around a table"""
        try:
            # Get all text from the page
            page_text = page.extract_text()
            
            # This is a simplified context extraction
            # In a more sophisticated implementation, you would use the table's
            # position to get text immediately before and after
            
            lines = page_text.split('\n')
            context_before = ""
            context_after = ""
            
            # Get first few non-empty lines as context before
            for line in lines[:10]:
                if line.strip():
                    context_before += line.strip() + " "
                    if len(context_before) > 200:
                        break
            
            # Get last few non-empty lines as context after
            for line in lines[-10:]:
                if line.strip():
                    context_after += line.strip() + " "
                    if len(context_after) > 200:
                        break
            
            return context_before.strip(), context_after.strip()
            
        except Exception as e:
            logger.warning(f"Failed to get table context: {e}")
            return "", ""
    
    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables from different extraction methods"""
        if not tables:
            return []
        
        unique_tables = []
        seen_signatures = set()
        
        for table in tables:
            # Create a signature based on table dimensions and first few cells
            signature = f"{table['rows']}x{table['cols']}"
            
            if 'data' in table and table['data']:
                # Add first few cell values to signature
                first_row = table['data'][0] if table['data'] else {}
                # Filter out None values and convert all to strings for comparison
                values = [str(v) for v in first_row.values() if v is not None]
                if values:
                    signature += str(sorted(values)[:3])
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_tables.append(table)
        
        logger.info(f"Deduplicated {len(tables)} tables to {len(unique_tables)} unique tables")
        return unique_tables
    
    def _add_table_context(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add table references to relevant text chunks"""
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
