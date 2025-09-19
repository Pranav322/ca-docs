import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableProcessor:
    def __init__(self):
        self.financial_keywords = [
            'amount', 'balance', 'debit', 'credit', 'total', 'subtotal',
            'assets', 'liabilities', 'equity', 'revenue', 'expense',
            'profit', 'loss', 'cash', 'investment', 'depreciation',
            'interest', 'dividend', 'tax', 'provision', 'reserve'
        ]
        
        self.table_types = {
            'balance_sheet': ['assets', 'liabilities', 'equity', 'balance'],
            'income_statement': ['revenue', 'expense', 'profit', 'loss', 'income'],
            'cash_flow': ['cash', 'operating', 'investing', 'financing', 'flow'],
            'ratio_analysis': ['ratio', 'current', 'quick', 'debt', 'return'],
            'schedule': ['schedule', 'breakdown', 'details', 'analysis'],
            'general': ['table', 'data', 'information']
        }
    
    def process_table_for_embedding(self, table_data: Dict[str, Any]) -> str:
        """
        Convert table data to text format optimized for embeddings
        Preserves numerical relationships and structure
        """
        try:
            # Extract table information
            data = table_data.get('data', [])
            columns = table_data.get('columns', [])
            context_before = table_data.get('context_before', '')
            context_after = table_data.get('context_after', '')
            
            if not data or not columns:
                return f"Table from page {table_data.get('page_number', 'unknown')}: No data available"
            
            # Create DataFrame for processing
            df = pd.DataFrame(data)
            
            # Identify table type
            table_type = self._identify_table_type(table_data)
            
            # Build text representation
            text_parts = []
            
            # Add context
            if context_before:
                text_parts.append(f"Context before table: {context_before}")
            
            # Add table metadata
            text_parts.append(f"Table Type: {table_type}")
            text_parts.append(f"Table from page {table_data.get('page_number', 'unknown')}")
            text_parts.append(f"Dimensions: {len(data)} rows × {len(columns)} columns")
            
            # Add column headers with emphasis
            text_parts.append(f"Column Headers: {', '.join(columns)}")
            
            # Process table content
            processed_content = self._process_table_content(df, table_type)
            text_parts.append(processed_content)
            
            # Add numerical summaries if applicable
            numerical_summary = self._create_numerical_summary(df)
            if numerical_summary:
                text_parts.append(f"Numerical Summary: {numerical_summary}")
            
            # Add financial insights if it's a financial table
            if table_type in ['balance_sheet', 'income_statement', 'cash_flow', 'ratio_analysis']:
                financial_insights = self._extract_financial_insights(df, table_type)
                if financial_insights:
                    text_parts.append(f"Financial Insights: {financial_insights}")
            
            # Add context after
            if context_after:
                text_parts.append(f"Context after table: {context_after}")
            
            return ' | '.join(text_parts)
            
        except Exception as e:
            logger.error(f"Failed to process table for embedding: {e}")
            return f"Table processing error from page {table_data.get('page_number', 'unknown')}"
    
    def _identify_table_type(self, table_data: Dict[str, Any]) -> str:
        """Identify the type of table based on content and context"""
        try:
            # Combine all text sources for analysis
            text_to_analyze = ""
            
            if 'columns' in table_data:
                text_to_analyze += " ".join(table_data['columns'])
            
            if 'context_before' in table_data:
                text_to_analyze += " " + table_data.get('context_before', '')
            
            if 'context_after' in table_data:
                text_to_analyze += " " + table_data.get('context_after', '')
            
            if 'data' in table_data and table_data['data']:
                # Add first row content
                first_row = table_data['data'][0]
                if isinstance(first_row, dict):
                    text_to_analyze += " " + " ".join(str(v) for v in first_row.values())
            
            text_to_analyze = text_to_analyze.lower()
            
            # Score each table type
            type_scores = {}
            for table_type, keywords in self.table_types.items():
                score = sum(1 for keyword in keywords if keyword in text_to_analyze)
                type_scores[table_type] = score
            
            # Return the type with highest score
            if type_scores:
                best_type = max(type_scores, key=type_scores.get)
                if type_scores[best_type] > 0:
                    return best_type
            
            return 'general'
            
        except Exception as e:
            logger.warning(f"Failed to identify table type: {e}")
            return 'general'
    
    def _process_table_content(self, df: pd.DataFrame, table_type: str) -> str:
        """Process table content based on its type"""
        try:
            content_parts = []
            
            # Add row-by-row content with enhanced formatting for financial tables
            for idx, row in df.iterrows():
                row_text = f"Row {idx + 1}:"
                
                for col_name, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        # Special formatting for financial values
                        if self._is_financial_value(str(value)):
                            formatted_value = self._format_financial_value(str(value))
                            row_text += f" {col_name}: {formatted_value},"
                        else:
                            row_text += f" {col_name}: {str(value).strip()},"
                
                content_parts.append(row_text.rstrip(','))
            
            # Add totals and key figures if identifiable
            totals_info = self._identify_totals_and_keys(df)
            if totals_info:
                content_parts.append(f"Key Figures: {totals_info}")
            
            return ' | '.join(content_parts)
            
        except Exception as e:
            logger.warning(f"Failed to process table content: {e}")
            return str(df.to_string())
    
    def _is_financial_value(self, value: str) -> bool:
        """Check if a value appears to be a financial amount"""
        # Remove common formatting
        cleaned = re.sub(r'[₹$,\s]', '', value)
        
        # Check if it's a number (possibly with decimal)
        try:
            float(cleaned)
            return True
        except ValueError:
            pass
        
        # Check for percentage
        if '%' in value:
            return True
        
        # Check for negative values in brackets
        if re.match(r'\(.*\)', value.strip()):
            return True
        
        return False
    
    def _format_financial_value(self, value: str) -> str:
        """Format financial values for better understanding"""
        # Preserve original formatting but add context
        if '₹' in value or 'Rs' in value.upper():
            return f"{value} (Indian Rupees)"
        elif '$' in value:
            return f"{value} (Dollars)"
        elif '%' in value:
            return f"{value} (Percentage)"
        elif re.match(r'\(.*\)', value.strip()):
            return f"{value} (Negative amount in brackets)"
        
        return value
    
    def _create_numerical_summary(self, df: pd.DataFrame) -> str:
        """Create a summary of numerical data in the table"""
        try:
            numerical_cols = []
            
            for col in df.columns:
                # Try to convert column to numeric
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                non_null_count = numeric_data.count()
                
                if non_null_count > 0:  # Column has some numeric data
                    total = numeric_data.sum()
                    avg = numeric_data.mean()
                    
                    numerical_cols.append(f"{col}: Total={total:.2f}, Average={avg:.2f}")
            
            return ', '.join(numerical_cols) if numerical_cols else ""
            
        except Exception as e:
            logger.warning(f"Failed to create numerical summary: {e}")
            return ""
    
    def _identify_totals_and_keys(self, df: pd.DataFrame) -> str:
        """Identify total rows and key figures"""
        try:
            key_info = []
            
            # Look for rows that might contain totals
            for idx, row in df.iterrows():
                row_text = ' '.join(str(v).lower() for v in row.values if pd.notna(v))
                
                if any(keyword in row_text for keyword in ['total', 'subtotal', 'grand total', 'net']):
                    key_info.append(f"Total row {idx + 1}: {dict(row)}")
            
            # Look for the last row (often contains totals)
            if len(df) > 1:
                last_row = df.iloc[-1]
                if any(self._is_financial_value(str(v)) for v in last_row.values if pd.notna(v)):
                    key_info.append(f"Last row (possibly total): {dict(last_row)}")
            
            return ' | '.join(key_info) if key_info else ""
            
        except Exception as e:
            logger.warning(f"Failed to identify totals and keys: {e}")
            return ""
    
    def _extract_financial_insights(self, df: pd.DataFrame, table_type: str) -> str:
        """Extract financial insights based on table type"""
        try:
            insights = []
            
            if table_type == 'balance_sheet':
                # Look for basic balance sheet structure
                for col in df.columns:
                    col_lower = col.lower()
                    if 'asset' in col_lower:
                        insights.append(f"Assets column: {col}")
                    elif 'liabilit' in col_lower:
                        insights.append(f"Liabilities column: {col}")
                    elif 'equit' in col_lower:
                        insights.append(f"Equity column: {col}")
            
            elif table_type == 'income_statement':
                # Look for P&L structure
                for col in df.columns:
                    col_lower = col.lower()
                    if any(word in col_lower for word in ['revenue', 'income', 'sales']):
                        insights.append(f"Revenue/Income column: {col}")
                    elif any(word in col_lower for word in ['expense', 'cost', 'expenditure']):
                        insights.append(f"Expense column: {col}")
            
            elif table_type == 'cash_flow':
                # Look for cash flow components
                for col in df.columns:
                    col_lower = col.lower()
                    if 'operating' in col_lower:
                        insights.append(f"Operating activities: {col}")
                    elif 'investing' in col_lower:
                        insights.append(f"Investing activities: {col}")
                    elif 'financing' in col_lower:
                        insights.append(f"Financing activities: {col}")
            
            return ', '.join(insights) if insights else ""
            
        except Exception as e:
            logger.warning(f"Failed to extract financial insights: {e}")
            return ""
    
    def create_table_embedding_text(self, table_data: Dict[str, Any], 
                                  metadata: Dict[str, str]) -> str:
        """
        Create comprehensive text for table embedding that includes metadata context
        """
        try:
            # Start with metadata context
            metadata_parts = []
            
            if metadata.get('level'):
                metadata_parts.append(f"CA Level: {metadata['level']}")
            if metadata.get('paper'):
                metadata_parts.append(f"Paper: {metadata['paper']}")
            if metadata.get('module'):
                metadata_parts.append(f"Module: {metadata['module']}")
            if metadata.get('chapter'):
                metadata_parts.append(f"Chapter: {metadata['chapter']}")
            if metadata.get('unit'):
                metadata_parts.append(f"Unit: {metadata['unit']}")
            
            metadata_context = ' | '.join(metadata_parts)
            
            # Get table content
            table_content = self.process_table_for_embedding(table_data)
            
            # Combine metadata and content
            full_text = f"Metadata: {metadata_context} | Table Content: {table_content}"
            
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to create table embedding text: {e}")
            return f"Table from {metadata.get('level', 'unknown')} level"
    
    def chunk_table_aware_text(self, text: str, tables_info: List[Dict], 
                              chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Chunk text while preserving table context and references
        """
        try:
            chunks = []
            
            # Split text into paragraphs first
            paragraphs = text.split('\n\n')
            
            current_chunk = ""
            current_chunk_tables = []
            chunk_index = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Check if adding this paragraph exceeds chunk size
                if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        'content': current_chunk.strip(),
                        'chunk_index': chunk_index,
                        'table_references': current_chunk_tables.copy(),
                        'has_tables': bool(current_chunk_tables)
                    })
                    
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    if overlap > 0 and len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + "\n\n" + para
                    else:
                        current_chunk = para
                    current_chunk_tables = []
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                
                # Check if this paragraph references any tables
                for table_info in tables_info:
                    if self._paragraph_references_table(para, table_info):
                        if table_info not in current_chunk_tables:
                            current_chunk_tables.append(table_info)
            
            # Add final chunk if any content remains
            if current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'table_references': current_chunk_tables,
                    'has_tables': bool(current_chunk_tables)
                })
            
            logger.info(f"Created {len(chunks)} table-aware text chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to create table-aware chunks: {e}")
            return [{'content': text, 'chunk_index': 0, 'table_references': [], 'has_tables': False}]
    
    def _paragraph_references_table(self, paragraph: str, table_info: Dict) -> bool:
        """Check if a paragraph might reference a specific table"""
        try:
            para_lower = paragraph.lower()
            
            # Check for explicit table references
            table_keywords = ['table', 'schedule', 'statement', 'above', 'below', 'following']
            
            if any(keyword in para_lower for keyword in table_keywords):
                # If table has specific identifiers, check for those
                if 'extraction_method' in table_info:
                    method = table_info['extraction_method'].lower()
                    if any(word in para_lower for word in method.split('-')):
                        return True
                
                # Check for financial keywords if it's a financial table
                if table_info.get('rows', 0) > 1:
                    financial_refs = sum(1 for keyword in self.financial_keywords 
                                       if keyword in para_lower)
                    if financial_refs >= 2:  # Multiple financial terms suggest table reference
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to check table reference: {e}")
            return False
