from openai import AzureOpenAI
from typing import List, Dict, Any, Tuple
import logging
from database import VectorDatabase
from embeddings import EmbeddingManager
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_VERSION, AZURE_LLM_DEPLOYMENT
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        # Configure Azure OpenAI client for LLM
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_VERSION
        )
        
        self.llm_deployment = AZURE_LLM_DEPLOYMENT
        self.vector_db = VectorDatabase()
        self.embedding_manager = EmbeddingManager()
        
        # RAG configuration
        self.top_k_documents = 5
        self.top_k_tables = 3
        self.similarity_threshold = 0.5
        
    def answer_question(self, question: str, level: str = None, paper: str = None,
                       module: str = None, chapter: str = None, unit: str = None,
                       include_tables: bool = True) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline with table-aware retrieval
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Step 1: Generate query embedding
            query_embedding = self.embedding_manager.get_query_embedding_with_filters(
                question, level or '', paper or ''
            )
            
            # Step 2: Retrieve relevant documents
            document_results = self.vector_db.similarity_search_documents(
                query_embedding=query_embedding,
                top_k=self.top_k_documents,
                level=level,
                paper=paper,
                module=module,
                chapter=chapter,
                unit=unit
            )
            
            # Step 3: Retrieve relevant tables if requested
            table_results = []
            if include_tables:
                table_results = self.vector_db.similarity_search_tables(
                    query_embedding=query_embedding,
                    top_k=self.top_k_tables,
                    level=level,
                    paper=paper,
                    module=module,
                    chapter=chapter,
                    unit=unit
                )
            
            # Step 4: Filter by similarity threshold
            filtered_documents = [
                doc for doc in document_results 
                if doc['similarity'] >= self.similarity_threshold
            ]
            
            filtered_tables = [
                table for table in table_results 
                if table['similarity'] >= self.similarity_threshold
            ]
            
            # Step 5: Handle empty results gracefully
            if not filtered_documents and not filtered_tables:
                return {
                    'answer': f"I don't have any relevant information about '{question}' in my knowledge base yet. Please upload some CA study materials first, or try a more general question about accounting principles.",
                    'confidence': 0.0,
                    'sources': {'documents': [], 'tables': []},
                    'metadata': {
                        'level': level,
                        'paper': paper,
                        'module': module,
                        'chapter': chapter,
                        'unit': unit,
                        'documents_found': 0,
                        'tables_found': 0,
                        'processing_time': 0
                    },
                    'suggestions': self._generate_suggestions(question, level, paper)
                }
            
            # Step 6: Prepare context for LLM
            context = self._prepare_context(filtered_documents, filtered_tables)
            
            # Step 7: Generate answer using LLM
            answer_data = self._generate_answer(question, context, level, paper)
            
            # Step 7: Prepare response with citations
            response = {
                'answer': answer_data['answer'],
                'confidence': self._calculate_confidence(filtered_documents, filtered_tables),
                'sources': {
                    'documents': [self._format_document_source(doc) for doc in filtered_documents[:3]],
                    'tables': [self._format_table_source(table) for table in filtered_tables[:2]]
                },
                'metadata': {
                    'level': level,
                    'paper': paper,
                    'module': module,
                    'chapter': chapter,
                    'unit': unit,
                    'documents_found': len(filtered_documents),
                    'tables_found': len(filtered_tables),
                    'processing_time': answer_data.get('processing_time', 0)
                },
                'suggestions': self._generate_suggestions(question, level, paper)
            }
            
            logger.info(f"Question answered successfully with {len(filtered_documents)} documents and {len(filtered_tables)} tables")
            return response
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return {
                'answer': "I apologize, but I encountered an error while processing your question. Please try again or rephrase your question.",
                'confidence': 0.0,
                'sources': {'documents': [], 'tables': []},
                'metadata': {'error': str(e)},
                'suggestions': []
            }
    
    def _prepare_context(self, documents: List[Dict], tables: List[Dict]) -> str:
        """Prepare context from retrieved documents and tables"""
        try:
            context_parts = []
            
            # Add document context
            if documents:
                context_parts.append("=== RELEVANT CONTENT ===")
                for i, doc in enumerate(documents[:5]):  # Limit to top 5
                    context_parts.append(f"Document {i+1} (Similarity: {doc['similarity']:.2f}):")
                    context_parts.append(f"Source: {doc['file_name']}")
                    if doc.get('level'):
                        context_parts.append(f"Level: {doc['level']}, Paper: {doc['paper']}")
                    if doc.get('chapter'):
                        context_parts.append(f"Chapter: {doc['chapter']}")
                    context_parts.append(f"Content: {doc['content']}")
                    context_parts.append("---")
            
            # Add table context
            if tables:
                context_parts.append("=== RELEVANT TABLES ===")
                for i, table in enumerate(tables[:3]):  # Limit to top 3
                    context_parts.append(f"Table {i+1} (Similarity: {table['similarity']:.2f}):")
                    context_parts.append(f"Source: {table['file_name']}, Page: {table['page_number']}")
                    if table.get('level'):
                        context_parts.append(f"Level: {table['level']}, Paper: {table['paper']}")
                    
                    # Add table data
                    if table.get('table_html'):
                        context_parts.append("Table HTML:")
                        context_parts.append(table['table_html'][:1000])  # Limit HTML length
                    
                    if table.get('context_before'):
                        context_parts.append(f"Context before table: {table['context_before']}")
                    if table.get('context_after'):
                        context_parts.append(f"Context after table: {table['context_after']}")
                    
                    context_parts.append("---")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return "No relevant context found."
    
    def _generate_answer(self, question: str, context: str, level: str = None, 
                        paper: str = None) -> Dict[str, Any]:
        """Generate answer using Azure OpenAI LLM"""
        try:
            import time
            start_time = time.time()
            
            # Prepare system message based on CA context
            system_message = self._get_system_message(level, paper)
            
            # Prepare user message
            user_message = f"""
Question: {question}

Context:
{context}

Instructions:
1. Answer the question based on the provided context
2. If the context includes tables, reference specific data points
3. Provide explanations for financial calculations or formulas
4. Include references to relevant standards, sections, or regulations
5. If the answer involves numerical data from tables, be precise
6. If you cannot answer based on the context, say so clearly
7. For CA-specific topics, use appropriate terminology and concepts
8. Structure your answer clearly with headings if needed
"""
            
            response = self.client.chat.completions.create(
                model=self.llm_deployment,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=2000,
                temperature=0.3  # Lower temperature for more factual responses
            )
            
            answer = response.choices[0].message.content.strip()
            processing_time = time.time() - start_time
            
            return {
                'answer': answer,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                'answer': "I apologize, but I encountered an error while generating the answer. Please try again.",
                'processing_time': 0
            }
    
    def _get_system_message(self, level: str = None, paper: str = None) -> str:
        """Get system message based on CA level and paper"""
        base_message = """You are an expert CA (Chartered Accountancy) assistant specializing in helping Indian CA students with their studies. You have deep knowledge of:

- Indian Accounting Standards (Ind AS)
- Companies Act 2013
- Income Tax Act 1961
- GST regulations
- Auditing and Assurance Standards
- Financial Management and Economics
- Strategic Management
- Ethics and Corporate Governance

Your responses should be:
- Accurate and based on current Indian standards and regulations
- Educational and helpful for students
- Well-structured with clear explanations
- Include practical examples when relevant
- Reference specific standards or sections when applicable
- Focused on helping students understand concepts deeply"""
        
        if level:
            level_specific = f"\n\nYou are currently helping a student at the CA {level} level."
            
            if level == "Foundation":
                level_specific += " Focus on fundamental concepts, basic principles, and foundational knowledge."
            elif level == "Intermediate":
                level_specific += " Provide intermediate-level explanations with moderate complexity and practical applications."
            elif level == "Final":
                level_specific += " Deliver advanced, comprehensive explanations suitable for final-level CA students."
            
            base_message += level_specific
        
        if paper:
            base_message += f"\n\nThe question relates to {paper}. Tailor your response to this specific paper's syllabus and requirements."
        
        return base_message
    
    def _calculate_confidence(self, documents: List[Dict], tables: List[Dict]) -> float:
        """Calculate confidence score based on retrieval results"""
        try:
            if not documents and not tables:
                return 0.0
            
            # Calculate average similarity
            doc_similarities = [doc['similarity'] for doc in documents]
            table_similarities = [table['similarity'] for table in tables]
            
            all_similarities = doc_similarities + table_similarities
            
            if not all_similarities:
                return 0.0
            
            avg_similarity = sum(all_similarities) / len(all_similarities)
            
            # Boost confidence if we have both documents and tables
            if documents and tables:
                avg_similarity *= 1.1
            
            # Cap at 1.0
            return min(avg_similarity, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _format_document_source(self, doc: Dict) -> Dict[str, Any]:
        """Format document source for response"""
        return {
            'file_name': doc['file_name'],
            'level': doc.get('level', ''),
            'paper': doc.get('paper', ''),
            'chapter': doc.get('chapter', ''),
            'similarity': round(doc['similarity'], 2),
            'snippet': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
        }
    
    def _format_table_source(self, table: Dict) -> Dict[str, Any]:
        """Format table source for response"""
        return {
            'file_name': table['file_name'],
            'page_number': table['page_number'],
            'level': table.get('level', ''),
            'paper': table.get('paper', ''),
            'chapter': table.get('chapter', ''),
            'similarity': round(table['similarity'], 2),
            'rows': table.get('rows', 0),
            'cols': table.get('cols', 0),
            'context': table.get('context_before', '')[:100] + "..." if table.get('context_before') else ""
        }
    
    def _generate_suggestions(self, question: str, level: str = None, 
                            paper: str = None) -> List[str]:
        """Generate related question suggestions"""
        try:
            suggestions = []
            
            # Basic suggestions based on question type
            question_lower = question.lower()
            
            if 'balance sheet' in question_lower:
                suggestions.extend([
                    "What are the components of a balance sheet?",
                    "How to prepare a balance sheet as per Schedule III?",
                    "Explain the classification of assets and liabilities"
                ])
            
            elif 'profit' in question_lower or 'income' in question_lower:
                suggestions.extend([
                    "How to prepare Statement of Profit and Loss?",
                    "What are the components of comprehensive income?",
                    "Explain revenue recognition principles"
                ])
            
            elif 'tax' in question_lower:
                suggestions.extend([
                    "What are the provisions for income tax computation?",
                    "Explain deferred tax assets and liabilities",
                    "How to calculate tax on different types of income?"
                ])
            
            elif 'audit' in question_lower:
                suggestions.extend([
                    "What are the key auditing standards?",
                    "Explain the audit process and procedures",
                    "What are the types of audit opinions?"
                ])
            
            # Level-specific suggestions
            if level == "Foundation":
                suggestions.extend([
                    "What are the fundamental accounting principles?",
                    "Explain the accounting equation and its applications"
                ])
            elif level == "Intermediate":
                suggestions.extend([
                    "How to apply Ind AS in practical scenarios?",
                    "Explain advanced accounting treatments"
                ])
            elif level == "Final":
                suggestions.extend([
                    "What are the latest updates in accounting standards?",
                    "Explain complex financial reporting requirements"
                ])
            
            # Return unique suggestions, limited to 5
            return list(set(suggestions))[:5]
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []
    
    def get_progressive_learning_path(self, topic: str, current_level: str) -> Dict[str, Any]:
        """Generate a progressive learning path for a topic"""
        try:
            learning_path = {
                'current_level': current_level,
                'topic': topic,
                'prerequisites': [],
                'current_concepts': [],
                'next_steps': [],
                'related_topics': []
            }
            
            # This would be enhanced with actual curriculum mapping
            # For now, providing a basic structure
            
            if current_level == "Foundation":
                learning_path['prerequisites'] = ["Basic accounting concepts", "Double entry system"]
                learning_path['next_steps'] = ["Advanced accounting", "Financial statement analysis"]
            
            elif current_level == "Intermediate":
                learning_path['prerequisites'] = ["Foundation level knowledge", "Basic standards"]
                learning_path['next_steps'] = ["Advanced standards", "Practical applications"]
            
            elif current_level == "Final":
                learning_path['prerequisites'] = ["Intermediate level mastery", "Practical exposure"]
                learning_path['next_steps'] = ["Professional application", "Industry specialization"]
            
            return learning_path
            
        except Exception as e:
            logger.error(f"Failed to generate learning path: {e}")
            return {'error': str(e)}
