"""
CA Curriculum Manager for Hierarchical Navigation
Parses and manages the CA curriculum structure for cascading dropdowns
"""

import re
import json
import os
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CurriculumManager:
    def __init__(self):
        self.curriculum_data = {}
        # Configuration for curriculum source
        self.curriculum_source = os.getenv('CURRICULUM_SOURCE', 'json')  # 'json' or 'tree'
        self.json_curriculum_path = os.getenv('CURRICULUM_JSON_PATH', 'attached_assets/New document 1_1758323551064.json')
        self.tree_curriculum_path = 'attached_assets/output_1758321057482.txt'
        self._load_curriculum_structure()
    
    def _load_curriculum_structure(self):
        """Load and parse the curriculum structure from JSON or text file"""
        try:
            # Try JSON first if available
            if self.curriculum_source == 'json' and os.path.exists(self.json_curriculum_path):
                logger.info(f"Loading curriculum from JSON: {self.json_curriculum_path}")
                with open(self.json_curriculum_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                self.curriculum_data = self._parse_curriculum_json(json_data)
                
                # Check if JSON parsing was successful
                if self.curriculum_data and len(self.curriculum_data) > 0:
                    logger.info(f"Successfully loaded JSON curriculum with {len(self.curriculum_data)} levels")
                    return
                else:
                    logger.warning("JSON curriculum data is empty or malformed, falling back to tree parsing")
            
            # Fallback to tree parsing
            logger.info(f"Loading curriculum from tree structure: {self.tree_curriculum_path}")
            with open(self.tree_curriculum_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.curriculum_data = self._parse_curriculum_tree(content)
            logger.info(f"Loaded tree curriculum with {len(self.curriculum_data)} levels")
            
        except Exception as e:
            logger.error(f"Failed to load curriculum: {e}")
            # Fallback to basic structure
            self._create_fallback_structure()
    
    def _parse_curriculum_tree(self, content: str) -> Dict[str, Any]:
        """Parse the tree structure from the text content"""
        curriculum = {}
        current_level = None
        current_paper = None
        current_module = None
        current_chapter = None
        
        lines = content.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
                
            # Clean the line by removing tree characters and getting the content
            clean_line = line.strip()
            # Remove tree drawing characters
            clean_line = re.sub(r'^[│├└\s─]*', '', clean_line).strip()
            
            if not clean_line or clean_line == '.':
                continue
            
            # Count tree depth by counting the tree structure
            # Pattern is: │ + two non-breaking spaces + one regular space
            tree_pattern = '│\xa0\xa0 '  # │ + two non-breaking spaces (Unicode 160) + one regular space
            tree_depth = line.count(tree_pattern) + (1 if ('├──' in line or '└──' in line) else 0)
            
            # Level 0: CA Levels (Foundation, Intermediate, Final)
            if tree_depth == 1 and any(level in clean_line.lower() for level in ['foundation', 'intermediate', 'final']):
                current_level = self._normalize_level_name(clean_line)
                curriculum[current_level] = {}
                current_paper = None
                current_module = None
                current_chapter = None
                
            # Level 1: Papers
            elif tree_depth == 2 and current_level and 'paper' in clean_line.lower():
                current_paper = self._clean_paper_name(clean_line)
                curriculum[current_level][current_paper] = {}
                current_module = None
                current_chapter = None
                
            # Level 2: Modules or Chapters
            elif tree_depth == 3 and current_level and current_paper:
                if 'module' in clean_line.lower() or 'part' in clean_line.lower():
                    # This is a module
                    current_module = self._clean_module_name(clean_line)
                    curriculum[current_level][current_paper][current_module] = {}
                    current_chapter = None
                elif 'chapter' in clean_line.lower():
                    # This is a direct chapter (no modules)
                    current_chapter = self._clean_chapter_name(clean_line)
                    if 'chapters' not in curriculum[current_level][current_paper]:
                        curriculum[current_level][current_paper]['chapters'] = {}
                    curriculum[current_level][current_paper]['chapters'][current_chapter] = {}
                    
            # Level 3: Chapters (under modules) or Units
            elif tree_depth == 4 and current_level and current_paper:
                if current_module and 'chapter' in clean_line.lower():
                    # Chapter under module
                    current_chapter = self._clean_chapter_name(clean_line)
                    curriculum[current_level][current_paper][current_module][current_chapter] = {}
                elif current_chapter and 'unit' in clean_line.lower():
                    # Unit under direct chapter
                    unit_name = self._clean_unit_name(clean_line)
                    if 'units' not in curriculum[current_level][current_paper]['chapters'][current_chapter]:
                        curriculum[current_level][current_paper]['chapters'][current_chapter]['units'] = {}
                    curriculum[current_level][current_paper]['chapters'][current_chapter]['units'][unit_name] = {}
                    
            # Level 4: Units (under module chapters)
            elif tree_depth == 5 and current_level and current_paper and current_module and current_chapter:
                if 'unit' in clean_line.lower():
                    unit_name = self._clean_unit_name(clean_line)
                    if 'units' not in curriculum[current_level][current_paper][current_module][current_chapter]:
                        curriculum[current_level][current_paper][current_module][current_chapter]['units'] = {}
                    curriculum[current_level][current_paper][current_module][current_chapter]['units'][unit_name] = {}
        
        return curriculum
    
    def _parse_curriculum_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the curriculum structure from JSON data"""
        curriculum = {}
        
        try:
            if 'CA_Course_Structure' not in json_data:
                logger.error("JSON data missing 'CA_Course_Structure' key")
                return curriculum
            
            ca_structure = json_data['CA_Course_Structure']
            
            for level_name, level_data in ca_structure.items():
                # Normalize level name
                normalized_level = self._normalize_level_name(level_name)
                curriculum[normalized_level] = {}
                
                for paper_name, paper_data in level_data.items():
                    # Clean paper name
                    clean_paper = self._clean_paper_name(paper_name)
                    curriculum[normalized_level][clean_paper] = {}
                    
                    # Check if paper has modules or direct chapters
                    has_modules = any('module' in key.lower() for key in paper_data.keys())
                    
                    if has_modules:
                        # Paper has modules
                        for module_name, module_data in paper_data.items():
                            if 'module' in module_name.lower():
                                clean_module = self._clean_module_name(module_name)
                                curriculum[normalized_level][clean_paper][clean_module] = {}
                                
                                # Add chapters under module
                                if isinstance(module_data, dict):
                                    for chapter_name, chapter_data in module_data.items():
                                        clean_chapter = self._clean_chapter_name(chapter_name)
                                        curriculum[normalized_level][clean_paper][clean_module][clean_chapter] = {}
                                        
                                        # Check for units in chapter
                                        if isinstance(chapter_data, dict) and chapter_data:
                                            curriculum[normalized_level][clean_paper][clean_module][clean_chapter]['units'] = {}
                                            for unit_name in chapter_data.keys():
                                                clean_unit = self._clean_unit_name(unit_name)
                                                curriculum[normalized_level][clean_paper][clean_module][clean_chapter]['units'][clean_unit] = {}
                    else:
                        # Paper has direct chapters (no modules)
                        curriculum[normalized_level][clean_paper]['chapters'] = {}
                        
                        for chapter_name, chapter_data in paper_data.items():
                            clean_chapter = self._clean_chapter_name(chapter_name)
                            curriculum[normalized_level][clean_paper]['chapters'][clean_chapter] = {}
                            
                            # Check for units in chapter
                            if isinstance(chapter_data, dict) and chapter_data:
                                curriculum[normalized_level][clean_paper]['chapters'][clean_chapter]['units'] = {}
                                for unit_name in chapter_data.keys():
                                    clean_unit = self._clean_unit_name(unit_name)
                                    curriculum[normalized_level][clean_paper]['chapters'][clean_chapter]['units'][clean_unit] = {}
            
            return curriculum
            
        except Exception as e:
            logger.error(f"Error parsing JSON curriculum: {e}")
            return {}
    
    def _group_sub_papers(self):
        """Group sub-papers (e.g., Paper-3A, Paper-3B) into a single main paper."""
        for level_name, level_data in self.curriculum_data.items():
            papers_to_group = {}
            
            # Find papers that need grouping
            for paper_name in list(level_data.keys()):
                match = re.match(r'(Paper-\d+)[A-Z]:\s*(.*)', paper_name)
                if match:
                    main_paper_base = match.group(1)
                    if main_paper_base not in papers_to_group:
                        papers_to_group[main_paper_base] = []
                    papers_to_group[main_paper_base].append(paper_name)
            
            # Group the papers
            for main_paper_base, sub_paper_names in papers_to_group.items():
                # Create a new main paper name, e.g., "Paper-3: Quantitative Aptitude"
                # This assumes the first sub-paper's name is representative
                first_sub_paper_name = sub_paper_names[0]
                main_paper_title = re.sub(r'[A-Z]:\s*', ': ', first_sub_paper_name)
                
                # Create the new main paper entry
                if main_paper_title not in level_data:
                    level_data[main_paper_title] = {}
                
                # Move sub-paper content into the new main paper entry
                for sub_paper_name in sub_paper_names:
                    sub_paper_content = level_data.pop(sub_paper_name)
                    # Use the sub-paper name as a key under the main paper, like a module
                    level_data[main_paper_title][sub_paper_name] = sub_paper_content

    def _normalize_level_name(self, level: str) -> str:
        """Normalize level names to consistent format"""
        level_lower = level.lower()
        if 'foundation' in level_lower:
            return 'Foundation'
        elif 'intermediate' in level_lower:
            return 'Intermediate'
        elif 'final' in level_lower:
            return 'Final'
        return level.title()
    
    def _clean_paper_name(self, paper: str) -> str:
        """Clean paper names"""
        # Remove file extensions and clean up
        paper = re.sub(r'\.pdf$', '', paper)
        return paper.strip()
    
    def _clean_module_name(self, module: str) -> str:
        """Clean module names"""
        module = re.sub(r'\.pdf$', '', module)
        return module.strip()
    
    def _clean_chapter_name(self, chapter: str) -> str:
        """Clean chapter names"""
        chapter = re.sub(r'\.pdf$', '', chapter)
        return chapter.strip()
    
    def _clean_unit_name(self, unit: str) -> str:
        """Clean unit names"""
        unit = re.sub(r'\.pdf$', '', unit)
        return unit.strip()
    
    def _create_fallback_structure(self):
        """Create a basic fallback structure if parsing fails"""
        self.curriculum_data = {
            'Foundation': {
                'Paper 1: Accounting': {'chapters': {}},
                'Paper 2: Business Laws': {'chapters': {}},
                'Paper 3: Quantitative Aptitude': {'chapters': {}},
                'Paper 4: Business Economics': {'chapters': {}}
            },
            'Intermediate': {
                'Paper 1: Advanced Accounting': {'chapters': {}},
                'Paper 2: Corporate and Other Laws': {'chapters': {}},
                'Paper 3: Taxation': {'chapters': {}},
                'Paper 4: Cost and Management Accounting': {'chapters': {}}
            },
            'Final': {
                'Paper 1: Financial Reporting': {'chapters': {}},
                'Paper 2: Advanced Financial Management': {'chapters': {}},
                'Paper 3: Advanced Auditing and Professional Ethics': {'chapters': {}},
                'Paper 4: Direct Tax Laws': {'chapters': {}},
                'Paper 5: Indirect Tax Laws': {'chapters': {}}
            }
        }
    
    # Public API methods for UI components
    
    def get_levels(self) -> List[str]:
        """Get all available CA levels"""
        return list(self.curriculum_data.keys())
    
    def get_papers(self, level: str) -> List[str]:
        """Get papers for a specific level"""
        if level not in self.curriculum_data:
            return []
        return list(self.curriculum_data[level].keys())
    
    def get_modules(self, level: str, paper: str) -> List[str]:
        """Get modules for a specific level and paper"""
        if not level or not paper or level not in self.curriculum_data or paper not in self.curriculum_data[level]:
            return []
        
        paper_data = self.curriculum_data[level][paper]
        modules = []
        
        # Get all keys that are not 'chapters'
        for key in paper_data.keys():
            if key != 'chapters':
                modules.append(key)
        
        return modules
    
    def has_modules(self, level: str, paper: str) -> bool:
        """Check if a paper has modules"""
        if not level or not paper:
            return False
        modules = self.get_modules(level, paper)
        return len(modules) > 0
    
    def get_chapters(self, level: str, paper: str, module: Optional[str] = None) -> List[str]:
        """Get chapters for a specific level, paper, and optionally module"""
        if not level or not paper or level not in self.curriculum_data or paper not in self.curriculum_data[level]:
            return []
        
        paper_data = self.curriculum_data[level][paper]
        
        if module:
            # Get chapters from specific module
            if module not in paper_data:
                return []
            module_data = paper_data[module]
            return [key for key in module_data.keys() if key != 'units']
        else:
            # Get chapters directly from paper (no modules)
            if 'chapters' in paper_data:
                return list(paper_data['chapters'].keys())
            return []
    
    def get_units(self, level: str, paper: str, chapter: str, module: Optional[str] = None) -> List[str]:
        """Get units for a specific level, paper, chapter, and optionally module"""
        if not level or not paper or not chapter or level not in self.curriculum_data or paper not in self.curriculum_data[level]:
            return []
        
        paper_data = self.curriculum_data[level][paper]
        
        if module:
            # Get units from module chapter
            if module not in paper_data or chapter not in paper_data[module]:
                return []
            chapter_data = paper_data[module][chapter]
            if 'units' in chapter_data:
                return list(chapter_data['units'].keys())
        else:
            # Get units from direct chapter
            if 'chapters' not in paper_data or chapter not in paper_data['chapters']:
                return []
            chapter_data = paper_data['chapters'][chapter]
            if 'units' in chapter_data:
                return list(chapter_data['units'].keys())
        
        return []
    
    def has_units(self, level: str, paper: str, chapter: str, module: Optional[str] = None) -> bool:
        """Check if a chapter has units"""
        if not level or not paper or not chapter:
            return False
        units = self.get_units(level, paper, chapter, module)
        return len(units) > 0
    
    def get_hierarchy_info(self, level: str, paper: Optional[str] = None, module: Optional[str] = None, 
                          chapter: Optional[str] = None, unit: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive hierarchy information for a given path"""
        info = {
            'level': level,
            'paper': paper,
            'module': module,
            'chapter': chapter,
            'unit': unit,
            'has_modules': False,
            'has_units': False,
            'available_papers': [],
            'available_modules': [],
            'available_chapters': [],
            'available_units': []
        }
        
        if level:
            info['available_papers'] = self.get_papers(level)
            
        if paper:
            info['has_modules'] = self.has_modules(level, paper)
            info['available_modules'] = self.get_modules(level, paper)
            
            if not info['has_modules']:
                info['available_chapters'] = self.get_chapters(level, paper)
                
        if module and paper:
            info['available_chapters'] = self.get_chapters(level, paper, module)
            
        if chapter and paper:
            info['has_units'] = self.has_units(level, paper, chapter, module)
            info['available_units'] = self.get_units(level, paper, chapter, module)
            
        return info
    
    def validate_hierarchy(self, level: str, paper: Optional[str] = None, module: Optional[str] = None,
                          chapter: Optional[str] = None, unit: Optional[str] = None) -> bool:
        """Validate if the given hierarchy path is valid"""
        try:
            if level not in self.curriculum_data:
                return False
                
            if paper and paper not in self.curriculum_data[level]:
                return False
                
            if module and paper:
                if not self.has_modules(level, paper) or module not in self.get_modules(level, paper):
                    return False
                    
            if chapter and paper:
                valid_chapters = self.get_chapters(level, paper, module)
                if chapter not in valid_chapters:
                    return False
                    
            if unit and paper and chapter:
                valid_units = self.get_units(level, paper, chapter, module)
                if unit not in valid_units:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Hierarchy validation error: {e}")
            return False
    
    def get_display_path(self, level: str, paper: Optional[str] = None, module: Optional[str] = None,
                        chapter: Optional[str] = None, unit: Optional[str] = None) -> str:
        """Generate a human-readable display path"""
        path_parts = [level]
        
        if paper:
            path_parts.append(paper)
        if module:
            path_parts.append(module)
        if chapter:
            path_parts.append(chapter)
        if unit:
            path_parts.append(unit)
            
        return " → ".join(path_parts)
    
    def find_item_hierarchy(self, search_term: str, search_type: str = 'all') -> List[Dict[str, Any]]:
        """
        Find all items matching search term and return their full hierarchy
        Uses flexible partial matching - you can search with any part of the name
        
        Args:
            search_term: Term to search for (case-insensitive, partial matching)
            search_type: Type to search ('all', 'unit', 'chapter', 'module', 'paper', 'level')
        
        Returns:
            List of dictionaries with full hierarchy information
        """
        results = []
        search_term_lower = search_term.lower().strip()
        
        # Split search term into words for better matching
        search_words = search_term_lower.split()
        
        def _matches_search(text: str) -> bool:
            """Check if text matches search criteria with flexible partial matching"""
            text_lower = text.lower()
            
            # If no search words, return False
            if not search_words:
                return False
            
            # Check if all search words are found in the text (partial matching)
            for word in search_words:
                if word not in text_lower:
                    return False
            return True
        
        for level_name, level_data in self.curriculum_data.items():
            if search_type in ['all', 'level'] and _matches_search(level_name):
                results.append({
                    'type': 'level',
                    'match': level_name,
                    'level': level_name,
                    'paper': None,
                    'module': None,
                    'chapter': None,
                    'unit': None,
                    'display_path': level_name,
                    'relevance': self._calculate_relevance(level_name, search_term_lower)
                })
            
            for paper_name, paper_data in level_data.items():
                if search_type in ['all', 'paper'] and _matches_search(paper_name):
                    results.append({
                        'type': 'paper',
                        'match': paper_name,
                        'level': level_name,
                        'paper': paper_name,
                        'module': None,
                        'chapter': None,
                        'unit': None,
                        'display_path': f"{level_name} → {paper_name}",
                        'relevance': self._calculate_relevance(paper_name, search_term_lower)
                    })
                
                # Check if paper has modules or direct chapters
                if 'chapters' in paper_data:
                    # Direct chapters (no modules)
                    for chapter_name, chapter_data in paper_data['chapters'].items():
                        if search_type in ['all', 'chapter'] and _matches_search(chapter_name):
                            results.append({
                                'type': 'chapter',
                                'match': chapter_name,
                                'level': level_name,
                                'paper': paper_name,
                                'module': None,
                                'chapter': chapter_name,
                                'unit': None,
                                'display_path': f"{level_name} → {paper_name} → {chapter_name}",
                                'relevance': self._calculate_relevance(chapter_name, search_term_lower)
                            })
                        
                        # Check for units in direct chapters
                        if isinstance(chapter_data, dict) and 'units' in chapter_data:
                            for unit_name in chapter_data['units'].keys():
                                if search_type in ['all', 'unit'] and _matches_search(unit_name):
                                    results.append({
                                        'type': 'unit',
                                        'match': unit_name,
                                        'level': level_name,
                                        'paper': paper_name,
                                        'module': None,
                                        'chapter': chapter_name,
                                        'unit': unit_name,
                                        'display_path': f"{level_name} → {paper_name} → {chapter_name} → {unit_name}",
                                        'relevance': self._calculate_relevance(unit_name, search_term_lower)
                                    })
                else:
                    # Paper has modules
                    for module_name, module_data in paper_data.items():
                        if module_name != 'chapters':  # Skip the chapters key if it exists
                            if search_type in ['all', 'module'] and _matches_search(module_name):
                                results.append({
                                    'type': 'module',
                                    'match': module_name,
                                    'level': level_name,
                                    'paper': paper_name,
                                    'module': module_name,
                                    'chapter': None,
                                    'unit': None,
                                    'display_path': f"{level_name} → {paper_name} → {module_name}",
                                    'relevance': self._calculate_relevance(module_name, search_term_lower)
                                })
                            
                            # Check chapters in modules
                            if isinstance(module_data, dict):
                                for chapter_name, chapter_data in module_data.items():
                                    if chapter_name != 'units':  # Skip units key
                                        if search_type in ['all', 'chapter'] and _matches_search(chapter_name):
                                            results.append({
                                                'type': 'chapter',
                                                'match': chapter_name,
                                                'level': level_name,
                                                'paper': paper_name,
                                                'module': module_name,
                                                'chapter': chapter_name,
                                                'unit': None,
                                                'display_path': f"{level_name} → {paper_name} → {module_name} → {chapter_name}",
                                                'relevance': self._calculate_relevance(chapter_name, search_term_lower)
                                            })
                                        
                                        # Check for units in module chapters
                                        if isinstance(chapter_data, dict) and 'units' in chapter_data:
                                            for unit_name in chapter_data['units'].keys():
                                                if search_type in ['all', 'unit'] and _matches_search(unit_name):
                                                    results.append({
                                                        'type': 'unit',
                                                        'match': unit_name,
                                                        'level': level_name,
                                                        'paper': paper_name,
                                                        'module': module_name,
                                                        'chapter': chapter_name,
                                                        'unit': unit_name,
                                                        'display_path': f"{level_name} → {paper_name} → {module_name} → {chapter_name} → {unit_name}",
                                                        'relevance': self._calculate_relevance(unit_name, search_term_lower)
                                                    })
        
        # Sort results by relevance (higher relevance first)
        results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        return results
    
    def _calculate_relevance(self, text: str, search_term: str) -> float:
        """Calculate relevance score for search results"""
        text_lower = text.lower()
        search_lower = search_term.lower()
        
        # Exact match gets highest score
        if text_lower == search_lower:
            return 1.0
        
        # Starts with search term gets high score
        if text_lower.startswith(search_lower):
            return 0.9
        
        # Contains search term as whole word gets medium-high score
        if f" {search_lower} " in f" {text_lower} ":
            return 0.8
        
        # Contains search term gets medium score
        if search_lower in text_lower:
            return 0.7
        
        # Partial word matches get lower scores
        search_words = search_lower.split()
        matches = sum(1 for word in search_words if word in text_lower)
        return 0.5 * (matches / len(search_words)) if search_words else 0.0
    
    def get_all_items_by_type(self, item_type: str) -> List[Dict[str, Any]]:
        """
        Get all items of a specific type with their full hierarchy
        
        Args:
            item_type: Type of items to get ('level', 'paper', 'module', 'chapter', 'unit')
        
        Returns:
            List of dictionaries with full hierarchy information
        """
        return self.find_item_hierarchy('', item_type)
    
    def auto_fill_hierarchy_from_selection(self, selected_item: str, item_type: str) -> Optional[Dict[str, str]]:
        """
        Auto-fill hierarchy when a specific item is selected
        
        Args:
            selected_item: The selected item name
            item_type: Type of the selected item ('unit', 'chapter', 'module', 'paper', 'level')
        
        Returns:
            Dictionary with complete hierarchy or None if not found
        """
        matches = self.find_item_hierarchy(selected_item, item_type)
        
        # Find exact match
        for match in matches:
            if match['match'].lower() == selected_item.lower() and match['type'] == item_type:
                return {
                    'level': match['level'],
                    'paper': match['paper'],
                    'module': match['module'],
                    'chapter': match['chapter'],
                    'unit': match['unit']
                }
        
        return None

# Global curriculum manager instance
curriculum_manager = CurriculumManager()