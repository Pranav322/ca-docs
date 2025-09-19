"""
CA Curriculum Manager for Hierarchical Navigation
Parses and manages the CA curriculum structure for cascading dropdowns
"""

import re
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CurriculumManager:
    def __init__(self):
        self.curriculum_data = {}
        self._load_curriculum_structure()
    
    def _load_curriculum_structure(self):
        """Load and parse the curriculum structure from text file"""
        try:
            # Read the curriculum file
            with open('attached_assets/output_1758321057482.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.curriculum_data = self._parse_curriculum_tree(content)
            logger.info(f"Loaded curriculum with {len(self.curriculum_data)} levels")
            
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
                
            # Count indentation level
            indent_level = (len(line) - len(line.lstrip())) // 4
            clean_line = line.strip().replace('├── ', '').replace('└── ', '').replace('│   ', '').strip()
            
            if not clean_line or clean_line == '.':
                continue
            
            # Level 1: CA Levels (Foundation, Intermediate, Final)
            if indent_level == 1 and any(level in clean_line.lower() for level in ['foundation', 'intermediate', 'final']):
                current_level = self._normalize_level_name(clean_line)
                curriculum[current_level] = {}
                current_paper = None
                current_module = None
                current_chapter = None
                
            # Level 2: Papers
            elif indent_level == 2 and current_level and 'paper' in clean_line.lower():
                current_paper = self._clean_paper_name(clean_line)
                curriculum[current_level][current_paper] = {}
                current_module = None
                current_chapter = None
                
            # Level 3: Modules or Chapters
            elif indent_level == 3 and current_level and current_paper:
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
                    
            # Level 4: Chapters (under modules) or Units
            elif indent_level == 4 and current_level and current_paper:
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
                    
            # Level 5: Units (under module chapters)
            elif indent_level == 5 and current_level and current_paper and current_module and current_chapter:
                if 'unit' in clean_line.lower():
                    unit_name = self._clean_unit_name(clean_line)
                    if 'units' not in curriculum[current_level][current_paper][current_module][current_chapter]:
                        curriculum[current_level][current_paper][current_module][current_chapter]['units'] = {}
                    curriculum[current_level][current_paper][current_module][current_chapter]['units'][unit_name] = {}
        
        return curriculum
    
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
                
        if module:
            info['available_chapters'] = self.get_chapters(level, paper, module)
            
        if chapter:
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
                
            if module:
                if not self.has_modules(level, paper) or module not in self.get_modules(level, paper):
                    return False
                    
            if chapter:
                valid_chapters = self.get_chapters(level, paper, module)
                if chapter not in valid_chapters:
                    return False
                    
            if unit:
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

# Global curriculum manager instance
curriculum_manager = CurriculumManager()