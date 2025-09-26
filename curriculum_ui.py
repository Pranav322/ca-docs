"""
Curriculum UI Components for Hierarchical Navigation
Provides cascading dropdown components for CA curriculum structure
"""

import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
from curriculum_manager import curriculum_manager

class CurriculumSelector:
    """Handles hierarchical curriculum selection with cascading dropdowns"""
    
    def __init__(self, prefix: str = "curriculum"):
        """
        Initialize curriculum selector with a unique prefix for session state
        
        Args:
            prefix: Unique prefix for session state keys
        """
        self.prefix = prefix
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for curriculum selection"""
        default_values = {
            f"{self.prefix}_level": None,
            f"{self.prefix}_paper": None,
            f"{self.prefix}_module": None,
            f"{self.prefix}_chapter": None,
            f"{self.prefix}_unit": None
        }
        
        for key, default in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = default
    
    def _reset_dependent_selections(self, from_level: str):
        """Reset dependent selections when a higher level changes"""
        reset_map = {
            'level': ['paper', 'module', 'chapter', 'unit'],
            'paper': ['module', 'chapter', 'unit'],
            'module': ['chapter', 'unit'],
            'chapter': ['unit']
        }
        
        if from_level in reset_map:
            for dependent in reset_map[from_level]:
                key = f"{self.prefix}_{dependent}"
                if key in st.session_state:
                    st.session_state[key] = None
    
    def render_level_selector(self, label: str = "CA Level", help_text: str = None) -> Optional[str]:
        """Render CA level dropdown"""
        levels = curriculum_manager.get_levels()
        
        if not levels:
            st.error("No CA levels available. Please check curriculum data.")
            return None
        
        # Add empty option for selection
        options = ["Select Level"] + levels
        current_index = 0
        
        if st.session_state.get(f"{self.prefix}_level"):
            try:
                current_index = options.index(st.session_state[f"{self.prefix}_level"])
            except ValueError:
                current_index = 0
        
        selected = st.selectbox(
            label,
            options=options,
            index=current_index,
            help=help_text,
            key=f"{self.prefix}_level_selectbox"
        )
        
        if selected and selected != "Select Level":
            if st.session_state.get(f"{self.prefix}_level") != selected:
                self._reset_dependent_selections('level')
            st.session_state[f"{self.prefix}_level"] = selected
            return selected
        else:
            st.session_state[f"{self.prefix}_level"] = None
            self._reset_dependent_selections('level')
            return None
    
    def render_paper_selector(self, label: str = "Paper", help_text: str = None) -> Optional[str]:
        """Render paper dropdown based on selected level"""
        level = st.session_state.get(f"{self.prefix}_level")
        
        if not level:
            st.selectbox(label, ["Select Level First"], disabled=True, help="Please select a CA level first", key=f"{self.prefix}_paper_disabled")
            return None
        
        papers = curriculum_manager.get_papers(level)
        
        if not papers:
            st.selectbox(label, ["No papers available"], disabled=True, key=f"{self.prefix}_paper_no_available")
            return None
        
        options = ["Select Paper"] + papers
        current_index = 0
        
        if st.session_state.get(f"{self.prefix}_paper"):
            try:
                current_index = options.index(st.session_state[f"{self.prefix}_paper"])
            except ValueError:
                current_index = 0
        
        selected = st.selectbox(
            label,
            options=options,
            index=current_index,
            help=help_text,
            key=f"{self.prefix}_paper_selectbox"
        )
        
        if selected and selected != "Select Paper":
            if st.session_state.get(f"{self.prefix}_paper") != selected:
                self._reset_dependent_selections('paper')
            st.session_state[f"{self.prefix}_paper"] = selected
            return selected
        else:
            st.session_state[f"{self.prefix}_paper"] = None
            self._reset_dependent_selections('paper')
            return None
    
    def render_module_selector(self, label: str = "Module", help_text: str = None) -> Optional[str]:
        """Render module dropdown based on selected level and paper"""
        level = st.session_state.get(f"{self.prefix}_level")
        paper = st.session_state.get(f"{self.prefix}_paper")
        
        if not level or not paper:
            st.selectbox(label, ["Select Paper First"], disabled=True, help="Please select a paper first", key=f"{self.prefix}_chapter_paper_first")
            return None
        
        # Check if paper has modules
        if not curriculum_manager.has_modules(level, paper):
            # Skip module selection - no modules for this paper
            st.info("ℹ️ This paper has no modules - proceeding directly to chapters")
            st.session_state[f"{self.prefix}_module"] = None
            return None
        
        modules = curriculum_manager.get_modules(level, paper)
        
        if not modules:
            st.selectbox(label, ["No modules available"], disabled=True, key=f"{self.prefix}_module_no_available")
            return None
        
        options = ["Select Module"] + modules
        current_index = 0
        
        if st.session_state.get(f"{self.prefix}_module"):
            try:
                current_index = options.index(st.session_state[f"{self.prefix}_module"])
            except ValueError:
                current_index = 0
        
        selected = st.selectbox(
            label,
            options=options,
            index=current_index,
            help=help_text,
            key=f"{self.prefix}_module_selectbox"
        )
        
        if selected and selected != "Select Module":
            if st.session_state.get(f"{self.prefix}_module") != selected:
                self._reset_dependent_selections('module')
            st.session_state[f"{self.prefix}_module"] = selected
            return selected
        else:
            st.session_state[f"{self.prefix}_module"] = None
            self._reset_dependent_selections('module')
            return None
    
    def render_chapter_selector(self, label: str = "Chapter", help_text: str = None) -> Optional[str]:
        """Render chapter dropdown based on selected level, paper, and module"""
        level = st.session_state.get(f"{self.prefix}_level")
        paper = st.session_state.get(f"{self.prefix}_paper")
        module = st.session_state.get(f"{self.prefix}_module")
        
        if not level or not paper:
            st.selectbox(label, ["Select Paper First"], disabled=True, help="Please select a paper first", key=f"{self.prefix}_module_disabled")
            return None
        
        # Check if we need module selection
        has_modules = curriculum_manager.has_modules(level, paper)
        if has_modules and not module:
            st.selectbox(label, ["Select Module First"], disabled=True, help="Please select a module first", key=f"{self.prefix}_chapter_module_first")
            return None
        
        chapters = curriculum_manager.get_chapters(level, paper, module)
        
        if not chapters:
            st.selectbox(label, ["No chapters available"], disabled=True, key=f"{self.prefix}_chapter_no_available")
            return None
        
        options = ["Select Chapter"] + chapters
        current_index = 0
        
        if st.session_state.get(f"{self.prefix}_chapter"):
            try:
                current_index = options.index(st.session_state[f"{self.prefix}_chapter"])
            except ValueError:
                current_index = 0
        
        selected = st.selectbox(
            label,
            options=options,
            index=current_index,
            help=help_text,
            key=f"{self.prefix}_chapter_selectbox"
        )
        
        if selected and selected != "Select Chapter":
            if st.session_state.get(f"{self.prefix}_chapter") != selected:
                self._reset_dependent_selections('chapter')
            st.session_state[f"{self.prefix}_chapter"] = selected
            return selected
        else:
            st.session_state[f"{self.prefix}_chapter"] = None
            self._reset_dependent_selections('chapter')
            return None
    
    def render_unit_selector(self, label: str = "Unit", help_text: str = None) -> Optional[str]:
        """Render unit dropdown based on selected level, paper, module, and chapter"""
        level = st.session_state.get(f"{self.prefix}_level")
        paper = st.session_state.get(f"{self.prefix}_paper")
        module = st.session_state.get(f"{self.prefix}_module")
        chapter = st.session_state.get(f"{self.prefix}_chapter")
        
        if not level or not paper or not chapter:
            st.selectbox(label, ["Select Chapter First"], disabled=True, help="Please select a chapter first", key=f"{self.prefix}_unit_chapter_first")
            return None
        
        # Check if chapter has units
        if not curriculum_manager.has_units(level, paper, chapter, module):
            # Skip unit selection - no units for this chapter
            st.info("ℹ️ This chapter has no units - selection complete at chapter level")
            st.session_state[f"{self.prefix}_unit"] = None
            return None
        
        units = curriculum_manager.get_units(level, paper, chapter, module)
        
        if not units:
            st.selectbox(label, ["No units available"], disabled=True, key=f"{self.prefix}_unit_no_available")
            return None
        
        options = ["Select Unit"] + units
        current_index = 0
        
        if st.session_state.get(f"{self.prefix}_unit"):
            try:
                current_index = options.index(st.session_state[f"{self.prefix}_unit"])
            except ValueError:
                current_index = 0
        
        selected = st.selectbox(
            label,
            options=options,
            index=current_index,
            help=help_text,
            key=f"{self.prefix}_unit_selectbox"
        )
        
        if selected and selected != "Select Unit":
            st.session_state[f"{self.prefix}_unit"] = selected
            return selected
        else:
            st.session_state[f"{self.prefix}_unit"] = None
            return None
    
    def render_complete_selector(self, title: str = "Select Curriculum Hierarchy",
                               show_path: bool = True, columns: bool = True) -> Dict[str, Optional[str]]:
        """Render complete curriculum selector with all levels"""
        
        if title:
            st.subheader(title)
        
        if columns:
            # Render in columns following proper hierarchy: Level → Paper → Module → Chapter → Unit
            col1, col2 = st.columns(2)
            
            with col1:
                level = self.render_level_selector("CA Level")
                paper = self.render_paper_selector("Paper")
                module = self.render_module_selector("Module")
                
            with col2:
                chapter = self.render_chapter_selector("Chapter")
                unit = self.render_unit_selector("Unit")
        else:
            # Render in single column
            level = self.render_level_selector("CA Level")
            paper = self.render_paper_selector("Paper")
            module = self.render_module_selector("Module")
            chapter = self.render_chapter_selector("Chapter")
            unit = self.render_unit_selector("Unit")
        
        # Show current path
        if show_path and level:
            path = curriculum_manager.get_display_path(level, paper, module, chapter, unit)
            st.info(f"📍 **Current Path:** {path}")
        
        return {
            'level': level,
            'paper': paper,
            'module': module,
            'chapter': chapter,
            'unit': unit
        }
    
    def get_current_selection(self) -> Dict[str, Optional[str]]:
        """Get current curriculum selection"""
        return {
            'level': st.session_state.get(f"{self.prefix}_level"),
            'paper': st.session_state.get(f"{self.prefix}_paper"),
            'module': st.session_state.get(f"{self.prefix}_module"),
            'chapter': st.session_state.get(f"{self.prefix}_chapter"),
            'unit': st.session_state.get(f"{self.prefix}_unit")
        }
    
    def is_complete_selection(self) -> bool:
        """Check if we have a complete, valid selection for document upload"""
        selection = self.get_current_selection()
        level = selection['level']
        paper = selection['paper']
        module = selection['module']
        chapter = selection['chapter']
        unit = selection['unit']
        
        if not level or not paper:
            return False
        
        # Check if we need module but don't have it
        if curriculum_manager.has_modules(level, paper) and not module:
            return False
        
        # Must have chapter
        if not chapter:
            return False
        
        # Check if we need unit but don't have it
        if curriculum_manager.has_units(level, paper, chapter, module) and not unit:
            return False
        
        return True
    
    def get_missing_selections(self) -> List[str]:
        """Get list of missing required selections"""
        missing = []
        selection = self.get_current_selection()
        
        if not selection['level']:
            missing.append("CA Level")
        elif not selection['paper']:
            missing.append("Paper")
        else:
            level = selection['level']
            paper = selection['paper']
            module = selection['module']
            chapter = selection['chapter']
            
            # Check if module is required but missing
            if curriculum_manager.has_modules(level, paper) and not module:
                missing.append("Module")
            
            # Check if chapter is missing
            if not chapter:
                missing.append("Chapter")
            
            # Check if unit is required but missing
            if chapter and curriculum_manager.has_units(level, paper, chapter, module) and not selection['unit']:
                missing.append("Unit")
        
        return missing
    
    def render_selection_status(self):
        """Render current selection status with helpful feedback"""
        if self.is_complete_selection():
            st.success("✅ Complete curriculum selection - ready to proceed!")
        else:
            missing = self.get_missing_selections()
            if missing:
                st.warning(f"⚠️ Please select: {', '.join(missing)}")
    
    def clear_selection(self):
        """Clear all selections"""
        keys_to_clear = [
            f"{self.prefix}_level",
            f"{self.prefix}_paper", 
            f"{self.prefix}_module",
            f"{self.prefix}_chapter",
            f"{self.prefix}_unit"
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                st.session_state[key] = None

def render_curriculum_filter(prefix: str = "filter", title: str = "Filter by Curriculum",
                           show_clear: bool = True) -> Dict[str, Optional[str]]:
    """
    Render curriculum filter component for questions/search
    Returns current filter selection
    """
    selector = CurriculumSelector(prefix)
    
    with st.expander(title, expanded=False):
        selection = selector.render_complete_selector(title="", show_path=True, columns=True)
        
        if show_clear:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Clear Filters", key=f"{prefix}_clear"):
                    selector.clear_selection()
                    st.rerun()
    
    return selection

class SmartCurriculumSelector:
    """Enhanced curriculum selector with search and backward compatibility"""
    
    def __init__(self, prefix: str = "smart_curriculum"):
        """
        Initialize smart curriculum selector with search capabilities
        
        Args:
            prefix: Unique prefix for session state keys
        """
        self.prefix = prefix
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for smart curriculum selection"""
        default_values = {
            f"{self.prefix}_level": None,
            f"{self.prefix}_paper": None,
            f"{self.prefix}_module": None,
            f"{self.prefix}_chapter": None,
            f"{self.prefix}_unit": None,
            f"{self.prefix}_search_mode": False,
            f"{self.prefix}_search_term": "",
            f"{self.prefix}_search_results": []
        }
        
        for key, default in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = default
    
    def render_search_interface(self, title: str = "🔍 Smart Curriculum Search") -> Optional[Dict[str, str]]:
        """Render the smart search interface with autocomplete"""
        
        if title:
            st.subheader(title)
        
        # Toggle between search mode and traditional mode
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_mode = st.toggle(
                "Enable Smart Search",
                value=st.session_state.get(f"{self.prefix}_search_mode", False),
                key=f"{self.prefix}_search_toggle",
                help="Search across all curriculum levels and auto-fill hierarchy"
            )
            st.session_state[f"{self.prefix}_search_mode"] = search_mode
        
        with col2:
            if st.button("🗑️ Clear All", key=f"{self.prefix}_clear_all"):
                self._clear_all_selections()
                st.rerun()
        
        if search_mode:
            return self._render_search_mode()
        else:
            return self._render_traditional_mode()
    
    def _render_search_mode(self) -> Optional[Dict[str, str]]:
        """Render search mode with autocomplete"""
        
        st.markdown("**Type to search across all curriculum levels:**")
        
        # Search input with better examples
        search_term = st.text_input(
            "Search curriculum items",
            value=st.session_state.get(f"{self.prefix}_search_term", ""),
            placeholder="e.g., 'balance', 'tax', 'unit 1', 'audit', etc.",
            key=f"{self.prefix}_search_input",
            help="🔍 Smart Search Tips:\n• Type partial words: 'balance' finds 'Balance Sheet'\n• Multiple words: 'direct tax' finds 'Direct Tax Laws'\n• Numbers: 'unit 1' finds all Unit 1s\n• Just start typing - no need for exact names!"
        )
        
        # Show search examples
        with st.expander("💡 Search Examples", expanded=False):
            st.markdown("""
            **Try these search examples:**
            - `balance` → finds Balance Sheet, Trial Balance, etc.
            - `tax` → finds Taxation, Direct Tax Laws, Indirect Tax Laws
            - `audit` → finds Auditing, Advanced Auditing
            - `unit 1` → finds all Unit 1 across different chapters
            - `accounting` → finds Accounting, Advanced Accounting, etc.
            - `paper 1` → finds all Paper 1 across levels
            - `final` → finds Final level and related content
            
            **No need to type the full name - partial matching works!**
            """)
        
        # Update search term in session state
        st.session_state[f"{self.prefix}_search_term"] = search_term
        
        if search_term and len(search_term) >= 2:
            # Get search results
            search_results = curriculum_manager.find_item_hierarchy(search_term, 'all')
            st.session_state[f"{self.prefix}_search_results"] = search_results
            
            if search_results:
                st.markdown(f"**Found {len(search_results)} matches:**")
                
                # Group results by type for better organization
                results_by_type = {}
                for result in search_results:
                    result_type = result['type']
                    if result_type not in results_by_type:
                        results_by_type[result_type] = []
                    results_by_type[result_type].append(result)
                
                # Display results by type
                type_icons = {
                    'level': '🎓',
                    'paper': '📄',
                    'module': '📚',
                    'chapter': '📖',
                    'unit': '📝'
                }
                
                for result_type in ['unit', 'chapter', 'module', 'paper', 'level']:
                    if result_type in results_by_type:
                        with st.expander(f"{type_icons.get(result_type, '📋')} {result_type.title()}s ({len(results_by_type[result_type])})", expanded=(result_type == 'unit')):
                            for i, result in enumerate(results_by_type[result_type]):
                                col1, col2 = st.columns([4, 1])
                                
                                with col1:
                                    # Show relevance with visual indicator
                                    relevance = result.get('relevance', 0)
                                    relevance_indicator = "🎯" if relevance > 0.8 else "📍" if relevance > 0.6 else "📌"
                                    
                                    st.markdown(f"**{result['match']}** {relevance_indicator}")
                                    st.caption(f"📍 {result['display_path']}")
                                
                                with col2:
                                    if st.button("Select", key=f"{self.prefix}_select_{result_type}_{i}"):
                                        self._auto_fill_from_result(result)
                                        st.success(f"✅ Selected: {result['match']}")
                                        st.rerun()
            else:
                st.info("No matches found. Try different search terms.")
        
        elif search_term and len(search_term) < 2:
            st.info("Type at least 2 characters to search")
        
        # Show current selection if any
        current_selection = self._get_current_selection()
        if any(current_selection.values()):
            st.markdown("---")
            st.markdown("**📍 Current Selection:**")
            path = curriculum_manager.get_display_path(
                current_selection['level'],
                current_selection['paper'],
                current_selection['module'],
                current_selection['chapter'],
                current_selection['unit']
            )
            st.info(f"**{path}**")
            
            return current_selection
        
        return None
    
    def _render_traditional_mode(self) -> Optional[Dict[str, str]]:
        """Render traditional cascading dropdown mode"""
        
        st.markdown("**Traditional cascading selection:**")
        
        # Use the existing CurriculumSelector for traditional mode
        traditional_selector = CurriculumSelector(f"{self.prefix}_traditional")
        
        # Copy current selections to traditional selector
        current_selection = self._get_current_selection()
        for key, value in current_selection.items():
            if value:
                st.session_state[f"{self.prefix}_traditional_{key}"] = value
        
        selection = traditional_selector.render_complete_selector(
            title="",
            show_path=True,
            columns=True
        )
        
        # Copy back to smart selector
        for key, value in selection.items():
            st.session_state[f"{self.prefix}_{key}"] = value
        
        return selection
    
    def _auto_fill_from_result(self, result: Dict[str, Any]):
        """Auto-fill hierarchy from search result"""
        st.session_state[f"{self.prefix}_level"] = result['level']
        st.session_state[f"{self.prefix}_paper"] = result['paper']
        st.session_state[f"{self.prefix}_module"] = result['module']
        st.session_state[f"{self.prefix}_chapter"] = result['chapter']
        st.session_state[f"{self.prefix}_unit"] = result['unit']
    
    def _get_current_selection(self) -> Dict[str, Optional[str]]:
        """Get current selection"""
        return {
            'level': st.session_state.get(f"{self.prefix}_level"),
            'paper': st.session_state.get(f"{self.prefix}_paper"),
            'module': st.session_state.get(f"{self.prefix}_module"),
            'chapter': st.session_state.get(f"{self.prefix}_chapter"),
            'unit': st.session_state.get(f"{self.prefix}_unit")
        }
    
    def get_current_selection(self) -> Dict[str, Optional[str]]:
        """Public method to get current selection (for compatibility with CurriculumSelector)"""
        return self._get_current_selection()
    
    def _clear_all_selections(self):
        """Clear all selections"""
        keys_to_clear = [
            f"{self.prefix}_level",
            f"{self.prefix}_paper",
            f"{self.prefix}_module",
            f"{self.prefix}_chapter",
            f"{self.prefix}_unit",
            f"{self.prefix}_search_term",
            f"{self.prefix}_search_results"
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                st.session_state[key] = None if 'search_results' not in key else []
                if 'search_term' in key:
                    st.session_state[key] = ""
    
    def is_complete_selection(self) -> bool:
        """Check if we have a complete, valid selection"""
        selection = self._get_current_selection()
        level = selection['level']
        paper = selection['paper']
        module = selection['module']
        chapter = selection['chapter']
        unit = selection['unit']
        
        if not level or not paper:
            return False
        
        # Check if we need module but don't have it
        if curriculum_manager.has_modules(level, paper) and not module:
            return False
        
        # Must have chapter
        if not chapter:
            return False
        
        # Check if we need unit but don't have it
        if curriculum_manager.has_units(level, paper, chapter, module) and not unit:
            return False
        
        return True
    
    def get_missing_selections(self) -> List[str]:
        """Get list of missing required selections"""
        missing = []
        selection = self._get_current_selection()
        
        if not selection['level']:
            missing.append("CA Level")
        elif not selection['paper']:
            missing.append("Paper")
        else:
            level = selection['level']
            paper = selection['paper']
            module = selection['module']
            chapter = selection['chapter']
            
            # Check if module is required but missing
            if curriculum_manager.has_modules(level, paper) and not module:
                missing.append("Module")
            
            # Check if chapter is missing
            if not chapter:
                missing.append("Chapter")
            
            # Check if unit is required but missing
            if chapter and curriculum_manager.has_units(level, paper, chapter, module) and not selection['unit']:
                missing.append("Unit")
        
        return missing
    
    def render_selection_status(self):
        """Render current selection status with helpful feedback"""
        if self.is_complete_selection():
            st.success("✅ Complete curriculum selection - ready to proceed!")
        else:
            missing = self.get_missing_selections()
            if missing:
                st.warning(f"⚠️ Please select: {', '.join(missing)}")

def render_smart_curriculum_selector(prefix: str = "smart", title: str = "🔍 Smart Curriculum Selection") -> Dict[str, Optional[str]]:
    """
    Render smart curriculum selector with search and backward compatibility
    
    Args:
        prefix: Unique prefix for session state
        title: Title for the selector
    
    Returns:
        Current selection dictionary
    """
    selector = SmartCurriculumSelector(prefix)
    selection = selector.render_search_interface(title)
    
    # Show selection status
    selector.render_selection_status()
    
    return selection or selector._get_current_selection()