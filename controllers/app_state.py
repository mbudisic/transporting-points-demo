import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
from models.distribution import Distribution

class AppState:
    """
    Controller class that manages the application state
    """
    @staticmethod
    def initialize():
        """Initialize the application state in the Streamlit session state"""
        # Create main distributions if they don't exist
        import random
        
        if 'distribution_a' not in st.session_state:
            # Create distribution A with default blobs (Teal color)
            distribution_a = Distribution('A', '#009E73')  # Teal color
            # Add some default blobs with randomized positions and heights
            distribution_a.add_blob(
                x=3.0 + random.uniform(-0.5, 0.5), 
                y=3.0 + random.uniform(-0.5, 0.5), 
                variance=0.5, 
                height=1.0 + random.uniform(-0.2, 0.2)
            )
            distribution_a.add_blob(
                x=7.0 + random.uniform(-0.5, 0.5), 
                y=3.0 + random.uniform(-0.5, 0.5), 
                variance=0.5, 
                height=1.0 + random.uniform(-0.2, 0.2)
            )
            distribution_a.add_blob(
                x=5.0 + random.uniform(-0.5, 0.5), 
                y=7.0 + random.uniform(-0.5, 0.5), 
                variance=0.5, 
                height=-1.0 + random.uniform(-0.2, 0.2)
            )
            st.session_state.distribution_a = distribution_a
            
        if 'distribution_b' not in st.session_state:
            # Create distribution B with default blobs
            distribution_b = Distribution('B', '#E69F00')  # Orange color
            # Add some default blobs with randomized positions and heights
            distribution_b.add_blob(
                x=3.0 + random.uniform(-0.5, 0.5), 
                y=7.0 + random.uniform(-0.5, 0.5), 
                variance=0.5, 
                height=1.0 + random.uniform(-0.2, 0.2)
            )
            distribution_b.add_blob(
                x=7.0 + random.uniform(-0.5, 0.5), 
                y=7.0 + random.uniform(-0.5, 0.5), 
                variance=0.5, 
                height=1.0 + random.uniform(-0.2, 0.2)
            )
            distribution_b.add_blob(
                x=5.0 + random.uniform(-0.5, 0.5), 
                y=3.0 + random.uniform(-0.5, 0.5), 
                variance=0.5, 
                height=-1.0 + random.uniform(-0.2, 0.2)
            )
            st.session_state.distribution_b = distribution_b
        
        # Initialize state variables
        if 'active_distribution' not in st.session_state:
            st.session_state.active_distribution = 'A'
            
        if 'show_both' not in st.session_state:
            st.session_state.show_both = True
            
        # Spatial transport visualization flags
        if 'show_bottleneck_lines' not in st.session_state:
            st.session_state.show_bottleneck_lines = False
            
        if 'show_wasserstein_lines' not in st.session_state:
            st.session_state.show_wasserstein_lines = False
            
        # Height-based transport visualization flags
        if 'show_height_bottleneck_lines' not in st.session_state:
            st.session_state.show_height_bottleneck_lines = False
            
        if 'show_height_wasserstein_lines' not in st.session_state:
            st.session_state.show_height_wasserstein_lines = False
            
        # Spatial transport data
        if 'bottleneck_matching' not in st.session_state:
            st.session_state.bottleneck_matching = []
            
        if 'wasserstein_pairs' not in st.session_state:
            st.session_state.wasserstein_pairs = []
            
        # Height-based transport data
        if 'height_bottleneck_matching' not in st.session_state:
            st.session_state.height_bottleneck_matching = []
            
        if 'height_wasserstein_pairs' not in st.session_state:
            st.session_state.height_wasserstein_pairs = []
            
        # Visualization mode (standard or graph)
        if 'visualization_mode' not in st.session_state:
            st.session_state.visualization_mode = 'standard'
            
        if 'selected_element' not in st.session_state:
            st.session_state.selected_element = None
            
        # Contour plot visibility settings
        if 'show_contour_a' not in st.session_state:
            st.session_state.show_contour_a = False
            
        if 'show_contour_b' not in st.session_state:
            st.session_state.show_contour_b = False
    
    @staticmethod
    def get_distribution_a() -> Distribution:
        """Get distribution A from session state"""
        return st.session_state.distribution_a
    
    @staticmethod
    def get_distribution_b() -> Distribution:
        """Get distribution B from session state"""
        return st.session_state.distribution_b
    
    @staticmethod
    def toggle_active_distribution(dist_name: str):
        """Toggle the active distribution"""
        if dist_name in ['A', 'B']:
            st.session_state.active_distribution = dist_name
    
    @staticmethod
    def toggle_show_both():
        """Toggle whether to show both distributions"""
        st.session_state.show_both = not st.session_state.show_both
    
    @staticmethod
    def set_transport_visualization(mode: str):
        """Set which transport visualization to show"""
        # First reset all visualization modes
        st.session_state.show_bottleneck_lines = False
        st.session_state.show_wasserstein_lines = False
        st.session_state.show_height_bottleneck_lines = False
        st.session_state.show_height_wasserstein_lines = False
        
        # Then enable the selected one
        if mode == "hide":
            pass  # All already set to False
        elif mode == "bottleneck_spatial":
            st.session_state.show_bottleneck_lines = True
        elif mode == "wasserstein_spatial":
            st.session_state.show_wasserstein_lines = True
        elif mode == "bottleneck_height":
            st.session_state.show_height_bottleneck_lines = True
        elif mode == "wasserstein_height":
            st.session_state.show_height_wasserstein_lines = True
    
    @staticmethod
    def get_active_distribution() -> str:
        """Get the active distribution name"""
        return st.session_state.active_distribution
    
    @staticmethod
    def is_showing_both() -> bool:
        """Check if both distributions are being shown"""
        return st.session_state.show_both
    
    @staticmethod
    def is_showing_bottleneck() -> bool:
        """Check if bottleneck transport is being shown"""
        return st.session_state.show_bottleneck_lines
    
    @staticmethod
    def is_showing_wasserstein() -> bool:
        """Check if Wasserstein transport is being shown"""
        return st.session_state.show_wasserstein_lines
    
    @staticmethod
    def is_showing_height_bottleneck() -> bool:
        """Check if height-based bottleneck transport is being shown"""
        return st.session_state.show_height_bottleneck_lines
    
    @staticmethod
    def is_showing_height_wasserstein() -> bool:
        """Check if height-based Wasserstein transport is being shown"""
        return st.session_state.show_height_wasserstein_lines
    
    @staticmethod
    def store_bottleneck_matching(matching: List[Tuple[int, int]]):
        """Store bottleneck matching results"""
        st.session_state.bottleneck_matching = matching
    
    @staticmethod
    def store_wasserstein_pairs(pairs: List[Tuple[int, int, float]]):
        """Store Wasserstein transport plan results"""
        st.session_state.wasserstein_pairs = pairs
        
    @staticmethod
    def store_height_bottleneck_matching(matching: List[Tuple[int, int]]):
        """Store height-based bottleneck matching results"""
        st.session_state.height_bottleneck_matching = matching
    
    @staticmethod
    def store_height_wasserstein_pairs(pairs: List[Tuple[int, int, float]]):
        """Store height-based Wasserstein transport plan results"""
        st.session_state.height_wasserstein_pairs = pairs
    
    @staticmethod
    def get_bottleneck_matching():
        """Get bottleneck matching results"""
        return st.session_state.bottleneck_matching
    
    @staticmethod
    def get_wasserstein_pairs():
        """Get Wasserstein transport plan results"""
        return st.session_state.wasserstein_pairs
        
    @staticmethod
    def get_height_bottleneck_matching():
        """Get height-based bottleneck matching results"""
        return st.session_state.height_bottleneck_matching
    
    @staticmethod
    def get_height_wasserstein_pairs():
        """Get height-based Wasserstein transport plan results"""
        return st.session_state.height_wasserstein_pairs
    
    @staticmethod
    def set_selected_element(element_type: str, dist: str, element_id: int):
        """Set the selected element (blob)"""
        st.session_state.selected_element = {
            'type': element_type,
            'dist': dist,
            'id': element_id
        }
    
    @staticmethod
    def get_selected_element() -> Optional[Dict[str, Any]]:
        """Get the currently selected element"""
        return st.session_state.selected_element
    
    @staticmethod
    def clear_selected_element():
        """Clear the selected element"""
        st.session_state.selected_element = None
        
    @staticmethod
    def get_visualization_mode() -> str:
        """Get the current visualization mode (standard or graph)"""
        return st.session_state.visualization_mode
        
    @staticmethod
    def set_visualization_mode(mode: str):
        """Set the visualization mode (standard or graph)"""
        if mode in ['standard', 'graph']:
            st.session_state.visualization_mode = mode
    
    @staticmethod
    def is_showing_contour_a() -> bool:
        """Check if contour plot for distribution A is being shown"""
        return st.session_state.show_contour_a
    
    @staticmethod
    def is_showing_contour_b() -> bool:
        """Check if contour plot for distribution B is being shown"""
        return st.session_state.show_contour_b
    
    @staticmethod
    def toggle_contour_a():
        """Toggle visibility of contour plot for distribution A"""
        st.session_state.show_contour_a = not st.session_state.show_contour_a
    
    @staticmethod
    def toggle_contour_b():
        """Toggle visibility of contour plot for distribution B"""
        st.session_state.show_contour_b = not st.session_state.show_contour_b