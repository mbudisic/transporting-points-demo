import streamlit as st
import pandas as pd
import numpy as np
import base64
import io

# Import MVC components
from models.distribution import Distribution
from models.blob import Blob
from controllers.app_state import AppState
from controllers.distance_calculator import DistanceCalculator
from controllers.distribution_controller import DistributionController
from controllers.event_handler import EventHandler
from views.visualization import VisualizationService
from views.ui_components import UIComponents

# Set page configuration
st.set_page_config(
    page_title="Distribution Distance Visualizer",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS to make heading fonts only 2 points larger than body text
st.markdown("""
<style>
    /* Adjust heading sizes to be only 2pt larger than body text */
    h1 {
        font-size: 1.5rem !important;
    }
    h2 {
        font-size: 1.4rem !important;
    }
    h3 {
        font-size: 1.3rem !important;
    }
    h4 {
        font-size: 1.2rem !important;
    }
    h5 {
        font-size: 1.1rem !important;
    }
    
    /* Make distance metric values use heading font */
    .metric-value {
        font-family: "Source Sans Pro", sans-serif !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

def update_state():
    """Force an app state update by triggering a rerun"""
    st.rerun()

def main():
    # Display the app header
    st.title("Distribution Distance Visualizer")
    
    # Initialize application state
    AppState.initialize()
    
    # Create the three-column layout
    left_col, center_col, right_col = st.columns([1, 3, 1])
    
    # Get the distributions from app state
    distribution_a = AppState.get_distribution_a()
    distribution_b = AppState.get_distribution_b()
    
    # Render Distribution A controls in the left sidebar
    with left_col:
        UIComponents.render_sidebar_a(distribution_a, update_state)
    
    # Render the main content area
    with center_col:
        # Render plot and controls
        UIComponents.render_main_content(
            distribution_a,
            distribution_b,
            update_state,
            EventHandler.handle_plot_click,
            EventHandler.handle_drag_event,
            VisualizationService
        )
        
        # Render metrics
        UIComponents.render_metrics(
            distribution_a,
            distribution_b,
            DistanceCalculator,
            update_state
        )
        
        # Render import/export section
        UIComponents.render_import_export(
            distribution_a,
            distribution_b,
            DistributionController,
            update_state
        )
    
    # Render Distribution B controls in the right sidebar
    with right_col:
        UIComponents.render_sidebar_b(distribution_b, update_state)

if __name__ == "__main__":
    main()