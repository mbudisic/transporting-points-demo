import streamlit as st
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from controllers.app_state import AppState
from models.distribution import Distribution
from controllers.distribution_controller import DistributionController

class EventHandler:
    """
    Controller class for handling user interaction events
    """
    @staticmethod
    def handle_plot_click(trace: Any, points: Any, state: Any) -> None:
        """
        Handle click events on the plot
        
        Args:
            trace: The trace that was clicked
            points: Information about the clicked point
            state: Additional state information
        """
        if not points.point_inds:  # No point was clicked
            return
        
        point_index = points.point_inds[0]  # Get the first clicked point
        
        # Check if the clicked point has custom data
        if 'customdata' in points.trace and points.trace.customdata is not None:
            customdata = points.trace.customdata[point_index]
            
            # Blob ID is the first item in customdata for blob points
            if len(customdata) == 5:  # Blob point with [id, variance, height, sign, distribution]
                blob_id = int(customdata[0])
                dist_name = customdata[4]  # 'A' or 'B'
                
                # Set as selected element
                AppState.set_selected_element('center', dist_name, blob_id)
                
                # Force a rerun to update the UI
                st.rerun()
    
    @staticmethod
    def handle_drag_event(trace: Any, points: Any, state: Any) -> None:
        """
        This function handles drag events for blob movement
        
        Args:
            trace: The trace that was dragged
            points: Information about the dragged point
            state: Additional state information
        """
        if not points.point_inds:
            return
        
        point_index = points.point_inds[0]
        
        # Check if the dragged point has custom data (blob points do)
        if 'customdata' in points.trace and points.trace.customdata is not None:
            customdata = points.trace.customdata[point_index]
            
            # Only proceed if this is a blob center point
            if len(customdata) == 5:  # [id, variance, height, sign, distribution]
                blob_id = int(customdata[0])
                dist_name = customdata[4]  # 'A' or 'B'
                
                # Get x, y coordinates of the drag end point
                x = points.xs[point_index]
                y = points.ys[point_index]
                
                # Get the corresponding distribution
                if dist_name == 'A':
                    distribution = AppState.get_distribution_a()
                else:
                    distribution = AppState.get_distribution_b()
                
                # Update the blob position
                DistributionController.update_blob(distribution, blob_id, x=x, y=y)
                
                # Set as selected element
                AppState.set_selected_element('center', dist_name, blob_id)
                
                # Force a rerun to update the UI
                st.rerun()