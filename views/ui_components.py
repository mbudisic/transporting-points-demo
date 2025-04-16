import streamlit as st
import pandas as pd
import numpy as np
from models.distribution import Distribution
from controllers.distribution_controller import DistributionController
from controllers.app_state import AppState
from typing import Callable, Dict, Any, List

class UIComponents:
    """
    Class containing UI components for the application
    """
    @staticmethod
    def render_sidebar_a(distribution: Distribution, on_update: Callable):
        """
        Render the left sidebar for Distribution A
        
        Args:
            distribution: Distribution A
            on_update: Callback function to call when the distribution is updated
        """
        st.subheader("Distribution A (Teal)")
        
        # Create columns for Add/Remove buttons
        add_col, remove_col = st.columns(2)
        
        # Add new blob button
        with add_col:
            if st.button("Add Blob to A"):
                DistributionController.add_blob(distribution)
                on_update()
        
        # Display table for distribution A
        st.markdown("#### Distribution A Properties", help="View and edit blob properties")
        
        # Create DataFrame for displaying blob data
        blob_data = distribution.get_data_dicts()
        df = pd.DataFrame(blob_data)
        
        if not df.empty:
            # Show data editor for property display and editing
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed", 
                disabled=["id"],
                key="data_editor_a",
                column_config={
                    "id": st.column_config.NumberColumn("ID", min_value=0, format="%d"),
                    "x": st.column_config.NumberColumn("X", min_value=0, max_value=10, format="%.2f"),
                    "y": st.column_config.NumberColumn("Y", min_value=0, max_value=10, format="%.2f"),
                    "variance": st.column_config.NumberColumn("Variance", min_value=0.1, max_value=5, format="%.2f"),
                    "height": st.column_config.NumberColumn("Height", min_value=-10, max_value=10, format="%.2f"),
                    "sign": st.column_config.SelectboxColumn("Sign", options=[1, -1])
                }
            )
            
            # Add a selectbox to choose which blob to edit
            st.markdown("#### Select Blob", help="Choose a blob to manipulate with sliders")
            
            # Create a mapping of blob IDs to display names
            blob_id_map = {f"Blob A{blob['id']}": blob['id'] for blob in blob_data}
            blob_options = list(blob_id_map.keys())
            
            if blob_options:
                # Initialize the session state variable for selection if it doesn't exist
                if "selected_blob_a" not in st.session_state:
                    st.session_state.selected_blob_a = blob_options[0] if blob_options else None
                
                # Create function to handle selection change
                def on_blob_selection_change_a():
                    # Get the selected blob from session state
                    selected_blob = st.session_state.blob_selector_a
                    if selected_blob:
                        # Get the ID directly from our mapping
                        selected_id = blob_id_map[selected_blob]
                        # Update the global selected element
                        AppState.set_selected_element('center', 'A', selected_id)
                        # Don't call on_update() here to avoid recursive rerun
                
                # Find current selection index if there is one
                selected_element = AppState.get_selected_element()
                selected_index = 0
                
                # Update session state based on current app state
                if (selected_element is not None and selected_element['dist'] == 'A'):
                    for i, opt in enumerate(blob_options):
                        if f"A{selected_element['id']}" in opt:
                            selected_index = i
                            st.session_state.selected_blob_a = blob_options[i]
                            break
                
                # Use the selectbox with on_change handler
                selected_blob = st.selectbox(
                    "Select a blob to manipulate:",
                    options=blob_options,
                    index=selected_index,
                    key="blob_selector_a",
                    on_change=on_blob_selection_change_a
                )
            
            # Remove selected blob button
            with remove_col:
                if st.button("Remove Selected Blob", key="remove_selected_a"):
                    selected_element = AppState.get_selected_element()
                    if selected_element and selected_element['dist'] == 'A':
                        DistributionController.remove_blob(distribution, selected_element['id'])
                        AppState.clear_selected_element()
                        on_update()
                    
            # Add sliders for direct manipulation
            st.markdown("### Direct Controls")
            
            # Get the current blob if any is selected
            selected_element = AppState.get_selected_element()
            if selected_element and selected_element['dist'] == 'A':
                blob_id = selected_element['id']
                blob = distribution.get_blob(blob_id)
                
                if blob:
                    st.write(f"**Selected Blob: A{blob_id}** (Height: {blob.height:.2f})")
                    
                    # Row-based controls with labels aligned with their sliders
                    st.markdown("#### Blob Controls")
                    
                    # X Position - in a row with label
                    x_row = st.container()
                    x_label, x_slider = x_row.columns([1, 3])
                    with x_label:
                        st.markdown("X:")
                    with x_slider:
                        # Create the slider with default value
                        new_x = st.slider(
                            "##", 
                            min_value=0.0, 
                            max_value=10.0, 
                            value=float(blob.x), 
                            step=0.1, 
                            key=f"x_slider_A{blob_id}",
                            on_change=lambda: DistributionController.update_blob(
                                distribution, blob_id, 
                                x=st.session_state.get(f"x_slider_A{blob_id}", blob.x)
                            )
                        )
                    
                    # Y Position - in a row with label
                    y_row = st.container()
                    y_label, y_slider = y_row.columns([1, 3])
                    with y_label:
                        st.markdown("Y:")
                    with y_slider:
                        # Create slider with default value
                        new_y = st.slider(
                            "##", 
                            min_value=0.0, 
                            max_value=10.0, 
                            value=float(blob.y), 
                            step=0.1,
                            key=f"y_slider_A{blob_id}",
                            on_change=lambda: DistributionController.update_blob(
                                distribution, blob_id, 
                                y=st.session_state.get(f"y_slider_A{blob_id}", blob.y)
                            )
                        )
                    
                    # Variance control - in a row with label
                    var_row = st.container()
                    var_label, var_slider = var_row.columns([1, 3])
                    with var_label:
                        st.markdown("Var:")
                    with var_slider:
                        # Create slider with default value
                        new_variance = st.slider(
                            "##", 
                            min_value=0.1, 
                            max_value=5.0, 
                            value=float(blob.variance), 
                            step=0.1,
                            key=f"var_slider_A{blob_id}",
                            on_change=lambda: DistributionController.update_blob(
                                distribution, blob_id, 
                                variance=st.session_state.get(f"var_slider_A{blob_id}", blob.variance)
                            )
                        )
                    
                    # Height control (double-sided slider) - in a row with label
                    height_row = st.container()
                    height_label, height_slider = height_row.columns([1, 3])
                    with height_label:
                        st.markdown("Height:")
                    with height_slider:
                        # Create height slider that allows for negative values (double-sided)
                        new_height = st.slider(
                            "##", 
                            min_value=-10.0, 
                            max_value=10.0, 
                            value=float(blob.height), 
                            step=0.1,
                            key=f"height_slider_A{blob_id}",
                            on_change=lambda: DistributionController.update_blob(
                                distribution, blob_id, 
                                height=st.session_state.get(f"height_slider_A{blob_id}", blob.height)
                            )
                        )
                else:
                    st.warning("No blob selected. Use the dropdown menu to select a blob.")
            else:
                st.info("No blob selected. Use the dropdown menu to select a blob.")
                
            # Update distribution based on edited dataframe
            for index, row in edited_df.iterrows():
                original_row = df.iloc[index]
                if not pd.DataFrame([row.values], columns=row.index).equals(
                   pd.DataFrame([original_row.values], columns=original_row.index)):
                    DistributionController.update_blob(
                        distribution, int(row['id']), 
                        row['x'], row['y'], row['variance'], row['height'], row['sign']
                    )
                    on_update()
    
    @staticmethod
    def render_sidebar_b(distribution: Distribution, on_update: Callable):
        """
        Render the right sidebar for Distribution B
        
        Args:
            distribution: Distribution B
            on_update: Callback function to call when the distribution is updated
        """
        st.subheader("Distribution B (Orange)")
        
        # Create columns for Add/Remove buttons
        add_col, remove_col = st.columns(2)
        
        # Add new blob button
        with add_col:
            if st.button("Add Blob to B"):
                DistributionController.add_blob(distribution)
                on_update()
        
        # Display table for distribution B
        st.markdown("#### Distribution B Properties", help="View and edit blob properties")
        
        # Create DataFrame for displaying blob data
        blob_data = distribution.get_data_dicts()
        df = pd.DataFrame(blob_data)
        
        if not df.empty:
            # Show data editor for property display and editing
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed", 
                disabled=["id"],
                key="data_editor_b",
                column_config={
                    "id": st.column_config.NumberColumn("ID", min_value=0, format="%d"),
                    "x": st.column_config.NumberColumn("X", min_value=0, max_value=10, format="%.2f"),
                    "y": st.column_config.NumberColumn("Y", min_value=0, max_value=10, format="%.2f"),
                    "variance": st.column_config.NumberColumn("Variance", min_value=0.1, max_value=5, format="%.2f"),
                    "height": st.column_config.NumberColumn("Height", min_value=-10, max_value=10, format="%.2f"),
                    "sign": st.column_config.SelectboxColumn("Sign", options=[1, -1])
                }
            )
            
            # Add a selectbox to choose which blob to edit
            st.markdown("#### Select Blob", help="Choose a blob to manipulate with sliders")
            
            # Create a mapping of blob IDs to display names
            blob_id_map = {f"Blob B{blob['id']}": blob['id'] for blob in blob_data}
            blob_options = list(blob_id_map.keys())
            
            if blob_options:
                # Initialize the session state variable for selection if it doesn't exist
                if "selected_blob_b" not in st.session_state:
                    st.session_state.selected_blob_b = blob_options[0] if blob_options else None
                
                # Create function to handle selection change
                def on_blob_selection_change_b():
                    # Get the selected blob from session state
                    selected_blob = st.session_state.blob_selector_b
                    if selected_blob:
                        # Get the ID directly from our mapping
                        selected_id = blob_id_map[selected_blob]
                        # Update the global selected element
                        AppState.set_selected_element('center', 'B', selected_id)
                        # Don't call on_update() here to avoid recursive rerun
                
                # Find current selection index if there is one
                selected_element = AppState.get_selected_element()
                selected_index = 0
                
                # Update session state based on current app state
                if (selected_element is not None and selected_element['dist'] == 'B'):
                    for i, opt in enumerate(blob_options):
                        if f"B{selected_element['id']}" in opt:
                            selected_index = i
                            st.session_state.selected_blob_b = blob_options[i]
                            break
                
                # Use the selectbox with on_change handler
                selected_blob = st.selectbox(
                    "Select a blob to manipulate:",
                    options=blob_options,
                    index=selected_index,
                    key="blob_selector_b",
                    on_change=on_blob_selection_change_b
                )
            
            # Remove selected blob button
            with remove_col:
                if st.button("Remove Selected Blob", key="remove_selected_b"):
                    selected_element = AppState.get_selected_element()
                    if selected_element and selected_element['dist'] == 'B':
                        DistributionController.remove_blob(distribution, selected_element['id'])
                        AppState.clear_selected_element()
                        on_update()
                    
            # Add sliders for direct manipulation
            st.markdown("### Direct Controls")
            
            # Get the current blob if any is selected
            selected_element = AppState.get_selected_element()
            if selected_element and selected_element['dist'] == 'B':
                blob_id = selected_element['id']
                blob = distribution.get_blob(blob_id)
                
                if blob:
                    st.write(f"**Selected Blob: B{blob_id}** (Height: {blob.height:.2f})")
                    
                    # Row-based controls with labels aligned with their sliders
                    st.markdown("#### Blob Controls")
                    
                    # X Position - in a row with label
                    x_row = st.container()
                    x_label, x_slider = x_row.columns([1, 3])
                    with x_label:
                        st.markdown("X:")
                    with x_slider:
                        # Create slider with default value
                        new_x = st.slider(
                            "##", 
                            min_value=0.0, 
                            max_value=10.0, 
                            value=float(blob.x), 
                            step=0.1,
                            key=f"x_slider_B{blob_id}",
                            on_change=lambda: DistributionController.update_blob(
                                distribution, blob_id, 
                                x=st.session_state.get(f"x_slider_B{blob_id}", blob.x)
                            )
                        )
                    
                    # Y Position - in a row with label
                    y_row = st.container()
                    y_label, y_slider = y_row.columns([1, 3])
                    with y_label:
                        st.markdown("Y:")
                    with y_slider:
                        # Create slider with default value
                        new_y = st.slider(
                            "##", 
                            min_value=0.0, 
                            max_value=10.0, 
                            value=float(blob.y), 
                            step=0.1,
                            key=f"y_slider_B{blob_id}",
                            on_change=lambda: DistributionController.update_blob(
                                distribution, blob_id, 
                                y=st.session_state.get(f"y_slider_B{blob_id}", blob.y)
                            )
                        )
                    
                    # Variance control - in a row with label
                    var_row = st.container()
                    var_label, var_slider = var_row.columns([1, 3])
                    with var_label:
                        st.markdown("Var:")
                    with var_slider:
                        # Create slider with default value
                        new_variance = st.slider(
                            "##", 
                            min_value=0.1, 
                            max_value=5.0, 
                            value=float(blob.variance), 
                            step=0.1,
                            key=f"var_slider_B{blob_id}",
                            on_change=lambda: DistributionController.update_blob(
                                distribution, blob_id, 
                                variance=st.session_state.get(f"var_slider_B{blob_id}", blob.variance)
                            )
                        )
                    
                    # Height control (double-sided slider) - in a row with label
                    height_row = st.container()
                    height_label, height_slider = height_row.columns([1, 3])
                    with height_label:
                        st.markdown("Height:")
                    with height_slider:
                        # Create height slider that allows for negative values (double-sided)
                        new_height = st.slider(
                            "##", 
                            min_value=-10.0, 
                            max_value=10.0, 
                            value=float(blob.height), 
                            step=0.1,
                            key=f"height_slider_B{blob_id}",
                            on_change=lambda: DistributionController.update_blob(
                                distribution, blob_id, 
                                height=st.session_state.get(f"height_slider_B{blob_id}", blob.height)
                            )
                        )
                else:
                    st.warning("No blob selected. Use the dropdown menu to select a blob.")
            else:
                st.info("No blob selected. Use the dropdown menu to select a blob.")
                
            # Update distribution based on edited dataframe
            for index, row in edited_df.iterrows():
                original_row = df.iloc[index]
                if not pd.DataFrame([row.values], columns=row.index).equals(
                   pd.DataFrame([original_row.values], columns=original_row.index)):
                    DistributionController.update_blob(
                        distribution, int(row['id']), 
                        row['x'], row['y'], row['variance'], row['height'], row['sign']
                    )
                    on_update()
    
    @staticmethod
    def render_main_content(
        distribution_a: Distribution, 
        distribution_b: Distribution,
        on_update: Callable,
        handle_plot_click: Callable,
        handle_drag_event: Callable,
        visualization_service,
        calculator
    ):
        """
        Render the main content area with the plot and distance metrics
        
        Args:
            distribution_a: Distribution A
            distribution_b: Distribution B
            on_update: Callback function to call when state changes
            handle_plot_click: Function to handle plot click events
            handle_drag_event: Function to handle plot drag events
            visualization_service: Service for creating visualizations
            calculator: Calculator for computing distances
        """
        st.title("Distribution Distance Visualization")
        
        # Determine which distributions to show
        show_a = not distribution_a.is_empty
        show_b = not distribution_b.is_empty
        
        # First, show transport visualization selector (moved outside tabs)
        # Dropdown for selecting transport visualization mode
        current_plan = "hide"
        if AppState.is_showing_bottleneck():
            current_plan = "bottleneck_spatial"
        elif AppState.is_showing_wasserstein():
            current_plan = "wasserstein_spatial"
        elif AppState.is_showing_height_bottleneck():
            current_plan = "bottleneck_height"
        elif AppState.is_showing_height_wasserstein():
            current_plan = "wasserstein_height"
        
        transport_options = {
            "hide": "Hide All Transport",
            "bottleneck_spatial": "Spatial Bottleneck",
            "wasserstein_spatial": "Spatial Wasserstein",
            "bottleneck_height": "Height-Based Bottleneck",
            "wasserstein_height": "Height-Based Wasserstein"
        }
        
        st.markdown("### Transport Plan Selection")
        selected_transport = st.selectbox(
            "Transport Visualization",
            options=list(transport_options.keys()),
            format_func=lambda x: transport_options[x],
            index=list(transport_options.keys()).index(current_plan),
            key="transport_plan_selector"
        )
        
        # Apply the selection
        if selected_transport == "bottleneck_spatial":
            AppState.set_transport_visualization("bottleneck_spatial")
        elif selected_transport == "wasserstein_spatial":
            AppState.set_transport_visualization("wasserstein_spatial")
        elif selected_transport == "bottleneck_height":
            AppState.set_transport_visualization("bottleneck_height")
        elif selected_transport == "wasserstein_height":
            AppState.set_transport_visualization("wasserstein_height")
        else:
            AppState.set_transport_visualization("hide")
        
        # Display explanation for the selected transport mode
        if selected_transport == "bottleneck_spatial":
            st.info("Spatial Bottleneck: Minimizes the maximum distance between any paired points.")
        elif selected_transport == "wasserstein_spatial":
            st.info("Spatial Wasserstein: Minimizes the overall transportation cost based on spatial positions.")
        elif selected_transport == "bottleneck_height":
            st.info("Height-Based Bottleneck: Pairs blobs to minimize the maximum height difference, ignoring positions.")
        elif selected_transport == "wasserstein_height":
            st.info("Height-Based Wasserstein: Optimal transport based only on blob heights, ignoring positions.")
        
        # Add tabs for distribution visualization and distances
        viz_tab, distance_tab = st.tabs(["Visualization", "Distances"])
        
        with viz_tab:
            st.subheader("Visual Representation")
            
            # Display a message if no distributions are created
            if not show_a and not show_b:
                st.info("Use the sidebars to add blobs to the distributions.")
            
            # Visualization mode selection
            mode = AppState.get_visualization_mode()
            
            # Create the appropriate visualization
            if show_a or show_b:
                # Create the standard visualization
                fig = visualization_service.create_standard_visualization(
                    dist_a=distribution_a if show_a else None,
                    dist_b=distribution_b if show_b else None,
                    show_both=AppState.is_showing_both(),
                )
                
                # Get matching information
                wasserstein_pairs = AppState.get_wasserstein_pairs()
                bottleneck_pairs = AppState.get_bottleneck_matching()
                height_wasserstein_pairs = AppState.get_height_wasserstein_pairs()
                height_bottleneck_pairs = AppState.get_height_bottleneck_matching()
                
                # Add transportation plan connections based on selected mode
                if AppState.is_showing_bottleneck() and bottleneck_pairs:
                    fig = visualization_service.add_transport_plan_to_figure(
                        fig, distribution_a, distribution_b, bottleneck_pairs, mode='bottleneck'
                    )
                    
                elif AppState.is_showing_wasserstein() and wasserstein_pairs:
                    fig = visualization_service.add_transport_plan_to_figure(
                        fig, distribution_a, distribution_b, wasserstein_pairs, mode='wasserstein'
                    )
                
                elif AppState.is_showing_height_bottleneck() and height_bottleneck_pairs:
                    fig = visualization_service.add_transport_plan_to_figure(
                        fig, distribution_a, distribution_b, height_bottleneck_pairs, mode='bottleneck'
                    )
                    
                elif AppState.is_showing_height_wasserstein() and height_wasserstein_pairs:
                    fig = visualization_service.add_transport_plan_to_figure(
                        fig, distribution_a, distribution_b, height_wasserstein_pairs, mode='wasserstein'
                    )
                
                # Display the figure with a fixed height
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
                
                # Add explanation for the visualization
                with st.expander("Visualization Explanation"):
                    st.markdown("""
                    ### How to Read This Chart
                    
                    - **Blobs**: Each blob represents a Gaussian distribution centered at (x,y) with a variance (spread).
                    - **Colors**: Teal circles (Distribution A) and Orange circles (Distribution B).
                    - **Circle Size**: The absolute value of the height (larger height = larger circle).
                    - **Line Style**: Dashed circles represent positive heights, dotted circles represent negative heights.
                    - **Connecting Lines**: When showing a transport plan, lines connect matching blobs.
                    - **Line Thickness**: For Wasserstein transport, thicker lines indicate higher weight transport.
                    """)
        
        # Display distance metrics and matrices
        with distance_tab:
            # Display metrics 
            UIComponents.render_metrics(distribution_a, distribution_b, calculator, on_update)
            
            # Display distance matrices 
            UIComponents.render_distance_matrices(
                distribution_a, distribution_b, calculator, selected_transport
            )
    
    @staticmethod
    def render_distance_matrices(distribution_a: Distribution, distribution_b: Distribution, 
                               calculator, current_transport_mode: str):
        """
        Render tables of all-to-all distances between blobs based on the selected transport mode
        
        Args:
            distribution_a: Distribution A
            distribution_b: Distribution B
            calculator: Calculator for computing distances
            current_transport_mode: Current transport visualization mode
                (bottleneck_spatial, wasserstein_spatial, bottleneck_height, wasserstein_height, hide)
        """
        st.subheader("Distance Matrices")
        
        has_a = not distribution_a.is_empty
        has_b = not distribution_b.is_empty
        
        if not has_a or not has_b:
            st.warning("Add blobs to both distributions to see distance matrices.")
            return
        
        # Skip if no transport visualization is selected
        if current_transport_mode == "hide":
            st.info("Select a transport plan visualization to see the corresponding distance matrices.")
            return
        
        # Get the matching information for highlighting
        highlight_pairs = []
        if current_transport_mode == "bottleneck_spatial":
            highlight_pairs = AppState.get_bottleneck_matching()
        elif current_transport_mode == "wasserstein_spatial":
            highlight_pairs = [(a, b) for a, b, _ in AppState.get_wasserstein_pairs()]
        elif current_transport_mode == "bottleneck_height":
            highlight_pairs = AppState.get_height_bottleneck_matching()
        elif current_transport_mode == "wasserstein_height":
            highlight_pairs = [(a, b) for a, b, _ in AppState.get_height_wasserstein_pairs()]
        
        # Separate blobs by sign for separate tables
        pos_blobs_a = [b for b in distribution_a.blobs if b.height > 0]
        neg_blobs_a = [b for b in distribution_a.blobs if b.height < 0]
        pos_blobs_b = [b for b in distribution_b.blobs if b.height > 0]
        neg_blobs_b = [b for b in distribution_b.blobs if b.height < 0]
        
        # Show matrices only if we have blobs of the respective sign
        if pos_blobs_a and pos_blobs_b:
            # Create mapping of blob IDs to indices
            blob_a_id_to_idx = {blob.id: i for i, blob in enumerate(pos_blobs_a)}
            blob_b_id_to_idx = {blob.id: i for i, blob in enumerate(pos_blobs_b)}
            
            # Create distance matrix for positive-to-positive blobs
            distance_matrix = np.zeros((len(pos_blobs_a), len(pos_blobs_b)))
            
            # Compute distances based on the selected mode
            if current_transport_mode in ['bottleneck_spatial', 'wasserstein_spatial']:
                # For spatial distance, use Euclidean distance
                for i, blob_a in enumerate(pos_blobs_a):
                    for j, blob_b in enumerate(pos_blobs_b):
                        distance_matrix[i, j] = np.sqrt((blob_a.x - blob_b.x)**2 + (blob_a.y - blob_b.y)**2)
            else:
                # For height-based distance, use absolute difference in heights
                for i, blob_a in enumerate(pos_blobs_a):
                    for j, blob_b in enumerate(pos_blobs_b):
                        distance_matrix[i, j] = abs(blob_a.height - blob_b.height)
            
            # Create DataFrame column/row labels with property information
            if current_transport_mode in ['bottleneck_spatial', 'wasserstein_spatial']:
                # For spatial distance, include coordinates in labels
                col_labels = [f"B{b.id} [{b.x:.2f}, {b.y:.2f}]" for b in pos_blobs_b]
                row_labels = [f"A{b.id} [{b.x:.2f}, {b.y:.2f}]" for b in pos_blobs_a]
            else:
                # For height-based distance, include height values in labels
                col_labels = [f"B{b.id} [{b.height:.2f}]" for b in pos_blobs_b]
                row_labels = [f"A{b.id} [{b.height:.2f}]" for b in pos_blobs_a]
                
            # Create the DataFrame for display
            pos_df = pd.DataFrame(distance_matrix, columns=col_labels, index=row_labels)
            
            # Format the values to 4 decimal places
            pos_df = pos_df.round(4)
            
            # Display the table header
            st.markdown("**Positive-to-Positive Blob Distances:**")
            
            # Create a list to track which cells should be highlighted
            highlighted_cells = []
            for blob_a_id, blob_b_id in highlight_pairs:
                # Check if both IDs are in our positive blob collections
                if (blob_a_id in blob_a_id_to_idx and blob_b_id in blob_b_id_to_idx):
                    # Get the matrix indices for these blob IDs
                    row_idx = blob_a_id_to_idx[blob_a_id]
                    col_idx = blob_b_id_to_idx[blob_b_id]
                    highlighted_cells.append((row_idx, col_idx))
            
            # Display the table with custom HTML to highlight matching pairs
            html_table = "<table style='width:100%; border-collapse: collapse;'>"
            
            # Add header row
            html_table += "<tr><th></th>"
            for col in col_labels:
                html_table += f"<th style='border: 1px solid gray; padding: 8px; text-align: center;'>{col}</th>"
            html_table += "</tr>"
            
            # Add data rows
            for i, row_idx in enumerate(row_labels):
                html_table += f"<tr><th style='border: 1px solid gray; padding: 8px; text-align: center;'>{row_idx}</th>"
                
                for j in range(len(col_labels)):
                    # Check if this cell should be highlighted (matched pair)
                    is_highlighted = (i, j) in highlighted_cells
                    
                    # Apply appropriate style
                    if is_highlighted:
                        cell_style = "background-color: rgba(255, 255, 0, 0.3); font-weight: bold;"
                    else:
                        cell_style = ""
                    
                    # Get the value and format it
                    cell_value = pos_df.iloc[i, j]
                    
                    # Add the cell with the value
                    html_table += f"<td style='border: 1px solid gray; padding: 8px; text-align: center; {cell_style}'>{cell_value}</td>"
                
                html_table += "</tr>"
            
            html_table += "</table>"
            
            # Display the HTML table
            st.markdown(html_table, unsafe_allow_html=True)
        
        # Show negative-to-negative table if we have negative blobs
        if neg_blobs_a and neg_blobs_b:
            # Create mapping of blob IDs to indices
            blob_a_id_to_idx = {blob.id: i for i, blob in enumerate(neg_blobs_a)}
            blob_b_id_to_idx = {blob.id: i for i, blob in enumerate(neg_blobs_b)}
            
            # Create distance matrix for negative-to-negative blobs
            distance_matrix = np.zeros((len(neg_blobs_a), len(neg_blobs_b)))
            
            # Compute distances based on the selected mode
            if current_transport_mode in ['bottleneck_spatial', 'wasserstein_spatial']:
                # For spatial distance, use Euclidean distance
                for i, blob_a in enumerate(neg_blobs_a):
                    for j, blob_b in enumerate(neg_blobs_b):
                        distance_matrix[i, j] = np.sqrt((blob_a.x - blob_b.x)**2 + (blob_a.y - blob_b.y)**2)
            else:
                # For height-based distance, use absolute difference in heights
                for i, blob_a in enumerate(neg_blobs_a):
                    for j, blob_b in enumerate(neg_blobs_b):
                        distance_matrix[i, j] = abs(blob_a.height - blob_b.height)
                        
            # Create DataFrame column/row labels with property information
            if current_transport_mode in ['bottleneck_spatial', 'wasserstein_spatial']:
                # For spatial distance, include coordinates in labels
                col_labels = [f"B{b.id} [{b.x:.2f}, {b.y:.2f}]" for b in neg_blobs_b]
                row_labels = [f"A{b.id} [{b.x:.2f}, {b.y:.2f}]" for b in neg_blobs_a]
            else:
                # For height-based distance, include height values in labels
                col_labels = [f"B{b.id} [{b.height:.2f}]" for b in neg_blobs_b]
                row_labels = [f"A{b.id} [{b.height:.2f}]" for b in neg_blobs_a]
                
            # Create the DataFrame for display
            neg_df = pd.DataFrame(distance_matrix, columns=col_labels, index=row_labels)
            
            # Format the values to 4 decimal places
            neg_df = neg_df.round(4)
            
            # Display the table header
            st.markdown("**Negative-to-Negative Blob Distances:**")
            
            # Create a list to track which cells should be highlighted
            highlighted_cells = []
            for blob_a_id, blob_b_id in highlight_pairs:
                # Check if both IDs are in our negative blob collections
                if (blob_a_id in blob_a_id_to_idx and blob_b_id in blob_b_id_to_idx):
                    # Get the matrix indices for these blob IDs
                    row_idx = blob_a_id_to_idx[blob_a_id]
                    col_idx = blob_b_id_to_idx[blob_b_id]
                    highlighted_cells.append((row_idx, col_idx))
            
            # Display the table with custom HTML to highlight matching pairs
            html_table = "<table style='width:100%; border-collapse: collapse;'>"
            
            # Add header row
            html_table += "<tr><th></th>"
            for col in col_labels:
                html_table += f"<th style='border: 1px solid gray; padding: 8px; text-align: center;'>{col}</th>"
            html_table += "</tr>"
            
            # Add data rows
            for i, row_idx in enumerate(row_labels):
                html_table += f"<tr><th style='border: 1px solid gray; padding: 8px; text-align: center;'>{row_idx}</th>"
                
                for j in range(len(col_labels)):
                    # Check if this cell should be highlighted (matched pair)
                    is_highlighted = (i, j) in highlighted_cells
                    
                    # Apply appropriate style
                    if is_highlighted:
                        cell_style = "background-color: rgba(255, 255, 0, 0.3); font-weight: bold;"
                    else:
                        cell_style = ""
                    
                    # Get the value and format it
                    cell_value = neg_df.iloc[i, j]
                    
                    # Add the cell with the value
                    html_table += f"<td style='border: 1px solid gray; padding: 8px; text-align: center; {cell_style}'>{cell_value}</td>"
                
                html_table += "</tr>"
            
            html_table += "</table>"
            
            # Display the HTML table
            st.markdown(html_table, unsafe_allow_html=True)
    
    @staticmethod
    def render_metrics(distribution_a: Distribution, distribution_b: Distribution, calculator, on_update: Callable):
        """
        Render distance metrics between the distributions
        
        Args:
            distribution_a: Distribution A
            distribution_b: Distribution B
            calculator: Calculator for computing distances
            on_update: Callback function for state updates
        """
        has_a = not distribution_a.is_empty
        has_b = not distribution_b.is_empty
        
        if has_a and has_b:
            # Calculate all distances
            wasserstein_continuous = calculator.calculate_wasserstein_continuous(distribution_a, distribution_b)
            wasserstein_discrete, wasserstein_pairs = calculator.calculate_wasserstein_plan(distribution_a, distribution_b)
            bottleneck_value, bottleneck_pairs = calculator.calculate_bottleneck(distribution_a, distribution_b)
            
            # Calculate distances based on heights only
            wasserstein_heights, height_wasserstein_pairs = calculator.calculate_height_wasserstein_plan(distribution_a, distribution_b)
            bottleneck_heights, height_bottleneck_pairs = calculator.calculate_height_bottleneck_plan(distribution_a, distribution_b)
            
            # Store all transportation plans in session state for visualization
            AppState.store_bottleneck_matching(bottleneck_pairs)
            AppState.store_wasserstein_pairs(wasserstein_pairs)
            AppState.store_height_wasserstein_pairs(height_wasserstein_pairs)
            AppState.store_height_bottleneck_matching(height_bottleneck_pairs)
            
            # Determine which transport plan is currently selected
            current_plan = "hide"
            if AppState.is_showing_bottleneck():
                current_plan = "bottleneck_spatial"
            elif AppState.is_showing_wasserstein():
                current_plan = "wasserstein_spatial"
            elif AppState.is_showing_height_bottleneck():
                current_plan = "bottleneck_height"
            elif AppState.is_showing_height_wasserstein():
                current_plan = "wasserstein_height"
            
            # Display only the selected metric
            if current_plan != "hide":
                st.markdown("### Selected Distance Metric")
                
                if current_plan == "bottleneck_spatial":
                    st.metric("Spatial Bottleneck Distance", f"{bottleneck_value:.4f}")
                    st.info("Largest minimum distance between points when transforming one distribution into another.")
                    
                elif current_plan == "wasserstein_spatial":
                    st.metric("Spatial Wasserstein Distance", f"{wasserstein_discrete:.4f}")
                    st.info("Minimum 'cost' of transforming centers of one distribution into another, based on spatial positions.")
                    
                elif current_plan == "bottleneck_height":
                    st.metric("Height-Based Bottleneck Distance", f"{bottleneck_heights:.4f}")
                    st.info("Maximum difference between sorted heights, ignoring spatial positions.")
                    
                elif current_plan == "wasserstein_height":
                    st.metric("Height-Based Wasserstein Distance", f"{wasserstein_heights:.4f}")
                    st.info("Minimum 'cost' of transforming heights, ignoring spatial positions.")
            
            # Additional metrics with collapsible sections
            with st.expander("Show All Distance Metrics"):
                # Display all metrics in a single table format
                st.markdown("#### Spatial Distribution Distances")
                metrics_data = [
                    ["Wasserstein (Continuous)", f"{wasserstein_continuous:.4f}"],
                    ["Wasserstein (Centers)", f"{wasserstein_discrete:.4f}"],
                    ["Bottleneck Distance", f"{bottleneck_value:.4f}"]
                ]
                # Create a DataFrame for display
                metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])
                st.table(metrics_df)
                
                # Display height-based metrics
                st.markdown("#### Height-Based Distances")
                height_metrics_data = [
                    ["Wasserstein (Heights)", f"{wasserstein_heights:.4f}"],
                    ["Bottleneck (Heights)", f"{bottleneck_heights:.4f}"]
                ]
                # Create a DataFrame for display
                height_metrics_df = pd.DataFrame(height_metrics_data, columns=["Metric", "Value"])
                st.table(height_metrics_df)
        else:
            st.warning("Add blobs to both distributions to calculate distances.")
    
    @staticmethod
    def render_import_export(distribution_a: Distribution, distribution_b: Distribution, controller, on_update: Callable):
        """
        Render import/export section for distributions
        
        Args:
            distribution_a: Distribution A
            distribution_b: Distribution B
            controller: Controller for handling distribution operations
            on_update: Callback function for state updates
        """
        st.markdown("---")
        st.subheader("Import/Export Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Export Data")
            if not distribution_a.is_empty or not distribution_b.is_empty:
                export_link = controller.export_distributions_to_csv(distribution_a, distribution_b)
                st.markdown(export_link, unsafe_allow_html=True)
            else:
                st.warning("Add some blobs to export data.")
        
        with col2:
            st.markdown("### Import Data")
            uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key="distribution_csv_uploader")
            if uploaded_file is not None:
                if controller.import_distributions_from_csv(distribution_a, distribution_b, uploaded_file):
                    st.success("Data imported successfully.")
                    on_update()
                else:
                    st.error("Failed to import data. Make sure the CSV file has the correct format.")