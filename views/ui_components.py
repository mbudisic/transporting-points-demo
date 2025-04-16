import streamlit as st
import pandas as pd
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
                        # Set the session state value first
                        st.session_state[f"x_slider_B{blob_id}"] = float(blob.x)
                        
                        # Then create the slider with that value
                        new_x = st.slider(
                            "##", 
                            min_value=0.0, 
                            max_value=10.0, 
                            value=float(blob.x), 
                            step=0.1, 
                            key=f"x_slider_B{blob_id}",
                            on_change=lambda: DistributionController.update_blob(
                                distribution, blob_id, 
                                x=st.session_state[f"x_slider_B{blob_id}"]
                            )
                        )
                    
                    # Y Position - in a row with label
                    y_row = st.container()
                    y_label, y_slider = y_row.columns([1, 3])
                    with y_label:
                        st.markdown("Y:")
                    with y_slider:
                        # Set the session state value first 
                        st.session_state[f"y_slider_B{blob_id}"] = float(blob.y)
                        
                        # Then create the slider with that value
                        new_y = st.slider(
                            "##", 
                            min_value=0.0, 
                            max_value=10.0, 
                            value=float(blob.y), 
                            step=0.1,
                            key=f"y_slider_B{blob_id}",
                            on_change=lambda: DistributionController.update_blob(
                                distribution, blob_id, 
                                y=st.session_state[f"y_slider_B{blob_id}"]
                            )
                        )
                    
                    # Variance control - in a row with label
                    var_row = st.container()
                    var_label, var_slider = var_row.columns([1, 3])
                    with var_label:
                        st.markdown("Var:")
                    with var_slider:
                        # Set the session state value first
                        st.session_state[f"var_slider_B{blob_id}"] = float(blob.variance)
                        
                        # Then create the slider with that value
                        new_variance = st.slider(
                            "##", 
                            min_value=0.1, 
                            max_value=5.0, 
                            value=float(blob.variance), 
                            step=0.1,
                            key=f"var_slider_B{blob_id}",
                            on_change=lambda: DistributionController.update_blob(
                                distribution, blob_id, 
                                variance=st.session_state[f"var_slider_B{blob_id}"]
                            )
                        )
                    
                    # Height control (double-sided slider) - in a row with label
                    height_row = st.container()
                    height_label, height_slider = height_row.columns([1, 3])
                    with height_label:
                        st.markdown("Height:")
                    with height_slider:
                        # Set the session state value directly with the signed height
                        st.session_state[f"height_slider_B{blob_id}"] = float(blob.height)
                        
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
                                height=st.session_state[f"height_slider_B{blob_id}"]
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
        visualization_service
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
        """
        # Display control buttons in a row
        col1, col2, col3 = st.columns(3)
        with col1:
            active_a = AppState.get_active_distribution() == 'A'
            if st.button("Edit Distribution A", type="primary" if active_a else "secondary"):
                AppState.toggle_active_distribution('A')
                on_update()
        
        with col2:
            active_b = AppState.get_active_distribution() == 'B'
            if st.button("Edit Distribution B", type="primary" if active_b else "secondary"):
                AppState.toggle_active_distribution('B')
                on_update()
        
        with col3:
            show_both = AppState.is_showing_both()
            if st.button("Toggle View " + ("Both" if show_both else "Active Only")):
                AppState.toggle_show_both()
                on_update()
        
        # Create interactive plot with all transport plan options
        fig = visualization_service.create_interactive_plot(
            distribution_a,
            distribution_b,
            active_distribution=AppState.get_active_distribution(),
            show_both=AppState.is_showing_both(),
            # Spatial transport plans
            show_bottleneck_lines=AppState.is_showing_bottleneck(),
            bottleneck_pairs=AppState.get_bottleneck_matching(),
            show_wasserstein_lines=AppState.is_showing_wasserstein(),
            wasserstein_pairs=AppState.get_wasserstein_pairs(),
            # Height-based transport plans
            show_height_bottleneck_lines=AppState.is_showing_height_bottleneck(),
            height_bottleneck_pairs=AppState.get_height_bottleneck_matching(),
            show_height_wasserstein_lines=AppState.is_showing_height_wasserstein(),
            height_wasserstein_pairs=AppState.get_height_wasserstein_pairs()
        )
        
        # Set the dragmode to 'drag' to support element dragging
        fig.update_layout(
            dragmode='zoom',  # Use zoom mode which allows for regular drag behavior
            modebar_add=['drawopenpath', 'eraseshape']
        )
        
        # Add event callbacks
        st.plotly_chart(
            fig, 
            use_container_width=True, 
            on_click=handle_plot_click,
            on_dragend=handle_drag_event
        )
    
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
            # Calculate distances based on spatial positions
            wasserstein_continuous = calculator.calculate_wasserstein_continuous(distribution_a, distribution_b)
            wasserstein_discrete = calculator.calculate_wasserstein_discrete(distribution_a, distribution_b)
            bottleneck_value, bottleneck_pairs = calculator.calculate_bottleneck(distribution_a, distribution_b)
            
            # Calculate distances based on heights only
            wasserstein_heights = calculator.calculate_wasserstein_by_heights(distribution_a, distribution_b)
            bottleneck_heights = calculator.calculate_bottleneck_by_heights(distribution_a, distribution_b)
            
            # Store the bottleneck matching in session state for visualization
            AppState.store_bottleneck_matching(bottleneck_pairs)
            
            # Display spatial metrics in a box
            st.markdown("### Spatial Distribution Distances")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Wasserstein (Continuous)", f"{wasserstein_continuous:.4f}")
                st.info("Measures minimum 'cost' of transforming one continuous distribution into another.")
                
            with metrics_col2:
                st.metric("Wasserstein (Centers)", f"{wasserstein_discrete:.4f}")
                st.info("Measures minimum 'cost' of transforming centers of one distribution into another.")
                
            with metrics_col3:
                st.metric("Bottleneck Distance", f"{bottleneck_value:.4f}")
                st.info("Largest minimum distance to transform one distribution into another.")
            
            # Display height-based metrics in a separate box
            st.markdown("### Height-Based Distances")
            height_col1, height_col2 = st.columns(2)
            
            with height_col1:
                st.metric("Wasserstein (Heights)", f"{wasserstein_heights:.4f}")
                st.info("Measures minimum 'cost' of transforming heights, ignoring positions.")
                
            with height_col2:
                st.metric("Bottleneck (Heights)", f"{bottleneck_heights:.4f}")
                st.info("Maximum difference between sorted heights, ignoring positions.")
                
                # Calculate all transportation plans
                # Spatial transport plan for Wasserstein
                wasserstein_value, wasserstein_pairs = calculator.calculate_wasserstein_plan(distribution_a, distribution_b)
                AppState.store_wasserstein_pairs(wasserstein_pairs)
                
                # Height-based transportation plans
                height_wasserstein_value, height_wasserstein_pairs = calculator.calculate_height_wasserstein_plan(distribution_a, distribution_b)
                AppState.store_height_wasserstein_pairs(height_wasserstein_pairs)
                
                height_bottleneck_value, height_bottleneck_pairs = calculator.calculate_height_bottleneck_plan(distribution_a, distribution_b)
                AppState.store_height_bottleneck_matching(height_bottleneck_pairs)
                
                # Transport plan visualization with dropdown
                st.markdown("#### Transport Plan Visualization")
                transport_options = [
                    {"label": "Hide Transportation Plans", "value": "hide"},
                    {"label": "Spatial Bottleneck", "value": "bottleneck_spatial"},
                    {"label": "Spatial Wasserstein", "value": "wasserstein_spatial"},
                    {"label": "Height-Based Bottleneck", "value": "bottleneck_height"},
                    {"label": "Height-Based Wasserstein", "value": "wasserstein_height"}
                ]
                
                # Map state to option value
                current_value = "hide"  # Default
                if AppState.is_showing_bottleneck():
                    current_value = "bottleneck_spatial"
                elif AppState.is_showing_wasserstein():
                    current_value = "wasserstein_spatial"
                elif AppState.is_showing_height_bottleneck():
                    current_value = "bottleneck_height"
                elif AppState.is_showing_height_wasserstein():
                    current_value = "wasserstein_height"
                
                # Create map of labels to values for lookup
                option_map = {opt["label"]: opt["value"] for opt in transport_options}
                option_labels = [opt["label"] for opt in transport_options]
                
                # Find the current index
                current_index = 0
                for i, opt in enumerate(transport_options):
                    if opt["value"] == current_value:
                        current_index = i
                        break
                
                # Display dropdown for transport plan selection
                transport_selection = st.selectbox(
                    "Select visualization type",
                    options=option_labels,
                    index=current_index,
                    key="transport_plan_dropdown"
                )
                
                # Update visualization based on selection
                selected_value = option_map[transport_selection]
                if selected_value != current_value:
                    AppState.set_transport_visualization(selected_value)
                    on_update()
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
            uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
            if uploaded_file is not None:
                if controller.import_distributions_from_csv(distribution_a, distribution_b, uploaded_file):
                    st.success("Data imported successfully.")
                    on_update()
                else:
                    st.error("Failed to import data. Make sure the CSV file has the correct format.")
