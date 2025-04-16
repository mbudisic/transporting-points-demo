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
        with st.sidebar:
            st.subheader("Distribution A (Red)")
            
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
                    # Find current selection index if there is one
                    selected_element = AppState.get_selected_element()
                    selected_index = 0
                    if (selected_element is not None and selected_element['dist'] == 'A'):
                        for i, opt in enumerate(blob_options):
                            if f"A{selected_element['id']}" in opt:
                                selected_index = i
                                break
                                
                    selected_blob = st.selectbox(
                        "Select a blob to manipulate:",
                        options=blob_options,
                        index=selected_index,
                        key="blob_selector_a"
                    )
                    
                    # Get the ID directly from our mapping
                    if selected_blob:
                        selected_id = blob_id_map[selected_blob]
                        
                        # Update the global selected element if needed
                        if (selected_element is None or 
                            selected_element['dist'] != 'A' or 
                            selected_element['id'] != selected_id):
                            AppState.set_selected_element('center', 'A', selected_id)
                            on_update()  # Trigger update to reflect selection change
                
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
                        st.write(f"**Selected Blob: A{blob_id}** (Height: {blob.height:.2f}, Sign: {blob.sign})")
                        
                        # Stacked controls with labels on the left (for Distribution A)
                        st.markdown("#### Blob Controls")
                        
                        # Create label and slider columns
                        label_col, slider_col = st.columns([1, 3])
                        
                        # Initialize slider state keys if they don't exist
                        slider_keys = [
                            f"x_slider_A{blob_id}", 
                            f"y_slider_A{blob_id}", 
                            f"var_slider_A{blob_id}", 
                            f"height_slider_A{blob_id}"
                        ]
                        for key in slider_keys:
                            if key not in st.session_state:
                                st.session_state[key] = 0.0  # Default will be immediately overwritten
                        
                        # Initialize sign selector state if it doesn't exist
                        sign_key = f"sign_selector_A{blob_id}"
                        if sign_key not in st.session_state:
                            st.session_state[sign_key] = 1  # Default will be immediately overwritten
                        
                        # X Position
                        with label_col:
                            st.markdown("X Position:")
                        with slider_col:
                            # Set the session state value first
                            st.session_state[f"x_slider_A{blob_id}"] = float(blob.x)
                            
                            # Then create the slider with that value
                            new_x = st.slider(
                                "##", 
                                min_value=0.0, 
                                max_value=10.0, 
                                value=float(blob.x), 
                                step=0.1, 
                                key=f"x_slider_A{blob_id}",
                                on_change=lambda: DistributionController.update_blob(
                                    distribution, blob_id, 
                                    x=st.session_state[f"x_slider_A{blob_id}"]
                                )
                            )
                        
                        # Y Position
                        with label_col:
                            st.markdown("Y Position:")
                        with slider_col:
                            # Set the session state value first
                            st.session_state[f"y_slider_A{blob_id}"] = float(blob.y)
                            
                            # Then create the slider with that value
                            new_y = st.slider(
                                "##", 
                                min_value=0.0, 
                                max_value=10.0, 
                                value=float(blob.y), 
                                step=0.1,
                                key=f"y_slider_A{blob_id}",
                                on_change=lambda: DistributionController.update_blob(
                                    distribution, blob_id, 
                                    y=st.session_state[f"y_slider_A{blob_id}"]
                                )
                            )
                        
                        # Variance control
                        with label_col:
                            st.markdown("Variance:")
                        with slider_col:
                            # Set the session state value first
                            st.session_state[f"var_slider_A{blob_id}"] = float(blob.variance)
                            
                            # Then create the slider with that value
                            new_variance = st.slider(
                                "##", 
                                min_value=0.1, 
                                max_value=5.0, 
                                value=float(blob.variance), 
                                step=0.1,
                                key=f"var_slider_A{blob_id}",
                                on_change=lambda: DistributionController.update_blob(
                                    distribution, blob_id, 
                                    variance=st.session_state[f"var_slider_A{blob_id}"]
                                )
                            )
                        
                        # Height control
                        with label_col:
                            st.markdown("Magnitude:")
                        with slider_col:
                            # Set the session state value first
                            st.session_state[f"height_slider_A{blob_id}"] = float(blob.height)
                            
                            # Then create the slider with that value
                            new_height = st.slider(
                                "##", 
                                min_value=0.1, 
                                max_value=10.0, 
                                value=float(blob.height), 
                                step=0.1,
                                key=f"height_slider_A{blob_id}",
                                on_change=lambda: DistributionController.update_blob(
                                    distribution, blob_id, 
                                    height=st.session_state[f"height_slider_A{blob_id}"]
                                )
                            )
                        
                        # Sign control
                        with label_col:
                            st.markdown("Sign:")
                        with slider_col:
                            # Set the session state value first
                            st.session_state[f"sign_selector_A{blob_id}"] = blob.sign
                            
                            # Then create the selector with that value
                            new_sign = st.selectbox(
                                "##", 
                                options=[1, -1],
                                index=0 if blob.sign > 0 else 1,
                                key=f"sign_selector_A{blob_id}",
                                on_change=lambda: DistributionController.update_blob(
                                    distribution, blob_id, 
                                    sign=st.session_state[f"sign_selector_A{blob_id}"]
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
        with st.sidebar:
            st.subheader("Distribution B (Blue)")
            
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
                    # Find current selection index if there is one
                    selected_element = AppState.get_selected_element()
                    selected_index = 0
                    if (selected_element is not None and selected_element['dist'] == 'B'):
                        for i, opt in enumerate(blob_options):
                            if f"B{selected_element['id']}" in opt:
                                selected_index = i
                                break
                                
                    selected_blob = st.selectbox(
                        "Select a blob to manipulate:",
                        options=blob_options,
                        index=selected_index,
                        key="blob_selector_b"
                    )
                    
                    # Get the ID directly from our mapping
                    if selected_blob:
                        selected_id = blob_id_map[selected_blob]
                        
                        # Update the global selected element if needed
                        if (selected_element is None or 
                            selected_element['dist'] != 'B' or 
                            selected_element['id'] != selected_id):
                            AppState.set_selected_element('center', 'B', selected_id)
                            on_update()  # Trigger update to reflect selection change
                
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
                        st.write(f"**Selected Blob: B{blob_id}** (Height: {blob.height:.2f}, Sign: {blob.sign})")
                        
                        # Stacked controls with labels on the right (for Distribution B)
                        st.markdown("#### Blob Controls")
                        
                        # Create slider and label columns (reverse order for B side)
                        slider_col, label_col = st.columns([3, 1])
                        
                        # Initialize slider state keys if they don't exist
                        slider_keys = [
                            f"x_slider_B{blob_id}", 
                            f"y_slider_B{blob_id}", 
                            f"var_slider_B{blob_id}", 
                            f"height_slider_B{blob_id}"
                        ]
                        for key in slider_keys:
                            if key not in st.session_state:
                                st.session_state[key] = 0.0  # Default will be immediately overwritten
                        
                        # Initialize sign selector state if it doesn't exist
                        sign_key = f"sign_selector_B{blob_id}"
                        if sign_key not in st.session_state:
                            st.session_state[sign_key] = 1  # Default will be immediately overwritten
                        
                        # X Position
                        with slider_col:
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
                        with label_col:
                            st.markdown("X Position:")
                        
                        # Y Position
                        with slider_col:
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
                        with label_col:
                            st.markdown("Y Position:")
                        
                        # Variance control
                        with slider_col:
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
                        with label_col:
                            st.markdown("Variance:")
                        
                        # Height control
                        with slider_col:
                            # Set the session state value first
                            st.session_state[f"height_slider_B{blob_id}"] = float(blob.height)
                            
                            # Then create the slider with that value
                            new_height = st.slider(
                                "##", 
                                min_value=0.1, 
                                max_value=10.0, 
                                value=float(blob.height), 
                                step=0.1,
                                key=f"height_slider_B{blob_id}",
                                on_change=lambda: DistributionController.update_blob(
                                    distribution, blob_id, 
                                    height=st.session_state[f"height_slider_B{blob_id}"]
                                )
                            )
                        with label_col:
                            st.markdown("Magnitude:")
                        
                        # Sign control
                        with slider_col:
                            # Set the session state value first
                            st.session_state[f"sign_selector_B{blob_id}"] = blob.sign
                            
                            # Then create the selector with that value
                            new_sign = st.selectbox(
                                "##", 
                                options=[1, -1],
                                index=0 if blob.sign > 0 else 1,
                                key=f"sign_selector_B{blob_id}",
                                on_change=lambda: DistributionController.update_blob(
                                    distribution, blob_id, 
                                    sign=st.session_state[f"sign_selector_B{blob_id}"]
                                )
                            )
                        with label_col:
                            st.markdown("Sign:")
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
        
        # Create interactive plot
        fig = visualization_service.create_interactive_plot(
            distribution_a,
            distribution_b,
            active_distribution=AppState.get_active_distribution(),
            show_both=AppState.is_showing_both(),
            show_bottleneck_lines=AppState.is_showing_bottleneck(),
            bottleneck_pairs=AppState.get_bottleneck_matching(),
            show_wasserstein_lines=AppState.is_showing_wasserstein(),
            wasserstein_pairs=AppState.get_wasserstein_pairs()
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
            # Calculate distances
            wasserstein_continuous = calculator.calculate_wasserstein_continuous(distribution_a, distribution_b)
            wasserstein_discrete = calculator.calculate_wasserstein_discrete(distribution_a, distribution_b)
            bottleneck_value, matching_pairs = calculator.calculate_bottleneck(distribution_a, distribution_b)
            
            # Store the bottleneck matching in session state for visualization
            AppState.store_bottleneck_matching(matching_pairs)
            
            # Display metrics in a box
            st.markdown("### Distribution Distances")
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
                
                # Calculate Wasserstein transportation plan
                wasserstein_value, wasserstein_pairs = calculator.calculate_wasserstein_plan(distribution_a, distribution_b)
                AppState.store_wasserstein_pairs(wasserstein_pairs)
                
                # Transport plan visualization with radio buttons
                st.markdown("#### Transport Plan Visualization")
                transport_options = ["Hide Transportation Plans", "Bottleneck Transport", "Wasserstein Transport"]
                
                # Determine the current selection index based on current state
                current_option = 0  # Default to "Hide"
                if AppState.is_showing_bottleneck():
                    current_option = 1  # "Bottleneck Transport"
                elif AppState.is_showing_wasserstein():
                    current_option = 2  # "Wasserstein Transport"
                    
                # Display radio buttons for transport plan selection
                transport_selection = st.radio(
                    "Select visualization",
                    options=transport_options,
                    index=current_option,
                    horizontal=True,
                    key="transport_plan_radio"
                )
                
                # Update visualization based on selection
                if transport_selection == "Hide Transportation Plans":
                    if AppState.is_showing_bottleneck() or AppState.is_showing_wasserstein():
                        AppState.set_transport_visualization("hide")
                        on_update()
                elif transport_selection == "Bottleneck Transport":
                    if not AppState.is_showing_bottleneck() or AppState.is_showing_wasserstein():
                        AppState.set_transport_visualization("bottleneck")
                        on_update()
                elif transport_selection == "Wasserstein Transport":
                    if AppState.is_showing_bottleneck() or not AppState.is_showing_wasserstein():
                        AppState.set_transport_visualization("wasserstein")
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
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")
            if uploaded_file is not None:
                if st.button("Import"):
                    success = controller.import_distributions_from_csv(distribution_a, distribution_b, uploaded_file)
                    if success:
                        st.success("Data imported successfully.")
                        on_update()
                    else:
                        st.error("Failed to import data. Please check the file format.")