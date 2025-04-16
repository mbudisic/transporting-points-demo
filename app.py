import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.distribution_utils import create_gaussian_mixture, Distribution
from utils.distance_calculator import calculate_wasserstein_continuous, calculate_wasserstein_discrete, calculate_bottleneck, calculate_wasserstein_plan
from utils.visualization import create_distribution_plot, create_interactive_plot, create_bottleneck_transport_plot, create_wasserstein_transport_plot
import base64
import io

# Set page configuration
st.set_page_config(
    page_title="Distribution Distance Visualizer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables
if 'distribution_a' not in st.session_state:
    # Create distribution A with default blobs
    st.session_state.distribution_a = Distribution('A', 'red')
    
    # Add some default positive blobs to distribution A
    st.session_state.distribution_a.add_blob(x=3.0, y=3.0, variance=0.8, height=1.5, sign=1)
    st.session_state.distribution_a.add_blob(x=7.0, y=7.0, variance=1.2, height=2.0, sign=1)
    
    # Add a negative blob to distribution A
    st.session_state.distribution_a.add_blob(x=5.0, y=2.0, variance=0.5, height=1.0, sign=-1)
    
if 'distribution_b' not in st.session_state:
    # Create distribution B with default blobs
    st.session_state.distribution_b = Distribution('B', 'blue')
    
    # Add some default positive blobs to distribution B
    st.session_state.distribution_b.add_blob(x=2.0, y=7.0, variance=1.0, height=1.8, sign=1)
    st.session_state.distribution_b.add_blob(x=8.0, y=3.0, variance=0.7, height=1.2, sign=1)
    
    # Add a negative blob to distribution B
    st.session_state.distribution_b.add_blob(x=5.0, y=8.0, variance=0.6, height=1.5, sign=-1)
    
if 'active_distribution' not in st.session_state:
    st.session_state.active_distribution = 'A'

if 'show_both' not in st.session_state:
    st.session_state.show_both = True
    
# Initialize the bottleneck matching in the session state
if 'bottleneck_matching' not in st.session_state:
    st.session_state.bottleneck_matching = []
    
# Initialize the bottleneck transport line visibility
if 'show_bottleneck_lines' not in st.session_state:
    st.session_state.show_bottleneck_lines = False

# Initialize the Wasserstein transport plan
if 'wasserstein_pairs' not in st.session_state:
    st.session_state.wasserstein_pairs = []

# Initialize the Wasserstein transport line visibility
if 'show_wasserstein_lines' not in st.session_state:
    st.session_state.show_wasserstein_lines = False
    
# Initialize selected element for direct manipulation
if 'selected_element' not in st.session_state:
    st.session_state.selected_element = None

def toggle_active_distribution(dist_name):
    st.session_state.active_distribution = dist_name
    st.rerun()

def toggle_show_both():
    st.session_state.show_both = not st.session_state.show_both
    st.rerun()

def add_blob(distribution, x=None, y=None, variance=0.5, height=1.0, sign=1):
    if x is None:
        x = np.random.uniform(0, 10)
    if y is None:
        y = np.random.uniform(0, 10)
    
    if distribution == 'A':
        st.session_state.distribution_a.add_blob(x, y, variance, height, sign)
    else:
        st.session_state.distribution_b.add_blob(x, y, variance, height, sign)
    st.rerun()

def remove_blob(distribution, index):
    if distribution == 'A':
        st.session_state.distribution_a.remove_blob(index)
    else:
        st.session_state.distribution_b.remove_blob(index)
    st.rerun()

def update_blob(distribution, index, x=None, y=None, variance=None, height=None, sign=None):
    """
    Update a blob's properties and ensure the state is properly updated to trigger visualization changes.
    If both height and sign are specified, make sure they are consistent.
    """
    # If both height and sign are changing, ensure they work together correctly
    if height is not None and sign is not None:
        # Keep the original intention of the height (negative height should retain its sign)
        if height < 0:
            # If height is negative, we want sign to be 1 to preserve the negative value
            # This is because the actual weight = height * sign
            height = abs(height)
            sign = -1
    
    # Update the distribution
    if distribution == 'A':
        st.session_state.distribution_a.update_blob(index, x, y, variance, height, sign)
    else:
        st.session_state.distribution_b.update_blob(index, x, y, variance, height, sign)
    
    # Force a rerun if we changed important visual properties
    if height is not None or sign is not None:
        st.rerun()

def handle_plot_click(trace, points, state):
    """Handle click events on the plot"""
    if len(points.point_inds) == 0:
        return
    
    # Check if we clicked on a center or variance circle by examining customdata
    if points.point_inds and hasattr(points, 'customdata') and len(points.customdata) > 0:
        customdata = points.customdata[0]
        
        # If we have customdata, check what type of element was clicked
        if isinstance(customdata, dict) and 'type' in customdata:
            if customdata['type'] == 'center' or customdata['type'] == 'variance':
                # Store the selected element info in the session state for direct manipulation
                st.session_state.selected_element = {
                    'type': customdata['type'],
                    'dist': customdata['dist'],
                    'id': customdata['id']
                }
                st.rerun()
                return
    
    # If we didn't click on an existing element, add a new blob
    x, y = points.xs[0], points.ys[0]
    add_blob(st.session_state.active_distribution, x, y)

def handle_drag_event(trace, points, state):
    """
    This function doesn't effectively handle drag events in the current Streamlit/Plotly integration.
    We'll use direct controls instead.
    """
    pass

def export_to_csv():
    output = io.StringIO()
    
    # Export distribution A
    df_a = pd.DataFrame(st.session_state.distribution_a.get_data())
    df_a['distribution'] = 'A'
    
    # Export distribution B
    df_b = pd.DataFrame(st.session_state.distribution_b.get_data())
    df_b['distribution'] = 'B'
    
    # Combine and export
    df_combined = pd.concat([df_a, df_b])
    df_combined.to_csv(output, index=False)
    
    # Create download link
    csv_bytes = output.getvalue().encode()
    b64 = base64.b64encode(csv_bytes).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="distributions.csv">Download CSV</a>'
    
    return href

def import_from_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        
        # Clear existing distributions
        st.session_state.distribution_a = Distribution('A', 'red')
        st.session_state.distribution_b = Distribution('B', 'blue')
        
        # Add blobs from the CSV
        for _, row in df.iterrows():
            dist_name = row['distribution']
            if dist_name == 'A':
                st.session_state.distribution_a.add_blob(
                    row['x'], row['y'], row['variance'], row['height'], row['sign']
                )
            elif dist_name == 'B':
                st.session_state.distribution_b.add_blob(
                    row['x'], row['y'], row['variance'], row['height'], row['sign']
                )
        
        st.success("Data imported successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error importing data: {str(e)}")

# Main app layout
st.title("Interactive Distribution Distance Visualizer")

# Information and explanation
with st.expander("About this tool"):
    st.markdown("""
    ## What is this tool for?
    
    This interactive tool helps you visualize and understand two important metrics for comparing probability distributions:
    
    - **Wasserstein Distance** (also known as Earth Mover's Distance): Measures the minimum "cost" of transforming one distribution into another, where cost is the amount of probability mass times the distance it needs to be moved.
    
    - **Bottleneck Distance**: Identifies the largest minimum distance that needs to be covered when matching points between two distributions.
    
    ## How to use:
    
    1. Create Gaussian blobs by clicking on the 2D plane or using the spreadsheet interface
    2. Click on a blob center or variance circle to select it for direct manipulation
    3. Use the sliders that appear to precisely adjust position or variance
    4. Toggle the bottleneck transport lines to visualize optimal matching between distributions
    5. Change height/intensity and sign using the spreadsheet
    6. Switch between distributions A and B or view both simultaneously
    7. Watch how the distances change in real-time
    8. Import/export your distributions as CSV files
    
    The tool calculates distances between both continuous Gaussian mixtures and the discrete weighted centers.
    """)

# Create three columns: left sidebar, main plot, right sidebar
left_col, center_col, right_col = st.columns([3, 6, 3])

# Left sidebar - Distribution A controls
with left_col:
    st.subheader("Distribution A (Red)")
    
    # Create columns for Add/Remove buttons
    add_col_a, remove_col_a = st.columns(2)
    
    # Add new blob button
    with add_col_a:
        if st.button("Add Blob to A"):
            add_blob('A')
    
    # Display table for distribution A
    st.markdown("### Distribution A Properties")
    df_a = pd.DataFrame(st.session_state.distribution_a.get_data())
    
    if not df_a.empty:
        # Initialize selected_id_a in session state if not present
        if 'selected_id_a' not in st.session_state:
            st.session_state.selected_id_a = df_a['id'].iloc[0] if not df_a.empty else None
            
        # Format the dataframe for display with highlighting for selected row
        def highlight_selected_row_a(row):
            if row.name in st.session_state.get('selected_rows_a', []):
                return ['background-color: rgba(255, 192, 192, 0.5); font-weight: bold'] * len(row)
            return [''] * len(row)
            
        # Show data editor with ability to select rows
        st.caption("Click on a row to select a blob for manipulation")
        edited_df_a = st.data_editor(
            df_a,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed", 
            disabled=["id"],
            key="data_editor_a",
            on_change=lambda: setattr(st.session_state, 'selected_id_a', 
                                      int(df_a['id'].iloc[st.session_state.data_editor_a['edited_rows'][0]]) 
                                      if st.session_state.data_editor_a.get('edited_rows') else 
                                      st.session_state.get('selected_id_a')),
            column_config={
                "id": st.column_config.NumberColumn("ID", min_value=0, format="%d"),
                "x": st.column_config.NumberColumn("X", min_value=0, max_value=10, format="%.2f"),
                "y": st.column_config.NumberColumn("Y", min_value=0, max_value=10, format="%.2f"),
                "variance": st.column_config.NumberColumn("Variance", min_value=0.1, max_value=5, format="%.2f"),
                "height": st.column_config.NumberColumn("Height", min_value=-10, max_value=10, format="%.2f"),
                "sign": st.column_config.SelectboxColumn("Sign", options=[1, -1])
            }
        )
        
        # Set the selected element for direct manipulation based on data editor interaction
        if 'selected_id_a' in st.session_state and st.session_state.selected_id_a is not None:
            selected_id = st.session_state.selected_id_a
            
            # Update the global selected element if needed
            if (st.session_state.selected_element is None or 
                st.session_state.selected_element['dist'] != 'A' or 
                st.session_state.selected_element['id'] != selected_id):
                st.session_state.selected_element = {
                    'type': 'center',  # Default to center
                    'dist': 'A',
                    'id': selected_id
                }
        
        # Remove selected blob button
        with remove_col_a:
            if st.button("Remove Selected Blob", key="remove_selected_a"):
                if st.session_state.selected_element and st.session_state.selected_element['dist'] == 'A':
                    remove_blob('A', st.session_state.selected_element['id'])
                    st.session_state.selected_element = None
                    # Rerun to update UI
                    st.rerun()
                
        # Add sliders for direct manipulation
        st.markdown("### Direct Controls")
        
        # Get the current blob if any is selected
        if st.session_state.selected_element and st.session_state.selected_element['dist'] == 'A':
            blob_id = st.session_state.selected_element['id']
            
            # Find the blob in the distribution
            for blob in st.session_state.distribution_a.blobs:
                if blob['id'] == blob_id:
                    st.write(f"**Selected Blob: A{blob_id}** (Height: {blob['height']:.2f}, Sign: {blob['sign']})")
                    
                    # Position controls (X, Y)
                    st.markdown("#### Position")
                    pos_col1, pos_col2 = st.columns(2)
                    
                    with pos_col1:
                        new_x = st.slider(
                            "X Position", 
                            min_value=0.0, 
                            max_value=10.0, 
                            value=float(blob['x']), 
                            step=0.1, 
                            key=f"x_slider_A{blob_id}",
                            on_change=lambda: update_blob('A', blob_id, 
                                                        x=st.session_state[f"x_slider_A{blob_id}"])
                        )
                    
                    with pos_col2:
                        new_y = st.slider(
                            "Y Position", 
                            min_value=0.0, 
                            max_value=10.0, 
                            value=float(blob['y']), 
                            step=0.1,
                            key=f"y_slider_A{blob_id}",
                            on_change=lambda: update_blob('A', blob_id, 
                                                        y=st.session_state[f"y_slider_A{blob_id}"])
                        )
                    
                    # Variance control
                    st.markdown("#### Spread")
                    new_variance = st.slider(
                        "Variance", 
                        min_value=0.1, 
                        max_value=5.0, 
                        value=float(blob['variance']), 
                        step=0.1,
                        key=f"var_slider_A{blob_id}",
                        on_change=lambda: update_blob('A', blob_id, 
                                                    variance=st.session_state[f"var_slider_A{blob_id}"])
                    )
                    
                    # Height and sign controls
                    st.markdown("#### Height")
                    height_col1, height_col2 = st.columns(2)
                    
                    with height_col1:
                        new_height = st.slider(
                            "Magnitude", 
                            min_value=0.1, 
                            max_value=10.0, 
                            value=float(blob['height']), 
                            step=0.1,
                            key=f"height_slider_A{blob_id}",
                            on_change=lambda: update_blob('A', blob_id, 
                                                        height=st.session_state[f"height_slider_A{blob_id}"])
                        )
                    
                    with height_col2:
                        new_sign = st.selectbox(
                            "Sign", 
                            options=[1, -1],
                            index=0 if blob['sign'] > 0 else 1,
                            key=f"sign_selector_A{blob_id}",
                            on_change=lambda: update_blob('A', blob_id, 
                                                        sign=st.session_state[f"sign_selector_A{blob_id}"])
                        )
                    break
            else:
                st.warning("No blob selected. Click on a row in the table to select a blob.")
        else:
            st.info("No blob selected. Click on a row in the table to select a blob.")
                
        # Update distribution based on edited dataframe
        for index, row in edited_df_a.iterrows():
            original_row = df_a.iloc[index]
            if not np.array_equal(row.values, original_row.values):
                update_blob('A', int(row['id']), row['x'], row['y'], row['variance'], row['height'], row['sign'])

# Right sidebar - Distribution B controls
with right_col:
    st.subheader("Distribution B (Blue)")
    
    # Create columns for Add/Remove buttons
    add_col_b, remove_col_b = st.columns(2)
    
    # Add new blob button
    with add_col_b:
        if st.button("Add Blob to B"):
            add_blob('B')
    
    # Display table for distribution B
    st.markdown("### Distribution B Properties")
    df_b = pd.DataFrame(st.session_state.distribution_b.get_data())
    
    if not df_b.empty:
        # Initialize selected_id_b in session state if not present
        if 'selected_id_b' not in st.session_state:
            st.session_state.selected_id_b = df_b['id'].iloc[0] if not df_b.empty else None
            
        # Format the dataframe for display with highlighting for selected row
        def highlight_selected_row_b(row):
            if row.name in st.session_state.get('selected_rows_b', []):
                return ['background-color: rgba(192, 192, 255, 0.5); font-weight: bold'] * len(row)
            return [''] * len(row)
            
        # Show data editor with ability to select rows
        st.caption("Click on a row to select a blob for manipulation")
        edited_df_b = st.data_editor(
            df_b,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed", 
            disabled=["id"],
            key="data_editor_b",
            on_change=lambda: setattr(st.session_state, 'selected_id_b', 
                                      int(df_b['id'].iloc[st.session_state.data_editor_b['edited_rows'][0]]) 
                                      if st.session_state.data_editor_b.get('edited_rows') else 
                                      st.session_state.get('selected_id_b')),
            column_config={
                "id": st.column_config.NumberColumn("ID", min_value=0, format="%d"),
                "x": st.column_config.NumberColumn("X", min_value=0, max_value=10, format="%.2f"),
                "y": st.column_config.NumberColumn("Y", min_value=0, max_value=10, format="%.2f"),
                "variance": st.column_config.NumberColumn("Variance", min_value=0.1, max_value=5, format="%.2f"),
                "height": st.column_config.NumberColumn("Height", min_value=-10, max_value=10, format="%.2f"),
                "sign": st.column_config.SelectboxColumn("Sign", options=[1, -1])
            }
        )
        
        # Set the selected element for direct manipulation based on data editor interaction
        if 'selected_id_b' in st.session_state and st.session_state.selected_id_b is not None:
            selected_id = st.session_state.selected_id_b
            
            # Update the global selected element if needed
            if (st.session_state.selected_element is None or 
                st.session_state.selected_element['dist'] != 'B' or 
                st.session_state.selected_element['id'] != selected_id):
                st.session_state.selected_element = {
                    'type': 'center',  # Default to center
                    'dist': 'B',
                    'id': selected_id
                }
        
        # Remove selected blob button
        with remove_col_b:
            if st.button("Remove Selected Blob", key="remove_selected_b"):
                if st.session_state.selected_element and st.session_state.selected_element['dist'] == 'B':
                    remove_blob('B', st.session_state.selected_element['id'])
                    st.session_state.selected_element = None
                    # Rerun to update UI
                    st.rerun()
                
        # Add sliders for direct manipulation
        st.markdown("### Direct Controls")
        
        # Get the current blob if any is selected
        if st.session_state.selected_element and st.session_state.selected_element['dist'] == 'B':
            blob_id = st.session_state.selected_element['id']
            
            # Find the blob in the distribution
            for blob in st.session_state.distribution_b.blobs:
                if blob['id'] == blob_id:
                    st.write(f"**Selected Blob: B{blob_id}** (Height: {blob['height']:.2f}, Sign: {blob['sign']})")
                    
                    # Position controls (X, Y)
                    st.markdown("#### Position")
                    pos_col1, pos_col2 = st.columns(2)
                    
                    with pos_col1:
                        new_x = st.slider(
                            "X Position", 
                            min_value=0.0, 
                            max_value=10.0, 
                            value=float(blob['x']), 
                            step=0.1, 
                            key=f"x_slider_B{blob_id}",
                            on_change=lambda: update_blob('B', blob_id, 
                                                        x=st.session_state[f"x_slider_B{blob_id}"])
                        )
                    
                    with pos_col2:
                        new_y = st.slider(
                            "Y Position", 
                            min_value=0.0, 
                            max_value=10.0, 
                            value=float(blob['y']), 
                            step=0.1,
                            key=f"y_slider_B{blob_id}",
                            on_change=lambda: update_blob('B', blob_id, 
                                                        y=st.session_state[f"y_slider_B{blob_id}"])
                        )
                    
                    # Variance control
                    st.markdown("#### Spread")
                    new_variance = st.slider(
                        "Variance", 
                        min_value=0.1, 
                        max_value=5.0, 
                        value=float(blob['variance']), 
                        step=0.1,
                        key=f"var_slider_B{blob_id}",
                        on_change=lambda: update_blob('B', blob_id, 
                                                    variance=st.session_state[f"var_slider_B{blob_id}"])
                    )
                    
                    # Height and sign controls
                    st.markdown("#### Height")
                    height_col1, height_col2 = st.columns(2)
                    
                    with height_col1:
                        new_height = st.slider(
                            "Magnitude", 
                            min_value=0.1, 
                            max_value=10.0, 
                            value=float(blob['height']), 
                            step=0.1,
                            key=f"height_slider_B{blob_id}",
                            on_change=lambda: update_blob('B', blob_id, 
                                                        height=st.session_state[f"height_slider_B{blob_id}"])
                        )
                    
                    with height_col2:
                        new_sign = st.selectbox(
                            "Sign", 
                            options=[1, -1],
                            index=0 if blob['sign'] > 0 else 1,
                            key=f"sign_selector_B{blob_id}",
                            on_change=lambda: update_blob('B', blob_id, 
                                                        sign=st.session_state[f"sign_selector_B{blob_id}"])
                        )
                    break
            else:
                st.warning("No blob selected. Click on a row in the table to select a blob.")
        else:
            st.info("No blob selected. Click on a row in the table to select a blob.")
                
        # Update distribution based on edited dataframe
        for index, row in edited_df_b.iterrows():
            original_row = df_b.iloc[index]
            if not np.array_equal(row.values, original_row.values):
                update_blob('B', int(row['id']), row['x'], row['y'], row['variance'], row['height'], row['sign'])

# Main area - Plot and distance metrics
with center_col:
    # Display control buttons in a row
    col1, col2, col3 = st.columns(3)
    with col1:
        active_a = st.session_state.active_distribution == 'A'
        if st.button("Edit Distribution A", type="primary" if active_a else "secondary"):
            toggle_active_distribution('A')
    
    with col2:
        active_b = st.session_state.active_distribution == 'B'
        if st.button("Edit Distribution B", type="primary" if active_b else "secondary"):
            toggle_active_distribution('B')
    
    with col3:
        if st.button("Toggle View Both" if st.session_state.show_both else "Toggle View Active Only"):
            toggle_show_both()
    
    # Create interactive plot
    fig = create_interactive_plot(
        st.session_state.distribution_a,
        st.session_state.distribution_b,
        active_distribution=st.session_state.active_distribution,
        show_both=st.session_state.show_both,
        show_bottleneck_lines=st.session_state.show_bottleneck_lines,
        bottleneck_pairs=st.session_state.bottleneck_matching,
        show_wasserstein_lines=st.session_state.show_wasserstein_lines,
        wasserstein_pairs=st.session_state.wasserstein_pairs
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
        on_dragend=handle_drag_event  # Add drag event handler
    )
    
    # Calculate and display distances
    has_a = len(st.session_state.distribution_a.blobs) > 0
    has_b = len(st.session_state.distribution_b.blobs) > 0
    
    if has_a and has_b:
        # Create the continuous distributions
        x_grid = np.linspace(0, 10, 100)
        y_grid = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        
        dist_a_continuous = create_gaussian_mixture(st.session_state.distribution_a.blobs, points)
        dist_b_continuous = create_gaussian_mixture(st.session_state.distribution_b.blobs, points)
        
        # Extract centers and weights for discrete calculation
        centers_a = [(b['x'], b['y']) for b in st.session_state.distribution_a.blobs]
        weights_a = [b['height'] * b['sign'] for b in st.session_state.distribution_a.blobs]
        
        centers_b = [(b['x'], b['y']) for b in st.session_state.distribution_b.blobs]
        weights_b = [b['height'] * b['sign'] for b in st.session_state.distribution_b.blobs]
        
        # Calculate distances
        wasserstein_continuous = calculate_wasserstein_continuous(dist_a_continuous, dist_b_continuous, points)
        wasserstein_discrete = calculate_wasserstein_discrete(centers_a, centers_b, weights_a, weights_b)
        bottleneck_value, matching_pairs = calculate_bottleneck(centers_a, centers_b, weights_a, weights_b)
        
        # Store the bottleneck matching in session state for visualization
        if 'bottleneck_matching' not in st.session_state:
            st.session_state.bottleneck_matching = matching_pairs
        else:
            st.session_state.bottleneck_matching = matching_pairs
        
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
            
            # Add Transport Line Toggle Buttons
            # Toggle for Bottleneck Transport Lines
            if st.button("Toggle Bottleneck Transport Lines"):
                st.session_state.show_bottleneck_lines = not st.session_state.show_bottleneck_lines
                st.rerun()
                
            # Calculate Wasserstein transportation plan
            wasserstein_value, wasserstein_pairs = calculate_wasserstein_plan(centers_a, centers_b, weights_a, weights_b)
            st.session_state.wasserstein_pairs = wasserstein_pairs
            
            # Toggle for Wasserstein Transport Lines
            if st.button("Toggle Wasserstein Transport Plan"):
                st.session_state.show_wasserstein_lines = not st.session_state.show_wasserstein_lines
                st.rerun()
    else:
        st.warning("Add blobs to both distributions to calculate distances.")

# Import/Export section at the bottom
st.markdown("---")
st.subheader("Import/Export Distributions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Export Data")
    if len(st.session_state.distribution_a.blobs) > 0 or len(st.session_state.distribution_b.blobs) > 0:
        st.markdown(export_to_csv(), unsafe_allow_html=True)
    else:
        st.warning("Add some blobs to export data.")

with col2:
    st.markdown("### Import Data")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        if st.button("Import"):
            import_from_csv(uploaded_file)