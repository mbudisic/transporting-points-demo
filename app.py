import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.distribution_utils import create_gaussian_mixture, Distribution
from utils.distance_calculator import calculate_wasserstein_continuous, calculate_wasserstein_discrete, calculate_bottleneck
from utils.visualization import create_distribution_plot, create_interactive_plot
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
    st.session_state.distribution_a = Distribution('A', 'red')
    
if 'distribution_b' not in st.session_state:
    st.session_state.distribution_b = Distribution('B', 'blue')
    
if 'active_distribution' not in st.session_state:
    st.session_state.active_distribution = 'A'

if 'show_both' not in st.session_state:
    st.session_state.show_both = True

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
    if distribution == 'A':
        st.session_state.distribution_a.update_blob(index, x, y, variance, height, sign)
    else:
        st.session_state.distribution_b.update_blob(index, x, y, variance, height, sign)

def handle_plot_click(trace, points, state):
    # This function will be called when a user clicks on the plot
    if len(points.point_inds) == 0:
        return
    
    x, y = points.xs[0], points.ys[0]
    add_blob(st.session_state.active_distribution, x, y)

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
    2. Drag points to reposition them
    3. Adjust the spread (variance) by dragging the dotted circles
    4. Change height/intensity and sign using the spreadsheet
    5. Switch between distributions A and B or view both simultaneously
    6. Watch how the distances change in real-time
    7. Import/export your distributions as CSV files
    
    The tool calculates distances between both continuous Gaussian mixtures and the discrete weighted centers.
    """)

# Create three columns: left sidebar, main plot, right sidebar
left_col, center_col, right_col = st.columns([3, 6, 3])

# Left sidebar - Distribution A controls
with left_col:
    st.subheader("Distribution A (Red)")
    
    # Add new blob button
    if st.button("Add Blob to A"):
        add_blob('A')
    
    # Display table for distribution A
    st.markdown("### Distribution A Properties")
    df_a = pd.DataFrame(st.session_state.distribution_a.get_data())
    
    if not df_a.empty:
        edited_df_a = st.data_editor(
            df_a,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=["id"],
            column_config={
                "id": st.column_config.NumberColumn("ID", min_value=0, format="%d"),
                "x": st.column_config.NumberColumn("X", min_value=0, max_value=10, format="%.2f"),
                "y": st.column_config.NumberColumn("Y", min_value=0, max_value=10, format="%.2f"),
                "variance": st.column_config.NumberColumn("Variance", min_value=0.1, max_value=5, format="%.2f"),
                "height": st.column_config.NumberColumn("Height", min_value=0.1, max_value=10, format="%.2f"),
                "sign": st.column_config.SelectboxColumn("Sign", options=[1, -1])
            }
        )
        
        # Update distribution based on edited dataframe
        for index, row in edited_df_a.iterrows():
            original_row = df_a.iloc[index]
            if not np.array_equal(row.values, original_row.values):
                update_blob('A', int(row['id']), row['x'], row['y'], row['variance'], row['height'], row['sign'])
        
        # Remove blob buttons
        for blob_id in edited_df_a['id']:
            if st.button(f"Remove Blob {int(blob_id)}", key=f"remove_a_{blob_id}"):
                remove_blob('A', int(blob_id))

# Right sidebar - Distribution B controls
with right_col:
    st.subheader("Distribution B (Blue)")
    
    # Add new blob button
    if st.button("Add Blob to B"):
        add_blob('B')
    
    # Display table for distribution B
    st.markdown("### Distribution B Properties")
    df_b = pd.DataFrame(st.session_state.distribution_b.get_data())
    
    if not df_b.empty:
        edited_df_b = st.data_editor(
            df_b,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=["id"],
            column_config={
                "id": st.column_config.NumberColumn("ID", min_value=0, format="%d"),
                "x": st.column_config.NumberColumn("X", min_value=0, max_value=10, format="%.2f"),
                "y": st.column_config.NumberColumn("Y", min_value=0, max_value=10, format="%.2f"),
                "variance": st.column_config.NumberColumn("Variance", min_value=0.1, max_value=5, format="%.2f"),
                "height": st.column_config.NumberColumn("Height", min_value=0.1, max_value=10, format="%.2f"),
                "sign": st.column_config.SelectboxColumn("Sign", options=[1, -1])
            }
        )
        
        # Update distribution based on edited dataframe
        for index, row in edited_df_b.iterrows():
            original_row = df_b.iloc[index]
            if not np.array_equal(row.values, original_row.values):
                update_blob('B', int(row['id']), row['x'], row['y'], row['variance'], row['height'], row['sign'])
        
        # Remove blob buttons
        for blob_id in edited_df_b['id']:
            if st.button(f"Remove Blob {int(blob_id)}", key=f"remove_b_{blob_id}"):
                remove_blob('B', int(blob_id))

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
        show_both=st.session_state.show_both
    )
    
    # Add click callback
    st.plotly_chart(fig, use_container_width=True, on_click=handle_plot_click)
    
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
        bottleneck = calculate_bottleneck(centers_a, centers_b, weights_a, weights_b)
        
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
            st.metric("Bottleneck Distance", f"{bottleneck:.4f}")
            st.info("Largest minimum distance to transform one distribution into another.")
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
