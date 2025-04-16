import numpy as np
import plotly.graph_objects as go
from utils.distribution_utils import Distribution, create_gaussian_mixture

def create_wasserstein_transport_plot(distribution_a, distribution_b, wasserstein_pairs):
    """
    Create a plot showing the Wasserstein transportation plan between two distributions.
    
    Args:
        distribution_a: Distribution object for the first distribution
        distribution_b: Distribution object for the second distribution
        wasserstein_pairs: List of tuples (idx_a, idx_b, weight) indicating matched blob indices and transport weight
        
    Returns:
        Plotly figure object with Wasserstein transport lines
    """
    # Get centers of mass if needed for virtual points
    centers_a = [(blob['x'], blob['y']) for blob in distribution_a.blobs]
    centers_b = [(blob['x'], blob['y']) for blob in distribution_b.blobs]
    
    # Calculate center of mass for positive and negative components of each distribution
    pos_centers_a = [c for i, c in enumerate(centers_a) if distribution_a.blobs[i]['sign'] > 0]
    neg_centers_a = [c for i, c in enumerate(centers_a) if distribution_a.blobs[i]['sign'] < 0]
    pos_centers_b = [c for i, c in enumerate(centers_b) if distribution_b.blobs[i]['sign'] > 0]
    neg_centers_b = [c for i, c in enumerate(centers_b) if distribution_b.blobs[i]['sign'] < 0]
    
    # Calculate centers of mass for positive/negative groups
    center_of_mass_pos_a = np.mean(pos_centers_a, axis=0) if pos_centers_a else np.array([2.5, 2.5])
    center_of_mass_neg_a = np.mean(neg_centers_a, axis=0) if neg_centers_a else np.array([7.5, 7.5])
    center_of_mass_pos_b = np.mean(pos_centers_b, axis=0) if pos_centers_b else np.array([2.5, 2.5])
    center_of_mass_neg_b = np.mean(neg_centers_b, axis=0) if neg_centers_b else np.array([7.5, 7.5])
    
    # Calculate overall centers of mass
    center_of_mass_all_a = np.mean(centers_a, axis=0) if centers_a else np.array([5, 5])
    center_of_mass_all_b = np.mean(centers_b, axis=0) if centers_b else np.array([5, 5])
    
    fig = go.Figure()
    
    # Find max weight for normalization
    max_weight = 0.01  # Small default to avoid division by zero
    if wasserstein_pairs:
        max_weight = max([w for _, _, w in wasserstein_pairs]) or 0.01
    
    # Add transport lines
    for pair in wasserstein_pairs:
        idx_a, idx_b, weight = pair
        
        # Normalize weight for visual scaling
        normalized_weight = weight / max_weight
        
        # Handle normal point-to-point connections
        if idx_a >= 0 and idx_b >= 0:
            blob_a = distribution_a.blobs[idx_a]
            blob_b = distribution_b.blobs[idx_b]
            
            # Determine line color based on sign
            if blob_a['sign'] > 0 and blob_b['sign'] > 0:
                # Positive to positive
                line_color = 'rgba(0, 128, 0, {})'.format(0.3 + 0.7 * normalized_weight)  # Green with opacity
            elif blob_a['sign'] < 0 and blob_b['sign'] < 0:
                # Negative to negative
                line_color = 'rgba(255, 165, 0, {})'.format(0.3 + 0.7 * normalized_weight)  # Orange with opacity
            else:
                # This shouldn't happen with our matching algorithm but handle it anyway
                line_color = 'rgba(128, 128, 128, {})'.format(0.3 + 0.7 * normalized_weight)  # Gray with opacity
                
            # Scale line width by flow weight
            line_width = 1 + 4 * normalized_weight
            
            # Draw a line connecting the matched blobs
            fig.add_trace(go.Scatter(
                x=[blob_a['x'], blob_b['x']],
                y=[blob_a['y'], blob_b['y']],
                mode='lines',
                line=dict(color=line_color, width=line_width),
                showlegend=False,
                hoverinfo='text',
                hovertext=f'Flow: A{idx_a} → B{idx_b}<br>Weight: {weight:.4f} ({normalized_weight*100:.1f}%)'
            ))
            
        # Handle virtual points from distribution A to B
        elif idx_a >= 0 and idx_b < 0:
            blob_a = distribution_a.blobs[idx_a]
            
            # Choose target based on virtual point code
            if idx_b == -1:  # Center of mass of positive points in B
                target_x, target_y = center_of_mass_pos_b
                target_name = "center(B+)"
                line_color = 'rgba(0, 128, 0, {})'.format(0.3 + 0.7 * normalized_weight) if blob_a['sign'] > 0 else 'rgba(128, 128, 128, {})'.format(0.3 + 0.7 * normalized_weight)
            elif idx_b == -2:  # Geometric center of all points in B
                target_x, target_y = center_of_mass_all_b
                target_name = "center(B)"
                line_color = 'rgba(128, 128, 128, {})'.format(0.3 + 0.7 * normalized_weight)
            elif idx_b == -3:  # Center of mass of negative points in B
                target_x, target_y = center_of_mass_neg_b
                target_name = "center(B-)"
                line_color = 'rgba(255, 165, 0, {})'.format(0.3 + 0.7 * normalized_weight) if blob_a['sign'] < 0 else 'rgba(128, 128, 128, {})'.format(0.3 + 0.7 * normalized_weight)
            else:  # Geometric center for negative (idx_b == -4)
                target_x, target_y = center_of_mass_all_b
                target_name = "center(B)"
                line_color = 'rgba(128, 128, 128, {})'.format(0.3 + 0.7 * normalized_weight)
            
            # Scale line width by flow weight
            line_width = 1 + 4 * normalized_weight
            line_dash = 'dash'  # Use dashed line for virtual points
            
            # Draw a line connecting the blob to the virtual point
            fig.add_trace(go.Scatter(
                x=[blob_a['x'], target_x],
                y=[blob_a['y'], target_y],
                mode='lines',
                line=dict(color=line_color, width=line_width, dash=line_dash),
                showlegend=False,
                hoverinfo='text',
                hovertext=f'Flow: A{idx_a} → {target_name}<br>Weight: {weight:.4f} ({normalized_weight*100:.1f}%)'
            ))
            
        # Handle virtual points from distribution B to A
        elif idx_a < 0 and idx_b >= 0:
            blob_b = distribution_b.blobs[idx_b]
            
            # Choose source based on virtual point code
            if idx_a == -1:  # Center of mass of positive points in A
                source_x, source_y = center_of_mass_pos_a
                source_name = "center(A+)"
                line_color = 'rgba(0, 128, 0, {})'.format(0.3 + 0.7 * normalized_weight) if blob_b['sign'] > 0 else 'rgba(128, 128, 128, {})'.format(0.3 + 0.7 * normalized_weight)
            elif idx_a == -2:  # Geometric center of all points in A
                source_x, source_y = center_of_mass_all_a
                source_name = "center(A)"
                line_color = 'rgba(128, 128, 128, {})'.format(0.3 + 0.7 * normalized_weight)
            elif idx_a == -3:  # Center of mass of negative points in A
                source_x, source_y = center_of_mass_neg_a
                source_name = "center(A-)"
                line_color = 'rgba(255, 165, 0, {})'.format(0.3 + 0.7 * normalized_weight) if blob_b['sign'] < 0 else 'rgba(128, 128, 128, {})'.format(0.3 + 0.7 * normalized_weight)
            else:  # Geometric center for negative (idx_a == -4)
                source_x, source_y = center_of_mass_all_a
                source_name = "center(A)"
                line_color = 'rgba(128, 128, 128, {})'.format(0.3 + 0.7 * normalized_weight)
            
            # Scale line width by flow weight
            line_width = 1 + 4 * normalized_weight
            line_dash = 'dash'  # Use dashed line for virtual points
            
            # Draw a line connecting the virtual point to the blob
            fig.add_trace(go.Scatter(
                x=[source_x, blob_b['x']],
                y=[source_y, blob_b['y']],
                mode='lines',
                line=dict(color=line_color, width=line_width, dash=line_dash),
                showlegend=False,
                hoverinfo='text',
                hovertext=f'Flow: {source_name} → B{idx_b}<br>Weight: {weight:.4f} ({normalized_weight*100:.1f}%)'
            ))
    
    # Add markers for center of mass points if needed
    if any(idx_a < 0 or idx_b < 0 for idx_a, idx_b, _ in wasserstein_pairs):
        center_points = []
        center_names = []
        center_colors = []
        
        if pos_centers_a:
            center_points.append((center_of_mass_pos_a[0], center_of_mass_pos_a[1]))
            center_names.append("center(A+)")
            center_colors.append("darkred")
        
        if neg_centers_a:
            center_points.append((center_of_mass_neg_a[0], center_of_mass_neg_a[1]))
            center_names.append("center(A-)")
            center_colors.append("orangered")
            
        if pos_centers_b:
            center_points.append((center_of_mass_pos_b[0], center_of_mass_pos_b[1]))
            center_names.append("center(B+)")
            center_colors.append("darkblue")
            
        if neg_centers_b:
            center_points.append((center_of_mass_neg_b[0], center_of_mass_neg_b[1]))
            center_names.append("center(B-)")
            center_colors.append("mediumpurple")
        
        # Add markers for centers of mass
        for (x, y), name, color in zip(center_points, center_names, center_colors):
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    symbol='x',
                    size=8,
                    color=color
                ),
                text=name,
                textposition="top center",
                textfont=dict(size=10),
                showlegend=False,
                hoverinfo='text',
                hovertext=f'{name}: ({x:.2f}, {y:.2f})'
            ))
    
    return fig

def create_bottleneck_transport_plot(distribution_a, distribution_b, matching_pairs):
    """
    Create a plot showing the bottleneck transportation plan between two distributions.
    
    Args:
        distribution_a: Distribution object for the first distribution
        distribution_b: Distribution object for the second distribution
        matching_pairs: List of tuples (idx_a, idx_b) indicating matched blob indices
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Plot background grid
    fig.add_trace(go.Scatter(
        x=[0, 10, 10, 0, 0],
        y=[0, 0, 10, 10, 0],
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False
    ))
    
    # Plot distribution A blobs
    for i, blob in enumerate(distribution_a.blobs):
        # Add center point
        marker_size = 10 + abs(blob['height']) * 2
        marker_color = 'darkred' if blob['sign'] > 0 else 'orangered'
        marker_symbol = 'circle' if blob['sign'] > 0 else 'circle-open'
        
        fig.add_trace(go.Scatter(
            x=[blob['x']],
            y=[blob['y']],
            mode='markers',
            marker=dict(
                symbol=marker_symbol,
                size=marker_size,
                color=marker_color,
                line=dict(width=1, color='darkred')
            ),
            name=f'A{blob["id"]}',
            hoverinfo='text',
            hovertext=f'ID: A{blob["id"]}<br>Position: ({blob["x"]:.2f}, {blob["y"]:.2f})<br>Height: {blob["height"]:.2f}<br>Sign: {blob["sign"]}'
        ))
        
        # Add variance circle
        theta = np.linspace(0, 2*np.pi, 30)
        radius = np.sqrt(blob['variance'] * 2)
        x_circle = blob['x'] + radius * np.cos(theta)
        y_circle = blob['y'] + radius * np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=x_circle,
            y=y_circle,
            mode='lines',
            line=dict(color='red', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='none'
        ))
        
        # Add sign indicator in the center of the blob
        sign_symbol = '+' if blob['sign'] > 0 else '-'
        fig.add_trace(go.Scatter(
            x=[blob['x']],
            y=[blob['y']],
            mode='text',
            text=sign_symbol,
            textfont=dict(
                size=16,
                color='white' if blob['sign'] > 0 else 'black'
            ),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # Plot distribution B blobs
    for i, blob in enumerate(distribution_b.blobs):
        # Add center point
        marker_size = 10 + abs(blob['height']) * 2
        marker_color = 'darkblue' if blob['sign'] > 0 else 'mediumpurple'
        marker_symbol = 'circle' if blob['sign'] > 0 else 'circle-open'
        
        fig.add_trace(go.Scatter(
            x=[blob['x']],
            y=[blob['y']],
            mode='markers',
            marker=dict(
                symbol=marker_symbol,
                size=marker_size,
                color=marker_color,
                line=dict(width=1, color='darkblue')
            ),
            name=f'B{blob["id"]}',
            hoverinfo='text',
            hovertext=f'ID: B{blob["id"]}<br>Position: ({blob["x"]:.2f}, {blob["y"]:.2f})<br>Height: {blob["height"]:.2f}<br>Sign: {blob["sign"]}'
        ))
        
        # Add variance circle
        theta = np.linspace(0, 2*np.pi, 30)
        radius = np.sqrt(blob['variance'] * 2)
        x_circle = blob['x'] + radius * np.cos(theta)
        y_circle = blob['y'] + radius * np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=x_circle,
            y=y_circle,
            mode='lines',
            line=dict(color='blue', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='none'
        ))
        
        # Add sign indicator in the center of the blob
        sign_symbol = '+' if blob['sign'] > 0 else '-'
        fig.add_trace(go.Scatter(
            x=[blob['x']],
            y=[blob['y']],
            mode='text',
            text=sign_symbol,
            textfont=dict(
                size=16,
                color='white' if blob['sign'] > 0 else 'black'
            ),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # Plot the bottleneck matching lines
    for idx_a, idx_b in matching_pairs:
        blob_a = distribution_a.blobs[idx_a]
        blob_b = distribution_b.blobs[idx_b]
        
        # Determine line color based on sign
        line_color = 'green' if blob_a['sign'] > 0 else 'orange'
        
        # Draw a line connecting the matched blobs
        fig.add_trace(go.Scatter(
            x=[blob_a['x'], blob_b['x']],
            y=[blob_a['y'], blob_b['y']],
            mode='lines',
            line=dict(color=line_color, width=2),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'Match: A{blob_a["id"]} ↔ B{blob_b["id"]}<br>Distance: {((blob_a["x"]-blob_b["x"])**2 + (blob_a["y"]-blob_b["y"])**2)**0.5:.2f}'
        ))
    
    # Configure layout
    fig.update_layout(
        title="Bottleneck Distance Transportation Plan",
        xaxis_title="X",
        yaxis_title="Y",
        autosize=True,
        margin=dict(l=50, r=50, b=50, t=70),
        plot_bgcolor="white",
        xaxis=dict(range=[0, 10], constrain="domain"),
        yaxis=dict(range=[0, 10], scaleanchor="x", scaleratio=1),
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Make the plot responsive
    fig.update_layout(
        autosize=True,
        height=600,
    )
    
    return fig

def create_distribution_plot(distribution_a, distribution_b, grid_size=100):
    """
    Create a plot of two distributions on a 2D plane.
    
    Args:
        distribution_a: Distribution object for the first distribution
        distribution_b: Distribution object for the second distribution
        grid_size: Number of points in the grid
        
    Returns:
        Plotly figure object
    """
    # Create evaluation grid
    x_grid = np.linspace(0, 10, grid_size)
    y_grid = np.linspace(0, 10, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Evaluate distributions on the grid
    if distribution_a.blobs:
        dist_a_values = create_gaussian_mixture(distribution_a.blobs, points)
        Z_a = dist_a_values.reshape(grid_size, grid_size)
    else:
        Z_a = np.zeros((grid_size, grid_size))
    
    if distribution_b.blobs:
        dist_b_values = create_gaussian_mixture(distribution_b.blobs, points)
        Z_b = dist_b_values.reshape(grid_size, grid_size)
    else:
        Z_b = np.zeros((grid_size, grid_size))
    
    # Create figure
    fig = go.Figure()
    
    # Add contour plots for the distributions
    fig.add_trace(go.Contour(
        z=Z_a,
        x=x_grid,
        y=y_grid,
        colorscale=[[0, 'rgba(255,0,0,0)'], [1, 'rgba(255,0,0,0.7)']],
        showscale=False,
        name='Distribution A'
    ))
    
    fig.add_trace(go.Contour(
        z=Z_b,
        x=x_grid,
        y=y_grid,
        colorscale=[[0, 'rgba(0,0,255,0)'], [1, 'rgba(0,0,255,0.7)']],
        showscale=False,
        name='Distribution B'
    ))
    
    # Configure layout
    fig.update_layout(
        title="Distribution Visualization",
        xaxis_title="X",
        yaxis_title="Y",
        autosize=True,
        margin=dict(l=50, r=50, b=50, t=70),
        plot_bgcolor="white",
        xaxis=dict(range=[0, 10]),
        yaxis=dict(range=[0, 10]),
    )
    
    return fig

def add_blob_labels(fig, blob, dist_name):
    """
    Add a label near the blob center for easier identification
    
    Args:
        fig: Plotly figure to add the label to
        blob: The blob dictionary containing position and id
        dist_name: The distribution name ('A' or 'B')
        
    Returns:
        None (modifies fig in place)
    """
    label_offset_x = 0.3
    label_offset_y = 0.3
    fig.add_trace(go.Scatter(
        x=[blob['x'] + label_offset_x],
        y=[blob['y'] + label_offset_y],
        mode='text',
        text=f"{dist_name}{blob['id']}",
        textfont=dict(
            size=12,
            color='darkred' if dist_name == 'A' else 'darkblue'
        ),
        showlegend=False,
        hoverinfo='none'
    ))

def create_interactive_plot(distribution_a, distribution_b, active_distribution='A', show_both=True, 
                      show_bottleneck_lines=False, bottleneck_pairs=None,
                      show_wasserstein_lines=False, wasserstein_pairs=None):
    """
    Create an interactive plot that allows users to manipulate the distributions.
    
    Args:
        distribution_a: Distribution object for the first distribution
        distribution_b: Distribution object for the second distribution
        active_distribution: Which distribution is being actively edited
        show_both: Whether to show both distributions or just the active one
        show_bottleneck_lines: Whether to show the bottleneck transport lines
        bottleneck_pairs: List of tuples (idx_a, idx_b) indicating matched blob indices for bottleneck
        show_wasserstein_lines: Whether to show the Wasserstein transport lines
        wasserstein_pairs: List of tuples (idx_a, idx_b, weight) indicating matched blob indices for Wasserstein
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Plot background grid
    fig.add_trace(go.Scatter(
        x=[0, 10, 10, 0, 0],
        y=[0, 0, 10, 10, 0],
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False
    ))
    
    # Create the base contour plot of distributions
    x_grid = np.linspace(0, 10, 100)
    y_grid = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Show distribution A
    if show_both or active_distribution == 'A':
        if distribution_a.blobs:
            dist_a_values = create_gaussian_mixture(distribution_a.blobs, points)
            Z_a = dist_a_values.reshape(100, 100)
            
            opacity = 0.7 if active_distribution == 'A' else 0.3
            
            # Use different color scale for positive and negative values
            # Positive values: red scale
            # Negative values: orange-pink scale
            # Create a custom colorscale
            colorscale = [
                [0, f'rgba(255,150,150,{opacity})'],           # Negative values (pink/orange-ish)
                [0.5, f'rgba(255,255,255,0)'],                 # Zero (transparent white)
                [1, f'rgba(255,0,0,{opacity})']                # Positive values (red)
            ]
            
            fig.add_trace(go.Contour(
                z=Z_a,
                x=x_grid,
                y=y_grid,
                colorscale=colorscale,
                showscale=False,
                name='Distribution A',
                hoverinfo='none',
                contours=dict(
                    start=-1,
                    end=1,
                    size=0.05,
                    showlabels=False
                )
            ))
            
            # Add markers and variance circles for distribution A
            for blob in distribution_a.blobs:
                # Add center point
                marker_size = 10 + abs(blob['height']) * 2  # Use absolute value for size
                # Different colors for positive vs negative height
                marker_color = 'darkred' if blob['sign'] > 0 else 'orangered'
                # Change marker symbol to indicate sign
                marker_symbol = 'circle' if blob['sign'] > 0 else 'circle-open'
                
                fig.add_trace(go.Scatter(
                    x=[blob['x']],
                    y=[blob['y']],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=marker_size,
                        color=marker_color,
                        line=dict(width=1, color='darkred')
                    ),
                    name=f'Blob A{blob["id"]}',
                    hoverinfo='text',
                    hovertext=f'ID: A{blob["id"]}<br>Position: ({blob["x"]:.2f}, {blob["y"]:.2f})<br>Variance: {blob["variance"]:.2f}<br>Height: {blob["height"]:.2f}<br>Sign: {blob["sign"]}',
                    customdata=[{'type': 'center', 'dist': 'A', 'id': blob['id']}]
                ))
                
                # Add sign indicator in the center of the blob
                sign_symbol = '+' if blob['sign'] > 0 else '-'
                fig.add_trace(go.Scatter(
                    x=[blob['x']],
                    y=[blob['y']],
                    mode='text',
                    text=sign_symbol,
                    textfont=dict(
                        size=16,
                        color='white' if blob['sign'] > 0 else 'black'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
                
                # Add variance circle
                theta = np.linspace(0, 2*np.pi, 50)
                radius = np.sqrt(blob['variance'] * 2)  # Scale factor for visualization
                x_circle = blob['x'] + radius * np.cos(theta)
                y_circle = blob['y'] + radius * np.sin(theta)
                
                # Different line style for positive vs negative blobs
                line_color = 'red' if blob['sign'] > 0 else 'orangered'
                line_dash = 'dot' if blob['sign'] > 0 else 'dash'
                
                fig.add_trace(go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    mode='lines',
                    line=dict(color=line_color, width=1, dash=line_dash),
                    showlegend=False,
                    hoverinfo='none',
                    customdata=[{'type': 'variance', 'dist': 'A', 'id': blob['id']}] * len(theta)
                ))
                
                # Add label for the blob
                add_blob_labels(fig, blob, 'A')
    
    # Show distribution B
    if show_both or active_distribution == 'B':
        if distribution_b.blobs:
            dist_b_values = create_gaussian_mixture(distribution_b.blobs, points)
            Z_b = dist_b_values.reshape(100, 100)
            
            opacity = 0.7 if active_distribution == 'B' else 0.3
            
            # Use different color scale for positive and negative values
            # Positive values: blue scale
            # Negative values: purple-ish scale
            # Create a custom colorscale
            colorscale = [
                [0, f'rgba(180,150,255,{opacity})'],           # Negative values (light purple)
                [0.5, f'rgba(255,255,255,0)'],                 # Zero (transparent white)
                [1, f'rgba(0,0,255,{opacity})']                # Positive values (blue)
            ]
            
            fig.add_trace(go.Contour(
                z=Z_b,
                x=x_grid,
                y=y_grid,
                colorscale=colorscale,
                showscale=False,
                name='Distribution B',
                hoverinfo='none',
                contours=dict(
                    start=-1,
                    end=1,
                    size=0.05,
                    showlabels=False
                )
            ))
            
            # Add markers and variance circles for distribution B
            for blob in distribution_b.blobs:
                # Add center point
                marker_size = 10 + abs(blob['height']) * 2  # Use absolute value for size
                # Different colors for positive vs negative height
                marker_color = 'darkblue' if blob['sign'] > 0 else 'mediumpurple'
                # Change marker symbol to indicate sign
                marker_symbol = 'circle' if blob['sign'] > 0 else 'circle-open'
                
                fig.add_trace(go.Scatter(
                    x=[blob['x']],
                    y=[blob['y']],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=marker_size,
                        color=marker_color,
                        line=dict(width=1, color='darkblue')
                    ),
                    name=f'Blob B{blob["id"]}',
                    hoverinfo='text',
                    hovertext=f'ID: B{blob["id"]}<br>Position: ({blob["x"]:.2f}, {blob["y"]:.2f})<br>Variance: {blob["variance"]:.2f}<br>Height: {blob["height"]:.2f}<br>Sign: {blob["sign"]}',
                    customdata=[{'type': 'center', 'dist': 'B', 'id': blob['id']}]
                ))
                
                # Add sign indicator in the center of the blob
                sign_symbol = '+' if blob['sign'] > 0 else '-'
                fig.add_trace(go.Scatter(
                    x=[blob['x']],
                    y=[blob['y']],
                    mode='text',
                    text=sign_symbol,
                    textfont=dict(
                        size=16,
                        color='white' if blob['sign'] > 0 else 'black'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
                
                # Add variance circle
                theta = np.linspace(0, 2*np.pi, 50)
                radius = np.sqrt(blob['variance'] * 2)  # Scale factor for visualization
                x_circle = blob['x'] + radius * np.cos(theta)
                y_circle = blob['y'] + radius * np.sin(theta)
                
                # Different line style for positive vs negative blobs
                line_color = 'blue' if blob['sign'] > 0 else 'mediumpurple'
                line_dash = 'dot' if blob['sign'] > 0 else 'dash'
                
                fig.add_trace(go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    mode='lines',
                    line=dict(color=line_color, width=1, dash=line_dash),
                    showlegend=False,
                    hoverinfo='none',
                    customdata=[{'type': 'variance', 'dist': 'B', 'id': blob['id']}] * len(theta)
                ))
                
                # Add label for the blob
                add_blob_labels(fig, blob, 'B')
    
    # Add instructions for interaction
    fig.add_annotation(
        x=5,
        y=0.5,
        text="Click to add a new blob to the active distribution",
        showarrow=False,
        bordercolor="#888",
        borderwidth=1,
        bgcolor="#f8f8f8",
        opacity=0.8
    )
    
    # Add bottleneck transport lines if requested
    if show_bottleneck_lines and bottleneck_pairs:
        for idx_a, idx_b in bottleneck_pairs:
            try:
                # Get the blob objects using their indices
                if idx_a < len(distribution_a.blobs) and idx_b < len(distribution_b.blobs):
                    blob_a = distribution_a.blobs[idx_a]
                    blob_b = distribution_b.blobs[idx_b]
                    
                    # Determine line color based on sign
                    line_color = 'green' if blob_a['sign'] > 0 else 'orange'
                    line_dash = 'solid'
                    line_width = 2
                    
                    # Draw a line connecting the matched blobs
                    fig.add_trace(go.Scatter(
                        x=[blob_a['x'], blob_b['x']],
                        y=[blob_a['y'], blob_b['y']],
                        mode='lines',
                        line=dict(color=line_color, width=line_width, dash=line_dash),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f'Bottleneck: A{blob_a["id"]} ↔ B{blob_b["id"]}<br>Distance: {((blob_a["x"]-blob_b["x"])**2 + (blob_a["y"]-blob_b["y"])**2)**0.5:.2f}'
                    ))
            except (IndexError, KeyError):
                # Skip invalid indices or missing data
                continue
    
    # Add Wasserstein transport lines if requested
    if show_wasserstein_lines and wasserstein_pairs:
        # Get centers of mass if needed for virtual points
        centers_a = [(blob['x'], blob['y']) for blob in distribution_a.blobs]
        centers_b = [(blob['x'], blob['y']) for blob in distribution_b.blobs]
        
        # Calculate center of mass for positive and negative components of each distribution
        pos_centers_a = [c for i, c in enumerate(centers_a) if distribution_a.blobs[i]['sign'] > 0]
        neg_centers_a = [c for i, c in enumerate(centers_a) if distribution_a.blobs[i]['sign'] < 0]
        pos_centers_b = [c for i, c in enumerate(centers_b) if distribution_b.blobs[i]['sign'] > 0]
        neg_centers_b = [c for i, c in enumerate(centers_b) if distribution_b.blobs[i]['sign'] < 0]
        
        # Calculate centers of mass for positive/negative groups
        center_of_mass_pos_a = np.mean(pos_centers_a, axis=0) if pos_centers_a else np.array([2.5, 2.5])
        center_of_mass_neg_a = np.mean(neg_centers_a, axis=0) if neg_centers_a else np.array([7.5, 7.5])
        center_of_mass_pos_b = np.mean(pos_centers_b, axis=0) if pos_centers_b else np.array([2.5, 2.5])
        center_of_mass_neg_b = np.mean(neg_centers_b, axis=0) if neg_centers_b else np.array([7.5, 7.5])
        
        # Calculate overall centers of mass
        center_of_mass_all_a = np.mean(centers_a, axis=0) if centers_a else np.array([5, 5])
        center_of_mass_all_b = np.mean(centers_b, axis=0) if centers_b else np.array([5, 5])
        
        for pair in wasserstein_pairs:
            idx_a, idx_b, weight = pair
            
            # Handle normal point-to-point connections
            if idx_a >= 0 and idx_b >= 0:
                try:
                    blob_a = distribution_a.blobs[idx_a]
                    blob_b = distribution_b.blobs[idx_b]
                    
                    # Determine line color and width based on sign and weight
                    line_color = 'teal' if blob_a['sign'] > 0 else 'purple'
                    line_width = 1 + 3 * weight  # Scale line width by flow weight
                    
                    # Draw a line connecting the matched blobs
                    fig.add_trace(go.Scatter(
                        x=[blob_a['x'], blob_b['x']],
                        y=[blob_a['y'], blob_b['y']],
                        mode='lines',
                        line=dict(color=line_color, width=line_width),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f'Wasserstein: A{idx_a} → B{idx_b}<br>Weight: {weight:.2f}'
                    ))
                except (IndexError, KeyError):
                    # Skip invalid indices or missing data
                    continue
                
            # Handle virtual points from distribution A to B
            elif idx_a >= 0 and idx_b < 0:
                try:
                    blob_a = distribution_a.blobs[idx_a]
                    
                    # Choose target based on virtual point code
                    if idx_b == -1:  # Center of mass of positive points in B
                        target_x, target_y = center_of_mass_pos_b
                        target_name = "center(B+)"
                    elif idx_b == -2:  # Geometric center of all points in B
                        target_x, target_y = center_of_mass_all_b
                        target_name = "center(B)"
                    elif idx_b == -3:  # Center of mass of negative points in B
                        target_x, target_y = center_of_mass_neg_b
                        target_name = "center(B-)"
                    else:  # Geometric center for negative (idx_b == -4)
                        target_x, target_y = center_of_mass_all_b
                        target_name = "center(B)"
                    
                    # Determine line color based on sign
                    line_color = 'teal' if blob_a['sign'] > 0 else 'purple'
                    line_width = 1 + 3 * weight  # Scale line width by flow weight
                    line_dash = 'dash'  # Use dashed line for virtual points
                    
                    # Draw a line connecting the blob to the virtual point
                    fig.add_trace(go.Scatter(
                        x=[blob_a['x'], target_x],
                        y=[blob_a['y'], target_y],
                        mode='lines',
                        line=dict(color=line_color, width=line_width, dash=line_dash),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f'Wasserstein: A{idx_a} → {target_name}<br>Weight: {weight:.2f}'
                    ))
                    
                    # Add marker for the virtual point if not already added
                    fig.add_trace(go.Scatter(
                        x=[target_x],
                        y=[target_y],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=8,
                            color=line_color
                        ),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f'{target_name}: ({target_x:.2f}, {target_y:.2f})'
                    ))
                except (IndexError, KeyError):
                    # Skip invalid indices or missing data
                    continue
                
            # Handle virtual points from distribution B to A
            elif idx_a < 0 and idx_b >= 0:
                try:
                    blob_b = distribution_b.blobs[idx_b]
                    
                    # Choose source based on virtual point code
                    if idx_a == -1:  # Center of mass of positive points in A
                        source_x, source_y = center_of_mass_pos_a
                        source_name = "center(A+)"
                    elif idx_a == -2:  # Geometric center of all points in A
                        source_x, source_y = center_of_mass_all_a
                        source_name = "center(A)"
                    elif idx_a == -3:  # Center of mass of negative points in A
                        source_x, source_y = center_of_mass_neg_a
                        source_name = "center(A-)"
                    else:  # Geometric center for negative (idx_a == -4)
                        source_x, source_y = center_of_mass_all_a
                        source_name = "center(A)"
                    
                    # Determine line color based on sign
                    line_color = 'teal' if blob_b['sign'] > 0 else 'purple'
                    line_width = 1 + 3 * weight  # Scale line width by flow weight
                    line_dash = 'dash'  # Use dashed line for virtual points
                    
                    # Draw a line connecting the virtual point to the blob
                    fig.add_trace(go.Scatter(
                        x=[source_x, blob_b['x']],
                        y=[source_y, blob_b['y']],
                        mode='lines',
                        line=dict(color=line_color, width=line_width, dash=line_dash),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f'Wasserstein: {source_name} → B{idx_b}<br>Weight: {weight:.2f}'
                    ))
                    
                    # Add marker for the virtual point if not already added
                    fig.add_trace(go.Scatter(
                        x=[source_x],
                        y=[source_y],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=8,
                            color=line_color
                        ),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f'{source_name}: ({source_x:.2f}, {source_y:.2f})'
                    ))
                except (IndexError, KeyError):
                    # Skip invalid indices or missing data
                    continue
    
    # Configure layout
    fig.update_layout(
        title="Interactive Distribution Editor",
        xaxis_title="X",
        yaxis_title="Y",
        autosize=True,
        margin=dict(l=50, r=50, b=50, t=70),
        plot_bgcolor="white",
        xaxis=dict(range=[0, 10], constrain="domain"),
        yaxis=dict(range=[0, 10], scaleanchor="x", scaleratio=1),
        hovermode='closest',
        dragmode='pan',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        modebar=dict(
            orientation='v'
        ),
        clickmode='event'
    )
    
    # Make the plot responsive
    fig.update_layout(
        autosize=True,
        height=600,
    )
    
    return fig
