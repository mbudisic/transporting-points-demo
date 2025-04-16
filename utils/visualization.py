import numpy as np
import plotly.graph_objects as go
from utils.distribution_utils import Distribution, create_gaussian_mixture

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

def create_interactive_plot(distribution_a, distribution_b, active_distribution='A', show_both=True):
    """
    Create an interactive plot that allows users to manipulate the distributions.
    
    Args:
        distribution_a: Distribution object for the first distribution
        distribution_b: Distribution object for the second distribution
        active_distribution: Which distribution is being actively edited
        show_both: Whether to show both distributions or just the active one
        
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
