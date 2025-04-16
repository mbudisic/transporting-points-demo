import numpy as np
import plotly.graph_objects as go
from models.distribution import Distribution
from typing import List, Tuple, Optional, Dict, Any

class VisualizationService:
    """
    Service for visualizing distributions and transport plans
    """
    @staticmethod
    def create_interactive_plot(
        distribution_a: Distribution, 
        distribution_b: Distribution,
        active_distribution: str = 'A',
        show_both: bool = True,
        show_bottleneck_lines: bool = False,
        bottleneck_pairs: Optional[List[Tuple[int, int]]] = None,
        show_wasserstein_lines: bool = False,
        wasserstein_pairs: Optional[List[Tuple[int, int, float]]] = None
    ):
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
        # Create the base figure
        fig = go.Figure()
        
        # Define grid for continuous distributions
        grid_size = 100
        x_grid = np.linspace(0, 10, grid_size)
        y_grid = np.linspace(0, 10, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        
        # Get distribution values on the grid
        if not distribution_a.is_empty:
            dist_a_values = distribution_a.create_gaussian_mixture(points)
            dist_a_grid = dist_a_values.reshape(grid_size, grid_size)
            
            # Show distribution A if it's active or if showing both
            if active_distribution == 'A' or show_both:
                # Add contour plot for distribution A
                fig.add_trace(go.Contour(
                    z=dist_a_grid,
                    x=x_grid,
                    y=y_grid,
                    colorscale=[[0, 'rgba(255,0,0,0)'],  # Transparent for negative values
                               [0.5, 'rgba(255,150,150,0.7)'],  # Light red for low positive values
                               [1, 'rgba(255,0,0,0.7)']],  # Red for high positive values
                    showscale=False,
                    contours=dict(
                        start=0,
                        end=np.max(dist_a_grid) if np.max(dist_a_grid) > 0 else 0.1,
                        size=(np.max(dist_a_grid) if np.max(dist_a_grid) > 0 else 0.1) / 10
                    ),
                    name="Distribution A (Positive)",
                    customdata=np.ones_like(dist_a_grid),
                    hoverinfo="skip"
                ))
                
                # Add contour for negative parts
                fig.add_trace(go.Contour(
                    z=-dist_a_grid,
                    x=x_grid,
                    y=y_grid,
                    colorscale=[[0, 'rgba(255,0,0,0)'],  # Transparent for positive values
                               [0.5, 'rgba(255,100,100,0.7)'],  # Light red for low negative values
                               [1, 'rgba(100,0,0,0.7)']],  # Dark red for high negative values
                    showscale=False,
                    contours=dict(
                        start=0,
                        end=np.max(-dist_a_grid) if np.max(-dist_a_grid) > 0 else 0.1,
                        size=(np.max(-dist_a_grid) if np.max(-dist_a_grid) > 0 else 0.1) / 10
                    ),
                    name="Distribution A (Negative)",
                    customdata=np.ones_like(dist_a_grid) * -1,
                    hoverinfo="skip"
                ))
                
                # Add markers for blob centers in distribution A
                for blob in distribution_a.blobs:
                    marker_symbol = "circle" if blob.sign > 0 else "x"
                    marker_color = 'rgba(255, 0, 0, 0.8)' if blob.sign > 0 else 'rgba(100, 0, 0, 0.8)'
                    
                    fig.add_trace(go.Scatter(
                        x=[blob.x],
                        y=[blob.y],
                        mode='markers+text',
                        marker=dict(
                            symbol=marker_symbol,
                            size=12 + 3 * blob.height,
                            color=marker_color,
                            line=dict(width=2, color='black')
                        ),
                        text=[f"A{blob.id}"],
                        textposition="top center",
                        name=f"Blob A{blob.id}",
                        customdata=[[blob.id, 
                                    blob.variance, 
                                    blob.height, 
                                    blob.sign, 
                                    'A']],
                        hovertemplate="<b>Blob A%{customdata[0]}</b><br>"
                                      "x: %{x:.2f}<br>"
                                      "y: %{y:.2f}<br>"
                                      "variance: %{customdata[1]:.2f}<br>"
                                      "height: %{customdata[2]:.2f}<br>"
                                      "sign: %{customdata[3]}<extra></extra>"
                    ))
                    
                    # Add a circle to represent variance
                    theta = np.linspace(0, 2*np.pi, 100)
                    radius = np.sqrt(2 * blob.variance)  # 2σ circle (captures ~95% of the distribution)
                    circle_x = blob.x + radius * np.cos(theta)
                    circle_y = blob.y + radius * np.sin(theta)
                    
                    fig.add_trace(go.Scatter(
                        x=circle_x,
                        y=circle_y,
                        mode='lines',
                        line=dict(
                            color='rgba(255, 0, 0, 0.4)',
                            width=2,
                            dash='dash'
                        ),
                        name=f"Variance A{blob.id}",
                        hoverinfo="skip",
                        showlegend=False
                    ))
        
        # Similar logic for distribution B
        if not distribution_b.is_empty:
            dist_b_values = distribution_b.create_gaussian_mixture(points)
            dist_b_grid = dist_b_values.reshape(grid_size, grid_size)
            
            # Show distribution B if it's active or if showing both
            if active_distribution == 'B' or show_both:
                # Add contour plot for distribution B
                fig.add_trace(go.Contour(
                    z=dist_b_grid,
                    x=x_grid,
                    y=y_grid,
                    colorscale=[[0, 'rgba(0,0,255,0)'],  # Transparent for negative values
                               [0.5, 'rgba(150,150,255,0.7)'],  # Light blue for low positive values
                               [1, 'rgba(0,0,255,0.7)']],  # Blue for high positive values
                    showscale=False,
                    contours=dict(
                        start=0,
                        end=np.max(dist_b_grid) if np.max(dist_b_grid) > 0 else 0.1,
                        size=(np.max(dist_b_grid) if np.max(dist_b_grid) > 0 else 0.1) / 10
                    ),
                    name="Distribution B (Positive)",
                    customdata=np.ones_like(dist_b_grid),
                    hoverinfo="skip"
                ))
                
                # Add contour for negative parts
                fig.add_trace(go.Contour(
                    z=-dist_b_grid,
                    x=x_grid,
                    y=y_grid,
                    colorscale=[[0, 'rgba(0,0,255,0)'],  # Transparent for positive values
                               [0.5, 'rgba(100,100,255,0.7)'],  # Light blue for low negative values
                               [1, 'rgba(0,0,100,0.7)']],  # Dark blue for high negative values
                    showscale=False,
                    contours=dict(
                        start=0,
                        end=np.max(-dist_b_grid) if np.max(-dist_b_grid) > 0 else 0.1,
                        size=(np.max(-dist_b_grid) if np.max(-dist_b_grid) > 0 else 0.1) / 10
                    ),
                    name="Distribution B (Negative)",
                    customdata=np.ones_like(dist_b_grid) * -1,
                    hoverinfo="skip"
                ))
                
                # Add markers for blob centers in distribution B
                for blob in distribution_b.blobs:
                    marker_symbol = "circle" if blob.sign > 0 else "x"
                    marker_color = 'rgba(0, 0, 255, 0.8)' if blob.sign > 0 else 'rgba(0, 0, 100, 0.8)'
                    
                    fig.add_trace(go.Scatter(
                        x=[blob.x],
                        y=[blob.y],
                        mode='markers+text',
                        marker=dict(
                            symbol=marker_symbol,
                            size=12 + 3 * blob.height,
                            color=marker_color,
                            line=dict(width=2, color='black')
                        ),
                        text=[f"B{blob.id}"],
                        textposition="top center",
                        name=f"Blob B{blob.id}",
                        customdata=[[blob.id, 
                                    blob.variance, 
                                    blob.height, 
                                    blob.sign, 
                                    'B']],
                        hovertemplate="<b>Blob B%{customdata[0]}</b><br>"
                                      "x: %{x:.2f}<br>"
                                      "y: %{y:.2f}<br>"
                                      "variance: %{customdata[1]:.2f}<br>"
                                      "height: %{customdata[2]:.2f}<br>"
                                      "sign: %{customdata[3]}<extra></extra>"
                    ))
                    
                    # Add a circle to represent variance
                    theta = np.linspace(0, 2*np.pi, 100)
                    radius = np.sqrt(2 * blob.variance)  # 2σ circle (captures ~95% of the distribution)
                    circle_x = blob.x + radius * np.cos(theta)
                    circle_y = blob.y + radius * np.sin(theta)
                    
                    fig.add_trace(go.Scatter(
                        x=circle_x,
                        y=circle_y,
                        mode='lines',
                        line=dict(
                            color='rgba(0, 0, 255, 0.4)',
                            width=2,
                            dash='dash'
                        ),
                        name=f"Variance B{blob.id}",
                        hoverinfo="skip",
                        showlegend=False
                    ))
        
        # Add bottleneck transport lines if requested
        if show_bottleneck_lines and bottleneck_pairs and not distribution_a.is_empty and not distribution_b.is_empty:
            for pair in bottleneck_pairs:
                idx_a, idx_b = pair
                # Find the corresponding blobs
                blob_a = distribution_a.get_blob(idx_a)
                blob_b = distribution_b.get_blob(idx_b)
                
                if blob_a and blob_b:
                    # Draw a line between the centers
                    fig.add_trace(go.Scatter(
                        x=[blob_a.x, blob_b.x],
                        y=[blob_a.y, blob_b.y],
                        mode='lines',
                        line=dict(
                            color='rgba(0, 0, 0, 0.8)',
                            width=2
                        ),
                        name=f"Bottleneck A{idx_a}-B{idx_b}",
                        hoverinfo="skip"
                    ))
        
        # Add Wasserstein transport lines if requested
        if show_wasserstein_lines and wasserstein_pairs and not distribution_a.is_empty and not distribution_b.is_empty:
            # Find max weight for normalization
            max_weight = max([w for _, _, w in wasserstein_pairs]) if wasserstein_pairs else 0.01
            
            for pair in wasserstein_pairs:
                idx_a, idx_b, weight = pair
                # Find the corresponding blobs
                blob_a = distribution_a.get_blob(idx_a)
                blob_b = distribution_b.get_blob(idx_b)
                
                if blob_a and blob_b:
                    # Normalize weight for visual scaling
                    normalized_weight = weight / max_weight
                    
                    # Determine line width and opacity based on weight
                    line_width = 1 + 4 * normalized_weight
                    line_opacity = 0.3 + 0.7 * normalized_weight
                    
                    # Draw a line between the centers
                    fig.add_trace(go.Scatter(
                        x=[blob_a.x, blob_b.x],
                        y=[blob_a.y, blob_b.y],
                        mode='lines',
                        line=dict(
                            color=f'rgba(0, 0, 0, {line_opacity})',
                            width=line_width
                        ),
                        name=f"Wasserstein A{idx_a}-B{idx_b}",
                        hoverinfo="skip"
                    ))
        
        # Set figure layout
        fig.update_layout(
            xaxis=dict(
                range=[0, 10],
                title="X",
                constrain="domain"
            ),
            yaxis=dict(
                range=[0, 10],
                title="Y",
                scaleanchor="x",
                scaleratio=1
            ),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='rgba(240,240,240,0.6)',
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def add_blob_labels(fig, blob, dist_name):
        """
        Add a label near the blob center for easier identification
        
        Args:
            fig: Plotly figure to add the label to
            blob: The blob object containing position and id
            dist_name: The distribution name ('A' or 'B')
            
        Returns:
            None (modifies fig in place)
        """
        fig.add_annotation(
            x=blob.x,
            y=blob.y + 0.3,  # Slightly above the center
            text=f"{dist_name}{blob.id}",
            showarrow=False,
            font=dict(
                family="Arial",
                size=10,
                color="black"
            )
        )