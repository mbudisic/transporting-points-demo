import numpy as np
import plotly.graph_objects as go
from models.distribution import Distribution
from typing import List, Tuple, Optional, Dict, Any

class VisualizationService:
    """
    Service for visualizing distributions and transport plans
    """
    @staticmethod
    def _scale_heights_for_markers(distribution_a: Distribution, distribution_b: Distribution):
        """
        Scale absolute height values to a range between 1 and 5
        
        Args:
            distribution_a: Distribution A
            distribution_b: Distribution B
            
        Returns:
            A function that scales an absolute height to the range [1, 5]
        """
        # Collect all heights from both distributions
        all_heights = []
        for blob in distribution_a.blobs:
            all_heights.append(abs(blob.height))
        for blob in distribution_b.blobs:
            all_heights.append(abs(blob.height))
            
        # If no heights (no blobs), return default size of 3
        if not all_heights:
            return lambda h: 3
            
        # Find min and max to create scaling
        min_height = min(all_heights) if all_heights else 0
        max_height = max(all_heights) if all_heights else 1
        
        # Handle case where all heights are the same
        if min_height == max_height:
            return lambda h: 3  # Middle of the range
            
        # Create scaling function
        def scale_height(height):
            abs_height = abs(height)
            # Scale between 1 and 5
            return 1 + 4 * (abs_height - min_height) / (max_height - min_height)
            
        return scale_height
        
    @staticmethod
    def _add_transport_edges(
        fig, 
        distribution_a: Distribution, 
        distribution_b: Distribution,
        show_bottleneck_lines: bool = False,
        bottleneck_pairs: Optional[List[Tuple[int, int]]] = None,
        show_wasserstein_lines: bool = False,
        wasserstein_pairs: Optional[List[Tuple[int, int, float]]] = None,
        show_height_bottleneck_lines: bool = False,
        height_bottleneck_pairs: Optional[List[Tuple[int, int]]] = None,
        show_height_wasserstein_lines: bool = False,
        height_wasserstein_pairs: Optional[List[Tuple[int, int, float]]] = None
    ):
        """
        Add transport edges between blobs in a better way using techniques from the graph visualization.
        
        Args:
            fig: The plotly figure to add edges to
            distribution_a: Distribution A
            distribution_b: Distribution B
            show_bottleneck_lines: Whether to show spatial bottleneck edges
            bottleneck_pairs: Spatial bottleneck matching pairs
            show_wasserstein_lines: Whether to show spatial Wasserstein edges
            wasserstein_pairs: Spatial Wasserstein transport plan
            show_height_bottleneck_lines: Whether to show height-based bottleneck edges
            height_bottleneck_pairs: Height-based bottleneck matching pairs
            show_height_wasserstein_lines: Whether to show height-based Wasserstein edges
            height_wasserstein_pairs: Height-based Wasserstein transport plan
        """
        # Calculate max weight for scaling (needed for Wasserstein)
        max_weight = 0.1  # Minimum to avoid division by zero
        
        if show_wasserstein_lines and wasserstein_pairs:
            for _, _, weight in wasserstein_pairs:
                max_weight = max(max_weight, abs(weight))
                
        if show_height_wasserstein_lines and height_wasserstein_pairs:
            for _, _, weight in height_wasserstein_pairs:
                max_weight = max(max_weight, abs(weight))
        
        # Add bottleneck matching lines (spatial)
        if show_bottleneck_lines and bottleneck_pairs:
            for pair in bottleneck_pairs:
                idx_a, idx_b = pair
                if idx_a < len(distribution_a.blobs) and idx_b < len(distribution_b.blobs):
                    blob_a = distribution_a.blobs[idx_a]
                    blob_b = distribution_b.blobs[idx_b]
                    
                    # Use solid black lines for bottleneck
                    fig.add_trace(go.Scatter(
                        x=[blob_a.x, blob_b.x],
                        y=[blob_a.y, blob_b.y],
                        mode='lines',
                        line=dict(
                            width=2,
                            color='rgba(0, 0, 0, 0.7)',
                            dash='solid'
                        ),
                        name=f"Bottleneck {idx_a}-{idx_b}",
                        hoverinfo="skip"
                    ))
        
        # Add Wasserstein transport plan lines (spatial)
        if show_wasserstein_lines and wasserstein_pairs:
            for pair in wasserstein_pairs:
                idx_a, idx_b, weight = pair
                if idx_a < len(distribution_a.blobs) and idx_b < len(distribution_b.blobs):
                    blob_a = distribution_a.blobs[idx_a]
                    blob_b = distribution_b.blobs[idx_b]
                    
                    # Scale width based on normalized weight
                    # Ensure weight is positive for line width
                    edge_width = max(1, 5 * (abs(weight) / max_weight))
                    
                    # Use dotted black lines for Wasserstein
                    fig.add_trace(go.Scatter(
                        x=[blob_a.x, blob_b.x],
                        y=[blob_a.y, blob_b.y],
                        mode='lines',
                        line=dict(
                            width=edge_width,
                            color='rgba(0, 0, 0, 0.7)',
                            dash='dot'
                        ),
                        name=f"Wasserstein {idx_a}-{idx_b}",
                        hoverinfo="skip"
                    ))
        
        # Add height-based bottleneck matching lines
        if show_height_bottleneck_lines and height_bottleneck_pairs:
            for pair in height_bottleneck_pairs:
                idx_a, idx_b = pair
                if idx_a < len(distribution_a.blobs) and idx_b < len(distribution_b.blobs):
                    blob_a = distribution_a.blobs[idx_a]
                    blob_b = distribution_b.blobs[idx_b]
                    
                    # Use solid magenta lines for height-based bottleneck
                    fig.add_trace(go.Scatter(
                        x=[blob_a.x, blob_b.x],
                        y=[blob_a.y, blob_b.y],
                        mode='lines',
                        line=dict(
                            width=2,
                            color='rgba(200, 30, 150, 0.7)',
                            dash='solid'
                        ),
                        name=f"Height Bottleneck {idx_a}-{idx_b}",
                        hoverinfo="skip"
                    ))
        
        # Add height-based Wasserstein transport plan lines
        if show_height_wasserstein_lines and height_wasserstein_pairs:
            for pair in height_wasserstein_pairs:
                idx_a, idx_b, weight = pair
                if idx_a < len(distribution_a.blobs) and idx_b < len(distribution_b.blobs):
                    blob_a = distribution_a.blobs[idx_a]
                    blob_b = distribution_b.blobs[idx_b]
                    
                    # Scale width based on normalized weight
                    # Ensure weight is positive for line width
                    edge_width = max(1, 5 * (abs(weight) / max_weight))
                    
                    # Use dotted magenta lines for height-based Wasserstein
                    fig.add_trace(go.Scatter(
                        x=[blob_a.x, blob_b.x],
                        y=[blob_a.y, blob_b.y],
                        mode='lines',
                        line=dict(
                            width=edge_width,
                            color='rgba(200, 30, 150, 0.7)',
                            dash='dot'
                        ),
                        name=f"Height Wasserstein {idx_a}-{idx_b}",
                        hoverinfo="skip"
                    ))
                    
        return fig
    
    @staticmethod
    def create_interactive_plot(
        distribution_a: Distribution, 
        distribution_b: Distribution,
        active_distribution: str = 'A',
        show_both: bool = True,
        show_bottleneck_lines: bool = False,
        bottleneck_pairs: Optional[List[Tuple[int, int]]] = None,
        show_wasserstein_lines: bool = False,
        wasserstein_pairs: Optional[List[Tuple[int, int, float]]] = None,
        show_height_bottleneck_lines: bool = False,
        height_bottleneck_pairs: Optional[List[Tuple[int, int]]] = None,
        show_height_wasserstein_lines: bool = False,
        height_wasserstein_pairs: Optional[List[Tuple[int, int, float]]] = None
    ):
        """
        Create an interactive plot that allows users to manipulate the distributions.
        
        Args:
            distribution_a: Distribution object for the first distribution
            distribution_b: Distribution object for the second distribution
            active_distribution: Which distribution is being actively edited
            show_both: Whether to show both distributions or just the active one
            show_bottleneck_lines: Whether to show the spatial bottleneck transport lines
            bottleneck_pairs: List of tuples (idx_a, idx_b) indicating matched blob indices for spatial bottleneck
            show_wasserstein_lines: Whether to show the spatial Wasserstein transport lines
            wasserstein_pairs: List of tuples (idx_a, idx_b, weight) indicating matched blob indices for spatial Wasserstein
            show_height_bottleneck_lines: Whether to show the height-based bottleneck transport lines
            height_bottleneck_pairs: List of tuples (idx_a, idx_b) indicating matched blob indices for height-based bottleneck
            show_height_wasserstein_lines: Whether to show the height-based Wasserstein transport lines
            height_wasserstein_pairs: List of tuples (idx_a, idx_b, weight) indicating matched blob indices for height-based Wasserstein
            
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
        
        # Create height scaling function
        height_scaler = VisualizationService._scale_heights_for_markers(distribution_a, distribution_b)
        
        # Get distribution values on the grid
        if not distribution_a.is_empty:
            dist_a_values = distribution_a.create_gaussian_mixture(points)
            dist_a_grid = dist_a_values.reshape(grid_size, grid_size)
            
            # Show distribution A if it's active or if showing both
            if active_distribution == 'A' or show_both:
                # We've removed the contour plots in favor of just the variance circles
                
                # Add markers for blob centers in distribution A
                for blob in distribution_a.blobs:
                    marker_symbol = "circle"
                    marker_color = 'rgba(0, 158, 115, 0.8)'  # Teal (#009E73)
                    sign_symbol = "+" if blob.height > 0 else "-"
                    
                    fig.add_trace(go.Scatter(
                        x=[blob.x],
                        y=[blob.y],
                        mode='markers+text',
                        marker=dict(
                            symbol=marker_symbol,
                            size=12 * height_scaler(blob.height),
                            color=marker_color,
                            line=dict(width=2, color='black')
                        ),
                        text=[f"A{blob.id}"],
                        textfont=dict(
                            color='rgba(0, 255, 0, 1)',  # Neon green
                            size=12,
                            family='Arial Black'
                        ),
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
                    
                    # Add sign symbol (+/-) in the center of the blob
                    fig.add_trace(go.Scatter(
                        x=[blob.x],
                        y=[blob.y],
                        mode='text',
                        text=[sign_symbol],
                        textfont=dict(
                            color='rgba(0, 255, 0, 1)',  # Neon green
                            size=16,
                            family='Arial Black'
                        ),
                        name=f"Sign A{blob.id}",
                        hoverinfo="skip",
                        showlegend=False
                    ))
                    
                    # Add a circle to represent variance (66% of initial magnitude)
                    theta = np.linspace(0, 2*np.pi, 100)
                    radius = np.sqrt(blob.variance * 0.66)  # 66% of initial magnitude
                    circle_x = blob.x + radius * np.cos(theta)
                    circle_y = blob.y + radius * np.sin(theta)
                    
                    # Line width based on height (1-4 pixels)
                    line_width = 1 + 3 * min(abs(blob.height), 1.0)
                    
                    # Dash type based on sign (dashed for positive, dotted for negative)
                    dash_type = 'dash' if blob.height > 0 else 'dot'
                    
                    fig.add_trace(go.Scatter(
                        x=circle_x,
                        y=circle_y,
                        mode='lines',
                        line=dict(
                            color='rgba(0, 158, 115, 0.6)',  # Teal with transparency
                            width=line_width,
                            dash=dash_type
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
                # We've removed the contour plots in favor of just the variance circles
                
                # Add markers for blob centers in distribution B
                for blob in distribution_b.blobs:
                    marker_symbol = "circle"
                    marker_color = 'rgba(230, 159, 0, 0.8)'  # Orange (#E69F00)
                    sign_symbol = "+" if blob.height > 0 else "-"
                    
                    fig.add_trace(go.Scatter(
                        x=[blob.x],
                        y=[blob.y],
                        mode='markers+text',
                        marker=dict(
                            symbol=marker_symbol,
                            size=12 * height_scaler(blob.height),
                            color=marker_color,
                            line=dict(width=2, color='black')
                        ),
                        text=[f"B{blob.id}"],
                        textfont=dict(
                            color='rgba(0, 255, 0, 1)',  # Neon green
                            size=12,
                            family='Arial Black'
                        ),
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
                    
                    # Add sign symbol (+/-) in the center of the blob
                    fig.add_trace(go.Scatter(
                        x=[blob.x],
                        y=[blob.y],
                        mode='text',
                        text=[sign_symbol],
                        textfont=dict(
                            color='rgba(0, 255, 0, 1)',  # Neon green
                            size=16,
                            family='Arial Black'
                        ),
                        name=f"Sign B{blob.id}",
                        hoverinfo="skip",
                        showlegend=False
                    ))
                    
                    # Add a circle to represent variance (66% of initial magnitude)
                    theta = np.linspace(0, 2*np.pi, 100)
                    radius = np.sqrt(blob.variance * 0.66)  # 66% of initial magnitude
                    circle_x = blob.x + radius * np.cos(theta)
                    circle_y = blob.y + radius * np.sin(theta)
                    
                    # Line width based on height (1-4 pixels)
                    line_width = 1 + 3 * min(abs(blob.height), 1.0)
                    
                    # Dash type based on sign (dashed for positive, dotted for negative)
                    dash_type = 'dash' if blob.height > 0 else 'dot'
                    
                    fig.add_trace(go.Scatter(
                        x=circle_x,
                        y=circle_y,
                        mode='lines',
                        line=dict(
                            color='rgba(230, 159, 0, 0.6)',  # Orange with transparency
                            width=line_width,
                            dash=dash_type
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
            # Find max weight for normalization, ensure it's positive to avoid division by zero
            max_weight = max(max([w for _, _, w in wasserstein_pairs]) if wasserstein_pairs else 0.01, 0.01)
            
            for pair in wasserstein_pairs:
                idx_a, idx_b, weight = pair
                # Find the corresponding blobs
                blob_a = distribution_a.get_blob(idx_a)
                blob_b = distribution_b.get_blob(idx_b)
                
                if blob_a and blob_b:
                    # Normalize weight for visual scaling
                    normalized_weight = weight / max_weight
                    
                    # Determine line width and opacity based on weight
                    line_width = max(0.5, 1 + 4 * abs(normalized_weight))
                    # Ensure opacity is between 0 and 1
                    line_opacity = max(0.1, min(1.0, 0.3 + 0.7 * abs(normalized_weight)))
                    
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
                    
        # Add height-based bottleneck transport lines if requested (using a different color/style)
        if show_height_bottleneck_lines and height_bottleneck_pairs and not distribution_a.is_empty and not distribution_b.is_empty:
            for pair in height_bottleneck_pairs:
                idx_a, idx_b = pair
                # Find the corresponding blobs
                blob_a = distribution_a.get_blob(idx_a)
                blob_b = distribution_b.get_blob(idx_b)
                
                if blob_a and blob_b:
                    # Draw a magenta line between the centers
                    fig.add_trace(go.Scatter(
                        x=[blob_a.x, blob_b.x],
                        y=[blob_a.y, blob_b.y],
                        mode='lines',
                        line=dict(
                            color='rgba(200, 30, 150, 0.8)',  # Magenta
                            width=2,
                            dash='dash'  # Dashed line for height-based
                        ),
                        name=f"Height Bottleneck A{idx_a}-B{idx_b}",
                        hoverinfo="skip"
                    ))
        
        # Add height-based Wasserstein transport lines if requested (using a different color/style)
        if show_height_wasserstein_lines and height_wasserstein_pairs and not distribution_a.is_empty and not distribution_b.is_empty:
            # Find max weight for normalization, ensure it's positive to avoid division by zero
            max_weight = max(max([w for _, _, w in height_wasserstein_pairs]) if height_wasserstein_pairs else 0.01, 0.01)
            
            for pair in height_wasserstein_pairs:
                idx_a, idx_b, weight = pair
                # Find the corresponding blobs
                blob_a = distribution_a.get_blob(idx_a)
                blob_b = distribution_b.get_blob(idx_b)
                
                if blob_a and blob_b:
                    # Normalize weight for visual scaling
                    normalized_weight = weight / max_weight
                    
                    # Determine line width and opacity based on weight
                    line_width = max(0.5, 1 + 4 * abs(normalized_weight))
                    # Ensure opacity is between 0 and 1
                    line_opacity = max(0.1, min(1.0, 0.3 + 0.7 * abs(normalized_weight)))
                    
                    # Draw a line between the centers
                    fig.add_trace(go.Scatter(
                        x=[blob_a.x, blob_b.x],
                        y=[blob_a.y, blob_b.y],
                        mode='lines',
                        line=dict(
                            color=f'rgba(200, 30, 150, {line_opacity})',  # Magenta
                            width=line_width,
                            dash='dot'  # Dotted line for height-based Wasserstein
                        ),
                        name=f"Height Wasserstein A{idx_a}-B{idx_b}",
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