import numpy as np
import plotly.graph_objects as go
import networkx as nx
from models.distribution import Distribution
from typing import List, Tuple, Optional, Dict, Any

class GraphVisualizationService:
    """
    Service for visualizing distributions as graphs with nodes representing blobs
    and edges representing transportation plans between distributions
    """
    
    @staticmethod
    def create_graph_visualization(
        distribution_a: Distribution, 
        distribution_b: Distribution,
        transport_mode: str = 'hide',
        bottleneck_pairs: Optional[List[Tuple[int, int]]] = None,
        wasserstein_pairs: Optional[List[Tuple[int, int, float]]] = None,
        height_bottleneck_pairs: Optional[List[Tuple[int, int]]] = None,
        height_wasserstein_pairs: Optional[List[Tuple[int, int, float]]] = None
    ):
        """
        Create a graph visualization where:
        - Nodes are blobs from both distributions 
        - Edges represent transport plans between blobs
        - Positive and negative blobs are visualized separately
        
        Args:
            distribution_a: Distribution A
            distribution_b: Distribution B
            transport_mode: Which transport plan to show ('hide', 'bottleneck_spatial', etc.)
            bottleneck_pairs: List of (idx_a, idx_b) pairs for spatial bottleneck transport
            wasserstein_pairs: List of (idx_a, idx_b, weight) tuples for spatial Wasserstein transport
            height_bottleneck_pairs: List of (idx_a, idx_b) pairs for height-based bottleneck transport
            height_wasserstein_pairs: List of (idx_a, idx_b, weight) tuples for height-based Wasserstein transport
            
        Returns:
            Plotly figure object
        """
        # Create two graphs: one for positive blobs, one for negative blobs
        G_pos = nx.Graph()
        G_neg = nx.Graph()
        
        # Node positions
        pos_node_positions = {}
        neg_node_positions = {}
        
        # Node attributes
        pos_node_attrs = {}
        neg_node_attrs = {}
        
        # Collect all blob information
        for blob in distribution_a.blobs:
            node_id = f"A{blob.id}"
            
            # Store attributes common to both positive and negative nodes
            node_attrs = {
                'dist': 'A',
                'blob_id': blob.id,
                'variance': blob.variance,
                'height': blob.height,
                'abs_height': abs(blob.height),
                'x': blob.x,
                'y': blob.y
            }
            
            # Assign to appropriate graph based on sign
            if blob.height >= 0:
                G_pos.add_node(node_id)
                pos_node_positions[node_id] = (blob.x, blob.y)
                pos_node_attrs[node_id] = node_attrs
            else:
                G_neg.add_node(node_id)
                neg_node_positions[node_id] = (blob.x, blob.y)
                neg_node_attrs[node_id] = node_attrs
        
        for blob in distribution_b.blobs:
            node_id = f"B{blob.id}"
            
            # Store attributes
            node_attrs = {
                'dist': 'B',
                'blob_id': blob.id,
                'variance': blob.variance,
                'height': blob.height,
                'abs_height': abs(blob.height),
                'x': blob.x,
                'y': blob.y
            }
            
            # Assign to appropriate graph based on sign
            if blob.height >= 0:
                G_pos.add_node(node_id)
                pos_node_positions[node_id] = (blob.x, blob.y)
                pos_node_attrs[node_id] = node_attrs
            else:
                G_neg.add_node(node_id)
                neg_node_positions[node_id] = (blob.x, blob.y)
                neg_node_attrs[node_id] = node_attrs
        
        # Add edges based on transport plan
        if transport_mode != 'hide':
            if transport_mode == 'bottleneck_spatial' and bottleneck_pairs:
                for idx_a, idx_b in bottleneck_pairs:
                    blob_a = distribution_a.get_blob(idx_a)
                    blob_b = distribution_b.get_blob(idx_b)
                    
                    if blob_a and blob_b:
                        node_a = f"A{idx_a}"
                        node_b = f"B{idx_b}"
                        
                        # Calculate distance for edge weight
                        distance = np.sqrt((blob_a.x - blob_b.x)**2 + (blob_a.y - blob_b.y)**2)
                        
                        # Add edge to appropriate graph based on signs
                        if blob_a.height >= 0 and blob_b.height >= 0:
                            G_pos.add_edge(node_a, node_b, weight=distance, type='bottleneck')
                        elif blob_a.height < 0 and blob_b.height < 0:
                            G_neg.add_edge(node_a, node_b, weight=distance, type='bottleneck')
            
            elif transport_mode == 'wasserstein_spatial' and wasserstein_pairs:
                for idx_a, idx_b, weight in wasserstein_pairs:
                    blob_a = distribution_a.get_blob(idx_a)
                    blob_b = distribution_b.get_blob(idx_b)
                    
                    if blob_a and blob_b:
                        node_a = f"A{idx_a}"
                        node_b = f"B{idx_b}"
                        
                        # Add edge to appropriate graph based on signs
                        if blob_a.height >= 0 and blob_b.height >= 0:
                            G_pos.add_edge(node_a, node_b, weight=abs(weight), type='wasserstein')
                        elif blob_a.height < 0 and blob_b.height < 0:
                            G_neg.add_edge(node_a, node_b, weight=abs(weight), type='wasserstein')
            
            elif transport_mode == 'bottleneck_height' and height_bottleneck_pairs:
                for idx_a, idx_b in height_bottleneck_pairs:
                    blob_a = distribution_a.get_blob(idx_a)
                    blob_b = distribution_b.get_blob(idx_b)
                    
                    if blob_a and blob_b:
                        node_a = f"A{idx_a}"
                        node_b = f"B{idx_b}"
                        
                        # Calculate height difference for edge weight
                        height_diff = abs(abs(blob_a.height) - abs(blob_b.height))
                        
                        # Add edge to appropriate graph based on signs
                        if blob_a.height >= 0 and blob_b.height >= 0:
                            G_pos.add_edge(node_a, node_b, weight=height_diff, type='height_bottleneck')
                        elif blob_a.height < 0 and blob_b.height < 0:
                            G_neg.add_edge(node_a, node_b, weight=height_diff, type='height_bottleneck')
            
            elif transport_mode == 'wasserstein_height' and height_wasserstein_pairs:
                for idx_a, idx_b, weight in height_wasserstein_pairs:
                    blob_a = distribution_a.get_blob(idx_a)
                    blob_b = distribution_b.get_blob(idx_b)
                    
                    if blob_a and blob_b:
                        node_a = f"A{idx_a}"
                        node_b = f"B{idx_b}"
                        
                        # Add edge to appropriate graph based on signs
                        if blob_a.height >= 0 and blob_b.height >= 0:
                            G_pos.add_edge(node_a, node_b, weight=abs(weight), type='height_wasserstein')
                        elif blob_a.height < 0 and blob_b.height < 0:
                            G_neg.add_edge(node_a, node_b, weight=abs(weight), type='height_wasserstein')
        
        # Create figure with two subplots (one for positive, one for negative)
        fig = go.Figure()
        
        # Function to scale node sizes based on height
        def scale_node_size(height):
            return 20 + 80 * (abs(height) / max(max_abs_height, 0.1))
        
        # Find max absolute height for scaling
        max_abs_height = 0
        for blob in distribution_a.blobs + distribution_b.blobs:
            max_abs_height = max(max_abs_height, abs(blob.height))
        
        # Visualize positive graph
        if pos_node_positions:
            # Add nodes
            node_x = []
            node_y = []
            node_size = []
            node_color = []
            node_text = []
            
            for node, pos in pos_node_positions.items():
                node_x.append(pos[0])
                node_y.append(pos[1])
                attrs = pos_node_attrs[node]
                
                # Size based on height
                node_size.append(scale_node_size(attrs['height']))
                
                # Color based on distribution
                node_color.append('rgba(0, 158, 115, 0.8)' if attrs['dist'] == 'A' else 'rgba(230, 159, 0, 0.8)')
                
                # Text label
                node_text.append(f"{node}: {attrs['height']:.2f}")
            
            # Add nodes trace
            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=2, color='black')
                ),
                text=node_text,
                textposition="top center",
                name="Positive Blobs",
                hoverinfo="text"
            ))
            
            # Add edges if transport plan is selected
            if transport_mode != 'hide':
                for u, v, data in G_pos.edges(data=True):
                    x0, y0 = pos_node_positions[u]
                    x1, y1 = pos_node_positions[v]
                    
                    # Edge width based on weight 
                    edge_width = max(1, 5 * (data['weight'] / max(max_edge_weight, 0.1))) if G_pos.edges else 1
                    
                    # Edge color and style based on transport type
                    if data.get('type') in ['bottleneck', 'wasserstein']:
                        edge_color = 'rgba(0, 0, 0, 0.7)'  # Black for spatial transports
                    else:
                        edge_color = 'rgba(200, 30, 150, 0.7)'  # Magenta for height-based transports
                    
                    edge_dash = 'solid' if data.get('type') in ['bottleneck', 'height_bottleneck'] else 'dot'
                    
                    fig.add_trace(go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode='lines',
                        line=dict(
                            width=edge_width,
                            color=edge_color,
                            dash=edge_dash
                        ),
                        name=f"Edge {u}-{v}",
                        hoverinfo="skip"
                    ))
        
        # Visualize negative graph
        if neg_node_positions:
            # Add nodes
            node_x = []
            node_y = []
            node_size = []
            node_color = []
            node_text = []
            
            for node, pos in neg_node_positions.items():
                node_x.append(pos[0])
                node_y.append(pos[1])
                attrs = neg_node_attrs[node]
                
                # Size based on absolute height
                node_size.append(scale_node_size(attrs['height']))
                
                # Color based on distribution
                node_color.append('rgba(0, 158, 115, 0.8)' if attrs['dist'] == 'A' else 'rgba(230, 159, 0, 0.8)')
                
                # Text label
                node_text.append(f"{node}: {attrs['height']:.2f}")
            
            # Add nodes trace
            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=2, color='black'),
                    symbol='x'  # Use X symbol for negative blobs
                ),
                text=node_text,
                textposition="top center",
                name="Negative Blobs",
                hoverinfo="text"
            ))
            
            # Add edges if transport plan is selected
            if transport_mode != 'hide':
                for u, v, data in G_neg.edges(data=True):
                    x0, y0 = neg_node_positions[u]
                    x1, y1 = neg_node_positions[v]
                    
                    # Calculate maximum edge weight for scaling
                    max_edge_weight = max([d.get('weight', 0) for _, _, d in G_neg.edges(data=True)]) if G_neg.edges else 1
                    
                    # Edge width based on weight (with minimum of 1)
                    edge_width = max(1, 5 * (data['weight'] / max(max_edge_weight, 0.1)))
                    
                    # Edge color and style based on transport type
                    if data.get('type') in ['bottleneck', 'wasserstein']:
                        edge_color = 'rgba(0, 0, 0, 0.7)'  # Black for spatial transports
                    else:
                        edge_color = 'rgba(200, 30, 150, 0.7)'  # Magenta for height-based transports
                    
                    edge_dash = 'solid' if data.get('type') in ['bottleneck', 'height_bottleneck'] else 'dot'
                    
                    fig.add_trace(go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode='lines',
                        line=dict(
                            width=edge_width,
                            color=edge_color,
                            dash=edge_dash
                        ),
                        name=f"Edge {u}-{v}",
                        hoverinfo="skip"
                    ))
        
        # Calculate maximum edge weight for positive graph (needed for edge width scaling)
        max_edge_weight = 0
        if G_pos.edges:
            max_edge_weight = max([d.get('weight', 0) for _, _, d in G_pos.edges(data=True)])
        if G_neg.edges:
            max_edge_weight = max(max_edge_weight, max([d.get('weight', 0) for _, _, d in G_neg.edges(data=True)]))
        
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
            title_text=f"Graph Visualization - {transport_mode.replace('_', ' ').title() if transport_mode != 'hide' else 'No Transport Plan'}",
            showlegend=True,
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor='rgba(240,240,240,0.6)',
            height=600,
            hovermode='closest'
        )
        
        return fig