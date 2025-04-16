import streamlit as st
from typing import Dict, Any, List, Optional, Tuple, Union, BinaryIO
from models.distribution import Distribution
from models.blob import Blob
import pandas as pd
import io
import csv
import base64
import plotly.graph_objects as go
from utils.export_utils import export_to_formats, generate_html_report

class DistributionController:
    """
    Controller class for manipulating distributions
    """
    @staticmethod
    def add_blob(distribution: Distribution, 
                 x: Optional[float] = None, 
                 y: Optional[float] = None, 
                 variance: float = 0.5, 
                 height: float = 1.0) -> Blob:
        """
        Add a new blob to the distribution
        
        Args:
            distribution: Distribution to add the blob to
            x: X-coordinate (defaults to random if None)
            y: Y-coordinate (defaults to random if None)
            variance: Variance of the Gaussian blob
            height: Height of the blob (can be positive or negative)
            
        Returns:
            The newly created Blob
        """
        import random
        
        # Ensure valid values (but allow x and y to be None for random positioning)
        if x is not None:
            # Add small random variation (±0.5) to specified position
            x = max(0.0, min(10.0, float(x) + random.uniform(-0.5, 0.5)))
        if y is not None:
            # Add small random variation (±0.5) to specified position
            y = max(0.0, min(10.0, float(y) + random.uniform(-0.5, 0.5)))
            
        variance = max(0.1, min(5.0, float(variance)))
        
        # Add small random variation to the height (±0.2)
        height = max(-10.0, min(10.0, float(height) + random.uniform(-0.2, 0.2)))
        
        # Add to the distribution (random positions will be generated if x or y is None)
        return distribution.add_blob(x, y, variance, height)
    
    @staticmethod
    def remove_blob(distribution: Distribution, blob_id: int) -> bool:
        """
        Remove a blob from the distribution
        
        Args:
            distribution: Distribution to remove the blob from
            blob_id: ID of the blob to remove
            
        Returns:
            True if the blob was removed, False otherwise
        """
        return distribution.remove_blob(blob_id)
    
    @staticmethod
    def update_blob(distribution: Distribution, 
                   blob_id: int, 
                   x: Optional[float] = None, 
                   y: Optional[float] = None, 
                   variance: Optional[float] = None, 
                   height: Optional[float] = None, 
                   sign: Optional[float] = None) -> bool:
        """
        Update properties of an existing blob
        
        Args:
            distribution: Distribution containing the blob
            blob_id: ID of the blob to update
            x: New X-coordinate (optional)
            y: New Y-coordinate (optional)
            variance: New variance (optional)
            height: New height (optional, can be positive or negative)
            sign: New sign (optional, only used for backwards compatibility)
            
        Returns:
            True if the blob was updated, False if the blob was not found
        """
        # Apply limits to inputs
        if x is not None:
            x = max(0.0, min(10.0, float(x)))
        if y is not None:
            y = max(0.0, min(10.0, float(y)))
        if variance is not None:
            variance = max(0.1, min(5.0, float(variance)))
            
        # Handle height with sign
        if height is not None:
            # If sign is specified, it influences the height's sign
            if sign is not None:
                # Apply sign to the absolute height value
                sign_value = 1 if float(sign) > 0 else -1
                height = abs(float(height)) * sign_value
            
            # Apply limits to the signed height
            height = max(-10.0, min(10.0, float(height)))
            
        # Update the blob with the new values
        return distribution.update_blob(blob_id, x, y, variance, height, sign)
    
    @staticmethod
    def export_distributions_to_csv(dist_a: Distribution, dist_b: Distribution) -> str:
        """
        Export distributions to a CSV download link
        
        Args:
            dist_a: Distribution A
            dist_b: Distribution B
            
        Returns:
            HTML string with a download link for the CSV
        """
        # Get data dictionaries from both distributions
        data_a = dist_a.get_data_dicts()
        data_b = dist_b.get_data_dicts()
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        
        # Write header
        writer.writerow(['distribution', 'id', 'x', 'y', 'variance', 'height', 'sign'])
        
        # Write distribution A data
        for blob in data_a:
            writer.writerow(['A', blob['id'], blob['x'], blob['y'], 
                           blob['variance'], blob['height'], blob['sign']])
        
        # Write distribution B data
        for blob in data_b:
            writer.writerow(['B', blob['id'], blob['x'], blob['y'], 
                           blob['variance'], blob['height'], blob['sign']])
        
        # Get CSV string
        csv_string = csv_buffer.getvalue()
        
        # Create a download link
        b64 = base64.b64encode(csv_string.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="distributions_export.csv">Download CSV</a>'
        return href
    
    @staticmethod
    def import_distributions_from_csv(dist_a: Distribution, 
                                     dist_b: Distribution, 
                                     uploaded_file: BinaryIO) -> bool:
        """
        Import distributions from a CSV file
        
        Args:
            dist_a: Distribution A to populate
            dist_b: Distribution B to populate
            uploaded_file: CSV file uploaded by the user
            
        Returns:
            True if import was successful, False otherwise
        """
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Clear existing blobs
            dist_a.clear_blobs()
            dist_b.clear_blobs()
            
            # Process rows
            for _, row in df.iterrows():
                dist_name = row['distribution']
                blob_id = int(row['id'])
                x = float(row['x'])
                y = float(row['y'])
                variance = float(row['variance'])
                
                # Handle the height/sign columns based on what's available
                if 'height' in row:
                    height = float(row['height'])
                    # Use sign to adjust height if both columns exist
                    if 'sign' in row:
                        sign = float(row['sign'])
                        # Only apply sign if height doesn't already have the correct sign
                        if (sign < 0 and height > 0) or (sign > 0 and height < 0):
                            height = -height
                else:
                    # Fallback if only the legacy format is available
                    weight = float(row['weight']) if 'weight' in row else 1.0
                    sign = float(row['sign']) if 'sign' in row else 1.0
                    height = weight * (1 if sign > 0 else -1)
                
                # Add to the appropriate distribution (preserving ID)
                if dist_name == 'A':
                    dist_a.add_blob(x, y, variance, height, blob_id)
                elif dist_name == 'B':
                    dist_b.add_blob(x, y, variance, height, blob_id)
            
            return True
        except Exception as e:
            st.error(f"Error importing distributions: {str(e)}")
            return False
    
    @staticmethod
    def export_as_document(dist_a: Distribution, 
                           dist_b: Distribution,
                           fig: Optional[Any] = None) -> Dict[str, str]:
        """
        Export distributions, metrics, and visualizations to a static document.
        
        Args:
            dist_a: Distribution A
            dist_b: Distribution B
            fig: Optional plotly figure to include in export
            
        Returns:
            Dictionary of HTML strings with download links for different formats
        """
        # Store the figure in session state if provided
        if fig is not None:
            st.session_state.current_figure = fig
        
        # Create download links for all supported formats directly
        return export_to_formats(dist_a, dist_b, fig)
        
    @staticmethod
    def generate_export_report(dist_a: Distribution, 
                              dist_b: Distribution, 
                              visualization_service: Any,
                              calculator: Any) -> str:
        """
        Export distributions, metrics, and visualizations to a static document.
        
        Args:
            dist_a: Distribution A
            dist_b: Distribution B
            visualization_service: Service for creating visualizations
            calculator: Distance calculator for metrics
            
        Returns:
            HTML string with a download link for the export file
        """
        # Export to HTML and create download link
        if 'export_html' not in st.session_state:
            st.session_state.export_html = generate_html_report(
                dist_a, dist_b, visualization_service, calculator)
        
        # Get the HTML content
        html_content = st.session_state.export_html
        
        # Create download links for all supported formats
        return export_to_formats(html_content)