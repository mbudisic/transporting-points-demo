import streamlit as st
from typing import Dict, Any, List, Optional
from models.distribution import Distribution
from models.blob import Blob
import pandas as pd
import io
import csv
import base64

class DistributionController:
    """
    Controller class for manipulating distributions
    """
    @staticmethod
    def add_blob(distribution: Distribution, x=None, y=None, variance=0.5, height=1.0) -> Blob:
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
    def update_blob(distribution: Distribution, blob_id: int, 
                    x=None, y=None, variance=None, height=None, sign=None) -> bool:
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
        # If only sign is specified, we'll flip the height's sign in the Blob model
            
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
    def import_distributions_from_csv(dist_a: Distribution, dist_b: Distribution, uploaded_file) -> bool:
        """
        Import distributions from a CSV file
        
        Args:
            dist_a: Distribution A to update
            dist_b: Distribution B to update
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            True if import was successful, False otherwise
        """
        try:
            # Read CSV data
            csv_data = pd.read_csv(uploaded_file)
            
            # Validate structure
            required_columns = ['distribution', 'x', 'y', 'variance', 'height', 'sign']
            if not all(col in csv_data.columns for col in required_columns):
                return False
            
            # Create new distributions
            new_dist_a = Distribution('A', 'red')
            new_dist_b = Distribution('B', 'blue')
            
            # Process data by distribution
            for _, row in csv_data.iterrows():
                if row['distribution'] == 'A':
                    # Calculate signed height from old-style data
                    signed_height = float(row['height']) * int(row['sign'])
                    new_dist_a.add_blob(
                        x=float(row['x']),
                        y=float(row['y']),
                        variance=float(row['variance']),
                        height=signed_height
                    )
                elif row['distribution'] == 'B':
                    # Calculate signed height from old-style data
                    signed_height = float(row['height']) * int(row['sign'])
                    new_dist_b.add_blob(
                        x=float(row['x']),
                        y=float(row['y']),
                        variance=float(row['variance']),
                        height=signed_height
                    )
            
            # Replace existing distributions in session state
            st.session_state.distribution_a = new_dist_a
            st.session_state.distribution_b = new_dist_b
            return True
            
        except Exception as e:
            st.error(f"Error importing data: {str(e)}")
            return False