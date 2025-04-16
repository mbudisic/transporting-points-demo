import numpy as np
from typing import List, Dict, Tuple, Any

class Distribution:
    def __init__(self, name: str, color: str):
        self.name = name
        self.color = color
        self.blobs = []
        self.next_id = 0
    
    def add_blob(self, x: float, y: float, variance: float = 0.5, height: float = 1.0, sign: int = 1):
        """Add a new Gaussian blob to the distribution"""
        blob = {
            'id': self.next_id,
            'x': float(x),
            'y': float(y),
            'variance': float(variance),
            'height': float(height),
            'sign': int(sign)
        }
        self.blobs.append(blob)
        self.next_id += 1
    
    def remove_blob(self, blob_id: int):
        """Remove a blob from the distribution by ID"""
        self.blobs = [b for b in self.blobs if b['id'] != blob_id]
    
    def update_blob(self, blob_id: int, x=None, y=None, variance=None, height=None, sign=None):
        """Update properties of an existing blob"""
        for blob in self.blobs:
            if blob['id'] == blob_id:
                if x is not None:
                    blob['x'] = float(x)
                if y is not None:
                    blob['y'] = float(y)
                if variance is not None:
                    blob['variance'] = float(variance)
                if height is not None:
                    blob['height'] = float(height)
                if sign is not None:
                    blob['sign'] = int(sign)
                break
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Return a list of dictionaries with blob data suitable for pandas DataFrame"""
        return self.blobs

def gaussian_2d(point: np.ndarray, center: Tuple[float, float], variance: float) -> float:
    """
    Calculate 2D Gaussian function value at a point.
    
    Args:
        point: Point coordinates (x, y)
        center: Gaussian center (x, y)
        variance: Variance of the Gaussian
        
    Returns:
        Value of Gaussian function at the point
    """
    return np.exp(-((point[0] - center[0])**2 + (point[1] - center[1])**2) / (2 * variance))

def create_gaussian_mixture(blobs: List[Dict[str, Any]], points: np.ndarray) -> np.ndarray:
    """
    Create a Gaussian mixture distribution from a list of blobs.
    
    Args:
        blobs: List of blob dictionaries with 'x', 'y', 'variance', 'height', and 'sign'
        points: Array of points (x, y) where the distribution will be evaluated
        
    Returns:
        Array of distribution values at each point
    """
    if not blobs:
        return np.zeros(len(points))
    
    # Initialize with zeros
    dist = np.zeros(len(points))
    
    # Add contribution from each Gaussian blob
    for blob in blobs:
        center = (blob['x'], blob['y'])
        variance = blob['variance']
        height = blob['height']  # This can be positive or negative
        sign = blob['sign']      # This can be +1 or -1
        
        # The effective weight is the product of height and sign
        # This allows for negative heights in the distribution
        effective_weight = height * sign
        
        # Calculate Gaussian values for all points
        gauss_values = np.array([gaussian_2d(point, center, variance) for point in points])
        
        # Add weighted contribution to the mixture
        dist += effective_weight * gauss_values
    
    # Calculate the sum of absolute values to normalize properly
    total_abs_sum = np.sum(np.abs(dist))
    if total_abs_sum > 0:
        # Normalize to make the sum of absolute values equal to 1
        # This ensures both positive and negative parts are normalized proportionally
        dist = dist / total_abs_sum
    
    return dist
