import numpy as np
from typing import List, Dict, Tuple, Any, Callable, Optional
from models.blob import Blob

class Distribution:
    """
    Model class representing a distribution of Gaussian blobs
    """
    def __init__(self, name: str, color: str):
        self.name = name
        self.color = color
        self._blobs: List[Blob] = []
        self._next_id = 0
        self._observers = []  # For observer pattern implementation
    
    @property
    def blobs(self) -> List[Blob]:
        """Get the list of blobs"""
        return self._blobs.copy()  # Return a copy to prevent direct modification
    
    @property
    def is_empty(self) -> bool:
        """Check if the distribution has no blobs"""
        return len(self._blobs) == 0
    
    @property
    def centers(self) -> List[Tuple[float, float]]:
        """Get a list of blob center coordinates"""
        return [blob.center for blob in self._blobs]
    
    @property
    def weights(self) -> List[float]:
        """Get a list of blob effective weights (height × sign)"""
        return [blob.effective_weight for blob in self._blobs]
    
    @property
    def positive_centers(self) -> List[Tuple[float, float]]:
        """Get a list of centers with positive height"""
        return [blob.center for blob in self._blobs if blob.height > 0]
    
    @property
    def negative_centers(self) -> List[Tuple[float, float]]:
        """Get a list of centers with negative height"""
        return [blob.center for blob in self._blobs if blob.height < 0]
    
    def add_blob(self, x: float = None, y: float = None, variance: float = 0.5, 
                 height: float = 1.0) -> Blob:
        """
        Add a new Gaussian blob to the distribution.
        If x or y are None, random positions within the plot area (0-10) are assigned.
        """
        import random
        
        # Generate random positions if not specified
        if x is None:
            x = random.uniform(1.0, 9.0)  # Avoid edges
        else:
            # Add small randomness to the specified position
            x = max(0.2, min(9.8, x + random.uniform(-0.2, 0.2)))
        
        if y is None:
            y = random.uniform(1.0, 9.0)  # Avoid edges
        else:
            # Add small randomness to the specified position
            y = max(0.2, min(9.8, y + random.uniform(-0.2, 0.2)))
            
        # Add a small random variation to the height to create more visual variety
        randomized_height = height + random.uniform(-0.2, 0.2)
        # Ensure the sign doesn't change by accident due to randomization
        if (randomized_height * height) < 0 and height != 0:
            # If the randomization changed the sign, revert to the original height
            randomized_height = height
            
        blob = Blob(self._next_id, x, y, variance, randomized_height)
        blob.add_observer(lambda _: self._notify_observers())  # Propagate changes
        self._blobs.append(blob)
        self._next_id += 1
        self._notify_observers()
        return blob
    
    def get_blob(self, blob_id: int) -> Optional[Blob]:
        """Get a blob by ID"""
        for blob in self._blobs:
            if blob.id == blob_id:
                return blob
        return None
    
    def remove_blob(self, blob_id: int) -> bool:
        """Remove a blob from the distribution by ID"""
        old_len = len(self._blobs)
        self._blobs = [b for b in self._blobs if b.id != blob_id]
        
        if len(self._blobs) < old_len:
            self._notify_observers()
            return True
        return False
    
    def update_blob(self, blob_id: int, x=None, y=None, variance=None, height=None, sign=None) -> bool:
        """Update properties of an existing blob"""
        blob = self.get_blob(blob_id)
        if blob:
            blob.update(x, y, variance, height, sign)
            return True
        return False
    
    def get_data_dicts(self) -> List[Dict[str, Any]]:
        """Return a list of dictionaries with blob data suitable for pandas DataFrame"""
        return [blob.to_dict() for blob in self._blobs]
    
    def add_observer(self, callback: Callable[['Distribution'], None]) -> None:
        """Add an observer callback function"""
        if callback not in self._observers:
            self._observers.append(callback)
    
    def remove_observer(self, callback: Callable[['Distribution'], None]) -> None:
        """Remove an observer callback function"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self) -> None:
        """Notify all observers of a change"""
        for callback in self._observers:
            callback(self)
    
    def create_gaussian_mixture(self, points: np.ndarray) -> np.ndarray:
        """
        Create a Gaussian mixture distribution from blobs.
        
        Args:
            points: Array of points (x, y) where the distribution will be evaluated
            
        Returns:
            Array of distribution values at each point
        """
        if not self._blobs:
            return np.zeros(len(points))
        
        # Initialize with zeros
        dist = np.zeros(len(points))
        
        # Add contribution from each Gaussian blob
        for blob in self._blobs:
            # Calculate Gaussian values for all points
            gauss_values = np.array([blob.calculate_gaussian_value(point) for point in points])
            
            # Add weighted contribution to the mixture
            dist += blob.effective_weight * gauss_values
        
        # Calculate the sum of absolute values to normalize properly
        total_abs_sum = np.sum(np.abs(dist))
        if total_abs_sum > 0:
            # Normalize to make the sum of absolute values equal to 1
            # This ensures both positive and negative parts are normalized proportionally
            dist = dist / total_abs_sum
        
        return dist