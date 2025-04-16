import numpy as np
from typing import Dict, Any, Tuple, List, Callable

class Blob:
    """
    Model class for a Gaussian blob in a distribution
    """
    def __init__(self, blob_id: int, x: float, y: float, variance: float = 0.5, height: float = 1.0):
        self.id = blob_id
        self._x = float(x)
        self._y = float(y)
        self._variance = float(variance)
        self._height = float(height)  # Height is now signed (can be positive or negative)
        self._observers = []  # For observer pattern implementation
    
    # Property getters and setters with observer notification
    @property
    def x(self) -> float:
        return self._x
    
    @x.setter
    def x(self, value: float):
        self._x = float(value)
        self._notify_observers()
    
    @property
    def y(self) -> float:
        return self._y
    
    @y.setter
    def y(self, value: float):
        self._y = float(value)
        self._notify_observers()
    
    @property
    def variance(self) -> float:
        return self._variance
    
    @variance.setter
    def variance(self, value: float):
        self._variance = float(value)
        self._notify_observers()
    
    @property
    def height(self) -> float:
        return self._height
    
    @height.setter
    def height(self, value: float):
        self._height = float(value)
        self._notify_observers()
    
    # For backwards compatibility, implemented as computed property
    @property
    def sign(self) -> int:
        return 1 if self._height >= 0 else -1
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center coordinates as a tuple"""
        return (self._x, self._y)
    
    @property
    def effective_weight(self) -> float:
        """Get the effective weight (just the height now, as it's already signed)"""
        return self._height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for DataFrames"""
        return {
            'id': self.id,
            'x': self._x,
            'y': self._y,
            'variance': self._variance,
            'height': self._height,
            'sign': self.sign  # Computed property for backwards compatibility
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Blob':
        """Create a Blob from a dictionary representation"""
        # If we have old-style data with separate height and sign, combine them
        height = data['height']
        if 'sign' in data:
            height = height * data['sign']
            
        return cls(
            blob_id=data['id'],
            x=data['x'],
            y=data['y'],
            variance=data['variance'],
            height=height
        )
        
    def update(self, x=None, y=None, variance=None, height=None, sign=None) -> None:
        """Update multiple properties at once and notify only once"""
        changed = False
        
        if x is not None and float(x) != self._x:
            self._x = float(x)
            changed = True
            
        if y is not None and float(y) != self._y:
            self._y = float(y)
            changed = True
            
        if variance is not None and float(variance) != self._variance:
            self._variance = float(variance)
            changed = True
            
        # Handle both height and sign updates
        if height is not None:
            # If sign is also provided, use it to determine the direction of height
            if sign is not None:
                new_height = abs(float(height)) * (1 if int(sign) > 0 else -1)
                if new_height != self._height:
                    self._height = new_height
                    changed = True
            # Otherwise just update height directly
            elif float(height) != self._height:
                self._height = float(height)
                changed = True
        # If only sign is provided, flip the height if needed
        elif sign is not None:
            new_sign = 1 if int(sign) > 0 else -1
            current_sign = 1 if self._height >= 0 else -1
            if new_sign != current_sign:
                self._height = -self._height
                changed = True
                
        if changed:
            self._notify_observers()
    
    def add_observer(self, callback: Callable[['Blob'], None]) -> None:
        """Add an observer callback function"""
        if callback not in self._observers:
            self._observers.append(callback)
    
    def remove_observer(self, callback: Callable[['Blob'], None]) -> None:
        """Remove an observer callback function"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self) -> None:
        """Notify all observers of a change"""
        for callback in self._observers:
            callback(self)
    
    def calculate_gaussian_value(self, point: np.ndarray) -> float:
        """Calculate the Gaussian function value at a given point"""
        dist_squared = (point[0] - self._x)**2 + (point[1] - self._y)**2
        return np.exp(-dist_squared / (2 * self._variance))