import numpy as np
from typing import Dict, Any, Tuple, List, Callable

class Blob:
    """
    Model class for a Gaussian blob in a distribution
    """
    def __init__(self, blob_id: int, x: float, y: float, variance: float = 0.5, height: float = 1.0, sign: int = 1):
        self.id = blob_id
        self._x = float(x)
        self._y = float(y)
        self._variance = float(variance)
        self._height = float(height)
        self._sign = int(sign)
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
    
    @property
    def sign(self) -> int:
        return self._sign
    
    @sign.setter
    def sign(self, value: int):
        self._sign = int(value)
        self._notify_observers()
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center coordinates as a tuple"""
        return (self._x, self._y)
    
    @property
    def effective_weight(self) -> float:
        """Get the effective weight (height × sign)"""
        return self._height * self._sign
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for DataFrames"""
        return {
            'id': self.id,
            'x': self._x,
            'y': self._y,
            'variance': self._variance,
            'height': self._height,
            'sign': self._sign
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Blob':
        """Create a Blob from a dictionary representation"""
        return cls(
            blob_id=data['id'],
            x=data['x'],
            y=data['y'],
            variance=data['variance'],
            height=data['height'],
            sign=data['sign']
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
            
        if height is not None and float(height) != self._height:
            self._height = float(height)
            changed = True
            
        if sign is not None and int(sign) != self._sign:
            self._sign = int(sign)
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