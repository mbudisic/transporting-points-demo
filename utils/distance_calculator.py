import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot

def calculate_wasserstein_continuous(dist_a: np.ndarray, dist_b: np.ndarray, points: np.ndarray) -> float:
    """
    Calculate Wasserstein distance between two continuous distributions.
    
    Args:
        dist_a: Values of distribution A at each point
        dist_b: Values of distribution B at each point
        points: Array of points where the distributions are evaluated
        
    Returns:
        Wasserstein distance between the distributions
    """
    # Ensure distributions are non-negative
    dist_a_pos = np.maximum(dist_a, 0)
    dist_b_pos = np.maximum(dist_b, 0)
    
    # Normalize distributions
    if np.sum(dist_a_pos) > 0:
        dist_a_pos = dist_a_pos / np.sum(dist_a_pos)
    if np.sum(dist_b_pos) > 0:
        dist_b_pos = dist_b_pos / np.sum(dist_b_pos)
    
    # Create cost matrix (Euclidean distances between all pairs of points)
    M = cdist(points, points)
    
    # Calculate Wasserstein distance using POT
    wasserstein_dist = ot.emd2(dist_a_pos, dist_b_pos, M)
    
    return wasserstein_dist

def calculate_wasserstein_discrete(centers_a: List[Tuple[float, float]], 
                                  centers_b: List[Tuple[float, float]],
                                  weights_a: List[float], 
                                  weights_b: List[float]) -> float:
    """
    Calculate Wasserstein distance between two discrete distributions.
    
    Args:
        centers_a: List of centers (x, y) for distribution A
        centers_b: List of centers (x, y) for distribution B
        weights_a: Weights for centers in distribution A
        weights_b: Weights for centers in distribution B
        
    Returns:
        Wasserstein distance between the distributions
    """
    if not centers_a or not centers_b:
        return 0.0
    
    # Convert centers to numpy arrays
    centers_a_np = np.array(centers_a)
    centers_b_np = np.array(centers_b)
    
    # Ensure weights are positive and normalized
    weights_a_np = np.array(weights_a)
    weights_b_np = np.array(weights_b)
    
    # Get positive weights for normalization
    weights_a_pos = np.maximum(weights_a_np, 0)
    weights_b_pos = np.maximum(weights_b_np, 0)
    
    # Normalize weights
    if np.sum(weights_a_pos) > 0:
        weights_a_pos = weights_a_pos / np.sum(weights_a_pos)
    if np.sum(weights_b_pos) > 0:
        weights_b_pos = weights_b_pos / np.sum(weights_b_pos)
    
    # Create cost matrix (Euclidean distances between all pairs of centers)
    M = cdist(centers_a_np, centers_b_np)
    
    # Calculate Wasserstein distance using POT
    wasserstein_dist = ot.emd2(weights_a_pos, weights_b_pos, M)
    
    return wasserstein_dist

def calculate_bottleneck(centers_a: List[Tuple[float, float]], 
                        centers_b: List[Tuple[float, float]],
                        weights_a: List[float], 
                        weights_b: List[float]) -> float:
    """
    Calculate bottleneck distance between two distributions.
    
    Args:
        centers_a: List of centers (x, y) for distribution A
        centers_b: List of centers (x, y) for distribution B
        weights_a: Weights for centers in distribution A
        weights_b: Weights for centers in distribution B
        
    Returns:
        Bottleneck distance between the distributions
    """
    if not centers_a or not centers_b:
        return 0.0
    
    # Convert centers to numpy arrays
    centers_a_np = np.array(centers_a)
    centers_b_np = np.array(centers_b)
    
    # Create cost matrix (Euclidean distances between all pairs of centers)
    cost_matrix = cdist(centers_a_np, centers_b_np)
    
    # Solve the linear assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Bottleneck distance is the maximum cost in the optimal assignment
    bottleneck_dist = cost_matrix[row_ind, col_ind].max()
    
    return bottleneck_dist
