import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

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
    
    # Discretize the distributions by treating each point as a weighted point mass
    # Use scipy's linear_sum_assignment to solve the transportation problem
    
    # Create cost matrix (Euclidean distances between all pairs of points)
    M = cdist(points, points)
    
    # Flatten the cost matrix and create sparse weights
    indices = np.where((dist_a_pos > 0) & (dist_b_pos > 0))
    if len(indices[0]) == 0:
        return 0.0
        
    # Create cost matrix between locations with non-zero mass
    sub_cost = M[indices]
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(sub_cost)
    
    # Calculate the weighted sum of distances
    weights = np.minimum(dist_a_pos[indices], dist_b_pos[indices])
    wasserstein_dist = np.sum(sub_cost[row_ind, col_ind] * weights)
    
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
    
    # Implement a simplified Earth Mover's Distance (Wasserstein) using linear_sum_assignment
    # Ensure we have equal number of points by duplicating points weighted by their mass
    
    # For simplicity, we'll solve the assignment problem directly
    row_ind, col_ind = linear_sum_assignment(M)
    
    # Calculate weighted sum of distances for matched pairs
    wasserstein_dist = 0.0
    for i, j in zip(row_ind, col_ind):
        wasserstein_dist += M[i, j] * min(weights_a_pos[i], weights_b_pos[j])
    
    return wasserstein_dist

def calculate_bottleneck(centers_a: List[Tuple[float, float]], 
                        centers_b: List[Tuple[float, float]],
                        weights_a: List[float], 
                        weights_b: List[float]) -> tuple:
    """
    Calculate bottleneck distance between two distributions, respecting the sign of the weights.
    Positive centers are matched only with positive centers, and negative centers with negative centers.
    
    Args:
        centers_a: List of centers (x, y) for distribution A
        centers_b: List of centers (x, y) for distribution B
        weights_a: Weights for centers in distribution A
        weights_b: Weights for centers in distribution B
        
    Returns:
        A tuple containing (bottleneck_distance, matching_pairs)
        where matching_pairs is a list of tuples (idx_a, idx_b) indicating the optimal matching
    """
    if not centers_a or not centers_b:
        return 0.0, []
    
    # Convert centers and weights to numpy arrays
    centers_a_np = np.array(centers_a)
    centers_b_np = np.array(centers_b)
    weights_a_np = np.array(weights_a)
    weights_b_np = np.array(weights_b)
    
    # Separate positive and negative centers
    pos_indices_a = [i for i, w in enumerate(weights_a_np) if w > 0]
    neg_indices_a = [i for i, w in enumerate(weights_a_np) if w < 0]
    pos_indices_b = [i for i, w in enumerate(weights_b_np) if w > 0]
    neg_indices_b = [i for i, w in enumerate(weights_b_np) if w < 0]
    
    # Initialize variables to store results
    bottleneck_dist = 0.0
    matching_pairs = []
    
    # Process positive centers if both distributions have them
    if pos_indices_a and pos_indices_b:
        # Extract positive centers from both distributions
        pos_centers_a = centers_a_np[pos_indices_a]
        pos_centers_b = centers_b_np[pos_indices_b]
        
        # Create cost matrix for positive centers
        pos_cost_matrix = cdist(pos_centers_a, pos_centers_b)
        
        # Solve the linear assignment problem for positive centers
        pos_row_ind, pos_col_ind = linear_sum_assignment(pos_cost_matrix)
        
        # Get the maximum cost in the optimal assignment for positive centers
        if len(pos_row_ind) > 0:
            pos_bottleneck_dist = pos_cost_matrix[pos_row_ind, pos_col_ind].max()
            bottleneck_dist = max(bottleneck_dist, pos_bottleneck_dist)
            
            # Store the matching pairs
            for i, j in zip(pos_row_ind, pos_col_ind):
                matching_pairs.append((pos_indices_a[i], pos_indices_b[j]))
    
    # Process negative centers if both distributions have them
    if neg_indices_a and neg_indices_b:
        # Extract negative centers from both distributions
        neg_centers_a = centers_a_np[neg_indices_a]
        neg_centers_b = centers_b_np[neg_indices_b]
        
        # Create cost matrix for negative centers
        neg_cost_matrix = cdist(neg_centers_a, neg_centers_b)
        
        # Solve the linear assignment problem for negative centers
        neg_row_ind, neg_col_ind = linear_sum_assignment(neg_cost_matrix)
        
        # Get the maximum cost in the optimal assignment for negative centers
        if len(neg_row_ind) > 0:
            neg_bottleneck_dist = neg_cost_matrix[neg_row_ind, neg_col_ind].max()
            bottleneck_dist = max(bottleneck_dist, neg_bottleneck_dist)
            
            # Store the matching pairs
            for i, j in zip(neg_row_ind, neg_col_ind):
                matching_pairs.append((neg_indices_a[i], neg_indices_b[j]))
    
    # If there are unmatched centers (due to different counts of positive/negative centers),
    # the bottleneck distance should reflect that, but we don't include them in matching_pairs
    
    return bottleneck_dist, matching_pairs
