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

def calculate_wasserstein_plan(centers_a: List[Tuple[float, float]], 
                           centers_b: List[Tuple[float, float]],
                           weights_a: List[float], 
                           weights_b: List[float]) -> tuple:
    """
    Calculate Wasserstein distance and transportation plan between two distributions,
    respecting the sign of the weights. Positive centers are matched only with positive centers,
    and negative centers with negative centers.
    
    If a particular sign doesn't have a direct pair, match it to:
    1. The center of the distribution of same-sign items in the opposite distribution
    2. If none available, to the geometric center of all points
    
    Args:
        centers_a: List of centers (x, y) for distribution A
        centers_b: List of centers (x, y) for distribution B
        weights_a: Weights for centers in distribution A
        weights_b: Weights for centers in distribution B
        
    Returns:
        A tuple containing (wasserstein_distance, matching_pairs)
        where matching_pairs is a list of tuples (idx_a, idx_b, weight)
        indicating the transportation plan with weights
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
    wasserstein_dist = 0.0
    matching_pairs = []
    
    # Process positive centers
    if pos_indices_a and pos_indices_b:
        # Extract positive centers and weights from both distributions
        pos_centers_a = centers_a_np[pos_indices_a]
        pos_centers_b = centers_b_np[pos_indices_b]
        pos_weights_a = weights_a_np[pos_indices_a]
        pos_weights_b = weights_b_np[pos_indices_b]
        
        # Create cost matrix for positive centers
        pos_cost_matrix = cdist(pos_centers_a, pos_centers_b)
        
        # Solve the optimal transport problem for positive centers
        pos_row_ind, pos_col_ind = linear_sum_assignment(pos_cost_matrix)
        
        # Calculate weighted distances for matched pairs
        for i, j in zip(pos_row_ind, pos_col_ind):
            flow = min(pos_weights_a[i], pos_weights_b[j])
            wasserstein_dist += pos_cost_matrix[i, j] * flow
            matching_pairs.append((pos_indices_a[i], pos_indices_b[j], flow))
    
    # Process negative centers
    if neg_indices_a and neg_indices_b:
        # Extract negative centers and weights from both distributions
        neg_centers_a = centers_a_np[neg_indices_a]
        neg_centers_b = centers_b_np[neg_indices_b]
        neg_weights_a = np.abs(weights_a_np[neg_indices_a])  # Use absolute values for computation
        neg_weights_b = np.abs(weights_b_np[neg_indices_b])
        
        # Create cost matrix for negative centers
        neg_cost_matrix = cdist(neg_centers_a, neg_centers_b)
        
        # Solve the optimal transport problem for negative centers
        neg_row_ind, neg_col_ind = linear_sum_assignment(neg_cost_matrix)
        
        # Calculate weighted distances for matched pairs
        for i, j in zip(neg_row_ind, neg_col_ind):
            flow = min(neg_weights_a[i], neg_weights_b[j])
            wasserstein_dist += neg_cost_matrix[i, j] * flow
            matching_pairs.append((neg_indices_a[i], neg_indices_b[j], flow))
    
    # Handle unpaired positive centers in distribution A
    for i in pos_indices_a:
        if not any(pair[0] == i for pair in matching_pairs):
            # If there are positive centers in B, match to their center of mass
            if pos_indices_b:
                pos_centers_b = centers_b_np[pos_indices_b]
                center_of_mass_b = np.mean(pos_centers_b, axis=0)
                dist = np.sqrt(np.sum((centers_a_np[i] - center_of_mass_b)**2))
                wasserstein_dist += dist * weights_a_np[i]
                matching_pairs.append((i, -1, weights_a_np[i]))  # Use -1 to indicate virtual point
            # Otherwise match to geometric center of all points
            else:
                center_of_all = np.mean(centers_b_np, axis=0) if len(centers_b_np) > 0 else np.array([5, 5])
                dist = np.sqrt(np.sum((centers_a_np[i] - center_of_all)**2))
                wasserstein_dist += dist * weights_a_np[i]
                matching_pairs.append((i, -2, weights_a_np[i]))  # Use -2 to indicate geometric center
    
    # Handle unpaired negative centers in distribution A
    for i in neg_indices_a:
        if not any(pair[0] == i for pair in matching_pairs):
            # If there are negative centers in B, match to their center of mass
            if neg_indices_b:
                neg_centers_b = centers_b_np[neg_indices_b]
                center_of_mass_b = np.mean(neg_centers_b, axis=0)
                dist = np.sqrt(np.sum((centers_a_np[i] - center_of_mass_b)**2))
                wasserstein_dist += dist * abs(weights_a_np[i])
                matching_pairs.append((i, -3, abs(weights_a_np[i])))  # Use -3 to indicate virtual negative point
            # Otherwise match to geometric center of all points
            else:
                center_of_all = np.mean(centers_b_np, axis=0) if len(centers_b_np) > 0 else np.array([5, 5])
                dist = np.sqrt(np.sum((centers_a_np[i] - center_of_all)**2))
                wasserstein_dist += dist * abs(weights_a_np[i])
                matching_pairs.append((i, -4, abs(weights_a_np[i])))  # Use -4 to indicate geometric center for negative
    
    # Do the same for unpaired centers in distribution B
    for i in pos_indices_b:
        if not any(pair[1] == i for pair in matching_pairs):
            if pos_indices_a:
                pos_centers_a = centers_a_np[pos_indices_a]
                center_of_mass_a = np.mean(pos_centers_a, axis=0)
                dist = np.sqrt(np.sum((centers_b_np[i] - center_of_mass_a)**2))
                wasserstein_dist += dist * weights_b_np[i]
                matching_pairs.append((-1, i, weights_b_np[i]))
            else:
                center_of_all = np.mean(centers_a_np, axis=0) if len(centers_a_np) > 0 else np.array([5, 5])
                dist = np.sqrt(np.sum((centers_b_np[i] - center_of_all)**2))
                wasserstein_dist += dist * weights_b_np[i]
                matching_pairs.append((-2, i, weights_b_np[i]))
    
    for i in neg_indices_b:
        if not any(pair[1] == i for pair in matching_pairs):
            if neg_indices_a:
                neg_centers_a = centers_a_np[neg_indices_a]
                center_of_mass_a = np.mean(neg_centers_a, axis=0)
                dist = np.sqrt(np.sum((centers_b_np[i] - center_of_mass_a)**2))
                wasserstein_dist += dist * abs(weights_b_np[i])
                matching_pairs.append((-3, i, abs(weights_b_np[i])))
            else:
                center_of_all = np.mean(centers_a_np, axis=0) if len(centers_a_np) > 0 else np.array([5, 5])
                dist = np.sqrt(np.sum((centers_b_np[i] - center_of_all)**2))
                wasserstein_dist += dist * abs(weights_b_np[i])
                matching_pairs.append((-4, i, abs(weights_b_np[i])))
    
    return wasserstein_dist, matching_pairs

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
