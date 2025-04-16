"""
Improved implementation of Wasserstein distance calculator using POT
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import ot
import networkx as nx
from networkx.algorithms.flow import maximum_flow
from scipy.optimize import linear_sum_assignment
from models.distribution import Distribution
from models.blob import Blob

def improved_wasserstein_plan(
    dist_a: Distribution, 
    dist_b: Distribution
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Calculate 2-Wasserstein distance and transportation plan using POT, respecting sign of weights.
    Uses the improved exact W2 calculator for more accurate results.
    
    Args:
        dist_a: First distribution
        dist_b: Second distribution
        
    Returns:
        A tuple containing (wasserstein_distance, matching_pairs)
        where matching_pairs is a list of tuples (idx_a, idx_b, weight)
    """
    blobs_a = dist_a.blobs
    blobs_b = dist_b.blobs
    
    # Handle empty distributions
    if not blobs_a or not blobs_b:
        return 0.0, []
    
    # Separate positive and negative blobs
    pos_blobs_a = [b for b in blobs_a if b.height > 0]
    neg_blobs_a = [b for b in blobs_a if b.height < 0]
    pos_blobs_b = [b for b in blobs_b if b.height > 0]
    neg_blobs_b = [b for b in blobs_b if b.height < 0]
    
    # Map indices for later reference
    pos_indices_a = [i for i, b in enumerate(blobs_a) if b.height > 0]
    neg_indices_a = [i for i, b in enumerate(blobs_a) if b.height < 0]
    pos_indices_b = [i for i, b in enumerate(blobs_b) if b.height > 0]
    neg_indices_b = [i for i, b in enumerate(blobs_b) if b.height < 0]
    
    pairs: List[Tuple[int, int, float]] = []
    total_distance = 0.0
    
    # Process positive blobs
    if pos_blobs_a and pos_blobs_b:
        # Create spatial cost matrix
        centers_a = np.array([blob.center for blob in pos_blobs_a])
        centers_b = np.array([blob.center for blob in pos_blobs_b])
        cost_matrix = ot.dist(centers_a, centers_b)
        
        # Square the cost matrix for 2-Wasserstein distance
        cost_matrix_squared = cost_matrix ** 2
        
        # Create weight vectors
        a = np.array([b.height for b in pos_blobs_a])
        b = np.array([b.height for b in pos_blobs_b])
        
        # Normalize weights to create probability distributions
        a = a / np.sum(a)
        b = b / np.sum(b)
        
        # Calculate optimal transport plan using the exact EMD solver
        transport_plan = ot.emd(a, b, cost_matrix_squared)
        
        # Calculate the Wasserstein-2 distance using the exact formula
        wasserstein_dist = np.sqrt(np.sum(transport_plan * cost_matrix_squared))
        total_distance += wasserstein_dist
        
        # Add matching pairs to result
        for i in range(len(pos_blobs_a)):
            for j in range(len(pos_blobs_b)):
                if transport_plan[i, j] > 1e-10:  # Threshold for numerical stability
                    orig_idx_a = pos_indices_a[i]
                    orig_idx_b = pos_indices_b[j]
                    pairs.append((orig_idx_a, orig_idx_b, float(transport_plan[i, j])))
    
    # Process negative blobs
    if neg_blobs_a and neg_blobs_b:
        # Create spatial cost matrix
        centers_a = np.array([blob.center for blob in neg_blobs_a])
        centers_b = np.array([blob.center for blob in neg_blobs_b])
        cost_matrix = ot.dist(centers_a, centers_b)
        
        # Square the cost matrix for 2-Wasserstein distance
        cost_matrix_squared = cost_matrix ** 2
        
        # Create weight vectors (use absolute values)
        a = np.array([abs(b.height) for b in neg_blobs_a])
        b = np.array([abs(b.height) for b in neg_blobs_b])
        
        # Normalize weights to create probability distributions
        a = a / np.sum(a)
        b = b / np.sum(b)
        
        # Calculate optimal transport plan using the exact EMD solver
        transport_plan = ot.emd(a, b, cost_matrix_squared)
        
        # Calculate the Wasserstein-2 distance using the exact formula
        wasserstein_dist = np.sqrt(np.sum(transport_plan * cost_matrix_squared))
        total_distance += wasserstein_dist
        
        # Add matching pairs to result
        for i in range(len(neg_blobs_a)):
            for j in range(len(neg_blobs_b)):
                if transport_plan[i, j] > 1e-10:  # Threshold for numerical stability
                    orig_idx_a = neg_indices_a[i]
                    orig_idx_b = neg_indices_b[j]
                    pairs.append((orig_idx_a, orig_idx_b, float(transport_plan[i, j])))
    
    return total_distance, pairs

def improved_bottleneck_plan(
    dist_a: Distribution, 
    dist_b: Distribution
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Calculate bottleneck distance and matching using our improved bottleneck algorithm
    that leverages max-flow for more accurate results.
    
    Args:
        dist_a: First distribution
        dist_b: Second distribution
        
    Returns:
        A tuple containing (bottleneck_distance, matching_pairs)
        where matching_pairs is a list of tuples (idx_a, idx_b)
    """
    blobs_a = dist_a.blobs
    blobs_b = dist_b.blobs
    
    # Handle empty distributions
    if not blobs_a or not blobs_b:
        return 0.0, []
    
    # Separate positive and negative blobs
    pos_blobs_a = [b for b in blobs_a if b.height > 0]
    neg_blobs_a = [b for b in blobs_a if b.height < 0]
    pos_blobs_b = [b for b in blobs_b if b.height > 0]
    neg_blobs_b = [b for b in blobs_b if b.height < 0]
    
    # Map indices for later reference
    pos_indices_a = [i for i, b in enumerate(blobs_a) if b.height > 0]
    neg_indices_a = [i for i, b in enumerate(blobs_a) if b.height < 0]
    pos_indices_b = [i for i, b in enumerate(blobs_b) if b.height > 0]
    neg_indices_b = [i for i, b in enumerate(blobs_b) if b.height < 0]
    
    pairs: List[Tuple[int, int]] = []
    max_distance = 0.0
    
    # Process positive blobs
    if pos_blobs_a and pos_blobs_b:
        # Create spatial cost matrix
        centers_a = np.array([blob.center for blob in pos_blobs_a])
        centers_b = np.array([blob.center for blob in pos_blobs_b])
        M = ot.dist(centers_a, centers_b)
        
        # Create weight arrays
        a = np.array([b.height for b in pos_blobs_a])
        b = np.array([b.height for b in pos_blobs_b])
        
        # Calculate bottleneck distance using our improved algorithm
        if len(a) <= len(b):  # Ensure a is the smaller array
            pos_bottleneck = winf_bottleneck(a, b, M)
            
            # Use scipy's linear_sum_assignment to get the matchings
            row_ind, col_ind = linear_sum_assignment(M)
            
            pos_matching: List[Tuple[int, int]] = []
            for i, j in zip(row_ind, col_ind):
                if i < len(pos_indices_a) and j < len(pos_indices_b):
                    orig_idx_a = pos_indices_a[i]
                    orig_idx_b = pos_indices_b[j]
                    # Only include matches that contribute to the bottleneck
                    if M[i, j] <= pos_bottleneck + 1e-10:  # Add small epsilon for numerical stability
                        pos_matching.append((orig_idx_a, orig_idx_b))
        else:
            pos_bottleneck = winf_bottleneck(b, a, M.T)
            
            # Use scipy's linear_sum_assignment to get the matchings
            row_ind, col_ind = linear_sum_assignment(M.T)
            
            pos_matching: List[Tuple[int, int]] = []
            for i, j in zip(row_ind, col_ind):
                if i < len(pos_indices_b) and j < len(pos_indices_a):
                    orig_idx_a = pos_indices_a[j]
                    orig_idx_b = pos_indices_b[i]
                    # Only include matches that contribute to the bottleneck
                    if M[j, i] <= pos_bottleneck + 1e-10:  # Add small epsilon for numerical stability
                        pos_matching.append((orig_idx_a, orig_idx_b))
                
        # Add the matching pairs and update the max distance
        pairs.extend(pos_matching)
        max_distance = max(max_distance, pos_bottleneck)
    
    # Process negative blobs
    if neg_blobs_a and neg_blobs_b:
        # Create spatial cost matrix
        centers_a = np.array([blob.center for blob in neg_blobs_a])
        centers_b = np.array([blob.center for blob in neg_blobs_b])
        M = ot.dist(centers_a, centers_b)
        
        # Create weight arrays (use absolute values)
        a = np.array([abs(b.height) for b in neg_blobs_a])
        b = np.array([abs(b.height) for b in neg_blobs_b])
        
        # Calculate bottleneck distance using our improved algorithm
        if len(a) <= len(b):  # Ensure a is the smaller array
            neg_bottleneck = winf_bottleneck(a, b, M)
            
            # Use scipy's linear_sum_assignment to get the matchings
            row_ind, col_ind = linear_sum_assignment(M)
            
            neg_matching: List[Tuple[int, int]] = []
            for i, j in zip(row_ind, col_ind):
                if i < len(neg_indices_a) and j < len(neg_indices_b):
                    orig_idx_a = neg_indices_a[i]
                    orig_idx_b = neg_indices_b[j]
                    # Only include matches that contribute to the bottleneck
                    if M[i, j] <= neg_bottleneck + 1e-10:  # Add small epsilon for numerical stability
                        neg_matching.append((orig_idx_a, orig_idx_b))
        else:
            neg_bottleneck = winf_bottleneck(b, a, M.T)
            
            # Use scipy's linear_sum_assignment to get the matchings
            row_ind, col_ind = linear_sum_assignment(M.T)
            
            neg_matching: List[Tuple[int, int]] = []
            for i, j in zip(row_ind, col_ind):
                if i < len(neg_indices_b) and j < len(neg_indices_a):
                    orig_idx_a = neg_indices_a[j]
                    orig_idx_b = neg_indices_b[i]
                    # Only include matches that contribute to the bottleneck
                    if M[j, i] <= neg_bottleneck + 1e-10:  # Add small epsilon for numerical stability
                        neg_matching.append((orig_idx_a, orig_idx_b))
        
        # Add the matching pairs and update the max distance
        pairs.extend(neg_matching)
        max_distance = max(max_distance, neg_bottleneck)
    
    return max_distance, pairs

def _has_feasible_flow(thresh: float, a: np.ndarray, b: np.ndarray, M: np.ndarray) -> bool:
    """
    Check if all mass can be transported using only edges ≤ thresh via max‑flow.
    
    Args:
        thresh: Distance threshold
        a: Source weights
        b: Target weights
        M: Cost matrix
        
    Returns:
        True if a feasible flow exists with all edges <= thresh, False otherwise
    """
    G = nx.DiGraph()
    src, sink = "S", "T"
    G.add_node(src)
    G.add_node(sink)

    # supply arcs (source → x_i)
    for i, ai in enumerate(a):
        if ai > 0:
            G.add_edge(src, ("u", i), capacity=float(ai))

    # demand arcs (y_j → sink)
    for j, bj in enumerate(b):
        if bj > 0:
            G.add_edge(("v", j), sink, capacity=float(bj))

    # transport arcs allowed by the threshold
    n, m = M.shape
    for i in range(n):
        ui = ("u", i)
        for j in range(m):
            if M[i, j] <= thresh:
                vi = ("v", j)
                G.add_edge(ui, vi, capacity=float("inf"))

    flow_val, _ = maximum_flow(G, src, sink)
    return flow_val >= a.sum() - 1e-12  # allow tiny numerical slack

def winf_bottleneck(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> float:
    """
    Exact Winf (bottleneck) distance via binary search + max‑flow feasibility oracle.
    
    Args:
        a: Source weights
        b: Target weights
        M: Cost matrix
        
    Returns:
        Bottleneck distance
    """
    levels = np.unique(M)
    lo, hi = 0, len(levels) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_feasible_flow(levels[mid], a, b, M):
            hi = mid  # feasible ⇒ tighten upper bound
        else:
            lo = mid + 1
    return float(levels[lo])