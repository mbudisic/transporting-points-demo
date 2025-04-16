import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any, Union
from models.distribution import Distribution
from models.blob import Blob
import ot
import networkx as nx
from scipy.optimize import linear_sum_assignment
from networkx.algorithms.flow import maximum_flow

class POTDistanceCalculator:
    """
    Controller class for calculating distances between distributions using Python Optimal Transport (POT) package.
    Implements both Wasserstein (Earth Mover's Distance) and Bottleneck distance metrics.
    """
    
    @staticmethod
    def calculate_wasserstein_continuous(dist_a: Distribution, dist_b: Distribution, grid_size: int = 100) -> float:
        """
        Calculate Wasserstein distance between two continuous distributions using POT.
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            grid_size: Number of points in the grid (higher = more accurate)
            
        Returns:
            Wasserstein distance between the distributions
        """
        if not dist_a.blobs or not dist_b.blobs:
            return 0.0
            
        # Extract positions and weights from distribution A
        positions_a = np.array([blob.center for blob in dist_a.blobs])
        weights_a = np.array([blob.height for blob in dist_a.blobs])
        
        # Extract positions and weights from distribution B
        positions_b = np.array([blob.center for blob in dist_b.blobs])
        weights_b = np.array([blob.height for blob in dist_b.blobs])
        
        # Normalize weights to create probability distributions
        weights_a = np.abs(weights_a) / np.sum(np.abs(weights_a))
        weights_b = np.abs(weights_b) / np.sum(np.abs(weights_b))
        
        # Calculate the distance matrix between positions
        M = ot.dist(positions_a, positions_b)
        
        # Calculate Wasserstein distance using EMD
        emd_value = ot.emd2(weights_a, weights_b, M)
        
        return float(emd_value)
    
    # ---------- Improved W2 and Winf Functions ----------
    
    @staticmethod
    def w2_exact(a: np.ndarray, b: np.ndarray, M2: np.ndarray) -> float:
        """
        Exact 2‑Wasserstein distance via POT's network simplex.
        
        Args:
            a: Source weights (normalized to sum to 1)
            b: Target weights (normalized to sum to 1)
            M2: Squared cost matrix
            
        Returns:
            2-Wasserstein distance (square root of the optimal cost)
        """
        cost_sq = ot.lp.emd2(a, b, M2)  # squared cost
        return float(np.sqrt(cost_sq))

    @staticmethod
    def w2_sinkhorn(a: np.ndarray, b: np.ndarray, M2: np.ndarray, reg: float = 1e-2) -> float:
        """
        Entropic‑regularised approximation of W2 (much faster for large distributions).
        
        Args:
            a: Source weights (normalized to sum to 1)
            b: Target weights (normalized to sum to 1)
            M2: Squared cost matrix
            reg: Regularization parameter (smaller values give more accurate results but are less stable)
            
        Returns:
            2-Wasserstein distance (square root of the optimal cost)
        """
        cost_sq, _ = ot.bregman.sinkhorn2(a, b, M2, reg=reg)
        return float(np.sqrt(cost_sq))
        
    @staticmethod
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
        
    @staticmethod
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
            if POTDistanceCalculator._has_feasible_flow(levels[mid], a, b, M):
                hi = mid  # feasible ⇒ tighten upper bound
            else:
                lo = mid + 1
        return float(levels[lo])
    
    # ---------- Utility Functions ----------
    
    @staticmethod
    def _separate_by_sign(blobs: List[Blob], get_value_fn: Callable = lambda b: b.height) -> Tuple[List[Tuple[int, Blob]], List[Tuple[int, Blob]]]:
        """
        Separate blobs by sign (positive/negative)
        
        Args:
            blobs: List of blobs to separate
            get_value_fn: Function to extract the value to check for sign
            
        Returns:
            Tuple containing (positive_blobs, negative_blobs)
        """
        positive = [(i, b) for i, b in enumerate(blobs) if get_value_fn(b) > 0]
        negative = [(i, b) for i, b in enumerate(blobs) if get_value_fn(b) < 0]
        return positive, negative
    
    @staticmethod
    def _create_cost_matrix_spatial(blobs_a: List[Blob], blobs_b: List[Blob]) -> np.ndarray:
        """
        Create a spatial cost matrix between two sets of blobs
        
        Args:
            blobs_a: First set of blobs
            blobs_b: Second set of blobs
            
        Returns:
            Distance matrix between all pairs of blobs
        """
        centers_a = np.array([blob.center for blob in blobs_a])
        centers_b = np.array([blob.center for blob in blobs_b])
        
        # Handle empty distributions
        if len(centers_a) == 0 or len(centers_b) == 0:
            return np.array([])
            
        # Calculate pairwise Euclidean distances
        return ot.dist(centers_a, centers_b)
    
    @staticmethod
    def _create_cost_matrix_heights(blobs_a: List[Blob], blobs_b: List[Blob]) -> np.ndarray:
        """
        Create a height difference cost matrix between two sets of blobs
        
        Args:
            blobs_a: First set of blobs
            blobs_b: Second set of blobs
            
        Returns:
            Matrix of absolute height differences between all pairs of blobs
        """
        if not blobs_a or not blobs_b:
            return np.array([])
            
        heights_a = np.array([blob.height for blob in blobs_a])
        heights_b = np.array([blob.height for blob in blobs_b])
        
        # Calculate pairwise absolute height differences
        n_a, n_b = len(heights_a), len(heights_b)
        cost_matrix = np.zeros((n_a, n_b))
        
        for i in range(n_a):
            for j in range(n_b):
                cost_matrix[i, j] = abs(heights_a[i] - heights_b[j])
                
        return cost_matrix
    
    # ---------- Spatial Metrics using POT ----------
    
    @staticmethod
    def calculate_wasserstein_plan(
        dist_a: Distribution, 
        dist_b: Distribution, 
        metric: str = 'euclidean', 
        p: int = 2
    ) -> Tuple[float, List[Tuple[int, int, float]]]:
        """
        Calculate Wasserstein distance and transportation plan using POT, respecting sign of weights.
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            metric: Distance metric to use ('euclidean', 'sqeuclidean', 'cityblock')
            p: Power for Wasserstein distance (p=1 for EMD, p=2 for Wasserstein-2)
            
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
            cost_matrix = POTDistanceCalculator._create_cost_matrix_spatial(pos_blobs_a, pos_blobs_b)
            
            # Create weight vectors
            a = np.array([b.height for b in pos_blobs_a])
            b = np.array([b.height for b in pos_blobs_b])
            
            # Normalize weights to create probability distributions
            a = a / np.sum(a)
            b = b / np.sum(b)
            
            # Calculate optimal transport plan using the exact EMD solver (more robust than Sinkhorn)
            transport_plan = ot.emd(a, b, cost_matrix)
            
            # Calculate the Wasserstein distance
            wasserstein_dist = np.sum(transport_plan * cost_matrix)
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
            cost_matrix = POTDistanceCalculator._create_cost_matrix_spatial(neg_blobs_a, neg_blobs_b)
            
            # Create weight vectors (use absolute values)
            a = np.array([abs(b.height) for b in neg_blobs_a])
            b = np.array([abs(b.height) for b in neg_blobs_b])
            
            # Normalize weights to create probability distributions
            a = a / np.sum(a)
            b = b / np.sum(b)
            
            # Calculate optimal transport plan using the exact EMD solver (more robust than Sinkhorn)
            transport_plan = ot.emd(a, b, cost_matrix)
            
            # Calculate the Wasserstein distance
            wasserstein_dist = np.sum(transport_plan * cost_matrix)
            total_distance += wasserstein_dist
            
            # Add matching pairs to result
            for i in range(len(neg_blobs_a)):
                for j in range(len(neg_blobs_b)):
                    if transport_plan[i, j] > 1e-10:  # Threshold for numerical stability
                        orig_idx_a = neg_indices_a[i]
                        orig_idx_b = neg_indices_b[j]
                        pairs.append((orig_idx_a, orig_idx_b, float(transport_plan[i, j])))
        
        return total_distance, pairs
    
    @staticmethod
    def calculate_bottleneck(dist_a: Distribution, dist_b: Distribution) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Calculate bottleneck distance and matching using worst-cost matching algorithm with
        POT-computed distance matrices, respecting sign of weights.
        
        The bottleneck distance is the cost of the worst-case assignment in an optimal matching.
        
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
            # Create spatial cost matrix using POT's distance function
            cost_matrix = POTDistanceCalculator._create_cost_matrix_spatial(pos_blobs_a, pos_blobs_b)
            
            # Use scipy's linear_sum_assignment for the assignment problem (Hungarian algorithm)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Find the maximum cost in the assignment (bottleneck distance)
            pos_bottleneck = 0.0
            pos_matching: List[Tuple[int, int]] = []
            
            for i, j in zip(row_ind, col_ind):
                if i < len(pos_indices_a) and j < len(pos_indices_b):
                    orig_idx_a = pos_indices_a[i]
                    orig_idx_b = pos_indices_b[j]
                    pos_matching.append((orig_idx_a, orig_idx_b))
                    pos_bottleneck = max(pos_bottleneck, cost_matrix[i, j])
            
            # Add the matching pairs and update the max distance
            pairs.extend(pos_matching)
            max_distance = max(max_distance, pos_bottleneck)
        
        # Process negative blobs
        if neg_blobs_a and neg_blobs_b:
            # Create spatial cost matrix using POT's distance function
            cost_matrix = POTDistanceCalculator._create_cost_matrix_spatial(neg_blobs_a, neg_blobs_b)
            
            # Use scipy's linear_sum_assignment for the assignment problem (Hungarian algorithm)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Find the maximum cost in the assignment (bottleneck distance)
            neg_bottleneck = 0.0
            neg_matching: List[Tuple[int, int]] = []
            
            for i, j in zip(row_ind, col_ind):
                if i < len(neg_indices_a) and j < len(neg_indices_b):
                    orig_idx_a = neg_indices_a[i]
                    orig_idx_b = neg_indices_b[j]
                    neg_matching.append((orig_idx_a, orig_idx_b))
                    neg_bottleneck = max(neg_bottleneck, cost_matrix[i, j])
            
            # Add the matching pairs and update the max distance
            pairs.extend(neg_matching)
            max_distance = max(max_distance, neg_bottleneck)
        
        return max_distance, pairs
    
    # ---------- Height-Based Metrics using POT ----------
    
    @staticmethod
    def calculate_height_wasserstein_plan(dist_a: Distribution, dist_b: Distribution) -> Tuple[float, List[Tuple[int, int, float]]]:
        """
        Calculate Wasserstein distance and transportation plan based only on blob heights using POT
        
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
            # Create height cost matrix
            cost_matrix = POTDistanceCalculator._create_cost_matrix_heights(pos_blobs_a, pos_blobs_b)
            
            # Create weight vectors
            a = np.array([b.height for b in pos_blobs_a])
            b = np.array([b.height for b in pos_blobs_b])
            
            # Normalize weights to create probability distributions
            a = a / np.sum(a)
            b = b / np.sum(b)
            
            # Calculate optimal transport plan using the exact EMD solver (more robust than Sinkhorn)
            transport_plan = ot.emd(a, b, cost_matrix)
            
            # Calculate the Wasserstein distance
            wasserstein_dist = np.sum(transport_plan * cost_matrix)
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
            # Create height cost matrix
            cost_matrix = POTDistanceCalculator._create_cost_matrix_heights(neg_blobs_a, neg_blobs_b)
            
            # Create weight vectors (use absolute values)
            a = np.array([abs(b.height) for b in neg_blobs_a])
            b = np.array([abs(b.height) for b in neg_blobs_b])
            
            # Normalize weights to create probability distributions
            a = a / np.sum(a)
            b = b / np.sum(b)
            
            # Calculate optimal transport plan using the exact EMD solver (more robust than Sinkhorn)
            transport_plan = ot.emd(a, b, cost_matrix)
            
            # Calculate the Wasserstein distance
            wasserstein_dist = np.sum(transport_plan * cost_matrix)
            total_distance += wasserstein_dist
            
            # Add matching pairs to result
            for i in range(len(neg_blobs_a)):
                for j in range(len(neg_blobs_b)):
                    if transport_plan[i, j] > 1e-10:  # Threshold for numerical stability
                        orig_idx_a = neg_indices_a[i]
                        orig_idx_b = neg_indices_b[j]
                        pairs.append((orig_idx_a, orig_idx_b, float(transport_plan[i, j])))
        
        return total_distance, pairs
    
    @staticmethod
    def calculate_height_bottleneck_plan(dist_a: Distribution, dist_b: Distribution) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Calculate bottleneck distance and matching based only on blob heights using POT
        for distance computation and custom worst-cost matching algorithm.
        
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
            # Create height cost matrix using POT's distance function
            cost_matrix = POTDistanceCalculator._create_cost_matrix_heights(pos_blobs_a, pos_blobs_b)
            
            # Use scipy's linear_sum_assignment for the assignment problem (Hungarian algorithm)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Find the maximum cost in the assignment (bottleneck distance)
            pos_bottleneck = 0.0
            pos_matching: List[Tuple[int, int]] = []
            
            for i, j in zip(row_ind, col_ind):
                if i < len(pos_indices_a) and j < len(pos_indices_b):
                    orig_idx_a = pos_indices_a[i]
                    orig_idx_b = pos_indices_b[j]
                    pos_matching.append((orig_idx_a, orig_idx_b))
                    pos_bottleneck = max(pos_bottleneck, cost_matrix[i, j])
            
            # Add the matching pairs and update the max distance
            pairs.extend(pos_matching)
            max_distance = max(max_distance, pos_bottleneck)
        
        # Process negative blobs
        if neg_blobs_a and neg_blobs_b:
            # Create height cost matrix using POT's distance function
            cost_matrix = POTDistanceCalculator._create_cost_matrix_heights(neg_blobs_a, neg_blobs_b)
            
            # Use scipy's linear_sum_assignment for the assignment problem (Hungarian algorithm)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Find the maximum cost in the assignment (bottleneck distance)
            neg_bottleneck = 0.0
            neg_matching: List[Tuple[int, int]] = []
            
            for i, j in zip(row_ind, col_ind):
                if i < len(neg_indices_a) and j < len(neg_indices_b):
                    orig_idx_a = neg_indices_a[i]
                    orig_idx_b = neg_indices_b[j]
                    neg_matching.append((orig_idx_a, orig_idx_b))
                    neg_bottleneck = max(neg_bottleneck, cost_matrix[i, j])
            
            # Add the matching pairs and update the max distance
            pairs.extend(neg_matching)
            max_distance = max(max_distance, neg_bottleneck)
        
        return max_distance, pairs
    
    # ---------- Helper Methods for Mathematical Validation ----------
    
    @staticmethod
    def get_distance_matrix(
        dist_a: Distribution, 
        dist_b: Distribution, 
        metric: str = 'spatial'
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Generate a distance matrix between blobs from two distributions.
        Useful for validating the calculations manually.
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            metric: 'spatial' for Euclidean distance or 'height' for height differences
            
        Returns:
            Tuple containing (distance_matrix, blob_indices_a, blob_indices_b)
        """
        blobs_a = dist_a.blobs
        blobs_b = dist_b.blobs
        
        if not blobs_a or not blobs_b:
            return np.array([]), [], []
        
        indices_a = list(range(len(blobs_a)))
        indices_b = list(range(len(blobs_b)))
        
        if metric == 'spatial':
            # Create distance matrix based on spatial distances
            cost_matrix = POTDistanceCalculator._create_cost_matrix_spatial(blobs_a, blobs_b)
        else:  # 'height'
            # Create distance matrix based on height differences
            cost_matrix = POTDistanceCalculator._create_cost_matrix_heights(blobs_a, blobs_b)
        
        return cost_matrix, indices_a, indices_b
    
    @staticmethod
    def explain_matching(
        dist_a: Distribution, 
        dist_b: Distribution, 
        matching_pairs: List[Tuple[int, int]], 
        weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a human-readable explanation of a matching between two distributions.
        Useful for validating the calculations and understanding the results.
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            matching_pairs: List of pairs (idx_a, idx_b) indicating the matching
            weights: Optional list of weights for Wasserstein transport
            
        Returns:
            List of dictionaries with detailed information about each match
        """
        blobs_a = dist_a.blobs
        blobs_b = dist_b.blobs
        
        explanations: List[Dict[str, Any]] = []
        
        if weights is None:
            weights = [1.0] * len(matching_pairs)
        
        for (idx_a, idx_b), weight in zip(matching_pairs, weights):
            if idx_a < len(blobs_a) and idx_b < len(blobs_b):
                blob_a = blobs_a[idx_a]
                blob_b = blobs_b[idx_b]
                
                # Calculate metrics
                spatial_distance = np.sqrt((blob_a.center[0] - blob_b.center[0])**2 + 
                                          (blob_a.center[1] - blob_b.center[1])**2)
                height_difference = abs(blob_a.height - blob_b.height)
                
                explanation = {
                    "blob_a_id": idx_a,
                    "blob_b_id": idx_b,
                    "blob_a_position": blob_a.center,
                    "blob_b_position": blob_b.center,
                    "blob_a_height": blob_a.height,
                    "blob_b_height": blob_b.height,
                    "spatial_distance": spatial_distance,
                    "height_distance": height_difference,
                    "weight": weight
                }
                explanations.append(explanation)
        
        return explanations