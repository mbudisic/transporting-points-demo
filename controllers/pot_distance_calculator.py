import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from models.distribution import Distribution
import ot
from scipy.optimize import linear_sum_assignment
from itertools import zip_longest

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
    
    # ---------- Utility Functions ----------
    
    @staticmethod
    def _separate_by_sign(blobs, get_value_fn: Callable = lambda b: b.height):
        """Separate blobs by sign (positive/negative)"""
        positive = [(i, b) for i, b in enumerate(blobs) if get_value_fn(b) > 0]
        negative = [(i, b) for i, b in enumerate(blobs) if get_value_fn(b) < 0]
        return positive, negative
    
    @staticmethod
    def _create_cost_matrix_spatial(blobs_a, blobs_b):
        """Create a spatial cost matrix between two sets of blobs"""
        centers_a = np.array([blob.center for blob in blobs_a])
        centers_b = np.array([blob.center for blob in blobs_b])
        
        # Handle empty distributions
        if len(centers_a) == 0 or len(centers_b) == 0:
            return np.array([])
            
        # Calculate pairwise Euclidean distances
        return ot.dist(centers_a, centers_b)
    
    @staticmethod
    def _create_cost_matrix_heights(blobs_a, blobs_b):
        """Create a height difference cost matrix between two sets of blobs"""
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
    def calculate_wasserstein_plan(dist_a: Distribution, dist_b: Distribution, 
                                   metric='euclidean', p=2, 
                                   regularization=0.01) -> Tuple[float, List[Tuple[int, int, float]]]:
        """
        Calculate Wasserstein distance and transportation plan using POT, respecting sign of weights.
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            metric: Distance metric to use ('euclidean', 'sqeuclidean', 'cityblock')
            p: Power for Wasserstein distance (p=1 for EMD, p=2 for Wasserstein-2)
            regularization: Regularization parameter for Sinkhorn algorithm
            
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
        
        pairs = []
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
            
            # Calculate optimal transport plan
            # Using Sinkhorn algorithm for regularized OT
            transport_plan = ot.sinkhorn(a, b, cost_matrix, regularization)
            
            # Add matching pairs to result
            for i in range(len(pos_blobs_a)):
                for j in range(len(pos_blobs_b)):
                    if transport_plan[i, j] > 1e-10:  # Threshold for numerical stability
                        orig_idx_a = pos_indices_a[i]
                        orig_idx_b = pos_indices_b[j]
                        pairs.append((orig_idx_a, orig_idx_b, float(transport_plan[i, j])))
                        total_distance += cost_matrix[i, j] * transport_plan[i, j]
        
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
            
            # Calculate optimal transport plan
            transport_plan = ot.sinkhorn(a, b, cost_matrix, regularization)
            
            # Add matching pairs to result
            for i in range(len(neg_blobs_a)):
                for j in range(len(neg_blobs_b)):
                    if transport_plan[i, j] > 1e-10:  # Threshold for numerical stability
                        orig_idx_a = neg_indices_a[i]
                        orig_idx_b = neg_indices_b[j]
                        pairs.append((orig_idx_a, orig_idx_b, float(transport_plan[i, j])))
                        total_distance += cost_matrix[i, j] * transport_plan[i, j]
        
        return total_distance, pairs
    
    @staticmethod
    def calculate_bottleneck(dist_a: Distribution, dist_b: Distribution) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Calculate bottleneck distance and matching using POT, respecting sign of weights
        
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
        
        pairs = []
        max_distance = 0.0
        
        # Process positive blobs
        if pos_blobs_a and pos_blobs_b:
            # Create spatial cost matrix
            cost_matrix = POTDistanceCalculator._create_cost_matrix_spatial(pos_blobs_a, pos_blobs_b)
            
            # Use scipy's linear_sum_assignment function (Hungarian algorithm)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for i, j in zip(row_ind, col_ind):
                if i < len(pos_indices_a) and j < len(pos_indices_b):
                    orig_idx_a = pos_indices_a[i]
                    orig_idx_b = pos_indices_b[j]
                    pairs.append((orig_idx_a, orig_idx_b))
                    max_distance = max(max_distance, cost_matrix[i, j])
        
        # Process negative blobs
        if neg_blobs_a and neg_blobs_b:
            # Create spatial cost matrix
            cost_matrix = POTDistanceCalculator._create_cost_matrix_spatial(neg_blobs_a, neg_blobs_b)
            
            # Use POT's linear assignment function
            assignment = ot.linear_assignment(cost_matrix)
            
            for i, j in zip(assignment[0], assignment[1]):
                if i < len(neg_indices_a) and j < len(neg_indices_b):
                    orig_idx_a = neg_indices_a[i]
                    orig_idx_b = neg_indices_b[j]
                    pairs.append((orig_idx_a, orig_idx_b))
                    max_distance = max(max_distance, cost_matrix[i, j])
        
        return max_distance, pairs
    
    # ---------- Height-Based Metrics using POT ----------
    
    @staticmethod
    def calculate_height_wasserstein_plan(dist_a: Distribution, dist_b: Distribution,
                                         regularization=0.01) -> Tuple[float, List[Tuple[int, int, float]]]:
        """
        Calculate Wasserstein distance and transportation plan based only on blob heights using POT
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            regularization: Regularization parameter for Sinkhorn algorithm
            
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
        
        pairs = []
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
            
            # Calculate optimal transport plan
            transport_plan = ot.sinkhorn(a, b, cost_matrix, regularization)
            
            # Add matching pairs to result
            for i in range(len(pos_blobs_a)):
                for j in range(len(pos_blobs_b)):
                    if transport_plan[i, j] > 1e-10:  # Threshold for numerical stability
                        orig_idx_a = pos_indices_a[i]
                        orig_idx_b = pos_indices_b[j]
                        pairs.append((orig_idx_a, orig_idx_b, float(transport_plan[i, j])))
                        total_distance += cost_matrix[i, j] * transport_plan[i, j]
        
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
            
            # Calculate optimal transport plan
            transport_plan = ot.sinkhorn(a, b, cost_matrix, regularization)
            
            # Add matching pairs to result
            for i in range(len(neg_blobs_a)):
                for j in range(len(neg_blobs_b)):
                    if transport_plan[i, j] > 1e-10:  # Threshold for numerical stability
                        orig_idx_a = neg_indices_a[i]
                        orig_idx_b = neg_indices_b[j]
                        pairs.append((orig_idx_a, orig_idx_b, float(transport_plan[i, j])))
                        total_distance += cost_matrix[i, j] * transport_plan[i, j]
        
        return total_distance, pairs
    
    @staticmethod
    def calculate_height_bottleneck_plan(dist_a: Distribution, dist_b: Distribution) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Calculate bottleneck distance and matching based only on blob heights using POT
        
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
        
        pairs = []
        max_distance = 0.0
        
        # Process positive blobs
        if pos_blobs_a and pos_blobs_b:
            # Create height cost matrix
            cost_matrix = POTDistanceCalculator._create_cost_matrix_heights(pos_blobs_a, pos_blobs_b)
            
            # Use POT's linear assignment function
            assignment = ot.linear_assignment(cost_matrix)
            
            for i, j in zip(assignment[0], assignment[1]):
                if i < len(pos_indices_a) and j < len(pos_indices_b):
                    orig_idx_a = pos_indices_a[i]
                    orig_idx_b = pos_indices_b[j]
                    pairs.append((orig_idx_a, orig_idx_b))
                    max_distance = max(max_distance, cost_matrix[i, j])
        
        # Process negative blobs
        if neg_blobs_a and neg_blobs_b:
            # Create height cost matrix
            cost_matrix = POTDistanceCalculator._create_cost_matrix_heights(neg_blobs_a, neg_blobs_b)
            
            # Use POT's linear assignment function
            assignment = ot.linear_assignment(cost_matrix)
            
            for i, j in zip(assignment[0], assignment[1]):
                if i < len(neg_indices_a) and j < len(neg_indices_b):
                    orig_idx_a = neg_indices_a[i]
                    orig_idx_b = neg_indices_b[j]
                    pairs.append((orig_idx_a, orig_idx_b))
                    max_distance = max(max_distance, cost_matrix[i, j])
        
        return max_distance, pairs
    
    # ---------- Helper Methods for Mathematical Validation ----------
    
    @staticmethod
    def get_distance_matrix(dist_a: Distribution, dist_b: Distribution, 
                           metric: str = 'spatial') -> Tuple[np.ndarray, List[int], List[int]]:
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
    def explain_matching(dist_a: Distribution, dist_b: Distribution, 
                        matching_pairs: List[Tuple[int, int]], 
                        weights: Optional[List[float]] = None) -> List[Dict]:
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
        
        explanations = []
        
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
                    "blob_a_idx": idx_a,
                    "blob_b_idx": idx_b,
                    "blob_a_center": blob_a.center,
                    "blob_b_center": blob_b.center,
                    "blob_a_height": blob_a.height,
                    "blob_b_height": blob_b.height,
                    "spatial_distance": spatial_distance,
                    "height_difference": height_difference,
                    "weight": weight
                }
                explanations.append(explanation)
        
        return explanations