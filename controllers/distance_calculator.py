import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from models.distribution import Distribution
import scipy.spatial.distance as dist
import scipy.optimize as opt
from itertools import zip_longest

class DistanceCalculator:
    """
    Controller class for calculating distances between distributions
    """
    
    # Common utility functions for distance calculations
    
    @staticmethod
    def _euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def _create_distance_matrix(centers_a: List[Tuple[float, float]], centers_b: List[Tuple[float, float]]) -> np.ndarray:
        """Create a distance matrix between two sets of centers"""
        n_a = len(centers_a)
        n_b = len(centers_b)
        cost_matrix = np.zeros((n_a, n_b))
        
        for i in range(n_a):
            for j in range(n_b):
                cost_matrix[i, j] = DistanceCalculator._euclidean_distance(centers_a[i], centers_b[j])
                
        return cost_matrix
    
    @staticmethod
    def _solve_assignment_problem(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the assignment problem using the Hungarian algorithm"""
        return opt.linear_sum_assignment(cost_matrix)
    
    @staticmethod
    def _separate_by_sign(blobs, get_value_fn: Callable = lambda b: b.height):
        """Separate blobs by sign (positive/negative)"""
        positive = [(i, b) for i, b in enumerate(blobs) if get_value_fn(b) > 0]
        negative = [(i, b) for i, b in enumerate(blobs) if get_value_fn(b) < 0]
        return positive, negative
    @staticmethod
    def calculate_wasserstein_continuous(dist_a: Distribution, dist_b: Distribution, 
                                        grid_size: int = 100) -> float:
        """
        Calculate Wasserstein distance between two continuous distributions.
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            grid_size: Number of points in the grid (higher = more accurate)
            
        Returns:
            Wasserstein distance between the distributions
        """
        # Create the grid for evaluation
        x_grid = np.linspace(0, 10, grid_size)
        y_grid = np.linspace(0, 10, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        
        # Calculate distribution values at the grid points
        dist_a_values = dist_a.create_gaussian_mixture(points)
        dist_b_values = dist_b.create_gaussian_mixture(points)
        
        # Use the earth mover's distance
        return np.sum(np.abs(dist_a_values - dist_b_values))
    
    @staticmethod
    def calculate_wasserstein_by_heights(dist_a: Distribution, dist_b: Distribution) -> float:
        """
        Calculate Wasserstein distance between two distributions based only on blob heights,
        ignoring spatial positions.
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            
        Returns:
            Wasserstein distance between the height distributions
        """
        # Get heights from each distribution
        heights_a = [blob.height for blob in dist_a.blobs]
        heights_b = [blob.height for blob in dist_b.blobs]
        
        # If either distribution is empty, return 0
        if not heights_a or not heights_b:
            return 0.0
        
        # Sort heights by absolute value (descending) but keep their signs
        heights_a.sort(key=lambda h: -abs(h))
        heights_b.sort(key=lambda h: -abs(h))
        
        # Normalize heights
        sum_abs_a = sum(abs(h) for h in heights_a)
        sum_abs_b = sum(abs(h) for h in heights_b)
        
        if sum_abs_a > 0:
            heights_a = [h / sum_abs_a for h in heights_a]
        if sum_abs_b > 0:
            heights_b = [h / sum_abs_b for h in heights_b]
        
        # Calculate Wasserstein distance between height distributions
        # Pad the shorter list with zeros
        if len(heights_a) < len(heights_b):
            heights_a.extend([0] * (len(heights_b) - len(heights_a)))
        elif len(heights_b) < len(heights_a):
            heights_b.extend([0] * (len(heights_a) - len(heights_b)))
        
        # Sum of absolute differences
        return sum(abs(a - b) for a, b in zip(heights_a, heights_b))
    
    @staticmethod
    def calculate_bottleneck_by_heights(dist_a: Distribution, dist_b: Distribution) -> float:
        """
        Calculate bottleneck distance between two distributions based only on blob heights,
        ignoring spatial positions.
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            
        Returns:
            Bottleneck distance between the height distributions
        """
        # Get heights from each distribution
        heights_a = [blob.height for blob in dist_a.blobs]
        heights_b = [blob.height for blob in dist_b.blobs]
        
        # If either distribution is empty, return 0
        if not heights_a or not heights_b:
            return 0.0
        
        # Sort heights by absolute value (descending) but keep their signs
        heights_a.sort(key=lambda h: -abs(h))
        heights_b.sort(key=lambda h: -abs(h))
        
        # Pad the shorter list with zeros
        if len(heights_a) < len(heights_b):
            heights_a.extend([0] * (len(heights_b) - len(heights_a)))
        elif len(heights_b) < len(heights_a):
            heights_b.extend([0] * (len(heights_a) - len(heights_b)))
        
        # Maximum absolute difference
        return max(abs(a - b) for a, b in zip(heights_a, heights_b))
        
    @staticmethod
    def calculate_height_bottleneck_plan(dist_a: Distribution, dist_b: Distribution) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Calculate bottleneck distance and matching based only on blob heights (ignores positions).
        
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
            
        # Create sorted lists of blob indices by absolute height (descending)
        indices_a = list(range(len(blobs_a)))
        indices_b = list(range(len(blobs_b)))
        
        indices_a.sort(key=lambda i: -abs(blobs_a[i].height))
        indices_b.sort(key=lambda i: -abs(blobs_b[i].height))
        
        # Match positives with positives and negatives with negatives
        pos_indices_a = [i for i in indices_a if blobs_a[i].height > 0]
        neg_indices_a = [i for i in indices_a if blobs_a[i].height < 0]
        pos_indices_b = [i for i in indices_b if blobs_b[i].height > 0]
        neg_indices_b = [i for i in indices_b if blobs_b[i].height < 0]
        
        matching_pairs = []
        max_difference = 0.0
        
        # Match positive to positive in order of decreasing height
        pos_pairs = zip_longest(pos_indices_a, pos_indices_b, fillvalue=None)
        for idx_a, idx_b in pos_pairs:
            if idx_a is None or idx_b is None:
                break
            matching_pairs.append((idx_a, idx_b))
            height_diff = abs(blobs_a[idx_a].height - blobs_b[idx_b].height)
            max_difference = max(max_difference, height_diff)
            
        # Match negative to negative in order of increasing height (most negative first)
        neg_pairs = zip_longest(neg_indices_a, neg_indices_b, fillvalue=None)
        for idx_a, idx_b in neg_pairs:
            if idx_a is None or idx_b is None:
                break
            matching_pairs.append((idx_a, idx_b))
            height_diff = abs(blobs_a[idx_a].height - blobs_b[idx_b].height)
            max_difference = max(max_difference, height_diff)
            
        return max_difference, matching_pairs
        
    @staticmethod
    def calculate_height_wasserstein_plan(dist_a: Distribution, dist_b: Distribution) -> Tuple[float, List[Tuple[int, int, float]]]:
        """
        Calculate Wasserstein distance and transportation plan based only on blob heights (ignores positions).
        
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
            
        # Get heights and indices
        heights_a = [(i, blob.height) for i, blob in enumerate(blobs_a)]
        heights_b = [(i, blob.height) for i, blob in enumerate(blobs_b)]
        
        # Sort by absolute height (descending)
        heights_a.sort(key=lambda x: -abs(x[1]))
        heights_b.sort(key=lambda x: -abs(x[1]))
        
        # Separate positive and negative heights
        pos_heights_a = [(i, h) for i, h in heights_a if h > 0]
        neg_heights_a = [(i, h) for i, h in heights_a if h < 0]
        pos_heights_b = [(i, h) for i, h in heights_b if h > 0]
        neg_heights_b = [(i, h) for i, h in heights_b if h < 0]
        
        # Create the matching pairs with weights
        pairs = []
        total_distance = 0.0
        
        # Match positive heights
        pos_idx_a = 0
        pos_idx_b = 0
        
        while pos_idx_a < len(pos_heights_a) and pos_idx_b < len(pos_heights_b):
            a_index, a_height = pos_heights_a[pos_idx_a]
            b_index, b_height = pos_heights_b[pos_idx_b]
            
            if a_height <= b_height:
                weight = a_height
                pairs.append((a_index, b_index, weight))
                total_distance += abs(a_height - weight)
                pos_idx_a += 1
                pos_heights_b[pos_idx_b] = (b_index, b_height - weight)
                if b_height - weight <= 0:
                    pos_idx_b += 1
            else:
                weight = b_height
                pairs.append((a_index, b_index, weight))
                total_distance += abs(b_height - weight)
                pos_idx_b += 1
                pos_heights_a[pos_idx_a] = (a_index, a_height - weight)
                if a_height - weight <= 0:
                    pos_idx_a += 1
        
        # Match negative heights
        neg_idx_a = 0
        neg_idx_b = 0
        
        while neg_idx_a < len(neg_heights_a) and neg_idx_b < len(neg_heights_b):
            a_index, a_height = neg_heights_a[neg_idx_a]
            b_index, b_height = neg_heights_b[neg_idx_b]
            
            if abs(a_height) <= abs(b_height):
                weight = abs(a_height)
                pairs.append((a_index, b_index, weight))
                total_distance += abs(abs(a_height) - weight)
                neg_idx_a += 1
                neg_heights_b[neg_idx_b] = (b_index, b_height + weight)
                if b_height + weight >= 0:
                    neg_idx_b += 1
            else:
                weight = abs(b_height)
                pairs.append((a_index, b_index, weight))
                total_distance += abs(abs(b_height) - weight)
                neg_idx_b += 1
                neg_heights_a[neg_idx_a] = (a_index, a_height + weight)
                if a_height + weight >= 0:
                    neg_idx_a += 1
        
        return total_distance, pairs
    
    @staticmethod
    def calculate_wasserstein_discrete(dist_a: Distribution, dist_b: Distribution) -> float:
        """
        Calculate Wasserstein distance between two discrete distributions.
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            
        Returns:
            Wasserstein distance between the distributions
        """
        # Get centers and weights
        centers_a = dist_a.centers
        centers_b = dist_b.centers
        weights_a = dist_a.weights
        weights_b = dist_b.weights
        
        # Handle empty distributions
        if not centers_a or not centers_b:
            return 0.0
        
        # Calculate the distance matrix between all points
        n_a = len(centers_a)
        n_b = len(centers_b)
        cost_matrix = np.zeros((n_a, n_b))
        
        for i in range(n_a):
            for j in range(n_b):
                # Euclidean distance between centers
                cost_matrix[i, j] = np.sqrt((centers_a[i][0] - centers_b[j][0])**2 + 
                                           (centers_a[i][1] - centers_b[j][1])**2)
        
        # Normalize weights for the earth mover's distance calculation
        weights_a_normalized = np.abs(weights_a) / np.sum(np.abs(weights_a))
        weights_b_normalized = np.abs(weights_b) / np.sum(np.abs(weights_b))
        
        # Use scipy's implementation of the Earth Mover's Distance (Wasserstein)
        try:
            return opt.linear_sum_assignment(cost_matrix)[1].sum()
        except:
            return np.mean(cost_matrix)  # Fallback if optimization fails
    
    @staticmethod
    def calculate_wasserstein_plan(dist_a: Distribution, dist_b: Distribution) -> Tuple[float, List[Tuple[int, int, float]]]:
        """
        Calculate Wasserstein distance and transportation plan between two distributions,
        respecting the sign of the weights.
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            
        Returns:
            A tuple containing (wasserstein_distance, matching_pairs)
            where matching_pairs is a list of tuples (idx_a, idx_b, weight)
            indicating the transportation plan with weights
        """
        blobs_a = dist_a.blobs
        blobs_b = dist_b.blobs
        
        # Handle empty distributions
        if not blobs_a or not blobs_b:
            return 0.0, []
        
        # Separate positive and negative centers
        pos_indices_a = [i for i, blob in enumerate(blobs_a) if blob.height > 0]
        neg_indices_a = [i for i, blob in enumerate(blobs_a) if blob.height < 0]
        pos_indices_b = [i for i, blob in enumerate(blobs_b) if blob.height > 0]
        neg_indices_b = [i for i, blob in enumerate(blobs_b) if blob.height < 0]
        
        # Create matching pairs
        pairs = []
        total_distance = 0.0
        
        # Match positive to positive
        if pos_indices_a and pos_indices_b:
            pos_centers_a = [blobs_a[i].center for i in pos_indices_a]
            pos_centers_b = [blobs_b[i].center for i in pos_indices_b]
            pos_weights_a = [blobs_a[i].height for i in pos_indices_a]
            pos_weights_b = [blobs_b[i].height for i in pos_indices_b]
            
            # Calculate distance matrix for positive pairs
            cost_matrix = np.zeros((len(pos_indices_a), len(pos_indices_b)))
            for i, idx_a in enumerate(pos_indices_a):
                for j, idx_b in enumerate(pos_indices_b):
                    cost_matrix[i, j] = np.sqrt(
                        (pos_centers_a[i][0] - pos_centers_b[j][0])**2 + 
                        (pos_centers_a[i][1] - pos_centers_b[j][1])**2
                    )
            
            # Use the Hungarian algorithm to find the optimal matching
            row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
            
            # Add matched pairs to the result
            for i, j in zip(row_ind, col_ind):
                orig_idx_a = pos_indices_a[i]
                orig_idx_b = pos_indices_b[j]
                weight = min(pos_weights_a[i], pos_weights_b[j])
                pairs.append((orig_idx_a, orig_idx_b, weight))
                total_distance += cost_matrix[i, j] * weight
        
        # Match negative to negative
        if neg_indices_a and neg_indices_b:
            neg_centers_a = [blobs_a[i].center for i in neg_indices_a]
            neg_centers_b = [blobs_b[i].center for i in neg_indices_b]
            neg_weights_a = [blobs_a[i].height for i in neg_indices_a]
            neg_weights_b = [blobs_b[i].height for i in neg_indices_b]
            
            # Calculate distance matrix for negative pairs
            cost_matrix = np.zeros((len(neg_indices_a), len(neg_indices_b)))
            for i, idx_a in enumerate(neg_indices_a):
                for j, idx_b in enumerate(neg_indices_b):
                    cost_matrix[i, j] = np.sqrt(
                        (neg_centers_a[i][0] - neg_centers_b[j][0])**2 + 
                        (neg_centers_a[i][1] - neg_centers_b[j][1])**2
                    )
            
            # Use the Hungarian algorithm to find the optimal matching
            row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
            
            # Add matched pairs to the result
            for i, j in zip(row_ind, col_ind):
                orig_idx_a = neg_indices_a[i]
                orig_idx_b = neg_indices_b[j]
                weight = min(neg_weights_a[i], neg_weights_b[j])
                pairs.append((orig_idx_a, orig_idx_b, weight))
                total_distance += cost_matrix[i, j] * weight
        
        return total_distance, pairs
    
    @staticmethod
    def calculate_bottleneck(dist_a: Distribution, dist_b: Distribution) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Calculate bottleneck distance between two distributions, respecting the sign of the weights.
        
        Args:
            dist_a: First distribution
            dist_b: Second distribution
            
        Returns:
            A tuple containing (bottleneck_distance, matching_pairs)
            where matching_pairs is a list of tuples (idx_a, idx_b) indicating the optimal matching
        """
        blobs_a = dist_a.blobs
        blobs_b = dist_b.blobs
        
        # Handle empty distributions
        if not blobs_a or not blobs_b:
            return 0.0, []
        
        # Separate positive and negative centers
        pos_indices_a = [i for i, blob in enumerate(blobs_a) if blob.height > 0]
        neg_indices_a = [i for i, blob in enumerate(blobs_a) if blob.height < 0]
        pos_indices_b = [i for i, blob in enumerate(blobs_b) if blob.height > 0]
        neg_indices_b = [i for i, blob in enumerate(blobs_b) if blob.height < 0]
        
        matching_pairs = []
        max_distance = 0.0
        
        # Match positive to positive
        if pos_indices_a and pos_indices_b:
            pos_centers_a = [blobs_a[i].center for i in pos_indices_a]
            pos_centers_b = [blobs_b[i].center for i in pos_indices_b]
            
            # Calculate distance matrix for positive pairs
            cost_matrix = np.zeros((len(pos_indices_a), len(pos_indices_b)))
            for i, idx_a in enumerate(pos_indices_a):
                for j, idx_b in enumerate(pos_indices_b):
                    cost_matrix[i, j] = np.sqrt(
                        (pos_centers_a[i][0] - pos_centers_b[j][0])**2 + 
                        (pos_centers_a[i][1] - pos_centers_b[j][1])**2
                    )
            
            # Use the Hungarian algorithm to find the optimal matching
            row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
            
            # Add matched pairs to the result
            for i, j in zip(row_ind, col_ind):
                orig_idx_a = pos_indices_a[i]
                orig_idx_b = pos_indices_b[j]
                matching_pairs.append((orig_idx_a, orig_idx_b))
                max_distance = max(max_distance, cost_matrix[i, j])
        
        # Match negative to negative
        if neg_indices_a and neg_indices_b:
            neg_centers_a = [blobs_a[i].center for i in neg_indices_a]
            neg_centers_b = [blobs_b[i].center for i in neg_indices_b]
            
            # Calculate distance matrix for negative pairs
            cost_matrix = np.zeros((len(neg_indices_a), len(neg_indices_b)))
            for i, idx_a in enumerate(neg_indices_a):
                for j, idx_b in enumerate(neg_indices_b):
                    cost_matrix[i, j] = np.sqrt(
                        (neg_centers_a[i][0] - neg_centers_b[j][0])**2 + 
                        (neg_centers_a[i][1] - neg_centers_b[j][1])**2
                    )
            
            # Use the Hungarian algorithm to find the optimal matching
            row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
            
            # Add matched pairs to the result
            for i, j in zip(row_ind, col_ind):
                orig_idx_a = neg_indices_a[i]
                orig_idx_b = neg_indices_b[j]
                matching_pairs.append((orig_idx_a, orig_idx_b))
                max_distance = max(max_distance, cost_matrix[i, j])
        
        return max_distance, matching_pairs
    
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
            centers_a = [blobs_a[i].center for i in indices_a]
            centers_b = [blobs_b[i].center for i in indices_b]
            distance_matrix = DistanceCalculator._create_distance_matrix(centers_a, centers_b)
        else:  # 'height'
            # Create distance matrix based on height differences
            distance_matrix = np.zeros((len(indices_a), len(indices_b)))
            for i, idx_a in enumerate(indices_a):
                for j, idx_b in enumerate(indices_b):
                    distance_matrix[i, j] = abs(blobs_a[idx_a].height - blobs_b[idx_b].height)
        
        return distance_matrix, indices_a, indices_b
    
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
        if weights is None:
            weights = [1.0] * len(matching_pairs)
            
        explanations = []
        
        for (idx_a, idx_b), weight in zip(matching_pairs, weights):
            blob_a = dist_a.blobs[idx_a]
            blob_b = dist_b.blobs[idx_b]
            
            spatial_distance = DistanceCalculator._euclidean_distance(blob_a.center, blob_b.center)
            height_distance = abs(blob_a.height - blob_b.height)
            
            explanations.append({
                'blob_a_id': idx_a,
                'blob_b_id': idx_b,
                'blob_a_position': blob_a.center,
                'blob_b_position': blob_b.center,
                'blob_a_height': blob_a.height,
                'blob_b_height': blob_b.height,
                'spatial_distance': spatial_distance,
                'height_distance': height_distance,
                'weight': weight
            })
        
        return explanations