import numpy as np
from typing import List, Tuple, Dict, Optional
from models.distribution import Distribution
import scipy.spatial.distance as dist
import scipy.optimize as opt

class DistanceCalculator:
    """
    Controller class for calculating distances between distributions
    """
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