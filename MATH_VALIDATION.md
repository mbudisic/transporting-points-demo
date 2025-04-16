# Mathematical Validation Guide

This guide explains how to verify the mathematical calculations used in the Distribution Distance Visualization application, particularly focusing on the transportation plans and distance metrics.

## Overview of Mathematical Components

The application implements several distance metrics between probability distributions:

1. **Spatial Bottleneck Distance**: Measures the maximum minimum distance between paired points
2. **Spatial Wasserstein Distance**: Measures the overall transportation cost between distributions
3. **Height-Based Bottleneck Distance**: Similar to spatial bottleneck, but based on blob heights
4. **Height-Based Wasserstein Distance**: Similar to spatial Wasserstein, but based on blob heights

## Validating the Calculations

### Prerequisites

- Basic familiarity with optimal transport theory
- Understanding of distance metrics between distributions
- Knowledge of linear programming (for Wasserstein distance)
- Knowledge of bottleneck matching (for bottleneck distance)

### Step 1: Accessing the Raw Distance Calculation Code

The core calculation code is located in `controllers/distance_calculator.py`. This file contains the implementation of:

- `calculate_wasserstein_plan()`: Computes the Wasserstein distance and transportation plan
- `calculate_bottleneck()`: Computes the bottleneck distance and matching
- `calculate_height_wasserstein_plan()`: Computes the height-based Wasserstein distance
- `calculate_height_bottleneck_plan()`: Computes the height-based bottleneck distance

### Step 2: Manual Calculation for Simple Examples

To validate the implementation:

1. Create a simple distribution (2-3 blobs) in both Distribution A and B
2. Record the coordinates and heights of each blob
3. Manually calculate the expected distances:
   - For bottleneck: Find the min-max matching by hand
   - For Wasserstein: Solve the transportation problem using the Hungarian algorithm or linear programming

### Step 3: Export Data and Verify

1. Use the "Export Data" button to download the current distributions as a CSV file
2. The file contains the exact coordinates, heights, and variances used in calculations
3. Perform your own calculations using this data
4. Compare with the displayed results in the application

### Bottleneck Distance Verification

For bottleneck distance, the application:

1. Creates a bipartite graph where:
   - Nodes on one side are blobs from Distribution A
   - Nodes on the other side are blobs from Distribution B
   - Edges have weights equal to the Euclidean distance between blob centers (for spatial) or the absolute difference in heights (for height-based)
2. Finds a complete matching that minimizes the maximum edge weight

You can verify this by:
1. Constructing the distance matrix between all pairs of blobs
2. Finding a matching that minimizes the maximum distance
3. Compare the result with what's displayed in the "Distance Matrices" tab

### Wasserstein Distance Verification

For Wasserstein distance, the application:

1. Creates a transportation problem where:
   - Supply nodes are blobs from Distribution A with supply equal to their height
   - Demand nodes are blobs from Distribution B with demand equal to their height
   - Costs are the Euclidean distances between blob centers (for spatial) or the absolute differences in heights (for height-based)
2. Solves the transportation problem to minimize the total cost

You can verify this by:
1. Formulating the linear program with the exported blob data
2. Solving it using a linear programming solver
3. Comparing the result with what's displayed in the application

## Sample Validation Calculation

For a simple example with:

**Distribution A:**
- Blob 1: Position (1, 2), Height 1.0
- Blob 2: Position (4, 5), Height -0.5

**Distribution B:**
- Blob 1: Position (2, 3), Height 0.8
- Blob 2: Position (5, 6), Height -0.3

**Expected Spatial Bottleneck Distance:**
The distances between pairs are:
- A1 to B1: √[(1-2)² + (2-3)²] = √2 ≈ 1.414
- A1 to B2: √[(1-5)² + (2-6)²] = √32 ≈ 5.657
- A2 to B1: √[(4-2)² + (5-3)²] = √8 ≈ 2.828
- A2 to B2: √[(4-5)² + (5-6)²] = √2 ≈ 1.414

Since positive blobs match with positive and negative with negative, the bottleneck matching is:
- A1 to B1 (distance 1.414)
- A2 to B2 (distance 1.414)
The bottleneck value is 1.414.

**Expected Height-Based Bottleneck Distance:**
The height differences are:
- A1 to B1: |1.0 - 0.8| = 0.2
- A2 to B2: |-0.5 - (-0.3)| = 0.2
The bottleneck value is 0.2.

## Getting Support

If you find discrepancies between your calculations and the application:

1. Check the "Distance Matrices" tab to see the raw distance values used
2. Verify that you're using the same mathematical formulation
3. If issues persist, check the calculation methods in `controllers/distance_calculator.py`