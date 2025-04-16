# Mathematical Validation Guide

This guide explains how to verify the mathematical calculations used in the Distribution Distance Visualization application, particularly focusing on the transportation plans and distance metrics.

## Overview of Mathematical Components

The application implements several distance metrics between probability distributions:

1. **Spatial Bottleneck Distance**: Measures the maximum minimum distance between paired points
2. **Spatial Wasserstein Distance**: Measures the overall transportation cost between distributions (Earth Mover's Distance)
3. **Height-Based Bottleneck Distance**: Similar to spatial bottleneck, but based on blob heights
4. **Height-Based Wasserstein Distance**: Similar to spatial Wasserstein, but based on blob heights

## Validating the Calculations

### Prerequisites

- Basic familiarity with optimal transport theory
- Understanding of distance metrics between distributions
- Knowledge of the Earth Mover's Distance (EMD) algorithm
- Knowledge of the Hungarian algorithm for assignment problems
- Knowledge of bottleneck matching (for bottleneck distance)

### Step 1: Accessing the Raw Distance Calculation Code

The core calculation code is located in `controllers/pot_distance_calculator.py`. This file contains the implementation of:

- `calculate_wasserstein_plan()`: Computes the Wasserstein distance (Earth Mover's Distance) and transportation plan
  - Uses the Python Optimal Transport (POT) package's `ot.emd()` function to compute exact EMD solutions
  - Handles positive and negative heights separately to maintain sign preservation
  - Normalizes distributions to create proper probability distributions for POT algorithms
  
- `calculate_bottleneck()`: Computes the bottleneck distance and matching
  - Uses POT to compute distance matrices but implements a custom worst-cost matching algorithm
  - Leverages `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) to find optimal assignments
  - Computes the maximum distance (bottleneck) from the optimal assignment
  
- `calculate_height_wasserstein_plan()`: Computes the height-based Wasserstein distance
  - Similar to spatial Wasserstein but uses a height-difference cost matrix
  - Uses `ot.emd()` for exact Earth Mover's Distance calculation
  
- `calculate_height_bottleneck_plan()`: Computes the height-based bottleneck distance
  - Similar to spatial bottleneck but uses a height-difference cost matrix
  - Uses a combination of POT distance computation and custom worst-cost matching

Additionally, the following helper methods are useful for validation:

- `_create_cost_matrix_spatial()`: Creates distance matrices for spatial metrics using `ot.dist()`
- `_create_cost_matrix_heights()`: Creates distance matrices for height-based metrics
- `get_distance_matrix()`: Generates the raw distance matrices used in calculations
- `explain_matching()`: Creates human-readable explanations of matchings

### Step 2: Manual Calculation for Simple Examples

To validate the implementation:

1. Create a simple distribution (2-3 blobs) in both Distribution A and B
2. Record the coordinates and heights of each blob
3. Manually calculate the expected distances:
   - For bottleneck: Find the min-max matching by hand
   - For Wasserstein: Solve the transportation problem using the Hungarian algorithm or linear programming

### Step 3: Export Data and Verify

1. Use the "Export Data" section to download the current distributions:
   - HTML Report: Contains a comprehensive report focusing on the currently selected transport plan
   - JSON Data: Contains structured data of distributions and the selected transport plan
   - CSV File: Contains the exact coordinates, heights, and variances of all blobs
2. The exports only include information about the currently selected transport plan, its distance matrices, and optimal value
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

## Validating Export Documents

The application provides three types of export documents, all focused on the currently selected transport plan:

1. **HTML Report**: Contains a comprehensive report of the distributions and the selected transport plan
   - Check that exported distance matrices match what's shown in the UI
   - Verify that the transport plan shown is only for the selected visualization mode
   - Check the metric value displayed in the report against the value in the UI

2. **JSON Data**: Contains structured data for programmatic analysis
   - All distance calculations and matchings are available for further analysis
   - The structure includes only data for the currently selected transport plan
   - Use this for automated validation scripts or more complex verifications

3. **CSV Export**: Contains blob-level data for all blobs in both distributions
   - Use this to reconstruct the distributions in other software
   - Perform independent distance calculations and compare the results

### Source Code for Export Functionality

The export functionality is implemented in `utils/export_utils.py` (lines 12-489). The key functions are:

- `generate_distribution_data()` (lines 12-121): Creates a structured data object with only the selected transport plan
- `generate_html_report()` (lines 123-385): Generates an HTML report with visualizations and tables
- `export_to_formats()` (lines 429-489): Creates downloadable files in multiple formats

## Python Optimal Transport (POT) Package

The application uses the [Python Optimal Transport (POT)](https://pythonot.github.io/) package for computing distance matrices and solving optimal transport problems. Key functions used include:

1. **`ot.dist()`**: Computes the distance matrix between two sets of points
   - Documentation: [POT Distance Functions](https://pythonot.github.io/gen_modules/ot.dist.html)
   - Used to compute Euclidean distances between blob centers in `_create_cost_matrix_spatial()`

2. **`ot.emd()`**: Solves the Earth Mover's Distance (EMD) problem exactly
   - Documentation: [POT EMD Solver](https://pythonot.github.io/gen_modules/ot.emd.html)
   - Used to compute transportation plans in all Wasserstein calculations
   - Provides more stability than entropy-regularized methods like Sinkhorn

3. **`ot.emd2()`**: Computes the Earth Mover's Distance value only
   - Documentation: [POT EMD2 Function](https://pythonot.github.io/gen_modules/ot.emd2.html)
   - Used in the `calculate_wasserstein_continuous()` method

For bottleneck distances, the application uses a combination of POT for distance computation and a custom worst-cost matching algorithm based on the Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`).

## Getting Support

If you find discrepancies between your calculations and the application:

1. Check the "Distance Matrices" tab to see the raw distance values used
2. Verify that you're using the same mathematical formulation 
3. Examine the exported documents for more detailed information
4. If issues persist, check the calculation methods in `controllers/pot_distance_calculator.py`
5. Refer to the [POT documentation](https://pythonot.github.io/) for details on the underlying algorithms