# Mathematical Developer Guide

This guide outlines the process for extending the Distribution Distance Visualization application with new distance metrics or transportation plan computations.

## Architecture Overview

The application follows an MVC (Model-View-Controller) architecture:

- **Models** (`models/`): Define the data structures (blobs, distributions)
  - `models/blob.py`: The `Blob` class defines properties like position, variance, and height
  - `models/distribution.py`: The `Distribution` class manages collections of blobs
- **Views** (`views/`): Handle UI components and visualizations
  - `views/visualization.py`: Contains visualization services for rendering distributions and transport plans
  - `views/ui_components.py`: Defines UI components like sliders, buttons, and tabs
- **Controllers** (`controllers/`): Implement business logic and calculations
  - `controllers/pot_distance_calculator.py`: Contains all distance metric implementations using POT
  - `controllers/app_state.py`: Manages application state across the Streamlit app
  - `controllers/distribution_controller.py`: Handles distribution operations and export functionality
- **Utilities** (`utils/`): Provide helper functions
  - `utils/export_utils.py`: Contains the focused export functionality for reporting results

Mathematical operations are primarily located in the `controllers/pot_distance_calculator.py` file, which contains the `POTDistanceCalculator` class with the following key methods:

- `calculate_wasserstein_continuous()`: Calculates Wasserstein distance using POT's continuous formulation
- `_create_cost_matrix_spatial()`: Creates distance matrices for spatial metrics using POT's distance functions
- `_create_cost_matrix_heights()`: Creates distance matrices for height-based metrics
- `calculate_wasserstein_plan()`: Implementation of Wasserstein distance using POT's Earth Mover's Distance (EMD)
- `calculate_bottleneck()`: Implementation of bottleneck distance using POT for distance matrices and custom worst-cost matching
- `calculate_height_wasserstein_plan()`: Height-based Wasserstein implementation using POT's EMD
- `calculate_height_bottleneck_plan()`: Height-based bottleneck implementation using POT and custom worst-cost matching
- Helper methods for mathematical validation and explanation

## Python Optimal Transport (POT) Package

The application uses the [Python Optimal Transport (POT)](https://pythonot.github.io/) package for computing distance matrices and solving optimal transport problems. Key functions include:

1. **`ot.dist()`**: Computes the distance matrix between two sets of points
2. **`ot.emd()`**: Solves the Earth Mover's Distance (EMD) problem exactly
3. **`ot.emd2()`**: Computes the Earth Mover's Distance value only

For bottleneck distances, the application uses a combination of POT for distance computation and a custom worst-cost matching algorithm based on the Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`).

## Adding a New Distance Metric

### Step 1: Implement the Distance Calculation

1. Open `controllers/pot_distance_calculator.py`
2. Add a new method to the `POTDistanceCalculator` class that leverages the POT package:

```python
@staticmethod
def calculate_new_distance(dist_a: Distribution, dist_b: Distribution) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Calculate your new distance between two distributions using POT package functions.
    
    Args:
        dist_a: First distribution
        dist_b: Second distribution
        
    Returns:
        A tuple containing (distance_value, matching_pairs)
        where matching_pairs is a list of tuples (idx_a, idx_b, weight) for the transportation plan
    """
    # Handle empty distributions
    if not dist_a.blobs or not dist_b.blobs:
        return 0.0, []
        
    # Extract positions and weights from distributions
    positions_a = np.array([blob.center for blob in dist_a.blobs])
    weights_a = np.array([blob.height for blob in dist_a.blobs])
    
    positions_b = np.array([blob.center for blob in dist_b.blobs])
    weights_b = np.array([blob.height for blob in dist_b.blobs])
    
    # Example: Use POT distance function to create cost matrix
    cost_matrix = ot.dist(positions_a, positions_b)
    
    # Example: Use POT solvers for the transportation problem
    # Assuming all weights are positive for this example
    weights_a_norm = np.abs(weights_a) / np.sum(np.abs(weights_a))
    weights_b_norm = np.abs(weights_b) / np.sum(np.abs(weights_b))
    
    # Calculate transport plan using EMD
    transport_plan = ot.emd(weights_a_norm, weights_b_norm, cost_matrix)
    
    # Calculate distance value
    distance_value = np.sum(transport_plan * cost_matrix)
    
    # Convert to matching pairs format
    matching_pairs = []
    for i in range(len(dist_a.blobs)):
        for j in range(len(dist_b.blobs)):
            if transport_plan[i, j] > 1e-10:  # Threshold for numerical stability
                matching_pairs.append((i, j, float(transport_plan[i, j])))
    
    return distance_value, matching_pairs
```

### Step 2: Update the Application State

1. Open `controllers/app_state.py`
2. Add new methods for storing and retrieving your matching pairs:

```python
@staticmethod
def store_new_matching(matching: List[Tuple[int, int, float]]):
    """Store new matching results in session state"""
    if 'new_matching' not in st.session_state:
        st.session_state.new_matching = []
    st.session_state.new_matching = matching

@staticmethod
def get_new_matching():
    """Get new matching results from session state"""
    if 'new_matching' not in st.session_state:
        st.session_state.new_matching = []
    return st.session_state.new_matching

@staticmethod
def is_showing_new_matching() -> bool:
    """Check if new matching is being shown"""
    return st.session_state.get('transport_visualization', '') == 'new_matching'
```

3. Update the `set_transport_visualization` method to handle your new visualization mode

### Step 3: Update the UI Components

1. Open `views/ui_components.py`
2. Add your new option to the transport options dictionary in `render_main_content`:

```python
transport_options = {
    "hide": "Hide All Transport",
    "bottleneck_spatial": "Spatial Bottleneck",
    "wasserstein_spatial": "Spatial Wasserstein", 
    "bottleneck_height": "Height-Based Bottleneck",
    "wasserstein_height": "Height-Based Wasserstein",
    "new_distance": "Your New Distance"  # Add this line
}
```

3. Update the transport visualization logic in the same function:

```python
# Apply the selection
if selected_transport == "bottleneck_spatial":
    AppState.set_transport_visualization("bottleneck_spatial")
elif selected_transport == "wasserstein_spatial":
    AppState.set_transport_visualization("wasserstein_spatial")
elif selected_transport == "bottleneck_height":
    AppState.set_transport_visualization("bottleneck_height")
elif selected_transport == "wasserstein_height":
    AppState.set_transport_visualization("wasserstein_height")
elif selected_transport == "new_distance":  # Add this block
    AppState.set_transport_visualization("new_distance")
else:
    AppState.set_transport_visualization("hide")
```

4. Update the explanation text:

```python
if selected_transport == "new_distance":
    st.info("Your New Distance: Brief explanation of what this distance measures.")
```

5. Update the `render_metrics` method to display your new distance metric:

```python
elif current_plan == "new_distance":
    st.metric("Your New Distance", f"{new_distance_value:.4f}")
```

### Step 4: Update the Visualization

1. Open `views/visualization.py`
2. If your distance requires custom visualization, update the `add_transport_plan_to_figure` method:

```python
# Add your custom visualization code as needed
```

### Step 5: Update the Distance Matrices Display

1. In `views/ui_components.py`, update the `render_distance_matrices` method to handle your new distance:

```python
elif current_transport_mode == "new_distance":
    highlight_pairs = [(a, b) for a, b, _ in AppState.get_new_matching()]
```

## Full Example: Adding a Normalized Wasserstein Distance

Here's a complete example of adding a normalized Wasserstein distance that scales the regular Wasserstein distance by the sum of distribution weights.

### Step 1: Implement the Distance Calculation

```python
@staticmethod
def calculate_normalized_wasserstein(dist_a: Distribution, dist_b: Distribution) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Calculate Wasserstein distance normalized by the total distribution weight.
    Uses POT package for transportation plans.
    
    Args:
        dist_a: First distribution
        dist_b: Second distribution
        
    Returns:
        A tuple containing (normalized_distance, matching_pairs)
    """
    # First calculate regular Wasserstein distance and plan using POT
    wasserstein_val, matching_pairs = POTDistanceCalculator.calculate_wasserstein_plan(dist_a, dist_b)
    
    # Calculate total weights
    total_weight_a = sum(abs(b.height) for b in dist_a.blobs)
    total_weight_b = sum(abs(b.height) for b in dist_b.blobs)
    normalization_factor = max(total_weight_a, total_weight_b)
    
    # Normalize the distance if the factor is non-zero
    if normalization_factor > 0:
        normalized_distance = wasserstein_val / normalization_factor
    else:
        normalized_distance = 0
        
    return normalized_distance, matching_pairs
```

### Step 2: Update Application State

```python
@staticmethod
def store_normalized_wasserstein_pairs(pairs: List[Tuple[int, int, float]]):
    """Store normalized Wasserstein plan results"""
    if 'normalized_wasserstein_pairs' not in st.session_state:
        st.session_state.normalized_wasserstein_pairs = []
    st.session_state.normalized_wasserstein_pairs = pairs

@staticmethod
def get_normalized_wasserstein_pairs():
    """Get normalized Wasserstein plan results"""
    if 'normalized_wasserstein_pairs' not in st.session_state:
        st.session_state.normalized_wasserstein_pairs = []
    return st.session_state.normalized_wasserstein_pairs

@staticmethod
def is_showing_normalized_wasserstein() -> bool:
    """Check if normalized Wasserstein transport is being shown"""
    return st.session_state.get('transport_visualization', '') == 'normalized_wasserstein'
```

### Step 3: Update the UI Components

```python
transport_options = {
    # ... existing options
    "normalized_wasserstein": "Normalized Wasserstein"
}

# Add to the transport selection logic
elif selected_transport == "normalized_wasserstein":
    AppState.set_transport_visualization("normalized_wasserstein")
    
# Add to explanation text
elif selected_transport == "normalized_wasserstein":
    st.info("Normalized Wasserstein: Wasserstein distance divided by total distribution weight.")
    
# Add to metrics display
elif current_plan == "normalized_wasserstein":
    normalized_wasserstein, normalized_pairs = calculator.calculate_normalized_wasserstein(distribution_a, distribution_b)
    AppState.store_normalized_wasserstein_pairs(normalized_pairs)
    st.metric("Normalized Wasserstein Distance", f"{normalized_wasserstein:.4f}")
```

## Testing Your Implementation

To test your new distance metric:

1. Add some blobs to both distributions A and B
2. Select your new distance from the transport visualization dropdown
3. Verify that:
   - The distance value is displayed correctly
   - The transportation plan is visualized correctly
   - The distance matrices show the correct values and highlighted cells

## Extending the Export Functionality

The export functionality is designed to focus only on the currently selected transport plan. If you add a new distance metric, you'll need to update the export functionality to include it:

1. Open `utils/export_utils.py`
2. Update the `generate_distribution_data` function (lines 12-121) to handle your new transport plan:

```python
elif selected_transport == "your_new_distance":
    distance_value, matching_pairs = POTDistanceCalculator.calculate_your_new_distance(dist_a, dist_b)
    plan_explanation = POTDistanceCalculator.explain_matching(dist_a, dist_b, matching_pairs)
    distance_matrix, idx_a, idx_b = POTDistanceCalculator.get_distance_matrix(dist_a, dist_b, 'spatial')
    metric_name = "Your New Distance"
```

3. Update the `export_to_formats` function (lines 429-489) to detect your new transport mode:

```python
elif AppState.is_showing_your_new_distance():
    current_plan = "your_new_distance"
```

4. Update the `generate_html_report` function (lines 123-385) to include your new transport plan:

```python
elif AppState.is_showing_your_new_distance():
    current_plan = "your_new_distance"
```

## Best Practices

1. **Documentation**: Include clear documentation for your new method, explaining the mathematical formulation
2. **Edge Cases**: Handle edge cases (empty distributions, single-point distributions, etc.)
3. **Efficiency**: Optimize your algorithm for performance, especially for larger distributions
4. **Visual Consistency**: Follow the existing visual language for displaying transportation plans
5. **Export Consistency**: Ensure your new distance metric integrates with the focused export functionality

## Additional Resources

- [Python Optimal Transport (POT) Documentation](https://pythonot.github.io/): The official documentation for the POT package
  - [POT API Reference](https://pythonot.github.io/gen_modules/ot.html): Detailed API documentation for all POT functions
  - [POT Tutorials](https://pythonot.github.io/auto_examples/index.html): Tutorials and examples using the POT package
  - [POT Distance Functions](https://pythonot.github.io/gen_modules/ot.dist.html): Documentation for distance matrix computation
  - [POT EMD Solver](https://pythonot.github.io/gen_modules/ot.emd.html): Documentation for the Earth Mover's Distance solver

- [Optimal Transport Theory](https://optimaltransport.github.io/): General resources on optimal transport theory
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html): Documentation for optimization functions (used for the Hungarian algorithm)
- [NetworkX Documentation](https://networkx.org/documentation/stable/): Documentation for graph algorithms (useful for custom implementations)