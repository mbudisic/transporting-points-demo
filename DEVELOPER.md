# Distribution Distance Visualizer - Developer Documentation

## Overview

The Distribution Distance Visualizer is an interactive educational web application that helps users understand the mathematical concepts of Wasserstein and bottleneck distances through intuitive visualization of 2D probability distributions. The application allows users to create, modify, and compare two distributions by manipulating Gaussian blobs, and visualize different transportation plans between them.

This application follows the Model-View-Controller (MVC) architectural pattern for clear separation of concerns and maintainability.

## Project Structure

```
├── models/                  # Data models
│   ├── blob.py              # Representation of a Gaussian blob
│   └── distribution.py      # Collection of blobs forming a distribution
├── controllers/             # Application logic
│   ├── app_state.py         # Manages application state
│   ├── pot_distance_calculator.py # Calculates metrics and transport plans using POT package
│   ├── distribution_controller.py # Manipulates distributions
│   └── event_handler.py     # Handles UI events
├── views/                   # User interface components
│   ├── ui_components.py     # Streamlit UI components
│   ├── visualization.py     # Plotting and visualization
│   └── graph_visualization.py # Additional visualization for complex graph structures
├── utils/                   # Utilities and helpers
│   └── export_utils.py      # Utilities for exporting data
├── app.py                   # Main application entry point
```

## Core Components

### Models

#### Blob (models/blob.py)
- Represents a single Gaussian blob with properties:
  - Position (x, y)
  - Variance (spread)
  - Height (magnitude, can be positive or negative)
- Implements observer pattern for state change notifications
- Methods for calculating Gaussian values at specific points

#### Distribution (models/distribution.py)
- Collection of Blob objects
- Manages blob creation, deletion, and updates
- Maintains list of blobs with unique IDs
- Provides methods for distribution evaluation on a grid

### Controllers

#### AppState (controllers/app_state.py)
- Manages global application state using Streamlit's session_state
- Stores and retrieves application settings
- Tracks which distribution is active
- Controls transportation plan visualization settings
- Manages contour plot visibility and opacity settings
- Tracks selected elements for user interaction

#### POTDistanceCalculator (controllers/pot_distance_calculator.py)
- Calculates Wasserstein and bottleneck distances between distributions using the Python Optimal Transport (POT) package
- Implements both spatial and height-based distance metrics
- Computes transportation plans for visualization
- Provides helper methods for mathematical validation

#### DistributionController (controllers/distribution_controller.py)
- High-level operations for manipulating distributions
- Adding, removing, and updating blobs
- Importing and exporting distributions to CSV

#### EventHandler (controllers/event_handler.py)
- Handles user interactions with the plot
- Processes click events for blob selection
- Manages drag events for blob repositioning

### Views

#### UIComponents (views/ui_components.py)
- Renders Streamlit UI components
- Creates sidebars with distribution controls
- Displays transportation plan selection dropdown
- Shows distance metrics
- Provides import/export interface

#### VisualizationService (views/visualization.py)
- Creates interactive plots using Plotly
- Visualizes blobs with varying size, color, and style
- Renders transportation plans between distributions
- Configures plot layout and appearance

## Key Implementation Details

### Blob Representation
- Gaussian blobs are represented by their center position, variance, and height
- Height can be positive or negative, affecting the visualization style
- Blob visualization scales with the absolute height value

### Distance Metrics
1. **Spatial Metrics**
   - **Bottleneck Distance**: Maximum minimum distance in the optimal matching
   - **Wasserstein Distance**: Minimum cost of transforming one distribution into another based on position

2. **Height-Based Metrics**
   - **Height Bottleneck**: Maximum difference between sorted heights
   - **Height Wasserstein**: Minimum cost of transforming heights, ignoring spatial positions

### Transportation Plans
- **Spatial Bottleneck**: Optimal matching between blobs based on position
- **Spatial Wasserstein**: Optimal transportation plan with weighted connections
- **Height-Based Bottleneck**: Matching of blobs based only on heights
- **Height-Based Wasserstein**: Transportation plan based only on heights

### User Interface
- **Left Sidebar**: Controls for Distribution A
- **Right Sidebar**: Controls for Distribution B
- **Main Content**: Interactive plot and metrics display
- **Transport Selection**: Dropdown with description of each transport plan
- **Distance Value**: Real-time calculated distance for the selected plan

## State Management

The application state is managed using Streamlit's session_state with the following components:
- Active distribution (A or B)
- Show both flag
- Selected blob for each distribution
- Transportation plan selection
- Calculated transportation plans

## Extending the Application

### Adding a New Distance Metric
1. Add calculation method to `DistanceCalculator` class
2. Update the transport options in `UIComponents.render_main_content`
3. Add handling for the new metric in the appropriate sections

### Adding New Visualization Features
1. Extend the `VisualizationService.create_interactive_plot` method
2. Add parameters for the new visualization features
3. Update the UI to control these features

## Best Practices

1. **State Management**: Use `AppState` for all state changes
2. **Separation of Concerns**: Follow MVC pattern
3. **Consistent Styling**: Use the established color scheme (Teal for A, Orange for B)
4. **Performance**: Pre-calculate values when possible to avoid redundant calculations
5. **Error Handling**: Validate user inputs and handle edge cases
6. **Documentation**: Document complex algorithms and key functions

## Technical Dependencies

- **Streamlit**: Web application framework
- **Plotly**: Interactive plotting library
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing (optimizers and distance calculations)
- **Pandas**: Data manipulation and analysis

## Performance Considerations

- Limit the number of blobs to maintain responsiveness
- Optimize grid resolution for continuous distance calculations
- Consider caching expensive calculations using Streamlit's caching mechanism