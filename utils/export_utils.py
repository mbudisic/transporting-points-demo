import pandas as pd
import numpy as np
import json
import io
import base64
from typing import Dict, Any, List, Tuple, Optional
import datetime
import plotly.graph_objects as go
from models.distribution import Distribution
from controllers.pot_distance_calculator import POTDistanceCalculator
from controllers.app_state import AppState

def generate_distribution_data(dist_a: Distribution, dist_b: Distribution, selected_transport: str = "hide") -> Dict[str, Any]:
    """
    Generate a dictionary with distribution data for export, focusing on the selected transport plan.
    
    Args:
        dist_a: Distribution A
        dist_b: Distribution B
        selected_transport: Currently selected transport plan ("bottleneck_spatial", "wasserstein_spatial", 
                           "bottleneck_height", "wasserstein_height", or "hide")
        
    Returns:
        Dictionary with focused distribution data
    """
    # Get basic distribution data
    data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "distribution_a": {
            "name": dist_a.name,
            "color": dist_a.color,
            "blobs": []
        },
        "distribution_b": {
            "name": dist_b.name,
            "color": dist_b.color,
            "blobs": []
        },
        "selected_transport_plan": selected_transport,
        "distance_metrics": {},
        "transport_plans": {},
        "distance_matrices": {}
    }
    
    # Add blob data
    for blob in dist_a.blobs:
        data["distribution_a"]["blobs"].append({
            "id": blob.id,
            "x": blob.x,
            "y": blob.y,
            "variance": blob.variance,
            "height": blob.height
        })
    
    for blob in dist_b.blobs:
        data["distribution_b"]["blobs"].append({
            "id": blob.id,
            "x": blob.x,
            "y": blob.y,
            "variance": blob.variance,
            "height": blob.height
        })
    
    # Calculate and add only the selected distance metric if both distributions have blobs
    if not dist_a.is_empty and not dist_b.is_empty and selected_transport != "hide":
        # Calculate only the selected metric and plan
        if selected_transport == "bottleneck_spatial":
            distance_value, matching_pairs = POTDistanceCalculator.calculate_bottleneck(dist_a, dist_b)
            plan_explanation = POTDistanceCalculator.explain_matching(dist_a, dist_b, matching_pairs)
            distance_matrix, idx_a, idx_b = POTDistanceCalculator.get_distance_matrix(dist_a, dist_b, 'spatial')
            metric_name = "Spatial Bottleneck Distance"
            
        elif selected_transport == "wasserstein_spatial":
            distance_value, transport_pairs = POTDistanceCalculator.calculate_wasserstein_plan(dist_a, dist_b)
            matching_pairs = [(a, b) for a, b, _ in transport_pairs]
            weights = [w for _, _, w in transport_pairs]
            plan_explanation = POTDistanceCalculator.explain_matching(dist_a, dist_b, matching_pairs, weights)
            distance_matrix, idx_a, idx_b = POTDistanceCalculator.get_distance_matrix(dist_a, dist_b, 'spatial')
            metric_name = "Spatial Wasserstein Distance"
            
        elif selected_transport == "bottleneck_height":
            distance_value, matching_pairs = POTDistanceCalculator.calculate_height_bottleneck_plan(dist_a, dist_b)
            plan_explanation = POTDistanceCalculator.explain_matching(dist_a, dist_b, matching_pairs)
            distance_matrix, idx_a, idx_b = POTDistanceCalculator.get_distance_matrix(dist_a, dist_b, 'height')
            metric_name = "Height-Based Bottleneck Distance"
            
        elif selected_transport == "wasserstein_height":
            distance_value, transport_pairs = POTDistanceCalculator.calculate_height_wasserstein_plan(dist_a, dist_b)
            matching_pairs = [(a, b) for a, b, _ in transport_pairs]
            weights = [w for _, _, w in transport_pairs]
            plan_explanation = POTDistanceCalculator.explain_matching(dist_a, dist_b, matching_pairs, weights)
            distance_matrix, idx_a, idx_b = POTDistanceCalculator.get_distance_matrix(dist_a, dist_b, 'height')
            metric_name = "Height-Based Wasserstein Distance"
            
        else:
            # Default case - shouldn't happen with proper selections
            distance_value = 0
            plan_explanation = []
            distance_matrix = np.array([])
            idx_a = []
            idx_b = []
            metric_name = "Unknown Distance"
            
        # Add only the selected metric to the output
        data["distance_metrics"] = {
            "name": metric_name,
            "value": float(distance_value)
        }
        
        # Add only the selected transport plan
        data["transport_plans"] = plan_explanation
        
        # Add only the relevant distance matrix
        data["distance_matrices"] = {
            "matrix": distance_matrix.tolist() if len(distance_matrix) > 0 else [],
            "blob_indices_a": idx_a,
            "blob_indices_b": idx_b,
            "metric_type": "spatial" if selected_transport in ["bottleneck_spatial", "wasserstein_spatial"] else "height"
        }
    
    return data

def generate_html_report(dist_a: Distribution, dist_b: Distribution, 
                         fig: Optional[go.Figure] = None) -> str:
    """
    Generate an HTML report with distribution data and visualizations.
    Includes only the currently selected transport plan, its distance matrices, and optimal value.
    
    Args:
        dist_a: Distribution A
        dist_b: Distribution B
        fig: Optional plotly figure to include
        
    Returns:
        HTML string with the report
    """
    # Determine which transport plan is currently selected
    current_plan = "hide"
    if AppState.is_showing_bottleneck():
        current_plan = "bottleneck_spatial"
    elif AppState.is_showing_wasserstein():
        current_plan = "wasserstein_spatial"
    elif AppState.is_showing_height_bottleneck():
        current_plan = "bottleneck_height"
    elif AppState.is_showing_height_wasserstein():
        current_plan = "wasserstein_height"
    
    # Get distribution data with focus on selected transport plan
    data = generate_distribution_data(dist_a, dist_b, current_plan)
    
    # Add current plan for use in formatting
    data["current_plan"] = current_plan
    
    # Generate HTML content
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Distribution Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric {{ font-weight: bold; color: #009E73; }}
            .container {{ display: flex; justify-content: space-between; }}
            .column {{ flex: 1; margin: 0 10px; }}
            .separator {{ border-top: 1px solid #ccc; margin: 20px 0; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>Distribution Analysis Report</h1>
        <p>Generated on: {data['timestamp']}</p>
        
        <div class="container">
            <div class="column">
                <h2>Distribution A</h2>
                <p>Name: {data['distribution_a']['name']}, Color: {data['distribution_a']['color']}</p>
                <h3>Blobs:</h3>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>X</th>
                        <th>Y</th>
                        <th>Variance</th>
                        <th>Height</th>
                    </tr>
    """
    
    # Add Distribution A blobs
    for blob in data['distribution_a']['blobs']:
        html += f"""
                    <tr>
                        <td>{blob['id']}</td>
                        <td>{blob['x']:.2f}</td>
                        <td>{blob['y']:.2f}</td>
                        <td>{blob['variance']:.2f}</td>
                        <td>{blob['height']:.2f}</td>
                    </tr>
        """
    
    html += f"""
                </table>
            </div>
            
            <div class="column">
                <h2>Distribution B</h2>
                <p>Name: {data['distribution_b']['name']}, Color: {data['distribution_b']['color']}</p>
                <h3>Blobs:</h3>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>X</th>
                        <th>Y</th>
                        <th>Variance</th>
                        <th>Height</th>
                    </tr>
    """
    
    # Add Distribution B blobs
    for blob in data['distribution_b']['blobs']:
        html += f"""
                    <tr>
                        <td>{blob['id']}</td>
                        <td>{blob['x']:.2f}</td>
                        <td>{blob['y']:.2f}</td>
                        <td>{blob['variance']:.2f}</td>
                        <td>{blob['height']:.2f}</td>
                    </tr>
        """
    
    html += """
                </table>
            </div>
        </div>
        
        <div class="separator"></div>
    """
    
    # Add selected distance metric if available
    if "distance_metrics" in data and data["distance_metrics"]:
        metric = data["distance_metrics"]
        html += f"""
        <h2>Selected Distance Metric</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>{metric['name']}</td>
                <td class="metric">{metric['value']:.6f}</td>
            </tr>
        </table>
        
        <div class="separator"></div>
        """
    
    # Add distance matrices if available
    if "distance_matrices" in data and "matrix" in data["distance_matrices"] and len(data["distance_matrices"]["matrix"]) > 0:
        matrix = data["distance_matrices"]["matrix"]
        indices_a = data["distance_matrices"]["blob_indices_a"]
        indices_b = data["distance_matrices"]["blob_indices_b"]
        matrix_type = data["distance_matrices"]["metric_type"]
        
        # Choose the appropriate title based on matrix type
        matrix_title = "Spatial Distance Matrix" if matrix_type == "spatial" else "Height-Based Distance Matrix"
        
        html += f"""
        <h2>Distance Matrix</h2>
        <h3>{matrix_title}</h3>
        <table>
            <tr>
                <th></th>
        """
        
        # Add column headers
        for j, idx_b in enumerate(indices_b):
            blob_b = next((b for b in data['distribution_b']['blobs'] if b['id'] == idx_b), None)
            if blob_b:
                if matrix_type == "spatial":
                    html += f"<th>B{idx_b} [{blob_b['x']:.2f}, {blob_b['y']:.2f}]</th>"
                else:
                    html += f"<th>B{idx_b} [{blob_b['height']:.2f}]</th>"
            else:
                html += f"<th>B{idx_b}</th>"
        
        html += "</tr>"
        
        # Add matrix rows
        for i, idx_a in enumerate(indices_a):
            blob_a = next((b for b in data['distribution_a']['blobs'] if b['id'] == idx_a), None)
            if blob_a:
                if matrix_type == "spatial":
                    html += f"<tr><th>A{idx_a} [{blob_a['x']:.2f}, {blob_a['y']:.2f}]</th>"
                else:
                    html += f"<tr><th>A{idx_a} [{blob_a['height']:.2f}]</th>"
            else:
                html += f"<tr><th>A{idx_a}</th>"
            
            for j in range(len(indices_b)):
                html += f"<td>{matrix[i][j]:.4f}</td>"
            
            html += "</tr>"
        
        html += """
        </table>
        
        <div class="separator"></div>
        """
    
    # Add transport plan if available
    if "transport_plans" in data and data["transport_plans"]:
        plan_data = data["transport_plans"]
        transport_type = "Transport Plan"
        if current_plan.startswith("bottleneck"):
            transport_type = "Bottleneck Matching"
        elif current_plan.startswith("wasserstein"):
            transport_type = "Wasserstein Transport Plan"
            
        html += f"""
        <h2>Selected {transport_type}</h2>
        <table>
            <tr>
                <th>Blob A</th>
                <th>Blob B</th>
                <th>Spatial Distance</th>
                <th>Height Distance</th>
                <th>Weight</th>
            </tr>
        """
        
        for match in plan_data:
            html += f"""
            <tr>
                <td>A{match['blob_a_id']} [{match['blob_a_position'][0]:.2f}, {match['blob_a_position'][1]:.2f}] (h={match['blob_a_height']:.2f})</td>
                <td>B{match['blob_b_id']} [{match['blob_b_position'][0]:.2f}, {match['blob_b_position'][1]:.2f}] (h={match['blob_b_height']:.2f})</td>
                <td>{match['spatial_distance']:.4f}</td>
                <td>{match['height_distance']:.4f}</td>
                <td>{match['weight']:.4f}</td>
            </tr>
            """
        
        html += """
        </table>
        """
        
        html += """
        <div class="separator"></div>
        """
    
    # Add visualization if provided
    if fig is not None:
        try:
            fig_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
            html += f"""
            <h2>Visualization</h2>
            <div id="plotly-figure">
                {fig_html}
            </div>
            
            <div class="separator"></div>
            """
        except Exception as e:
            html += f"""
            <h2>Visualization</h2>
            <p>Error generating visualization: {str(e)}</p>
            
            <div class="separator"></div>
            """
    
    # Add raw data export
    json_data = json.dumps(data, indent=2)
    html += f"""
        <h2>Raw Data Export</h2>
        <p>You can copy the JSON data below for programmatic analysis:</p>
        <pre>{json_data}</pre>
    </body>
    </html>
    """
    
    return html

def get_html_download_link(html_content: str, filename: str = "distribution_report.html") -> str:
    """
    Generate a download link for HTML content.
    
    Args:
        html_content: HTML content to download
        filename: Name of the file to download
        
    Returns:
        HTML string with download link
    """
    # Encode HTML content as base64
    b64 = base64.b64encode(html_content.encode()).decode()
    
    # Generate download link
    href = f'data:text/html;base64,{b64}'
    download_link = f'<a href="{href}" download="{filename}" target="_blank">Download HTML Report</a>'
    
    return download_link

def export_to_formats(dist_a: Distribution, dist_b: Distribution, 
                     fig: Optional[go.Figure] = None) -> Dict[str, str]:
    """
    Export distribution data to multiple formats and return download links.
    
    Args:
        dist_a: Distribution A
        dist_b: Distribution B
        fig: Optional plotly figure to include
        
    Returns:
        Dictionary with download links for different formats
    """
    # Determine which transport plan is currently selected
    current_plan = "hide"
    if AppState.is_showing_bottleneck():
        current_plan = "bottleneck_spatial"
    elif AppState.is_showing_wasserstein():
        current_plan = "wasserstein_spatial"
    elif AppState.is_showing_height_bottleneck():
        current_plan = "bottleneck_height"
    elif AppState.is_showing_height_wasserstein():
        current_plan = "wasserstein_height"
    
    # Generate data with focus on selected transport plan
    data = generate_distribution_data(dist_a, dist_b, current_plan)
    
    # Generate HTML report
    html_content = generate_html_report(dist_a, dist_b, fig)
    html_link = get_html_download_link(html_content)
    
    # Generate JSON data
    json_str = json.dumps(data, indent=2)
    b64_json = base64.b64encode(json_str.encode()).decode()
    json_href = f'data:application/json;base64,{b64_json}'
    json_link = f'<a href="{json_href}" download="distribution_data.json" target="_blank">Download JSON Data</a>'
    
    # Generate CSV for blobs
    blob_data_a = [blob for blob in data["distribution_a"]["blobs"]]
    for blob in blob_data_a:
        blob["distribution"] = "A"
    
    blob_data_b = [blob for blob in data["distribution_b"]["blobs"]]
    for blob in blob_data_b:
        blob["distribution"] = "B"
    
    all_blobs = blob_data_a + blob_data_b
    if all_blobs:
        df = pd.DataFrame(all_blobs)
        csv_string = df.to_csv(index=False)
        b64_csv = base64.b64encode(csv_string.encode()).decode()
        csv_href = f'data:text/csv;base64,{b64_csv}'
        csv_link = f'<a href="{csv_href}" download="distribution_blobs.csv" target="_blank">Download CSV Data</a>'
    else:
        csv_link = "No data to export as CSV"
    
    return {
        "html": html_link,
        "json": json_link,
        "csv": csv_link
    }