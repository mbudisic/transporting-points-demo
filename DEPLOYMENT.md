# Deployment Guide

This guide will help you deploy the Distribution Distance Visualization application on your local machine.

## Prerequisites

- Python 3.11 or higher (required by the project)
- Git (to clone the repository)

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

Replace `<repository-url>` with the URL of the repository, and `<repository-directory>` with the name of the directory that was created.

## Step 2: Install uv (Faster Package Manager)

`uv` is a fast Python package installer that we recommend for managing dependencies.

### For Windows:

```bash
curl -sSf https://astral.sh/uv/install.ps1 | powershell
```

### For macOS/Linux:

```bash
curl -sSf https://astral.sh/uv/install.sh | bash
```

## Step 3: Create a Virtual Environment and Install Dependencies using pyproject.toml

The project uses a `pyproject.toml` file to define its dependencies, which makes installation more reliable.

```bash
# Create a virtual environment
uv venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv pip sync
```

This command will analyze the `pyproject.toml` file and install all the required dependencies with their correct versions.

## Step 4: Run the Application

```bash
streamlit run app_mvc.py
```

This will start the application and open it in your default web browser. If it doesn't open automatically, you can access it at [http://localhost:8501](http://localhost:8501).

## Updating Dependencies

If you need to update the application's dependencies in the future:

### Method 1: Using pyproject.toml

If the project's `pyproject.toml` file has been updated:

```bash
# Make sure your virtual environment is activated
uv pip sync
```

### Method 2: Manual Update

If you need to update specific packages:

```bash
# Make sure your virtual environment is activated
uv pip install --upgrade <package-name>

# Example:
uv pip install --upgrade streamlit
```

## Troubleshooting

### Common Issues

1. **Error: Port 8501 is already in use**
   - Another application is using port 8501
   - Solution: Close other Streamlit applications or specify a different port:
     ```bash
     streamlit run app_mvc.py --server.port 8502
     ```

2. **Import Error: No module named 'streamlit'**
   - The dependencies were not installed correctly
   - Solution: Make sure you've activated the virtual environment and installed the dependencies

3. **Visual display issues**
   - If the visualization doesn't render correctly, try using a different web browser (Chrome or Firefox recommended)

### Getting Help

If you encounter any issues not covered in this guide, please refer to:

- The [README.md](README.md) file for general information
- The [DEVELOPER.md](DEVELOPER.md) file for more technical details
- Check for open issues in the repository

## Advanced Configuration

To customize the server settings, you can create a `.streamlit/config.toml` file with the following content:

```toml
[server]
port = 8501
address = "localhost"
```

You can change the port and address as needed.