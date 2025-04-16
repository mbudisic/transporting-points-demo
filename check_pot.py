import sys

try:
    import ot
    print(f"POT version: {ot.__version__}")
    print("POT is installed correctly")
except ImportError:
    print("POT is not installed")
    sys.exit(1)