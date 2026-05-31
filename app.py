import os
import sys

# Ensure the 'src' directory is in the Python path so that imports resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Run the Streamlit app
import streamlit_app
