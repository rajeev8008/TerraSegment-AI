"""
TerraSegment AI - Entry Point
Run this file from the project root: python run.py
"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the app
from app.main import main

if __name__ == '__main__':
    main()
