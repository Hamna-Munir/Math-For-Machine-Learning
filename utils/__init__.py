"""
Utils Package
=============

This package contains helper modules for:

- 📊 Data Visualization
- ➗ Mathematical Operations

Modules:
--------
- visualization.py  → Plotting and graph utilities
- math_helpers.py   → Mathematical helper functions

Author: Hamna Munir
Repository: Math-for-Machine-Learning
"""

# Import commonly used functions for easy access
from .visualization import (
    plot_line,
    plot_scatter,
    plot_histogram
)

from .math_helpers import (
    sigmoid,
    mean_squared_error,
    normalize
)

# Define what gets imported with *
__all__ = [
    "plot_line",
    "plot_scatter",
    "plot_histogram",
    "sigmoid",
    "mean_squared_error",
    "normalize"
]
