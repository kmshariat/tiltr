"""
tepcat_lambda_psi
==================
A small package for fetching the TEPCat "all planet info" table and
plotting spin-orbit obliquity diagnostics (Lambda, Psi) against
semi-major axis, eccentricity, and stellar effective temperature.

Example
-------
>>> from tepcat_lambda_psi import TEPCatPlotter
>>> tp = TEPCatPlotter()                    # downloads/refreshes the CSV
>>> tp.plot_lambda_vs_teff(highlight="TOI-1842")
>>> tp.plot_grid(highlight="TOI-1842", save_path="obliquity_grid.png")
"""

from .fetch import fetch_tepcat_csv, DEFAULT_URL
from .plotter import TEPCatPlotter

__all__ = ["TEPCatPlotter", "fetch_tepcat_csv", "DEFAULT_URL"]
__version__ = "0.1.0"
