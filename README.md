# tiltr
TEPCat Obliquity Plotter

## Installation
```
pip install tiltr
```

## Usage
Plots the sky-projected spin-orbit angle and the true spin-orbit angle of an exoplanet. Data taken from TEPCat. Currently, the package supports plotting with respect to the effective temperature (K), the semi-major axis (AU), and the period of the planet (days). The plot for the effective temperature takes an additional, optional argument `kraft` to plot the Kraft Break. 

```
from tiltr import TEPCatPlotter

tp = TEPCatPlotter() 

# projected spin-orbit angle
tp.plot_lambda_vs_teff(highlight="TOI-1842", save_path='lambda_vs_teff.pdf', show=True, kraft=6050)  
tp.plot_lambda_vs_a(log_scale=True, highlight="TOI-1842", save_path='lambda_vs_a.png', show=True)
tp.plot_lambda_vs_period(highlight="TOI-1842", save_path='lambda_vs_period.svg', show=True)         

# true spin-orbit angle
tp.plot_psi_vs_teff(highlight="TOI-1842", save_path='psi_vs_teff.pdf', kraft=6050, show=True)
tp.plot_psi_vs_a(log_scale=True, highlight="TOI-1842", save_path='psi_vs_a.png', show=True)
tp.plot_psi_vs_period(highlight="TOI-1842", save_path='psi_vs_period.jpg', show=True)

# all combined
tp.plot_grid(log_scale_a=True, highlight="TOI-1842", save_path="obliquity_grid.pdf")
```

## Reference

- Southworth, J. (2011). Homogeneous studies of transiting extrasolar planets–IV. Thirty systems with space-based light curves. Monthly Notices of the Royal Astronomical Society, 417(3), 2166-2196.
