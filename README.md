# tiltr
TEPCat Obliquity Plotter

## Installation
```
pip install tiltr
```

## Usage
Plots the sky-projected spin-orbit angle and the true spin-orbit angle of an exoplanet. Data taken from TEPCat. Currently, the package supports plotting with respect to the effective temperature (K), the semi-major axis (AU), and the period of the planet (days). The plot for the effective temperature takes an additional, optional argument `kraft` to plot the Kraft Break. 

```
import tiltr 

highlights = [
    {
        'lambda': -62.3,
        'lambda_err_1': -10,
        'lambda_err_2': 10,
        'Teff': 6200,
        'marker': '*',
        'color': 'red',
        'markersize': 15,
        'label': 'Everybody et al. (2024)'
    },
    {
        'lambda': 45.0,
        'lambda_err_1': -5,
        'lambda_err_2': 5,
        'Teff': 6200,
        'marker': 's',
        'color': 'blue',
        'markersize': 10,
        'label': 'Nobody et al. (2009)'
    }
]

tiltr.lambda_T(highlight=highlights, Kraft_Break=6260, save_as='lambda_teff.png')
tiltr.lambda_a(save_as='lambda_a.png')
tiltr.lambda_M(save_as='lambda_M.png')
tiltr.psi_T(save_as='psi_T.png')
tiltr.psi_a(save_as='psi_a.png')
tiltr.psi_M(save_as='psi_mass.png')
```

## Reference

- Southworth, J. (2011). Homogeneous studies of transiting extrasolar planets–IV. Thirty systems with space-based light curves. Monthly Notices of the Royal Astronomical Society, 417(3), 2166-2196.
