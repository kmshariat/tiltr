"""
tiltr - Visualization package for exoplanet spin-orbit obliquity data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class Plotter:
    """
    Loads exoplanet obliquity data from the TEPCat catalogue and provides
    plotting methods for absolute lambda and psi against Teff, semi-major axis,
    and stellar mass.

    Parameters
    ----------
    data_source : str, optional
        URL or local file path to the CSV data. If None, uses the default
        TEPCat URL.
    """

    # Column indices based on the TEPCat allinfo-csv.csv format
    _COL_MAP = {
        'System': 0,
        'Type': 1,
        'RA': 2,
        'Dec': 3,
        'Vmag': 4,
        'Kmag': 5,
        'length': 6,
        'depth': 7,
        'T0': 8,
        'T0err': 9,
        'Period': 10,
        'Perioderr': 11,
        'Teff': 12,
        'Teff_err1': 13,
        'Teff_err2': 14,
        '[Fe/H]': 15,
        '[Fe/H]_erru': 16,
        '[Fe/H]_errd': 17,
        'M_A': 18,
        'M_A_errup': 19,
        'M_A_errdn': 20,
        'R_A': 21,
        'R_A_errup': 22,
        'R_A_errdn': 23,
        'loggA': 24,
        'loggA_errup': 25,
        'loggA_errdn': 26,
        'rho_A': 27,
        'rho_A_errup': 28,
        'rho_A_errdn': 29,
        'e': 30,
        'e_errup': 31,
        'e_errdown': 32,
        'a(AU)': 33,
        'a_errup': 34,
        'a_errdown': 35,
        'M_b': 36,
        'M_b_errup': 37,
        'M_b_errdn': 38,
        'R_b': 39,
        'R_b_errup': 40,
        'R_b_errdn': 41,
        'g_b': 42,
        'g_b_errup': 43,
        'g_b_errdn': 44,
        'rho_b': 45,
        'rho_b_errup': 46,
        'rho_b_errdn': 47,
        'Teq': 48,
        'Teq_err1': 49,
        'Teq_err2': 50,
        'Lambda': 51,
        'Lambda_err1': 52,
        'Lambda_err2': 53,
        'Psi': 54,
        'Psi_err1': 55,
        'Psi_err2': 56
    }

    _NEEDED_COLS = [
        'System', 'Teff', 'M_A', 'R_b', 'e', 'a(AU)',
        'Lambda', 'Lambda_err1', 'Lambda_err2',
        'Psi', 'Psi_err1', 'Psi_err2'
    ]

    def __init__(self, data_source=None):
        self.data_source = data_source if data_source else \
            "https://www.astro.keele.ac.uk/jkt/tepcat/allinfo-csv.csv"
        self.df_raw = None
        self.data = None
        self._load_data()
        self._prepare_data()

    def _load_data(self):
        """Load CSV data from the given source."""
        try:
            self.df_raw = pd.read_csv(self.data_source, comment='#')
        except Exception as e:
            raise ValueError(f"Could not load data from {self.data_source}: {e}")

    def _prepare_data(self):
        """
        Extract relevant columns, compute absolute obliquities with
        asymmetric error bars, and clean invalid entries.
        """
        df = self.df_raw
        indices = [self._COL_MAP[col] for col in self._NEEDED_COLS]
        data = df.iloc[:, indices].copy()
        data.columns = self._NEEDED_COLS

        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        def compute_abs_errors(val, err_low, err_high):
            low = val - err_low
            high = val + err_high
            abs_val = abs(val)
            if low * high <= 0:
                abs_min = 0.0
            else:
                abs_min = min(abs(low), abs(high))
            abs_max = max(abs(low), abs(high))
            return abs_val, abs_val - abs_min, abs_max - abs_val

        lambda_abs = []
        lambda_err_low = []
        lambda_err_high = []
        psi_abs = []
        psi_err_low = []
        psi_err_high = []

        for _, row in data.iterrows():
            # Lambda
            l_val = row['Lambda']
            l_err1 = row['Lambda_err1']
            l_err2 = row['Lambda_err2']
            if pd.isna(l_val) or l_val in (999, -1, -999) or pd.isna(l_err1) or pd.isna(l_err2):
                lambda_abs.append(np.nan); lambda_err_low.append(np.nan); lambda_err_high.append(np.nan)
            else:
                av, el, eh = compute_abs_errors(l_val, l_err1, l_err2)
                lambda_abs.append(av); lambda_err_low.append(el); lambda_err_high.append(eh)

            # Psi
            p_val = row['Psi']
            p_err1 = row['Psi_err1']
            p_err2 = row['Psi_err2']
            if pd.isna(p_val) or p_val in (999, -1, -999) or pd.isna(p_err1) or pd.isna(p_err2):
                psi_abs.append(np.nan); psi_err_low.append(np.nan); psi_err_high.append(np.nan)
            else:
                av, el, eh = compute_abs_errors(p_val, p_err1, p_err2)
                psi_abs.append(av); psi_err_low.append(el); psi_err_high.append(eh)

        data['lambda_abs'] = lambda_abs
        data['lambda_err_low'] = lambda_err_low
        data['lambda_err_high'] = lambda_err_high
        data['psi_abs'] = psi_abs
        data['psi_err_low'] = psi_err_low
        data['psi_err_high'] = psi_err_high

        data = data.dropna(subset=['lambda_abs', 'psi_abs'], how='all')

        data['e'] = data['e'].fillna(0.5)
        data['R_b'] = data['R_b'].fillna(1.0)

        for col in ['e', 'R_b', 'Teff', 'M_A', 'a(AU)']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        self.data = data

    def _prepare_plot(self, x_vals, y_vals, y_err_low, y_err_high,
                      sizes, colors, xlabel, ylabel, ylim=(0, 180)):
        """
        Internal helper to create a scatter plot with error bars and colorbar.
        Uses reversed grayscale: low e → white, high e → black.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        # Error bars behind markers
        ax.errorbar(x_vals, y_vals, yerr=[y_err_low, y_err_high],
                    fmt='none', ecolor='black', capsize=2, alpha=0.2,
                    linewidth=0.5, zorder=1)
        # Scatter on top with reversed gray colormap
        sc = ax.scatter(x_vals, y_vals, s=sizes, c=colors, cmap='gray_r',
                        edgecolors='black', linewidth=0.3, vmin=0, vmax=1, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        # Colorbar for eccentricity
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=30)
        cbar.set_label('Eccentricity', rotation=270, labelpad=15)
        return fig, ax, sc

    def _add_highlight(self, ax, highlights, x_axis):
        """
        Add one or more manually specified data points (highlights) to the plot.
        Returns a tuple (handles, labels) for legend creation.
        """
        if highlights is None:
            return [], []
        if not isinstance(highlights, list):
            highlights = [highlights]

        handles = []
        labels = []

        for highlight in highlights:
            if 'lambda' in highlight:
                y_val = highlight['lambda']
                err1 = highlight.get('lambda_err_1', 0)
                err2 = highlight.get('lambda_err_2', 0)
            elif 'psi' in highlight:
                y_val = highlight['psi']
                err1 = highlight.get('psi_err_1', 0)
                err2 = highlight.get('psi_err_2', 0)
            else:
                raise ValueError("Highlight dict must contain 'lambda' or 'psi'.")

            if x_axis == 'Teff':
                x_val = highlight.get('Teff')
            elif x_axis == 'a':
                x_val = highlight.get('a') or highlight.get('a(AU)')
            elif x_axis == 'M':
                x_val = highlight.get('M') or highlight.get('M_A')
            else:
                raise ValueError("Unknown x_axis.")
            if x_val is None:
                raise ValueError(f"Highlight dict missing x value for {x_axis}.")

            low = y_val + err1
            high = y_val + err2
            abs_y = abs(y_val)
            if low * high <= 0:
                abs_min = 0
            else:
                abs_min = min(abs(low), abs(high))
            abs_max = max(abs(low), abs(high))
            yerr_low_abs = abs_y - abs_min
            yerr_high_abs = abs_max - abs_y

            marker = highlight.get('marker', 'o')
            color = highlight.get('color', 'red')
            markersize = highlight.get('markersize', 10)
            label = highlight.get('label', None)

            # Error bars behind
            ax.errorbar(x_val, abs_y, yerr=[[yerr_low_abs], [yerr_high_abs]],
                        fmt='none', color=color, capsize=2, alpha=0.2,
                        linewidth=0.5, zorder=1)
            # Marker on top
            sc_h = ax.scatter(x_val, abs_y, marker=marker, color=color,
                              s=markersize**2, label=label, zorder=3,
                              edgecolors='black', linewidth=0.3)
            if label:
                handles.append(sc_h)
                labels.append(label)

        return handles, labels

    # ------------------------------------------------------------------
    # Public plotting methods
    # ------------------------------------------------------------------

    def plot_lambda_vs_Teff(self, highlight=None, Kraft_Break=None, save_as=None):
        """Plot |lambda| against Teff."""
        data = self.data.dropna(subset=['lambda_abs', 'Teff'])
        x = data['Teff']
        y = data['lambda_abs']
        yerr_low = data['lambda_err_low']
        yerr_high = data['lambda_err_high']
        sizes = (data['R_b'] * 4) ** 2
        colors = data['e']   # e=0 → white, e=1 → black with gray_r

        fig, ax, _ = self._prepare_plot(
            x, y, yerr_low, yerr_high, sizes, colors,
            xlabel='Stellar Effective Temperature (K)',
            ylabel='|λ| (deg)'
        )
        if Kraft_Break is not None:
            ax.axvline(x=Kraft_Break, color='red', linestyle='--', linewidth=1.5)
        handles, labels = self._add_highlight(ax, highlight, x_axis='Teff')
        if labels:
            ax.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, 1.08), ncol=len(labels),
                      frameon=False)
        self._save_or_show(save_as)

    def plot_lambda_vs_a(self, highlight=None, save_as=None):
        """Plot |lambda| against semi-major axis (log scale)."""
        data = self.data.dropna(subset=['lambda_abs', 'a(AU)'])
        x = data['a(AU)']
        y = data['lambda_abs']
        yerr_low = data['lambda_err_low']
        yerr_high = data['lambda_err_high']
        sizes = (data['R_b'] * 4) ** 2
        colors = data['e']

        fig, ax, _ = self._prepare_plot(
            x, y, yerr_low, yerr_high, sizes, colors,
            xlabel='Semi-major Axis (AU) [log scale]',
            ylabel='|λ| (deg)'
        )
        ax.set_xscale('log')
        handles, labels = self._add_highlight(ax, highlight, x_axis='a')
        if labels:
            ax.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, 1.08), ncol=len(labels),
                      frameon=False)
        self._save_or_show(save_as)

    def plot_lambda_vs_M(self, highlight=None, save_as=None):
        """Plot |lambda| against stellar mass (log scale) with line at 1.2 Msun."""
        data = self.data.dropna(subset=['lambda_abs', 'M_A'])
        x = data['M_A']
        y = data['lambda_abs']
        yerr_low = data['lambda_err_low']
        yerr_high = data['lambda_err_high']
        sizes = (data['R_b'] * 4) ** 2
        colors = data['e']

        fig, ax, _ = self._prepare_plot(
            x, y, yerr_low, yerr_high, sizes, colors,
            xlabel='Stellar Mass (M_sun) [log scale]',
            ylabel='|λ| (deg)'
        )
        ax.set_xscale('log')
        ax.axvline(x=1.2, color='red', linestyle='--', linewidth=1.5)
        handles, labels = self._add_highlight(ax, highlight, x_axis='M')
        if labels:
            ax.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, 1.08), ncol=len(labels),
                      frameon=False)
        self._save_or_show(save_as)

    # --- Psi versions ---

    def plot_psi_vs_Teff(self, highlight=None, Kraft_Break=None, save_as=None):
        """Plot |psi| against Teff."""
        data = self.data.dropna(subset=['psi_abs', 'Teff'])
        x = data['Teff']
        y = data['psi_abs']
        yerr_low = data['psi_err_low']
        yerr_high = data['psi_err_high']
        sizes = (data['R_b'] * 4) ** 2
        colors = data['e']

        fig, ax, _ = self._prepare_plot(
            x, y, yerr_low, yerr_high, sizes, colors,
            xlabel='Stellar Effective Temperature (K)',
            ylabel='|ψ| (deg)'
        )
        if Kraft_Break is not None:
            ax.axvline(x=Kraft_Break, color='red', linestyle='--', linewidth=1.5)
        handles, labels = self._add_highlight(ax, highlight, x_axis='Teff')
        if labels:
            ax.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, 1.08), ncol=len(labels),
                      frameon=False)
        self._save_or_show(save_as)

    def plot_psi_vs_a(self, highlight=None, save_as=None):
        """Plot |psi| against semi-major axis (log scale)."""
        data = self.data.dropna(subset=['psi_abs', 'a(AU)'])
        x = data['a(AU)']
        y = data['psi_abs']
        yerr_low = data['psi_err_low']
        yerr_high = data['psi_err_high']
        sizes = (data['R_b'] * 4) ** 2
        colors = data['e']

        fig, ax, _ = self._prepare_plot(
            x, y, yerr_low, yerr_high, sizes, colors,
            xlabel='Semi-major Axis (AU) [log scale]',
            ylabel='|ψ| (deg)'
        )
        ax.set_xscale('log')
        handles, labels = self._add_highlight(ax, highlight, x_axis='a')
        if labels:
            ax.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, 1.08), ncol=len(labels),
                      frameon=False)
        self._save_or_show(save_as)

    def plot_psi_vs_M(self, highlight=None, save_as=None):
        """Plot |psi| against stellar mass (log scale) with line at 1.2 Msun."""
        data = self.data.dropna(subset=['psi_abs', 'M_A'])
        x = data['M_A']
        y = data['psi_abs']
        yerr_low = data['psi_err_low']
        yerr_high = data['psi_err_high']
        sizes = (data['R_b'] * 4) ** 2
        colors = data['e']

        fig, ax, _ = self._prepare_plot(
            x, y, yerr_low, yerr_high, sizes, colors,
            xlabel='Stellar Mass (M_sun) [log scale]',
            ylabel='|ψ| (deg)'
        )
        ax.set_xscale('log')
        ax.axvline(x=1.2, color='red', linestyle='--', linewidth=1.5)
        handles, labels = self._add_highlight(ax, highlight, x_axis='M')
        if labels:
            ax.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, 1.08), ncol=len(labels),
                      frameon=False)
        self._save_or_show(save_as)

    def _save_or_show(self, save_as):
        if save_as:
            plt.savefig(save_as, dpi=150, bbox_inches='tight')
        else:
            plt.show()


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------

_plotter = None

def get_plotter():
    global _plotter
    if _plotter is None:
        _plotter = Plotter()
    return _plotter


def lambda_T(highlight=None, Kraft_Break=None, save_as=None):
    get_plotter().plot_lambda_vs_Teff(highlight, Kraft_Break, save_as)

def lambda_a(highlight=None, save_as=None):
    get_plotter().plot_lambda_vs_a(highlight, save_as)

def lambda_M(highlight=None, save_as=None):
    get_plotter().plot_lambda_vs_M(highlight, save_as)

def psi_T(highlight=None, Kraft_Break=None, save_as=None):
    get_plotter().plot_psi_vs_Teff(highlight, Kraft_Break, save_as)

def psi_a(highlight=None, save_as=None):
    get_plotter().plot_psi_vs_a(highlight, save_as)

def psi_M(highlight=None, save_as=None):
    get_plotter().plot_psi_vs_M(highlight, save_as)