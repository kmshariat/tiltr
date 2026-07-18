"""
plotter.py
----------
TEPCatPlotter: a class for loading the TEPCat "all planet info" table and
plotting the spin-orbit obliquity parameters Lambda (projected obliquity,
degrees) and Psi (3D/true obliquity, degrees) against:

    * semi-major axis, a (AU),   on a log scale (default)
    * orbital period, P (days),  on a log scale (default)
    * stellar effective temperature, Teff (K)

Missing values in TEPCat are encoded as -1 or -999 depending on the
column; these are filtered out automatically before plotting.
Error bars are added for both axes where available:
    - Lambda / Psi: uses the first error column after each (symmetric).
    - a(AU): uses errup/errdown (asymmetric).
    - Period(day): uses Perioderr (symmetric).
    - Teff: uses the first 'err' column (symmetric).
The colour bar shows eccentricity, clamped to the physical range [0, 1].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .fetch import DEFAULT_URL, fetch_tepcat_csv

# Sentinel / missing-data values used by TEPCat for various columns.
MISSING_VALUES = (-1, -999)

# Sensible physical bounds used to additionally screen out bad values.
DEFAULT_BOUNDS = {
    "Lambda": (-180.0, 180.0),
    "Psi": (0.0, 180.0),
    "e": (0.0, 1.0),
    "a(AU)": (0.0, None),
    "Period(day)": (0.0, None),
    "Teff": (0.0, None),
}


class TEPCatPlotter:
    """
    Load TEPCat data and make Lambda / Psi diagnostic plots.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to a local TEPCat CSV file. If not given, the file is
        (re)downloaded via :func:`tepcat_lambda_psi.fetch.fetch_tepcat_csv`.
    auto_download : bool, optional
        If True (default), download/refresh the CSV automatically when
        no ``csv_path`` is supplied, or when the given path doesn't exist.
    url : str, optional
        URL to fetch the CSV from, if downloading.

    Attributes
    ----------
    data : pandas.DataFrame
        The full TEPCat table, indexed as read from the CSV.
    """

    def __init__(
        self,
        csv_path: str | os.PathLike | None = None,
        auto_download: bool = True,
        url: str = DEFAULT_URL,
    ):
        if csv_path is None:
            if not auto_download:
                raise ValueError(
                    "csv_path was not given and auto_download=False; "
                    "nothing to load."
                )
            csv_path = fetch_tepcat_csv(url=url)
        else:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                if auto_download:
                    csv_path = fetch_tepcat_csv(dest_path=csv_path, url=url)
                else:
                    raise FileNotFoundError(csv_path)

        self.csv_path = Path(csv_path)
        self.data = self._load(self.csv_path)

    # ------------------------------------------------------------------ #
    # Data loading / cleaning
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        return df

    def refresh(self, url: str = DEFAULT_URL, force: bool = True) -> None:
        """Re-download the TEPCat CSV and reload ``self.data``."""
        self.csv_path = fetch_tepcat_csv(dest_path=self.csv_path, url=url, force=force)
        self.data = self._load(self.csv_path)

    def _valid_mask(
        self,
        columns: Sequence[str],
        extra_bounds: dict | None = None,
    ) -> pd.Series:
        """
        Build a boolean mask that is True where every column in
        `columns` has a real, physically-plausible, non-missing value.
        """
        df = self.data
        mask = pd.Series(True, index=df.index)
        bounds = dict(DEFAULT_BOUNDS)
        if extra_bounds:
            bounds.update(extra_bounds)

        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column {col!r} not found in TEPCat data")
            series = df[col]
            mask &= series.notna()
            mask &= ~series.isin(MISSING_VALUES)
            lo_hi = bounds.get(col)
            if lo_hi is not None:
                lo, hi = lo_hi
                if lo is not None:
                    mask &= series > lo if col == "e" else series >= lo
                if hi is not None:
                    mask &= series <= hi
        return mask

    # ------------------------------------------------------------------ #
    # Error column helpers
    # ------------------------------------------------------------------ #
    def _get_x_error_cols(self, x_col: str) -> Optional[Tuple[str, Optional[str]]]:
        """
        Return the error column(s) for the given x-axis quantity.

        Returns
        -------
        (lo_col, hi_col) for asymmetric errors, or (sym_col, None) for symmetric.
        If no error columns are found, returns None.
        """
        df = self.data
        idx = df.columns.get_loc(x_col)

        if x_col == "a(AU)":
            # The two columns after 'a(AU)' are 'errup' and 'errdown'
            if idx + 2 < len(df.columns):
                lo_col = df.columns[idx + 1]  # 'errup'
                hi_col = df.columns[idx + 2]  # 'errdown'
                return (lo_col, hi_col)
        elif x_col == "Period(day)":
            # The next column is 'Perioderr'
            if idx + 1 < len(df.columns):
                return (df.columns[idx + 1], None)
        elif x_col == "Teff":
            # The next two columns are both 'err' (the first is symmetric)
            if idx + 1 < len(df.columns):
                return (df.columns[idx + 1], None)
        return None

    # ------------------------------------------------------------------ #
    # Core plotting engine (no saving/showing)
    # ------------------------------------------------------------------ #
    def _scatter(
        self,
        x_col: str,
        y_col: str,
        *,
        log_x: bool = False,
        highlight: str | Iterable[str] | None = None,
        highlight_label: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple = (10, 6),
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        extra_bounds: dict | None = None,
        show_x_errors: bool = True,
        show_y_errors: bool = True,
        ylim: tuple | None = None,
    ) -> Tuple[plt.Axes, bool]:
        """
        Internal scatter plot with optional x‑ and y‑error bars.

        x‑errors are added for known columns (a, Period, Teff) if `show_x_errors`.
        y‑errors are added for Lambda/Psi using the first error column after each.

        If `ax` is None, a new figure is created with size `figsize`.

        Returns
        -------
        ax : plt.Axes
            The Axes object containing the plot.
        created_fig : bool
            True if a new figure was created (i.e., `ax` was None), False otherwise.
        """
        df = self.data
        name = df["System"]

        columns_needed = [x_col, y_col]
        mask = self._valid_mask(columns_needed, extra_bounds=extra_bounds)
        if log_x:
            mask &= df[x_col] > 0

        # Determine y-error column (for Lambda/Psi)
        yerr_col = None
        if show_y_errors and y_col in ["Lambda", "Psi"]:
            idx = df.columns.get_loc(y_col)
            if idx + 1 < len(df.columns):
                yerr_col = df.columns[idx + 1]
                mask &= df[yerr_col].notna()

        # Determine x-error columns (if applicable)
        xerr_info = None
        if show_x_errors:
            xerr_info = self._get_x_error_cols(x_col)
            if xerr_info is not None:
                lo_col, hi_col = xerr_info
                if hi_col is None:
                    # symmetric
                    mask &= df[lo_col].notna()
                else:
                    # asymmetric
                    mask &= df[lo_col].notna() & df[hi_col].notna()

        if isinstance(highlight, str):
            highlight = [highlight]
        highlight = set(highlight) if highlight else set()

        is_highlighted = mask & name.isin(highlight)
        is_normal = mask & ~name.isin(highlight)

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)

        # Prepare data for all points
        x_all = df.loc[mask, x_col]
        # Use absolute value only for Lambda
        if y_col == "Lambda":
            y_all = df.loc[mask, y_col].abs()
        else:
            y_all = df.loc[mask, y_col]

        # Prepare error arrays
        xerr = None
        if xerr_info is not None:
            lo_col, hi_col = xerr_info
            if hi_col is None:  # symmetric
                xerr = df.loc[mask, lo_col].abs()
            else:               # asymmetric
                lower = df.loc[mask, lo_col].abs()
                upper = df.loc[mask, hi_col].abs()
                xerr = (lower, upper)

        yerr = None
        if yerr_col is not None:
            yerr = df.loc[mask, yerr_col].abs()

        # Plot error bars for all points (both normal and highlighted)
        if xerr is not None or yerr is not None:
            ax.errorbar(
                x_all, y_all,
                xerr=xerr, yerr=yerr,
                fmt='none', ecolor='gray', alpha=0.5, capsize=2,
                zorder=1
            )

        # Scatter for normal points – colour bar clamped to [0, 1]
        sc = ax.scatter(
            df.loc[is_normal, x_col],
            y_all.loc[is_normal],
            c=df.loc[is_normal, "e"],
            s=100 * df.loc[is_normal, "R_b"],
            cmap="viridis",
            vmin=0.0, vmax=1.0,
            alpha=0.8,
            edgecolor="k",
            linewidth=0.4,
            zorder=2,
        )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Orbital eccentricity")

        if highlight:
            ax.scatter(
                df.loc[is_highlighted, x_col],
                y_all.loc[is_highlighted],
                marker="*",
                s=250,
                color="red",
                edgecolor="black",
                linewidth=0.8,
                zorder=10,
                label=highlight_label or ", ".join(sorted(highlight)),
            )

        if log_x:
            ax.set_xscale("log")

        ax.set_xlabel(xlabel or x_col)
        ax.set_ylabel(ylabel or y_col)
        if title:
            ax.set_title(title)
        ax.legend()

        if ylim is not None:
            ax.set_ylim(ylim)

        return ax, created_fig

    # ------------------------------------------------------------------ #
    # Helper to finalize a standalone plot
    # ------------------------------------------------------------------ #
    def _finalize_plot(self, ax: plt.Axes, created_fig: bool,
                       save_path: str | os.PathLike | None = None,
                       show: bool = True) -> None:
        """Call tight_layout, save, and show if a new figure was created."""
        if created_fig:
            fig = ax.figure
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            if show:
                plt.show()

    # ------------------------------------------------------------------ #
    # Lambda plots
    # ------------------------------------------------------------------ #
    def plot_lambda_vs_a(
        self,
        log_scale: bool = True,
        highlight: str | Iterable[str] | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple = (10, 6),
        save_path: str | os.PathLike | None = None,
        show: bool = True,
        ylim: tuple | None = (-10, 190),
        **kwargs,
    ) -> plt.Axes:
        """Plot projected obliquity Lambda vs semi-major axis a (AU)."""
        ax, created = self._scatter(
            "a(AU)",
            "Lambda",
            log_x=log_scale,
            highlight=highlight,
            ax=ax,
            figsize=figsize,
            xlabel=r"$a$ (AU)" + (" [log scale]" if log_scale else ""),
            ylabel=r"$|\lambda|$ (deg)",
            title=r"Sky-projected obliquity for Exoplanets",
            ylim=ylim,
            **kwargs,
        )
        self._finalize_plot(ax, created, save_path, show)
        return ax

    def plot_lambda_vs_period(
        self,
        log_scale: bool = True,
        highlight: str | Iterable[str] | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple = (10, 6),
        save_path: str | os.PathLike | None = None,
        show: bool = True,
        ylim: tuple | None = (-10, 190),
        **kwargs,
    ) -> plt.Axes:
        """Plot projected obliquity Lambda vs orbital period (days), log scale by default."""
        ax, created = self._scatter(
            "Period(day)",
            "Lambda",
            log_x=log_scale,
            highlight=highlight,
            ax=ax,
            figsize=figsize,
            xlabel=r"$P$ (days)" + (" [log scale]" if log_scale else ""),
            ylabel=r"$|\lambda|$ (deg)",
            title=r"Sky-projected obliquity for Exoplanets",
            ylim=ylim,
            **kwargs,
        )
        self._finalize_plot(ax, created, save_path, show)
        return ax

    def plot_lambda_vs_teff(
        self,
        highlight: str | Iterable[str] | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple = (10, 6),
        save_path: str | os.PathLike | None = None,
        show: bool = True,
        kraft: float | None = 6250.0,
        ylim: tuple | None = (-10, 190),
        **kwargs,
    ) -> plt.Axes:
        """
        Plot projected obliquity Lambda vs stellar Teff (K).

        Parameters
        ----------
        kraft : float or None, default 6250.0
            If not None, draw a vertical dashed line at this temperature.
            The canonical Kraft break is 6250 K.
        """
        ax, created = self._scatter(
            "Teff",
            "Lambda",
            log_x=False,
            highlight=highlight,
            ax=ax,
            figsize=figsize,
            xlabel=r"$T_{\rm eff}$ (K)",
            ylabel=r"$|\lambda|$ (deg)",
            title=r"Sky-projected obliquity for Exoplanets",
            ylim=ylim,
            **kwargs,
        )
        if kraft is not None:
            ax.axvline(
                x=kraft,
                color='darkred',
                linestyle='--',
                linewidth=2,
                alpha=0.9,
                zorder=10,
                label=f'Kraft break ({kraft} K)'
            )
            # Update legend to avoid duplicate entries
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys())

        self._finalize_plot(ax, created, save_path, show)
        return ax

    # ------------------------------------------------------------------ #
    # Psi plots
    # ------------------------------------------------------------------ #
    def plot_psi_vs_a(
        self,
        log_scale: bool = True,
        highlight: str | Iterable[str] | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple = (10, 6),
        save_path: str | os.PathLike | None = None,
        show: bool = True,
        ylim: tuple | None = (-10, 190),
        **kwargs,
    ) -> plt.Axes:
        """Plot true (3D) obliquity Psi vs semi-major axis a (AU)."""
        ax, created = self._scatter(
            "a(AU)",
            "Psi",
            log_x=log_scale,
            highlight=highlight,
            ax=ax,
            figsize=figsize,
            xlabel=r"$a$ (AU)" + (" [log scale]" if log_scale else ""),
            ylabel=r"$\psi$ (deg)",
            title=r"True obliquity for Exoplanets",
            ylim=ylim,
            **kwargs,
        )
        self._finalize_plot(ax, created, save_path, show)
        return ax

    def plot_psi_vs_period(
        self,
        log_scale: bool = True,
        highlight: str | Iterable[str] | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple = (10, 6),
        save_path: str | os.PathLike | None = None,
        show: bool = True,
        ylim: tuple | None = (-10, 190),
        **kwargs,
    ) -> plt.Axes:
        """Plot true (3D) obliquity Psi vs orbital period (days), log scale by default."""
        ax, created = self._scatter(
            "Period(day)",
            "Psi",
            log_x=log_scale,
            highlight=highlight,
            ax=ax,
            figsize=figsize,
            xlabel=r"$P$ (days)" + (" [log scale]" if log_scale else ""),
            ylabel=r"$\psi$ (deg)",
            title=r"True obliquity for Exoplanets",
            ylim=ylim,
            **kwargs,
        )
        self._finalize_plot(ax, created, save_path, show)
        return ax

    def plot_psi_vs_teff(
        self,
        highlight: str | Iterable[str] | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple = (10, 6),
        save_path: str | os.PathLike | None = None,
        show: bool = True,
        kraft: float | None = 6250.0,
        ylim: tuple | None = (-10, 190),
        **kwargs,
    ) -> plt.Axes:
        """
        Plot true (3D) obliquity Psi vs stellar Teff (K).

        Parameters
        ----------
        kraft : float or None, default 6250.0
            If not None, draw a vertical dashed line at this temperature.
            The canonical Kraft break is 6250 K.
        """
        ax, created = self._scatter(
            "Teff",
            "Psi",
            log_x=False,
            highlight=highlight,
            ax=ax,
            figsize=figsize,
            xlabel=r"$T_{\rm eff}$ (K)",
            ylabel=r"$\psi$ (deg)",
            title=r"True obliquity for Exoplanets",
            ylim=ylim,
            **kwargs,
        )
        if kraft is not None:
            ax.axvline(
                x=kraft,
                color='darkred',
                linestyle='--',
                linewidth=2,
                alpha=0.9,
                zorder=10,
                label=f'Kraft break ({kraft} K)'
            )
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys())

        self._finalize_plot(ax, created, save_path, show)
        return ax

    # ------------------------------------------------------------------ #
    # Convenience: full 2x3 grid
    # ------------------------------------------------------------------ #
    def plot_grid(
        self,
        log_scale_a: bool = True,
        log_scale_period: bool = True,
        highlight: str | Iterable[str] | None = None,
        figsize: tuple = (16, 9),
        save_path: str | os.PathLike | None = None,
        show: bool = True,
        kraft: float | None = 6250.0,
        ylim: tuple | None = (-10, 190),
    ) -> plt.Figure:
        """
        Make a 2x3 grid of plots: rows = Lambda, Psi; columns = a, Period, Teff.

        Parameters
        ----------
        log_scale_a : bool
            Use log scale for the semi‑major axis (default True).
        log_scale_period : bool
            Use log scale for the orbital period (default True).
        highlight : str or iterable of str, optional
            System name(s) to highlight with a red star marker.
        figsize : tuple
            Overall figure size.
        save_path : str or Path, optional
            If given, save the figure to this path.
        show : bool
            If True, call plt.show() at the end.
        kraft : float or None, default 6250.0
            Passed to the Teff subplots; draws a vertical line at this temperature.
        ylim : tuple or None, default (-10, 190)
            Y-axis limits for all obliquity subplots.
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Pass ax and disable individual saving/showing – grid handles that.
        self.plot_lambda_vs_a(log_scale=log_scale_a, highlight=highlight, ax=axes[0, 0], show=False, ylim=ylim)
        self.plot_lambda_vs_period(log_scale=log_scale_period, highlight=highlight, ax=axes[0, 1], show=False, ylim=ylim)
        self.plot_lambda_vs_teff(highlight=highlight, ax=axes[0, 2], show=False, kraft=kraft, ylim=ylim)

        self.plot_psi_vs_a(log_scale=log_scale_a, highlight=highlight, ax=axes[1, 0], show=False, ylim=ylim)
        self.plot_psi_vs_period(log_scale=log_scale_period, highlight=highlight, ax=axes[1, 1], show=False, ylim=ylim)
        self.plot_psi_vs_teff(highlight=highlight, ax=axes[1, 2], show=False, kraft=kraft, ylim=ylim)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig