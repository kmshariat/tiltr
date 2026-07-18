"""
fetch.py
--------
Utilities to download the TEPCat "all planet info" CSV file from Keele
University and cache it locally.

TEPCat homepage: https://www.astro.keele.ac.uk/jkt/tepcat/tepcat.html
Direct CSV link : https://www.astro.keele.ac.uk/jkt/tepcat/allinfo-csv.csv
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_URL = "https://www.astro.keele.ac.uk/jkt/tepcat/allinfo-csv.csv"
DEFAULT_CACHE_DIR = Path.home() / ".tepcat_lambda_psi"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "allinfo-csv.csv"


def fetch_tepcat_csv(
    dest_path: str | os.PathLike | None = None,
    url: str = DEFAULT_URL,
    force: bool = False,
    timeout: int = 30,
) -> Path:
    """
    Download the latest TEPCat all-info CSV file and save it locally.

    Parameters
    ----------
    dest_path : str or Path, optional
        Where to save the CSV file. Defaults to
        ``~/.tepcat_lambda_psi/allinfo-csv.csv``.
    url : str, optional
        The URL to fetch the CSV from. Defaults to the official TEPCat
        "all info" CSV link.
    force : bool, optional
        If True, always re-download even if a cached copy already
        exists. If False (default) and a cached file is already present,
        it is still refreshed (TEPCat is updated frequently), but network
        errors will fall back to the existing cached copy instead of
        raising, unless no cached copy exists at all.
    timeout : int, optional
        Timeout in seconds for the HTTP request.

    Returns
    -------
    Path
        Path to the downloaded (or cached, on failure) CSV file.

    Raises
    ------
    RuntimeError
        If the download fails and no cached copy is available to fall
        back on.
    """
    dest_path = Path(dest_path) if dest_path is not None else DEFAULT_CACHE_FILE
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        # We still try to refresh, but tolerate failure since we have a
        # cached copy to fall back on.
        try:
            _download(url, dest_path, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(
                f"[tepcat_lambda_psi] Warning: could not refresh TEPCat "
                f"data ({exc}). Using cached file at {dest_path} "
                f"(last modified {_mtime_str(dest_path)})."
            )
        return dest_path

    try:
        _download(url, dest_path, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if dest_path.exists():
            print(
                f"[tepcat_lambda_psi] Warning: download failed ({exc}); "
                f"using existing cached file at {dest_path}."
            )
            return dest_path
        raise RuntimeError(
            f"Failed to download TEPCat CSV from {url!r} and no cached "
            f"copy was found at {dest_path!r}: {exc}"
        ) from exc

    return dest_path


def _download(url: str, dest_path: Path, timeout: int = 30) -> None:
    """Download `url` to a temp file, then atomically move it to dest_path."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "tepcat_lambda_psi/1.0 (Python urllib)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()

    if not data or b"System" not in data[:200]:
        # Very lightweight sanity check that we actually got a CSV
        # with the expected header, not e.g. an HTML error page.
        raise urllib.error.URLError(
            "Downloaded content does not look like the TEPCat CSV"
        )

    fd, tmp_name = tempfile.mkstemp(dir=str(dest_path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        shutil.move(tmp_name, dest_path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _mtime_str(path: Path) -> str:
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.strftime("%Y-%m-%d %H:%M UTC")
