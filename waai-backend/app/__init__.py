"""Application package for WAAI backend.

This module bootstraps the FastAPI instance from the top-level app.py while
also exposing subpackages (api, util).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


def _load_main_app():
    """Load top-level app.py as a module and return its `app` attribute."""
    root = pathlib.Path(__file__).resolve().parent.parent
    main_path = root / "app.py"
    spec = importlib.util.spec_from_file_location("_app_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load app.py from {main_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_app_main"] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return getattr(module, "app")


app = _load_main_app()

__all__ = ["app"]
