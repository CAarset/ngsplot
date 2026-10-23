"""pyplot-style plots that look exactly like NGSolve's Draw.

The command to reach for is :func:`ngsplot.Plot`, the counterpart of ``ngsolve.webgui.Draw``::

    from ngsplot import Plot

    Plot(gfu, colorbar=True)

It lives in :mod:`ngsplot.pyplot`, with the rest of the pyplot-shaped surface around it, and is
handed out from here on request so that the matplotlib import it needs happens only when someone
asks to plot, and the renderer underneath stays importable without it.
"""

from importlib import import_module

from . import bezier, camera, colormap, geometry, labels, lighting, raster, scene

__version__ = "0.1.0"

#: names that live in a module a plain ``import ngsplot`` does not load
_LAZY = {"Plot": "pyplot"}

__all__ = ["bezier", "camera", "colormap", "geometry", "labels", "lighting", "raster", "scene",
           "Plot", "__version__"]


def __getattr__(name):
    """Load :mod:`ngsplot.pyplot` the first time someone reaches for ``ngsplot.Plot``."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{_LAZY[name]}", __name__), name)
