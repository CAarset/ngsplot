"""pyplot-style plots that look exactly like NGSolve's Draw.

The command to reach for is :func:`ngsplot.Plot`, the counterpart of ``ngsolve.webgui.Draw``::

    from ngsplot import Plot

    Plot(gfu, colorbar=True)

It lives in :mod:`ngsplot.pyplot`, with the rest of the pyplot-shaped surface around it, and is
handed out from here on request so that the matplotlib import it needs happens only when someone
asks to plot, and the renderer underneath stays importable without it.  The same goes for
:func:`ngsplot.render_data`, the payload ``Draw`` would have sent, which needs ngsolve: build it
where the field lives and plot it later, anywhere, with ``Plot(payload)``.  :func:`ngsplot.frame`
is the camera a plot of a payload would be painted with, and so the size of its canvas, found
without painting: ``Plot(payload, view=frame(payload, width=400, height=None, fit="box"))``.
"""

from importlib import import_module

from . import bezier, camera, colormap, geometry, labels, lighting, raster, scene
from .scene import frame

__version__ = "0.1.0"

#: names that live in a module a plain ``import ngsplot`` does not load
_LAZY = {"Plot": "pyplot", "render_data": "_ngsolve"}

__all__ = ["bezier", "camera", "colormap", "geometry", "labels", "lighting", "raster", "scene",
           "Plot", "frame", "render_data", "__version__"]


def __getattr__(name):
    """Load the module behind ``ngsplot.Plot`` or ``ngsplot.render_data`` when first asked."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{_LAZY[name]}", __name__), name)
