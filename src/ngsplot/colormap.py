"""Colour mapping identical to the webgui renderer.

The renderer stores the palette in an ``ncolors x 1`` float texture sampled with
``GL_NEAREST``, so the value->colour map is a step function: hard band edges, no
interpolation.  See docs/SPEC.md, "Colour".
"""

import numpy as np

#: default number of colour bands (gui.ts ``Colormap.ncolors``)
NCOLORS = 8


def ramp(t):
    """Webgui's blue-cyan-green-yellow-red ramp at positions ``t`` in [0, 1].

    Mirrors ``makeColormapTexture`` in colormap.ts: four linear segments of equal
    width, meeting at the quarter points.  Positions outside ``[0, 1]`` are clamped,
    which is what the renderer gets from the texture's ``GL_CLAMP_TO_EDGE`` wrapping.
    """
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    h = 4.0 * t
    r = np.zeros(t.shape)
    g = np.zeros(t.shape)
    b = np.zeros(t.shape)

    m = h < 1.0
    r[m], g[m], b[m] = 0.0, h[m], 1.0

    m = (h >= 1.0) & (h < 2.0)
    r[m], g[m], b[m] = 0.0, 1.0, 1.0 - h[m] + 1.0

    m = (h >= 2.0) & (h < 3.0)
    r[m], g[m], b[m] = h[m] - 2.0, 1.0, 0.0

    m = h >= 3.0
    r[m], g[m], b[m] = 1.0, 1.0 - h[m] + 3.0, 0.0

    return np.stack([r, g, b], axis=-1)


def palette(ncolors=NCOLORS):
    """The ``ncolors`` texture texels, i.e. ``ramp(i / (ncolors - 1))``.

    Returned in float32 because that is the texture type the renderer uploads, and
    the texel values are compared bit-for-bit in the tests.
    """
    if ncolors < 2:
        raise ValueError("ncolors must be at least 2")
    return ramp(np.arange(ncolors) / (ncolors - 1)).astype(np.float32)


def band_index(values, vmin, vmax, ncolors=NCOLORS):
    """Texture index of ``values`` under nearest filtering.

    ``x = (value - vmin) / (vmax - vmin)`` then ``floor(x * ncolors)`` clamped to the
    edge texels, which is what ``GL_NEAREST`` + ``GL_CLAMP_TO_EDGE`` does to a fetch at
    ``x``.  Values outside the range snap to the first/last band rather than wrapping.
    """
    values = np.asarray(values, dtype=np.float64)
    if not vmax > vmin:
        return np.zeros(values.shape, dtype=np.intp)
    x = (values - vmin) / (vmax - vmin)
    return np.clip(np.floor(x * ncolors), 0, ncolors - 1).astype(np.intp)


def lookup(values, vmin, vmax, pal=None):
    """RGB (float64, 0..1) of ``values`` for palette ``pal`` (default ``palette()``)."""
    if pal is None:
        pal = palette()
    return pal[band_index(values, vmin, vmax, len(pal))].astype(np.float64)


def to_matplotlib(pal=None, name="ngsolve"):
    """A ``ListedColormap`` reproducing the bands, for matplotlib-native colorbars."""
    from matplotlib.colors import ListedColormap

    if pal is None:
        pal = palette()
    rgba = np.concatenate([pal, np.ones((len(pal), 1))], axis=1)
    return ListedColormap(rgba, name=name)
