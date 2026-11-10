"""Decode the render payload of ``ngsolve.webgui.BuildRenderData``.

The webgui does not draw a mesh: Python fits every element to a Bezier triangle
and JavaScript rasterises the result (``ngsolve/webgui.py``, ``three/mesh.ts``,
``src/utils.ts``).  The payload is a dictionary of base64 float32 arrays plus a
handful of scalars, and this module is the Python counterpart of the JavaScript
unpacking, so the same bytes can feed a software rasteriser.  It never imports
ngsolve: a payload can be captured once and replayed offline.

``Bezier_trig_points`` holds the surface, ``Bezier_points`` the element outlines
and ``edges`` the boundary skeleton.  With ``n = order2d`` control points per
side, ``K = (n + 1)(n + 2) / 2`` points per element and ``nr`` rows
(``nr`` = elements, ``3 * elements`` and boundary segments, respectively), the
lists are laid out as

* ``per_row`` arrays of ``(nr, 4)``: array ``j`` is control point ``j`` of every
  row, its components being ``(x, y, z, value)`` -- the vec4 attribute ``p{a}{b}``
  of ``shader/utils.h``.
* for a multi-component function the rest, which carries the ``vec`` attributes
  (the remaining components).  The surface packs two control points per vec4
  when ``n > 1`` (``.xy`` for the even index, ``.zw`` for the odd one) and sends
  a vec2 per control point when ``n == 1``; the wireframe always sends a vec2 per
  control point, and the boundary skeleton never sends any.

Control point ``j`` is the pair ``(a, b)`` of
:func:`ngsplot.bezier.index_pairs`; its Greville abscissa is
``(u, v) = ((n - a - b) / n, b / n)`` in the reference coordinates of
:mod:`ngsplot.bezier`, which are a reflection of the mesh parametrization.

Only the Bezier payload is decoded here.  ``nodal_p1=True`` and ``order=0`` ask
ngsolve for a nodal mesh instead (``vertices``, ``trigs``, ``segs``), which the
renderer unpacks through ``src/utils.ts:unpackIndexedData``; that path is not
needed to reproduce the default look.
"""

import base64

import numpy as np

from . import bezier

__all__ = [
    "DBL_MIN",
    "DBL_MAX",
    "decode_array",
    "order_of",
    "greville",
    "trig_points",
    "wireframe_points",
    "edge_points",
    "sample_values",
    "minmax",
]

DBL_MIN = np.finfo(np.float64).tiny
"""Smallest normal double, ``numeric_limits<double>::min()``."""

DBL_MAX = np.finfo(np.float64).max
"""Largest double, ``numeric_limits<double>::max()``."""

_GREVILLE = {}


def decode_array(data):
    """Return one entry of the payload as a 1-D float32 array.

    Accepts the base64 text of the default ``encoding='b64'``, the raw bytes of
    ``encoding='binary'``, and arrays (cast, so that a re-encoded payload fits).
    """
    if isinstance(data, np.ndarray):
        return data.astype(np.float32, copy=False)
    if isinstance(data, str):
        buf = base64.b64decode(data, validate=True)
    elif isinstance(data, (bytes, bytearray, memoryview)):
        buf = bytes(data)
    else:
        raise TypeError(f"cannot decode a payload entry of type {type(data).__name__}")
    if len(buf) % np.dtype(np.float32).itemsize:
        raise ValueError("payload is not a whole number of float32 values")
    return np.frombuffer(buf, dtype=np.float32)


def order_of(data, order=None):
    """Return ``order2d`` of the payload, the degree of the surface and outlines."""
    if order is None:
        order = data["order2d"]
    order = int(order)
    bezier.names(order)  # raises for the degrees that carry no Bezier data
    return order


def greville(order):
    """``(u, v)`` of every control point, in the payload order."""
    pairs = bezier.index_pairs(order)  # validates order
    if order not in _GREVILLE:
        u = np.array([(order - a - b) / order for a, b in pairs], dtype=np.float64)
        v = np.array([b / order for a, b in pairs], dtype=np.float64)
        u.flags.writeable = False
        v.flags.writeable = False
        _GREVILLE[order] = (u, v)
    return _GREVILLE[order]


def _rows(payload, order, per_row, key, packed_vectors):
    arrays = [decode_array(a) for a in payload]
    if not arrays:
        raise ValueError(f"{key}: empty payload")
    if arrays[0].size % 4:
        raise ValueError(f"{key}: first array is not a whole number of vec4 control points")
    nrows = arrays[0].size // 4
    if len(arrays) < per_row:
        raise ValueError(f"{key}: expected at least {per_row} arrays, got {len(arrays)}")
    points = np.stack([a.reshape(nrows, 4) for a in arrays[:per_row]], axis=1)
    extra = arrays[per_row:]
    if not extra:
        return points, None
    packed = packed_vectors and order > 1
    ncomp = 4 if packed else 2
    nvec = (per_row + 1) // 2 if packed else per_row
    if len(extra) != nvec:
        raise ValueError(f"{key}: expected {nvec} vector arrays, got {len(extra)}")
    if extra[0].size != nrows * ncomp:
        raise ValueError(f"{key}: vector arrays hold {extra[0].size // nrows} components"
                         f" per point, expected {ncomp}")
    if not packed:
        return points, np.stack([a.reshape(nrows, 2) for a in extra], axis=1)
    pairs = np.stack([a.reshape(nrows, 4) for a in extra], axis=1)
    vectors = np.empty((nrows, per_row, 2), dtype=np.float32)
    vectors[:, 0::2] = pairs[:, :, :2]
    vectors[:, 1::2] = pairs[:, :, 2:]
    return points, vectors


def trig_points(data, order=None):
    """Control points of the surface, ``(ne, K, 4)``, and ``(ne, K, 2)`` vectors.

    The vectors are ``None`` for a scalar function.  For a function of two
    components the second vector component stays zero: only what does not fit in
    the ``value`` slot of the vec4 is sent.
    """
    order = order_of(data, order)
    return _rows(
        data["Bezier_trig_points"], order, len(bezier.names(order)), "Bezier_trig_points", True
    )


def wireframe_points(data, order=None):
    """Control points of the element outlines, ``(3 * ne, order + 1, 4)``.

    Row ``3 * element + edge`` is edge ``edge`` of that element, ``edge`` being
    the edge of the reference triangle that starts at ``(0, 0)``, ``(0, 0)`` and
    ``(0, 1)`` for ``edge`` 0, 1 and 2.
    """
    order = order_of(data, order)
    return _rows(data["Bezier_points"], order, order + 1, "Bezier_points", False)


def edge_points(data, order=None):
    """Control points of the boundary skeleton, ``(nseg, order + 1, 4)``.

    ngsolve builds the boundary function as ``CF((xyz, 0.0))``, so the value slot
    -- the colour of the thick edges -- is zero throughout and no vectors are
    sent.
    """
    order = order_of(data, order)
    return _rows(data["edges"], order, order + 1, "edges", True)


def sample_values(order, ctrl, comp=3):
    """Interpolate control points at their own Greville points.

    ``ctrl`` is a ``(ne, K, 4)`` array of :func:`trig_points` (its component
    ``comp`` is used) or a ``(ne, K)`` field.  The result are the values at
    ``greville(order)``, the points the payload was fitted from and the points
    ngsolve takes the colour range from; since the stored control points are
    float32, expect the float64 samples back to a few parts in 1e7.
    """
    order = int(order)
    npoints = len(bezier.index_pairs(order))
    field = np.asarray(ctrl, dtype=np.float64)
    if field.ndim == 3:
        field = field[..., comp]
    if field.shape[-1] != npoints:
        raise ValueError(f"expected {npoints} control points of order {order}, got {field.shape[-1]}")
    return field @ bezier.bernstein(order, *greville(order)).T


def minmax(values, ignore_inf=True):
    """``ngs.Vector.MinMax``: the bounds start at ``DBL_MAX`` and ``DBL_MIN``.

    NGSolve initialises the bounds with ``numeric_limits`` instead of ``+-inf``,
    so a function that never exceeds ``DBL_MIN`` -- typically a non-positive one
    -- reports ``DBL_MIN`` as its maximum, and the maximum of an empty column
    stays there.  Infinities are left out on request (what ngsolve does for the
    control points, whose ``MinMax`` is called with ``ignore_inf=True``); NaN
    never takes part, in either mode.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    keep = np.isfinite(values) if ignore_inf else ~np.isnan(values)
    values = values[keep]
    if values.size == 0:
        return (DBL_MAX, DBL_MIN)
    return (float(values.min()), max(float(values.max()), DBL_MIN))
