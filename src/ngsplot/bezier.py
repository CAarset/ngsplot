"""Bézier triangles exactly as the webgui shaders evaluate them.

The renderer sends one instance per mesh element together with the Bézier control
points of the mapped reference triangle, tessellates each instance into
``3 * subdivision**2`` sub-triangles in the vertex shader, and never re-evaluates
the polynomial in the fragment shader.  Reproducing that pipeline pixel for pixel
means evaluating the same polynomials at the same points, with the same algebra,
which is what this module does.

Naming follows the shader attribute names: the control point of degree ``n`` with
barycentric exponents ``(i, j, k)`` for ``(u, v, w)`` is called ``p{a}{b}`` with
``a = k``, ``b = j``; ``index_pairs`` lists them in the order the renderer packs
them into attributes (three/mesh.ts:102-131).
"""

import math

import numpy as np

__all__ = ["index_pairs", "names", "vec_names", "bernstein", "evaluate",
           "vector_values", "derivatives", "normal", "get_value", "lattice",
           "subtriangles", "curve_basis", "curve_vector_values", "curve"]

_NAMES = {
    1: ["p0", "p1", "p2"],
    2: ["p00", "p01", "p02", "p10", "p11", "p20"],
    3: ["p00", "p01", "p02", "p03", "p10", "p11", "p12", "p20", "p21", "p30"],
}
_VEC_NAMES = {
    1: ["v0", "v1", "v2"],
    2: ["vec00_01", "vec02_10", "vec11_20"],
    3: ["vec00_01", "vec02_03", "vec10_11", "vec12_20", "vec21_30"],
}
_CACHE = {}
_LATTICE = {}
_SUBTRIANGLES = {}


def _check_order(order):
    if order not in _NAMES:
        raise ValueError(f"order must be 1, 2 or 3, got {order}")
    return order


def index_pairs(order):
    """Control point indices (a, b) in shader attribute order."""
    _check_order(order)
    return _pairs(order)


def _pairs(order):
    """Same listing, also allowed for the degree-0 basis used by the derivative."""
    if order not in _CACHE:
        _CACHE[order] = [(a, b) for a in range(order + 1) for b in range(order - a + 1)]
    return _CACHE[order]


def names(order):
    """Names of the vec4 position/value attributes, three/mesh.ts:102-123."""
    return list(_NAMES[_check_order(order)])


def vec_names(order):
    """Names of the vector-value attributes, three/mesh.ts:124-131."""
    _check_order(order)
    return list(_VEC_NAMES[order])


def _bernstein(order, u, v):
    u = np.asarray(u)
    v = np.asarray(v)
    dtype = np.result_type(u, v, float)
    u, v = u.astype(dtype, copy=False), v.astype(dtype, copy=False)
    coord = np.stack([u, v, 1.0 - u - v], axis=-1)
    fact = math.factorial(order)
    out = []
    for a, b in _pairs(order):
        exp = (order - a - b, b, a)
        coef = fact
        for e in exp:
            coef //= math.factorial(e)
        term = np.full(u.shape, float(coef), dtype=dtype)
        for axis, e in enumerate(exp):
            if e:
                term = term * coord[..., axis] ** e
        out.append(term)
    return np.stack(out, axis=-1)


def bernstein(order, u, v):
    """Degree-`order` (1..3) Bernstein basis over the reference triangle.

    Matches the literals of GetPositionAndScalar in shader/utils.h:94-100 (order 1),
    :126-131 (order 2) and :250-259 (order 3).
    """
    return _bernstein(_check_order(order), u, v)


def _weighted(basis, field):
    return np.sum(basis[..., :, None] * np.asarray(field), axis=-2)


def _difference(order, u, v, field, direction):
    """order * sum of the degree-(order-1) basis over consecutive control values."""
    field = np.asarray(field)
    pos = _pairs(order - 1)
    index = {pair: i for i, pair in enumerate(_pairs(order))}
    basis = _bernstein(order - 1, u, v) * order
    terms = []
    for j, (a, b) in enumerate(pos):
        p10 = field[..., index[(a + 1, b)], :]
        if direction == "u":
            terms.append(basis[..., j, None] * (field[..., index[(a, b)], :] - p10))
        else:
            terms.append(basis[..., j, None] * (field[..., index[(a, b + 1)], :] - p10))
    return np.sum(terms, axis=0)


def lattice(n):
    """(u, v) of the sub-triangle lattice, in the order of three/mesh.ts:25-42."""
    n = int(n)
    if n not in _LATTICE:
        u, v = [], []
        for s in range(n + 1):
            for x in range(s + 1):
                u.append(x / n)
                v.append((s - x) / n)
        arrays = tuple(np.asarray(a, dtype=np.float64) for a in (u, v))
        for a in arrays:
            a.flags.writeable = False
        _LATTICE[n] = arrays
    return _LATTICE[n]


def subtriangles(n):
    """Lattice indices of the n*n sub-triangles, in shader draw order."""
    n = int(n)
    if n not in _SUBTRIANGLES:
        tris = []
        for i in range(n):
            for j in range(i + 1):
                tris.append([_lattice_index(j, i - j, n), _lattice_index(j + 1, i - j, n),
                             _lattice_index(j, i - j + 1, n)])
            for j in range(i):
                tris.append([_lattice_index(j + 1, i - j - 1, n),
                             _lattice_index(j + 1, i - j, n), _lattice_index(j, i - j, n)])
        out = np.asarray(tris, dtype=np.int32)
        out.flags.writeable = False
        _SUBTRIANGLES[n] = out
    return _SUBTRIANGLES[n]


def _lattice_index(x, y, n):
    s = x + y
    return s * (s + 1) // 2 + x


def get_value(value3, function_mode=0.0, complex_scale=(1.0, 0.0)):
    """GetValue from shader/utils.h:46-56: the scalar the renderer colours by."""
    value3 = np.asarray(value3)
    mode = float(function_mode)
    x, y, z = value3[..., 0], value3[..., 1], value3[..., 2]
    if mode == 0.0:
        return x
    if mode == 1.0:
        return y
    if mode == 2.0:
        return z
    if mode == 3.0:
        return np.sqrt(x * x + y * y + z * z)
    sx, sy = float(complex_scale[0]), float(complex_scale[1])
    if mode == 5.0:                      # real part
        return x * sx - y * sy
    if mode == 6.0:                      # imaginary part
        return x * sy + y * sx
    return np.zeros(value3.shape[:-1], dtype=value3.dtype)


def _unit(n):
    """Divide by the length, exactly as the shader's ``normalize`` does.

    A collapsed patch has a zero cross product, and dividing by zero is what GLSL
    does there: the NaNs that follow are what cull the element's whole lattice,
    because they spread to every vertex of it (:mod:`ngsplot.raster` drops those
    triangles, so the element simply is not in the picture).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return n / np.linalg.norm(n, axis=-1, keepdims=True)


def vector_values(order, vectors, u, v):
    """GetVectorValues: the two packed vector components, same basis as the position."""
    return _weighted(_bernstein(_check_order(order), u, v), vectors)


def derivatives(order, ctrl, u, v):
    """d/du and d/dv of the homogeneous patch, as built by GetNormal.

    The shader writes the degree-(order-1) coefficients pre-multiplied by the
    derivative factor, which is exactly the derivative of the Bézier patch:
    d/du = order * sum of B^(order-1)_(a,b) (P_(a,b) - P_(a+1,b)) and
    d/dv = order * sum of B^(order-1)_(a,b) (P_(a,b+1) - P_(a+1,b)).
    """
    order = _check_order(order)
    return (_difference(order, u, v, ctrl, "u"), _difference(order, u, v, ctrl, "v"))


def _phase(ctrl, vectors, function_mode, complex_scale):
    """The complex phase mix that GetNormal applies to p.w (shader/utils.h:150-171,
    :300-311) before differencing, under DEFORMATION_2D."""
    mode = float(function_mode)
    sx, sy = float(complex_scale[0]), float(complex_scale[1])
    if mode == 5.0:
        re, im = sx, -sy
    elif mode == 6.0:
        re, im = sy, sx
    else:
        re, im = 1.0, 0.0
    out = np.array(ctrl, copy=True)
    out[..., 3] = out[..., 3] * re + vectors[..., 0] * im
    return out


def evaluate(order, ctrl, u, v, *, vectors=None, deformation=0.0, mode="none",
             function_mode=0.0, complex_scale=(1.0, 0.0)):
    """Position and scalar value at (u, v), with the deformation displacement.

    ctrl holds the vec4 control points (x, y, z, value) in names() order, vectors
    the unpacked vector values of the same points.  mode is "none", "3d"
    (DEFORMATION) or "2d" (DEFORMATION_2D), following trigsplines.vert:25-30.
    Returns (xyz, value), value being the interpolated p.w.
    """
    order = _check_order(order)
    pv = _weighted(_bernstein(order, u, v), ctrl)
    xyz, value = pv[..., :3], pv[..., 3]
    deformation = float(deformation)
    if mode == "none" or deformation == 0.0:
        return xyz, value
    if vectors is None:
        vv = np.zeros(pv.shape[:-1] + (2,), dtype=pv.dtype)
    else:
        vv = vector_values(order, vectors, u, v)
    value3 = np.stack([value, vv[..., 0], vv[..., 1]], axis=-1)
    return _displace(xyz, value3, deformation, mode, function_mode, complex_scale), value


def _displace(xyz, value3, deformation, mode, function_mode, complex_scale):
    """The two statements a vertex program adds to the blended position.

    ``DEFORMATION`` pushes the point by ``deformation * value_`` and ``DEFORMATION_2D``
    lifts ``z`` by ``GetValue(deformation * value_)``: ``trigsplines.vert:25-30`` for a
    surface, ``splines.vert:73-77`` for a curve.  ``value3`` is the program's ``value_``,
    whose first component is the blended 4th control-point component and whose other two
    are the blended vector attributes.
    """
    disp = value3 * deformation
    if mode == "3d":
        return xyz + disp
    if mode == "2d":
        z = xyz[..., 2] + get_value(disp, function_mode, complex_scale)
        return np.concatenate([xyz[..., :2], z[..., None]], axis=-1)
    raise ValueError(f"unknown deformation mode {mode!r}")


def normal(order, ctrl, u, v, *, vectors=None, deformation=0.0, mode="none",
           function_mode=0.0, complex_scale=(1.0, 0.0), vertex_normals=None):
    """GetNormal from shader/utils.h:84-94 (order 1), :134-190 and :260-338.

    Order 1 takes the cross product of the two edge vectors, or blends the nodal
    normals when the mesh supplies them (HAVE_NORMALS); it carries no deformation
    terms.  Higher orders cross the parametric derivatives, with the displacement
    folded into them the way the shader does: the deformed position adds
    deformation * value, so d/du gains deformation * d/du of that quantity, whose
    components come from the vector values in the "3d" case.
    """
    order = _check_order(order)
    if vertex_normals is not None and order == 1:
        u = np.asarray(u)
        v = np.asarray(v)
        nrm = np.asarray(vertex_normals)
        n = (u[..., None] * nrm[..., 0, :] + v[..., None] * nrm[..., 1, :]
             + (1.0 - u - v)[..., None] * nrm[..., 2, :])
        return _unit(n)

    ctrl = np.asarray(ctrl)
    deformation = float(deformation)
    if vectors is None:
        vectors = np.zeros(ctrl.shape[:-1] + (2,), dtype=ctrl.dtype)
    vectors = np.asarray(vectors)
    if mode == "2d":
        ctrl = _phase(ctrl, vectors, function_mode, complex_scale)
    du, dv = derivatives(order, ctrl, u, v)
    if deformation != 0.0 and mode == "3d":
        duu = _difference(order, u, v, vectors, "u")
        dvv = _difference(order, u, v, vectors, "v")
        du = _deform(du, duu, deformation, slice(1, 3), slice(0, 2))
        dv = _deform(dv, dvv, deformation, slice(1, 3), slice(0, 2))
    elif deformation != 0.0 and mode == "2d":
        du = _deform(du, None, deformation, slice(2, 3), None)
        dv = _deform(dv, None, deformation, slice(2, 3), None)
    n = np.cross(du[..., :3], dv[..., :3])
    return _unit(n)


def _deform(d, dvec, deformation, target, src):
    """The statements the shader adds under DEFORMATION / DEFORMATION_2D: the
    displaced position gains deformation * value, so the derivative gains its
    derivative, in x (3d) or z (2d), plus deformation * d/du of the vector part."""
    out = np.array(d, copy=True)
    axis = 0 if src is not None else 2       # du.x in the 3d case, du.z in the 2d case
    out[..., axis] = d[..., axis] + deformation * d[..., 3]
    add = deformation * d[..., 3][..., None] if dvec is None else deformation * dvec[..., src]
    out[..., target] = d[..., target] + add
    return out


def curve_basis(order, t):
    """Degree-``order`` (1..3) Bernstein basis along a curve, in attribute order.

    Order 1 evaluates ``t * p0 + (1.0 - t) * p1`` (``splines.vert:28``), which is the
    reverse of the usual convention: an instance runs from ``p1`` to ``p0``, and the
    vector attributes at :29 blend the same way.  Orders 2 and 3 use the ordinary
    polynomials, ``:32-36`` and ``:40-46``.
    """
    order = _check_order(order)
    t = np.asarray(t)
    dtype = np.result_type(t, float)
    t = t.astype(dtype, copy=False)
    if order == 1:
        return np.stack([t, 1.0 - t], axis=-1)
    s = 1.0 - t
    if order == 2:
        terms = [s * s, 2.0 * t * s, t * t]
    else:
        terms = [s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t]
    return np.stack(terms, axis=-1)


def curve_vector_values(order, vectors, t):
    """The two packed vector components of a curve instance, same basis as its position."""
    return _weighted(curve_basis(order, t), np.asarray(vectors))


def curve(order, ctrl, t, *, vectors=None, deformation=0.0, mode="none",
          function_mode=0.0, complex_scale=(1.0, 0.0)):
    """Position and value of a curve instance at the parameter values ``t``.

    ``ctrl`` holds the vec4 control points ``(x, y, z, value)`` of the instance
    attributes ``p0..p3``, ``vectors`` the two components of ``v0..v3``, as
    ``three/edges.ts:114-140`` and ``three/mesh.ts:200-230`` fill them.  The return is
    ``(xyz, value)`` with ``value`` the program's ``value_``: the blended 4th component
    in ``x``, which ``splines.vert:54`` copies out of ``ps.w``, and the blended vectors
    in ``y`` and ``z``.  ``mode`` and ``function_mode`` select the displacement of
    :func:`_displace`.
    """
    order = _check_order(order)
    basis = curve_basis(order, t)
    pv = _weighted(basis, np.asarray(ctrl))
    xyz, scalar = pv[..., :3], pv[..., 3]
    if vectors is None:
        vv = np.zeros(pv.shape[:-1] + (2,), dtype=pv.dtype)
    else:
        vv = curve_vector_values(order, vectors, t)
    value3 = np.concatenate([scalar[..., None], vv], axis=-1)
    deformation = float(deformation)
    if mode == "none" or deformation == 0.0:
        return xyz, value3
    return _displace(xyz, value3, deformation, mode, function_mode, complex_scale), value3
