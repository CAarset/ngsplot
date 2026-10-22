"""Paint an NGSolve surface with pixels, the way the renderer's shaders do.

The renderer does not draw the Bézier patch smoothly: it cuts every element into a
lattice of ``n_segments * n_segments`` small triangles, evaluates the patch once per
lattice vertex, and lets the graphics hardware interpolate the results across each
sub-triangle (``trigsplines.vert`` writes ``p_``, ``value_`` and ``normal_`` as
varyings, ``function.frag`` reads them back per fragment).  This module does the same
in numpy, and adds what a hardware rasterizer adds: which pixels a sub-triangle
covers, which samples of a pixel it reaches and therefore what colour that pixel
resolves to, and which of two overlapping triangles wins.

Everything visual comes from the modules this one glues together -- :mod:`~ngsplot.bezier`
for the patch, :mod:`~ngsplot.camera` for the matrices, :mod:`~ngsplot.colormap` for
the banded colour, :mod:`~ngsplot.lighting` for the light.  Nothing here knows about
ngsolve or matplotlib.
"""

import numpy as np

from . import bezier, camera, colormap, lighting

#: lattice subdivisions per element edge (``n_segments`` in mesh.ts)
N_SEGMENTS = 5

#: the renderer clears to white, ``renderer.setClearColor(0xffffff)``
BACKGROUND = (1.0, 1.0, 1.0, 1.0)

#: grey of the ``NO_FUNCTION_VALUES`` fragment program, which ``draw_surf === false``
#: (and only that, ``mesh.ts:51``) compiles
MESH_GREY = 0.7


class Canvas:
    """A float64 RGBA framebuffer with a depth buffer, rows counted from the top.

    ``samples`` is the number of sample positions per pixel *per axis*.  The renderer
    asks WebGL for ``antialias: true``, which is coverage-sampled multisampling: each
    primitive is tested against a grid of samples per pixel, wins or loses the depth
    test per sample, and the samples are averaged when the framebuffer is resolved.
    A pixel covered by two neighbouring sub-triangles is therefore covered by their
    *union*, not blended twice, so the samples are kept in separate planes -- one
    colour and depth buffer per sample -- and averaged only on request.
    """

    def __init__(self, width, height, *, samples=2, background=BACKGROUND):
        width, height, samples = int(width), int(height), int(samples)
        if width < 1 or height < 1:
            raise ValueError("canvas needs a positive size")
        if samples < 1:
            raise ValueError("samples must be at least 1")
        self.width = width
        self.height = height
        self.samples = samples
        self.planes = np.empty((samples * samples, height, width, 4), dtype=np.float64)
        self.planes[...] = np.asarray(background, dtype=np.float64)
        self.depths = np.full((samples * samples, height, width), np.inf, dtype=np.float64)

    def offsets(self):
        """Sample positions within a pixel, relative to its centre, as ``(dx, dy)``.

        Plane ``iy * samples + ix`` of :attr:`planes` is the sample at the returned
        entry of the same index, so the whole grid is the regular one a GPU uses.
        """
        g = (np.arange(self.samples) + 0.5) / self.samples - 0.5
        return [(dx, dy) for dy in g for dx in g]

    @property
    def color(self):
        """The resolved image: the mean of the sample planes, pixel by pixel."""
        return self.planes.mean(axis=0)

    @property
    def depth(self):
        """The resolved depth, averaged over the samples a primitive reached."""
        finite = np.isfinite(self.depths)
        total = np.where(finite, self.depths, 0.0).sum(axis=0)
        count = finite.sum(axis=0)
        return np.where(count > 0, total / np.maximum(count, 1), np.inf)

    def to_uint8(self):
        """The resolved image as RGBA bytes, clamped and rounded like the framebuffer."""
        return np.rint(np.clip(self.color, 0.0, 1.0) * 255.0).astype(np.uint8)


def deformation_mode(mesh_dim, funcdim, is_complex=False):
    """Which deformation the shaders compile for: ``"3d"``, ``"2d"`` or ``"none"``.

    ``mesh.ts:13-14``: a vector function on a mesh of the same dimension displaces the
    surface in space (``DEFORMATION``), a scalar one on a surface mesh pushes it along
    ``z`` (``DEFORMATION_2D``).
    """
    mesh_dim, funcdim = int(mesh_dim), int(funcdim)
    if mesh_dim == funcdim and not bool(is_complex):
        return "3d"
    if mesh_dim == 2 and funcdim > 0:
        return "2d"
    return "none"


def lattice(order, ctrl, *, vectors=None, deformation=0.0, mode="none", function_mode=0.0,
            complex_scale=(1.0, 0.0), vertex_normals=None, n_segments=N_SEGMENTS):
    """Position, value and normal at every lattice vertex of the surface.

    ``ctrl`` holds the ``(ne, K, 4)`` control points of
    :func:`ngsplot.geometry.trig_points`; the result is three ``(ne, n_lattice, 3)``
    arrays: the *deformed* object-space position, the interpolated value slot together
    with the two vector components (``value_`` in the vertex program), and the
    object-space normal, which the vertex program takes at ``trigsplines.vert:23`` --
    before the deformation is added to the position, at 26 and 29.
    """
    order = int(order)
    ctrl = np.asarray(ctrl, dtype=np.float64)
    u, v = bezier.lattice(n_segments)
    points = ctrl[:, None]
    kw = dict(vectors=vectors, deformation=deformation, mode=mode,
              function_mode=function_mode, complex_scale=complex_scale)
    pos, w = bezier.evaluate(order, points, u, v, **kw)
    if vectors is None:
        value = np.zeros(pos.shape)
        value[..., 0] = w
    else:
        vv = bezier.vector_values(order, np.asarray(vectors, dtype=np.float64)[:, None], u, v)
        value = np.concatenate([w[..., None], vv], axis=-1)
    nrm = bezier.normal(order, points, u, v, vertex_normals=vertex_normals, **kw)
    return pos, value, nrm


def _fragment_colour(value, normal, pos, inside, *, function_mode, complex_scale,
                     function_values, vmin, vmax, pal, dark_backside):
    """``function.frag``: banded colour of the value, then ``calcLight``.

    Modes 4 and 7 take the value straight into the colour, unlit, exactly as the
    fragment program does; anything else colours by the banded value and lights it.
    """
    if function_mode == 4.0:
        return value
    if function_mode == 7.0:
        return np.concatenate([np.full(value.shape[:-1] + (1,), 2.0), value[..., :2]], axis=-1)
    scalar = bezier.get_value(value, function_mode, complex_scale)
    if function_values:
        rgb = colormap.lookup(scalar, vmin, vmax, pal)
    else:
        rgb = np.full(value.shape[:-1] + (3,), MESH_GREY)
    lit = lighting.shade(rgb.reshape(-1, 3), normal.reshape(-1, 3), pos.reshape(-1, 3),
                         inside=inside, dark_backside=dark_backside)
    return lit.reshape(rgb.shape)


def paint_surface(canvas, cam, order, ctrl, *, vmin, vmax, pal=None, n_segments=N_SEGMENTS,
                  vectors=None, deformation=0.0, mode="none", function_mode=0.0,
                  complex_scale=(1.0, 0.0), vertex_normals=None, function_values=True,
                  dark_backside=True, skip_face_check=False, less_equal=True):
    """Draw the surface of one function into ``canvas`` from camera ``cam``.

    ``vmin`` and ``vmax`` are the ``funcmin``/``funcmax`` of the render data, the range
    the renderer was told to colour by; ``function_mode`` and ``complex_scale`` say
    which part of the value that is (:func:`ngsplot.bezier.get_value`).
    ``function_values=False`` is ``NO_FUNCTION_VALUES``, the mesh drawn in plain grey.

    Back faces are dimmed by ``dark_backside``.  ``less_equal`` is the depth comparison,
    which three.js defaults to ``LessEqualDepth`` for every material: what makes the
    wireframe win over the surface it sits on is the surface's ``polygonOffset``, not a
    different comparison here.  Mode ``8``, which colours by depth for debugging, is not
    implemented.
    """
    function_mode = float(function_mode)
    if function_mode == 8.0:
        raise NotImplementedError("function_mode 8 colours by depth; not implemented")
    order = int(order)
    pos, value, nrm = lattice(order, ctrl, vectors=vectors, deformation=deformation,
                              mode=mode, function_mode=function_mode,
                              complex_scale=complex_scale, vertex_normals=vertex_normals,
                              n_segments=n_segments)

    mv = cam.model_view
    ne = value.shape[0]
    points = pos.reshape(-1, 3)
    px, py, depth, w = camera.to_pixels(
        camera.clip_points(points, cam.model_view_projection), canvas.width, canvas.height)
    # calcLight's vPos_ is modelViewMatrix * position (utils.h:422), the object point as
    # this camera sees it -- not viewMatrix * position, which is the other matrix here and
    # which the specular term punishes as soon as the object matrix moves the surface.
    p_view = camera.transform_points(points, mv)
    # the matching normalMatrix: the inverse transpose of that same 3x3
    n_view = np.einsum("ij,...j->...i", np.linalg.inv(mv[:3, :3]).T, nrm.reshape(-1, 3))

    shape = (ne,) + value.shape[1:-1]
    px, py, depth, w = (a.reshape(shape) for a in (px, py, depth, w))
    p_view, n_view = (a.reshape(value.shape) for a in (p_view, n_view))
    usable = (np.isfinite(px) & np.isfinite(py) & np.isfinite(depth) & np.isfinite(w) & (w > 0.0)
              & ~np.any(np.isnan(value), axis=-1))

    for e in range(ne):
        for tri in bezier.subtriangles(n_segments):
            if not np.all(usable[e, tri]):
                continue
            _paint_triangle(
                canvas, px[e, tri], py[e, tri], w[e, tri], depth[e, tri], p_view[e, tri],
                n_view[e, tri], value[e, tri], vmin=vmin, vmax=vmax, pal=pal,
                function_mode=function_mode, complex_scale=complex_scale,
                function_values=function_values, dark_backside=dark_backside,
                skip_face_check=skip_face_check, less_equal=less_equal)
    return canvas


def _paint_triangle(canvas, x, y, w, z, p_view, n_view, value, *, vmin, vmax, pal,
                    function_mode, complex_scale, function_values, dark_backside,
                    skip_face_check, less_equal):
    """Rasterize one sub-triangle: coverage, perspective-correct varyings, colour, light.

    ``x`` and ``y`` are pixel coordinates with the first row on top, ``w`` the clip
    weight, ``z`` the window depth, and the three remaining arrays hold the ``(3, 3)``
    vertex values the vertex program would have written.
    """
    den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
    if den == 0.0 or not np.isfinite(den):
        return

    # gl_FrontFacing counts counter-clockwise in window space, whose y grows upwards, so
    # clockwise -- a positive cross product here -- in the y-down pixel coordinates.
    # function.frag negates the normal for every back face and only calls it "inside",
    # which is what dims it, unless the program was compiled with SKIP_FACE_CHECK.
    back = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]) > 0.0
    inside = back and not skip_face_check

    n = canvas.samples
    reach = 0.5 + 0.5 / n
    j0 = max(int(np.floor(min(x) - reach)), 0)
    j1 = min(int(np.ceil(max(x) + reach)), canvas.width)
    i0 = max(int(np.floor(min(y) - reach)), 0)
    i1 = min(int(np.ceil(max(y) + reach)), canvas.height)
    if j0 >= j1 or i0 >= i1:
        return

    # Barycentric coordinates are affine on the screen, so a constant plus two slopes
    # over the bounding box; vertex 2 is the origin, where l2 is 1.
    den_inv = 1.0 / den
    b0, c0 = (y[1] - y[2]) * den_inv, (x[2] - x[1]) * den_inv
    b1, c1 = (y[2] - y[0]) * den_inv, (x[0] - x[2]) * den_inv
    gx = np.arange(j0, j1) + 0.5 - x[2]
    gy = np.arange(i0, i1) + 0.5 - y[2]
    l0 = (b0 * gx)[None, :] + (c0 * gy)[:, None]
    l1 = (b1 * gx)[None, :] + (c1 * gy)[:, None]
    l2 = 1.0 - l0 - l1

    # 1/w is affine in screen space, and a varying is the affine blend of v/w divided by
    # the same blend of 1/w -- perspective-correct interpolation, as the hardware does it.
    # z is already the window depth from camera.to_pixels, so it needs no further mapping.
    iw = 1.0 / w
    zn = z * iw
    wc = 1.0 / (l0 * iw[0] + l1 * iw[1] + l2 * iw[2])

    def interp(field):
        out = (l0 * iw[0])[:, :, None] * field[0] + (l1 * iw[1])[:, :, None] * field[1]
        return (out + (l2 * iw[2])[:, :, None] * field[2]) * wc[:, :, None]

    # colour and light come from the varyings at the pixel centre, once for the whole
    # pixel and not once per sample: a multisampled framebuffer runs the fragment program
    # once per pixel, at gl_FragCoord in its centre, and lets the sample mask say which
    # samples the result lands in.  Antialiasing therefore mixes coverage, not colour --
    # measured against the recorded picture, evaluating per sample gives a mean error
    # twelve times larger, and every pixel where the two disagree sits on a colour band.
    normal = interp(n_view)
    if back:
        normal = -normal
    rgb = _fragment_colour(interp(value), normal, interp(p_view), inside,
                           function_mode=function_mode, complex_scale=complex_scale,
                           function_values=function_values, vmin=vmin, vmax=vmax,
                           pal=pal, dark_backside=dark_backside)

    rows, cols = np.indices(l0.shape) + np.array([i0, j0])[:, None, None]
    for plane, (dx, dy) in enumerate(canvas.offsets()):
        # one sample plane at a time: the coverage of a pixel is the union over the
        # primitives that reach it, which is what averaging the planes gives
        s0 = l0 + b0 * dx + c0 * dy
        s1 = l1 + b1 * dx + c1 * dy
        s2 = 1.0 - s0 - s1
        depth = (s0 * zn[0] + s1 * zn[1] + s2 * zn[2]) / (s0 * iw[0] + s1 * iw[1]
                                                          + s2 * iw[2])
        keep = (s0 >= 0.0) & (s1 >= 0.0) & (s2 >= 0.0)
        old = canvas.depths[plane, i0:i1, j0:j1]
        hit = keep & (depth <= old if less_equal else depth < old)
        _write(canvas, plane, rows[hit], cols[hit], rgb[hit], depth[hit])
    return canvas


def _write(canvas, plane, rows, cols, rgb, depth):
    """Write the samples a sub-triangle wins into one sample plane.

    Opaque, as the surface is drawn: a sample the triangle covers takes the fragment
    colour and its depth outright, and the depth test of the next primitive sees it.
    """
    canvas.planes[plane, rows, cols, :3] = rgb
    canvas.planes[plane, rows, cols, 3] = 1.0
    canvas.depths[plane, rows, cols] = depth
