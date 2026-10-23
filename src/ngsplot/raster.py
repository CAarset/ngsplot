"""Paint an NGSolve picture with pixels, the way the renderer's shaders do.

The renderer does not draw the Bézier patch smoothly: it cuts every element into a
lattice of ``n_segments * n_segments`` small triangles, evaluates the patch once per
lattice vertex, and lets the graphics hardware interpolate the results across each
sub-triangle (``trigsplines.vert`` writes ``p_``, ``value_`` and ``normal_`` as
varyings, ``function.frag`` reads them back per fragment).  This module does the same
in numpy, and adds what a hardware rasterizer adds: which pixels a sub-triangle
covers, which samples of a pixel it reaches and therefore what colour that pixel
resolves to, and which of two overlapping triangles wins.

The surface is only part of the picture.  The mesh wireframe and the boundary edges are
two further objects with their own vertex program (``splines.vert``) and fragment
program (``splines.frag``), where a curve is not a thin triangle at all: the vertex
program spreads each of its samples sideways into a ribbon whose width is a uniform, and
the fragment program paints it flat black.  :func:`paint_wireframe` and
:func:`paint_edges` reproduce the two, one pixel-for-pixel difference apart -- the first
is a line, the second a ribbon, which is why they come out a pixel and 2.5 pixels wide.

Everything visual comes from the modules this one glues together -- :mod:`~ngsplot.bezier`
for the patch, :mod:`~ngsplot.camera` for the matrices, :mod:`~ngsplot.colormap` for
the banded colour, :mod:`~ngsplot.lighting` for the light.  Nothing here knows about
ngsolve or matplotlib.
"""

import numpy as np

from . import bezier, camera, colormap, lighting

#: lattice subdivisions per element edge (``n_segments`` in mesh.ts)
N_SEGMENTS = 5

#: the ``line_thickness`` slider as shipped, ``gui.ts:98``, which ``scene.ts:776``
#: divides by the height of the canvas to get the uniform the vertex program offsets by
LINE_THICKNESS = 5.0

#: half a device pixel: ``THREE.Line`` has no thickness of its own, so the rasterizer
#: that draws it -- ``three/mesh.ts:160-240`` compiles no ``THICK_LINES`` -- covers one
LINE_WIDTH = 1.0

#: the renderer clears to white, ``renderer.setClearColor(0xffffff)``
BACKGROUND = (1.0, 1.0, 1.0, 1.0)

#: grey of the ``NO_FUNCTION_VALUES`` fragment program, which ``draw_surf === false``
#: (and only that, ``mesh.ts:51``) compiles
MESH_GREY = 0.7

#: The four sample positions of the multisampled framebuffer, relative to the centre of
#: the pixel.  A desktop GL answers ``antialias: true`` with 4x coverage multisampling on
#: the *rotated* grid, not the
#: regular one, and the recorded pictures say so: an interior wireframe line of
#: ``square_scalar.png`` projects to window x = 576.209, and the two columns it straddles
#: resolve to 0.25 and 0.75 (``64`` and ``192`` over the surface, measured as
#: ``1 - picture/surface``).  A regular 2x2 grid splits any vertical line 0.5/0.5 or
#: 1/0, so it is out; the rotated grid puts its samples at +-0.125 and +-0.375 and lands
#: one inside the 1-pixel-wide strip and three in the next column, for exactly the pair
#: the picture shows, over a window of +-0.12 px around the projected position.
MSAA_4X = ((-0.375, -0.125), (-0.125, -0.375), (0.125, 0.375), (0.375, 0.125))

#: the two window depths closest together that a depth buffer can tell apart -- the ``r``
#: of the polygon-offset formula.  ``gl.polygonOffset`` measures the offset in units of it,
#: and a WebGL2 context is required to offer a 24-bit depth buffer (WebGL 2.0 spec,
#: "Required Texture Formats", ``DEPTH_COMPONENT24``), which on ``[0, 1]`` window depth is
#: ``2 ** -24``.  The recorded pictures cannot pin ``r`` itself: any offset between about
#: ``1e-8`` and ``1e-5`` gives the same pixels, since a coplanar line and the surface it
#: lies in differ by no more than ``1.2e-16`` there, and any real separation between two
#: surfaces is orders of magnitude larger.
DEPTH_RESOLUTION = 2.0 ** -24

#: ``polygonOffsetFactor`` and ``polygonOffsetUnits`` of the surface material, both 1
#: (``three/mesh.ts:65-67``) -- the only polygon offset the renderer asks for
POLYGON_OFFSET_FACTOR = 1.0
POLYGON_OFFSET_UNITS = 1.0


def polygon_offset(x, y, z, *, factor=POLYGON_OFFSET_FACTOR, units=POLYGON_OFFSET_UNITS):
    """The window depth ``gl.polygonOffset`` adds to the fragments of a triangle.

    ``x``, ``y`` and ``z`` are the three corners of the triangle in window space.  The
    offset is ``factor * m + units * r`` with ``r`` the :data:`DEPTH_RESOLUTION` and ``m``
    the greatest depth slope the polygon reaches on the screen, which the rasterizer is
    required to treat as a property of the whole primitive: window depth is affine across
    a triangle, so ``m`` is the larger of the two partial derivatives of the plane through
    the corners, and adding the result to the corners is the same as adding it to every
    fragment.  A triangle with no area on the screen has no slope; nothing is drawn from
    it either way.
    """
    dx1, dy1, dz1 = x[0] - x[2], y[0] - y[2], z[0] - z[2]
    dx2, dy2, dz2 = x[1] - x[2], y[1] - y[2], z[1] - z[2]
    det = dx1 * dy2 - dx2 * dy1
    if det == 0.0 or not np.isfinite(det):
        slope = 0.0
    else:
        slope = max(abs((dz1 * dy2 - dz2 * dy1) / det), abs((dx1 * dz2 - dx2 * dz1) / det))
    return factor * slope + units * DEPTH_RESOLUTION


def _polygon_offsets(x, y, z, *, factor=POLYGON_OFFSET_FACTOR, units=POLYGON_OFFSET_UNITS):
    """:func:`polygon_offset` for a stack of ``(n, 3)`` window-space triangles at once.

    Row for row the same arithmetic as the one above: the same differences, the same
    determinant, the same two quotients in the same order.  A triangle of no screen area, or
    of a determinant that is no number to order against, gets no slope; a slope that comes
    out of an overflowed depth comes out of here as no number too, which is what the one
    above returns and what puts the whole triangle out of the depth test.
    """
    dx1, dy1, dz1 = x[:, 0] - x[:, 2], y[:, 0] - y[:, 2], z[:, 0] - z[:, 2]
    dx2, dy2, dz2 = x[:, 1] - x[:, 2], y[:, 1] - y[:, 2], z[:, 1] - z[:, 2]
    det = dx1 * dy2 - dx2 * dy1
    with np.errstate(divide="ignore", invalid="ignore"):
        dz_dy = (dz1 * dy2 - dz2 * dy1) / det
        dz_dx = (dx1 * dz2 - dx2 * dz1) / det
    slope = np.where(np.isfinite(det) & (det != 0.0), np.maximum(abs(dz_dy), abs(dz_dx)), 0.0)
    return factor * slope + units * DEPTH_RESOLUTION


def sample_offsets(samples):
    """Normalise the ``samples`` of a :class:`Canvas` to a list of ``(dx, dy)`` offsets.

    An integer is the number of positions per axis of the regular grid, so ``1`` means
    one sample at the pixel centre and no antialiasing at all; anything else must be a
    non-empty sequence of offsets, each inside the pixel.
    """
    if isinstance(samples, (int, np.integer)):
        n = int(samples)
        if n < 1:
            raise ValueError("samples must be at least 1")
        g = (np.arange(n) + 0.5) / n - 0.5
        return [(float(dx), float(dy)) for dy in g for dx in g]
    grid = np.asarray(samples, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] != 2 or len(grid) == 0:
        raise ValueError("samples is a count per axis or a list of (dx, dy) offsets")
    if np.abs(grid).max() > 0.5:
        raise ValueError("sample offsets must lie inside the pixel")
    return [tuple(row) for row in grid]


class Canvas:
    """A float64 RGBA framebuffer with a depth buffer, rows counted from the top.

    The renderer asks WebGL for ``antialias: true``, which is coverage-sampled
    multisampling: each primitive is tested against a fixed set of sample positions per
    pixel, wins or loses the depth test per sample, and the samples are averaged when
    the framebuffer is resolved.  A pixel covered by two neighbouring sub-triangles is
    therefore covered by their *union*, not blended twice, so the samples are kept in
    separate planes -- one colour and depth buffer per sample -- and averaged only on
    request.

    ``samples`` says which sample positions to use: an integer is the number per axis of
    the regular grid -- ``1`` switches antialiasing off -- and a sequence of ``(dx, dy)``
    pairs is used as given.  The default is :data:`MSAA_4X`, the pattern the recorded
    pictures were rendered with.
    """

    def __init__(self, width, height, *, samples=MSAA_4X, background=BACKGROUND):
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError("canvas needs a positive size")
        offsets = sample_offsets(samples)
        self.width = width
        self.height = height
        self.samples = len(offsets)
        self.sample_offsets = offsets
        #: how far outside a primitive's geometry a sample can still sit, in pixels
        self.reach = 0.5 + max(max(abs(dx), abs(dy)) for dx, dy in offsets)
        self.planes = np.empty((self.samples, height, width, 4), dtype=np.float64)
        self.planes[...] = np.asarray(background, dtype=np.float64)
        self.depths = np.full((self.samples, height, width), np.inf, dtype=np.float64)

    def offsets(self):
        """Sample positions within a pixel, relative to its centre, as ``(dx, dy)``.

        Plane ``k`` of :attr:`planes` is the sample at the returned entry of index ``k``.
        """
        return self.sample_offsets

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


#: how much canvas one primitive of a shaded stack has to be worth before the per-primitive
#: sweep of :func:`_paint_triangle` is handed the stack instead of the tiles of
#: :func:`_paint_stack`.  The batch pays for every tile a box touches, which for a triangle
#: wider than a tile is several times the area of the triangle; the loop pays a dozen numpy calls
#: per primitive and for no pixel twice.  The boxes of a mesh that fills the frame add up to
#: about one canvas whatever its fineness, so the batch's cost is set by the picture and the
#: loop's by the number of primitives in it, and the two cross at a fixed number of canvas
#: pixels per primitive.  Measured in one process, best of three, on :mod:`tools.bench_paint`
#: meshes at 900x600 and on the recorded pictures: the loop is 1.08x behind at 415 pixels per
#: primitive (1300 of a square at maxh 0.2) and 1.19x ahead at 600 (900 of a cube at maxh 0.5),
#: and on ``mesh_only`` -- 52 triangles, 10212 pixels each -- it is twice ahead of a batch that
#: sweeps each of them over sixteen tiles.
_STACK_LOOP_AREA = 512


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
    wireframe win over the surface it sits on is the polygon offset of
    :func:`polygon_offset`, which the surface material alone asks for and which is applied
    to every sub-triangle here, not a different comparison.  Mode ``8``, which colours by
    depth for debugging, is not implemented.

    The sheet goes to :func:`_paint_stack` unless it is too coarse for a tile sweep to pay for
    itself, in which case :func:`_paint_triangle` draws it a triangle at a time; the two draw the
    same picture, which is what :data:`_STACK_LOOP_AREA` is free to choose between them on.
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

    # The whole sheet is one stack, in the order an instanced mesh draws it: instance by
    # instance, and within an instance in the order subtriangles returns.  That order is not
    # a detail -- it is what settles every tie between two sub-triangles that meet on an edge,
    # which for a surface this fine is most of the ink on the sheet.
    tris = bezier.subtriangles(n_segments)
    e, ti = np.nonzero(np.all(usable[:, tris], axis=-1))
    corner = tris[ti]                                    # the three lattice corners of each
    e = e[:, None]
    x, y, z, w = (a[e, corner] for a in (px, py, depth, w))
    p_view, n_view, val = (a[e, corner] for a in (p_view, n_view, value))

    # gl_FrontFacing counts counter-clockwise in window space, whose y grows upwards, so
    # clockwise -- a positive cross product here -- in the y-down pixel coordinates.
    # function.frag negates the normal for every back face and only calls it "inside", which
    # is what dims it, unless the program was compiled with SKIP_FACE_CHECK.
    back = ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
            - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]) > 0.0)
    face_check = not skip_face_check

    def shade(fields, constants):
        # colour and light come from the varyings at the pixel centre, once for the whole
        # pixel and not once per sample, as _paint_triangle explains
        val, pos, normal = fields
        (back,) = constants
        return _fragment_colour(val, np.where(back[:, None], -normal, normal), pos,
                                back & face_check, function_mode=function_mode,
                                complex_scale=complex_scale, function_values=function_values,
                                vmin=vmin, vmax=vmax, pal=pal, dark_backside=dark_backside)

    if x.shape[0] * _STACK_LOOP_AREA < canvas.width * canvas.height:
        # A stack too coarse to amortise a tile sweep: one triangle at a time, which is what
        # :func:`_paint_triangle` is, and the reason it is kept.  The order is the stack's, so a
        # pixel two triangles both want is tested twice instead of sorted once -- against what
        # the earlier one left, which gives the later one the tie under LessEqualDepth and the
        # earlier one under LessDepth, the same two outcomes _resolve picks from a stack.
        for here in range(x.shape[0]):
            _paint_triangle(canvas, x[here], y[here], w[here], z[here],
                            p_view[here], n_view[here], val[here], vmin=vmin, vmax=vmax, pal=pal,
                            function_mode=function_mode, complex_scale=complex_scale,
                            function_values=function_values, dark_backside=dark_backside,
                            skip_face_check=skip_face_check, less_equal=less_equal)
        return canvas

    # every fragment of the surface is pushed away from the camera by a polygon offset, which
    # the surface material is the only one to ask for (three/mesh.ts:65-67).  It is what makes
    # the wireframe, which is generated in the surface itself, pass the depth test: measured
    # on square_scalar.png, a wireframe line and the surface under it differ in window depth
    # by at most 1.1102230246251565e-16, so the comparison is decided by rounding noise, which
    # throws away nearly a quarter of the ink -- 3554 units of it counted inside one box of
    # the surface against the 4405 of the recording.
    _paint_stack(canvas, x, y, w, z, varyings=(val, p_view, n_view), attrs=(back,),
                 shade=shade, less_equal=less_equal, offset=True)
    return canvas


def _paint_triangle(canvas, x, y, w, z, p_view, n_view, value, *, vmin, vmax, pal,
                    function_mode, complex_scale, function_values, dark_backside,
                    skip_face_check, less_equal):
    """Rasterize one sub-triangle: coverage, perspective-correct varyings, colour, light.

    ``x`` and ``y`` are pixel coordinates with the first row on top, ``w`` the clip
    weight, ``z`` the window depth, and the three remaining arrays hold the ``(3, 3)``
    vertex values the vertex program would have written.
    """
    # gl_FrontFacing counts counter-clockwise in window space, whose y grows upwards, so
    # clockwise -- a positive cross product here -- in the y-down pixel coordinates.
    # function.frag negates the normal for every back face and only calls it "inside",
    # which is what dims it, unless the program was compiled with SKIP_FACE_CHECK.
    back = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]) > 0.0
    inside = back and not skip_face_check
    # every fragment of the surface is pushed away from the camera by a polygon offset,
    # which the surface material is the only one to ask for (three/mesh.ts:65-67).  It is
    # what makes the wireframe, which is generated in the surface itself, pass the depth
    # test: measured on square_scalar.png, a wireframe line and the surface under it differ
    # in window depth by at most 1.1102230246251565e-16, so the comparison is decided by
    # rounding noise, which throws away nearly a quarter of the ink -- 3554 units of it
    # counted inside one box of the surface against the 4405 of the recording.
    z = z + polygon_offset(x, y, z)

    def shade(interp):
        # colour and light come from the varyings at the pixel centre, once for the whole
        # pixel and not once per sample: a multisampled framebuffer runs the fragment
        # program once per pixel, at gl_FragCoord in its centre, and lets the sample mask
        # say which samples the result lands in.  Antialiasing therefore mixes coverage,
        # not colour -- measured against the recorded picture, evaluating per sample gives
        # a mean error twelve times larger, and every pixel where the two disagree sits on
        # a colour band.
        normal = interp(n_view)
        if back:
            normal = -normal
        return _fragment_colour(interp(value), normal, interp(p_view), inside,
                                function_mode=function_mode, complex_scale=complex_scale,
                                function_values=function_values, vmin=vmin, vmax=vmax,
                                pal=pal, dark_backside=dark_backside)

    for plane, rows, cols, hit, depth, rgb in _sweep(canvas, x, y, w, z,
                                                     less_equal=less_equal, shade=shade):
        # one sample plane at a time: the coverage of a pixel is the union over the
        # primitives that reach it, which is what averaging the planes gives
        _write(canvas, plane, rows[hit], cols[hit], rgb[hit], depth[hit])
    return canvas


def _sweep(canvas, x, y, w, z, *, less_equal=True, shade=None):
    """Walk a triangle in window space over the sample planes of ``canvas``.

    ``x`` and ``y`` are the pixel coordinates of the three vertices with the first row on
    top, ``w`` their clip weight and ``z`` their window depth.  One yield per sample
    plane, giving ``(plane, rows, cols, hit, depth, rgb)`` over the bounding box of the
    triangle: the absolute index of every pixel in it, the samples this plane still wins
    here, the depth of those samples, and the colour -- which ``shade``, if given, is
    asked for once, with the interpolator of the varyings as its argument.  A triangle
    that is degenerate or wholly off the canvas yields nothing, which is how both are
    culled.
    """
    den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
    if den == 0.0 or not np.isfinite(den):
        return

    reach = canvas.reach
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

    rgb = None if shade is None else shade(interp)

    rows, cols = np.indices(l0.shape) + np.array([i0, j0])[:, None, None]
    for plane, (dx, dy) in enumerate(canvas.offsets()):
        s0 = l0 + b0 * dx + c0 * dy
        s1 = l1 + b1 * dx + c1 * dy
        s2 = 1.0 - s0 - s1
        depth = (s0 * zn[0] + s1 * zn[1] + s2 * zn[2]) / (s0 * iw[0] + s1 * iw[1]
                                                          + s2 * iw[2])
        keep = (s0 >= 0.0) & (s1 >= 0.0) & (s2 >= 0.0)
        old = canvas.depths[plane, i0:i1, j0:j1]
        hit = keep & (depth <= old if less_equal else depth < old)
        yield plane, rows, cols, hit, depth, rgb


def _paint_flat(canvas, x, y, w, z, rgb, *, less_equal=True):
    """Paint one triangle of window-space geometry in a single colour, depth-tested.

    This is what a line is made of here: the vertex program has already turned the curve
    into a ribbon of triangles in clip space, so the fragment program has no varyings
    left to interpolate and no light to apply -- only coverage and depth.  Nothing is
    culled for its winding, either: ``splines.frag`` never looks at ``gl_FrontFacing``,
    and the ribbon is built the same way wherever the curve turns.
    """
    for plane, rows, cols, hit, depth, _ in _sweep(canvas, x, y, w, z, less_equal=less_equal):
        _write(canvas, plane, rows[hit], cols[hit], rgb, depth[hit])
    return canvas


def _write(canvas, plane, rows, cols, rgb, depth):
    """Write the samples a sub-triangle wins into one sample plane.

    Opaque, as the surface is drawn: a sample the triangle covers takes the fragment
    colour and its depth outright, and the depth test of the next primitive sees it.
    """
    canvas.planes[plane, rows, cols, :3] = rgb
    canvas.planes[plane, rows, cols, 3] = 1.0
    canvas.depths[plane, rows, cols] = depth


# --------------------------------------------- the stack, binned into tiles and windows

#: how many samples one chunk -- the primitives of one tile, or of one window, swept together
#: -- is walked over.  A chunk is a single array of ``(primitives, rows, columns, sample
#: planes)``, walked over a dozen times by the barycentrics, the depth and the depth test, so
#: what matters is that one of its temporaries is small enough that the next one finds it in
#: cache: a megabyte each, which on the meshes of :mod:`tools.bench_paint` is the fastest of
#: the sizes tried, by a few percent over either a sixteenth or a sixteenth of a gigabyte.
_STACK_SAMPLES = 1 << 16

#: the tile sides :func:`_stack_tiles` will use, in pixels.  The floor is what keeps the
#: tiles -- and with them the numpy calls -- few on a fine mesh, where a triangle of a few
#: pixels is swept over its tile either way.  The ceiling is what keeps a triangle whose box
#: spans the frame, which a volume shows plenty of, from being swept once over all of it.
_STACK_TILE_MIN, _STACK_TILE_MAX = 4, 32

#: the largest window :func:`_flat_windows` sweeps a primitive over at a time, in pixels, and
#: its number of bits.  A box coarser than that is swept as several windows, which is what
#: keeps the array of a chunk to a few tens of megabytes however far a ribbon reaches.  A
#: power of two, so that a window is a rounding of the sides of a box and not a search.
_STACK_WINDOW = 64
_STACK_WINDOW_BITS = _STACK_WINDOW.bit_length() - 1

#: how many winning samples :func:`_paint_stack` collects before it colours and writes
#: them; the colouring of the surface is a lighting call, which wants rows to chew on.
_STACK_WINNERS = 1 << 15


def _resolve(keys, depth, index, less_equal):
    """Which of the candidates for one sample the depth test keeps.

    ``keys`` names the sample every candidate wants to write, ``depth`` is its window
    depth and ``index`` its place in the stack, the three of them the same length.  The
    test keeps the nearer fragment, and on an exact depth it keeps the fragment written
    last where the test is ``LessEqualDepth`` -- what the renderer sets,
    ``three/mesh.ts:57`` -- and the one written first where it is ``LessDepth``.  One
    sort does both: order by sample, then by depth, then by place in the stack, and take
    the first candidate of every run of a sample.  Nothing with a depth that cannot be
    ordered against another reaches here, the comparison having already failed it.
    """
    order = np.lexsort((-index if less_equal else index, depth, keys))
    ranked = keys[order]
    first = np.r_[True, ranked[1:] != ranked[:-1]]
    return order[first]


def _stack_tiles(canvas, i0, i1, j0, j1):
    """Yield ``(rows, columns, primitives)`` for the tiles a stack of boxes overlaps.

    ``i0``, ``i1``, ``j0`` and ``j1`` are the pixel rows and columns of each primitive's
    bounding box, clipped to the canvas and half-open.  A primitive is listed in every
    tile its box touches, and that duplication is what makes the sweep vectorizable: the
    primitives of one tile are few enough to be swept as a single array over the tile's
    own rows and columns, and a tile's samples belong to that tile alone, so tiles neither
    conflict nor depend on one another and may be swept in any order.

    The side of a tile is the power-of-two size class the box fits into, so a primitive
    reaches at most two tiles per axis and the box one array sweeps is at most twice the
    primitive's own per axis -- four times the arithmetic at its worst, against the
    per-primitive call it would otherwise cost.  Boxes finer than that get the floor of
    :data:`_STACK_TILE_MIN`, which buys fewer tiles with a wider box, and a box as coarse as
    the canvas the cap, which is the one case where a primitive reaches more than two tiles
    per axis: the duplication of a triangle that spans the frame is cheaper than sweeping the
    frame over it.  Tiles come out in a fixed order and their primitives in stack order,
    which is the order depth ties go in.
    """
    side = np.maximum(i1 - i0, j1 - j0)
    tile = np.clip(np.left_shift(1, np.ceil(np.log2(side)).astype(np.int64)),
                   _STACK_TILE_MIN, _STACK_TILE_MAX)
    tiles, group = np.unique(tile, return_inverse=True)
    for number, side in enumerate(tiles):
        (part,) = np.nonzero(group == number)
        ci0, cj0 = i0[part] // side, j0[part] // side
        span_i = (i1[part] - 1) // side - ci0 + 1
        span_j = (j1[part] - 1) // side - cj0 + 1
        ncol = (canvas.width - 1) // side + 1
        cells, prims = [], []
        for di in range(int(span_i.max())):
            for dj in range(int(span_j.max())):
                (hit,) = np.nonzero((di < span_i) & (dj < span_j))
                if hit.size:
                    cells.append((ci0[hit] + di) * ncol + cj0[hit] + dj)
                    prims.append(part[hit])
        cells = np.concatenate(cells)
        prims = np.concatenate(prims)
        # by tile, and within it by stack place: the order a sequential painter would use
        order = np.lexsort((prims, cells))
        cells, prims = cells[order], prims[order]
        starts = np.r_[0, np.flatnonzero(cells[1:] != cells[:-1]) + 1]
        for a, b in zip(starts, np.r_[starts[1:], cells.size]):
            cell = int(cells[a])
            ra, ca = (cell // ncol) * side, (cell % ncol) * side
            yield ((ra, min(ra + side, canvas.height)),
                   (ca, min(ca + side, canvas.width)), prims[a:b])


def _flat_windows(canvas, i0, i1, j0, j1):
    """Yield ``(rows, columns, window rows, window columns, primitives)``, a class at a time.

    The four box arrays of :func:`_stack_tiles`, and the same job: hand the sweep the
    primitives that share a rectangle of the canvas to walk, so that one array walks them.
    The rectangle is a window of the primitive's own box here rather than a tile of the
    canvas, and the two sides of a window are rounded up separately, so that a ribbon one
    pixel long is swept over a pixel and not over the square of the coarsest box of the call:
    a ribbon of a pixel swept over the :data:`_STACK_TILE_MIN` floor of a tile asks for two
    hundred pixels of arithmetic for the one it paints, which is the whole cost of a wireframe.

    Windows of one shape are yielded together, which is what lets their arrays share a
    length.  A box coarser than :data:`_STACK_WINDOW` is handed out as several windows, and
    the last window of an axis is pulled back to the edge of the canvas so that every window
    lies inside the canvas: the sweep of a window asks no pixel whether it is off it.  Where
    a window is pulled back onto its neighbour a pixel of the box is swept twice, and the two
    answers are the same bits, so it is :func:`_resolve` that is left to pick one.

    That a box may be cut up, and that the windows of a class come out in no particular
    order, is owed to the stack being a flat one of a single colour: whichever of two
    primitives wins a pixel, the pixel takes the colour of the stack, and only a depth could
    tell them apart.  :func:`_stack_tiles` hands the primitives of a tile in stack order
    because there two of them can tie over one pixel and colour it differently.
    """
    cap_i = min(_STACK_WINDOW_BITS, int(np.log2(canvas.height)))
    cap_j = min(_STACK_WINDOW_BITS, int(np.log2(canvas.width)))
    bits_i = np.clip(np.ceil(np.log2(np.maximum(i1 - i0, 1))).astype(np.int64), 0, cap_i)
    bits_j = np.clip(np.ceil(np.log2(np.maximum(j1 - j0, 1))).astype(np.int64), 0, cap_j)
    key = (bits_i << 8) | bits_j
    order = np.argsort(key, kind="stable")
    keys = key[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    for a, b in zip(starts, np.r_[starts[1:], keys.size]):
        part, number = order[a:b], int(keys[a])
        h, wd = 1 << (number >> 8), 1 << (number & 255)
        span_i = -((i1[part] - i0[part]) // -h)
        span_j = -((j1[part] - j0[part]) // -wd)
        rows = np.minimum(i0[part][:, None] + h * np.arange(span_i.max()),
                          canvas.height - h)
        columns = np.minimum(j0[part][:, None] + wd * np.arange(span_j.max()),
                             canvas.width - wd)
        grid = ((np.arange(rows.shape[1])[None, :, None] < span_i[:, None, None])
                & (np.arange(columns.shape[1])[None, None, :] < span_j[:, None, None]))
        k, u, v = np.nonzero(grid)
        yield h, wd, rows[k, u], columns[k, v], part[k]


def _interpolate(field, l0, l1, l2, wc, iw):
    """The value of a varying at the pixel centre of every row of a list of winners.

    ``field`` is ``(n, 3, k)``, the three corners' values of one varying for each of the
    primitives that won; ``l0``, ``l1``, ``l2`` and ``wc`` are the barycentrics and the
    share of ``1/w`` of the pixel centre it won, and ``iw`` the corners' ``1/w``.  This is
    :func:`_sweep`'s ``interp`` with the primitive as the row instead of the pixel, term for
    term, so that a colour comes out of it the same as out of the fragment program.
    """
    out = (l0 * iw[:, 0])[:, None] * field[:, 0] + (l1 * iw[:, 1])[:, None] * field[:, 1]
    return (out + (l2 * iw[:, 2])[:, None] * field[:, 2]) * wc[:, None]


def _paint_stack(canvas, x, y, w, z, *, rgb=None, varyings=(), attrs=(), shade=None,
                 less_equal=True, offset=False):
    """Sweep a whole stack of window-space triangles over the canvas, a batch at a time.

    ``x``, ``y``, ``w`` and ``z`` are ``(n, 3)``: row ``k`` holds primitive ``k``, in the
    order it is drawn, and its three corners' pixel columns, pixel rows, clip weights and
    window depths.  The depth test is the ``less_equal`` of :func:`_sweep`.

    A flat stack paints in the colour ``rgb``, one for all of it or one per primitive.  A
    shaded stack paints with ``shade``, which is given the ``varyings`` and the ``attrs`` of
    the primitives that won the pixels and answers ``(n, 3)`` of colour: ``varyings`` are
    ``(n, 3, k)`` per corner, interpolated perspective-correctly, and ``attrs`` ``(n, k)``
    per primitive, gathered.  One of the two, and only one, says which stack this is.

    This is :func:`_paint_flat` and :func:`_paint_triangle` for a great many primitives at
    once, pixel for pixel.  The arithmetic of :func:`_sweep` is reproduced term for term so
    that the depths come out the same bits and a tie lands where the sequential painter puts
    it; what changes is only *when* it is evaluated: over one rectangle of the canvas for all
    the primitives that reach it, instead of over one primitive for the whole canvas.  A sample
    lies in one tile only, and every primitive covering it is listed in that tile, so
    resolving the stack per tile resolves it; the primitives of a tile are then ordered by
    depth and stack place, which is what the depth test would have decided write by write.

    A stack of one colour for all of it is binned by the boxes of its primitives instead of by
    the canvas (:func:`_flat_windows`), the two binnings paying for different things.  A tile
    makes a primitive pay for a rectangle of the canvas whether or not it covers it, so a
    ribbon a pixel wide that lands in the tile of a coarse triangle pays for the square of
    that triangle; a window makes a primitive pay for a rectangle of its own box, which is
    what a wireframe of a 900-element mesh spends most of its time on.  What a window costs is
    a sweep that may hand the windows of one primitive out in any order, and one colour for
    the stack is what pays for that: a candidate the depth test of a window throws away is
    thrown away against a fragment already written to that sample, which is therefore nearer,
    or of the same depth -- and a tie of the same depth leaves the same depth bits and, one
    colour serving both fragments, the same colour, so which of the two a sequential painter
    would have kept is nothing the canvas can show.  A colour per primitive makes a tie
    visible, so such a stack stays on the tiles, where the order of the sweep is the order of
    the draw.

    The varyings are interpolated for the winners of that resolution only.  A fragment
    program that lost the depth test never ran, and a varying interpolated for it would be
    arithmetic thrown away -- ten times over, by the usual depth complexity -- as well as
    colouring a pixel with a triangle that is not the one visible.

    ``offset`` asks for the polygon offset of the surface material, every primitive of the
    stack pushed away from the camera by the slope of its own plane :func:`polygon_offset`
    measures.  It belongs to the pass and not to the primitive, which is how the renderer
    sets it: the surface material asks for it and the line materials do not.
    """
    if (shade is None) == (rgb is None):
        raise ValueError("a stack is either flat, with rgb, or shaded, with shade")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    count = x.shape[0]
    if count == 0:
        return canvas
    colour = (None if shade is not None
              else np.broadcast_to(np.asarray(rgb, dtype=np.float64), (count, 3)))
    # One colour for the whole stack, as opposed to one per primitive: the difference decides
    # whether a tie between two primitives over one pixel can be seen at all, and with it how
    # far the sweep may reorder and cut up the stack.  See :func:`_flat_windows`.
    flat = colour is not None and np.ndim(rgb) == 1

    # The same primitives survive here as survive _sweep: a determinant of no area or of
    # no finite value, and a bounding box the canvas edge clips away.  A corner that is
    # not a number takes the determinant with it, so it is culled here as it is there.
    den = ((y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2])
           + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2]))
    reach = canvas.reach
    with np.errstate(invalid="ignore"):
        i0 = np.maximum(np.floor(y.min(axis=1) - reach), 0).astype(np.int64)
        i1 = np.minimum(np.ceil(y.max(axis=1) + reach), canvas.height).astype(np.int64)
        j0 = np.maximum(np.floor(x.min(axis=1) - reach), 0).astype(np.int64)
        j1 = np.minimum(np.ceil(x.max(axis=1) + reach), canvas.width).astype(np.int64)
    (part,) = np.nonzero(np.isfinite(den) & (den != 0.0) & (i0 < i1) & (j0 < j1))
    if part.size == 0:
        return canvas
    x, y, w, z, den = (a[part] for a in (x, y, w, z, den))
    if offset:
        # the push belongs to the plane, so the three corners taking it is the same as every
        # fragment of the triangle taking it -- and it is measured before it is applied, as
        # _paint_triangle measures it
        z = z + _polygon_offsets(x, y, z)[:, None]
    if colour is not None:
        colour = colour[part]
    i0, i1, j0, j1 = i0[part], i1[part], j0[part], j1[part]
    # the varyings and the attributes of the primitives, as those of the survivors: they are
    # asked for by the stack place of a winner, which is a place in the compacted stack now
    varyings = tuple(np.asarray(f, dtype=np.float64)[part] for f in varyings)
    attrs = tuple(np.asarray(a)[part] for a in attrs)

    # the sample offsets, lined up with the last axis of the swept array
    offsets = np.asarray(canvas.offsets(), dtype=np.float64)
    dx = offsets[:, 0][None, None, None, :]
    dy = offsets[:, 1][None, None, None, :]
    planes = len(offsets)

    names = ["plane", "row", "col", "depth", "index"]
    if shade is not None:
        # the colour of a fragment, kept for every sample of it: the fragment program runs
        # once for the pixel and not once for the sample, so a primitive and a pixel have
        # one colour to shade and four samples to put it in
        names += ["paint"]
    won = {name: [] for name in names}
    held = 0

    def flush():
        """Write what the tiles have won so far, once, with a sample's ties resolved."""
        nonlocal held
        if not held:
            return
        got = {name: np.concatenate(won[name]) for name in won}
        keys = ((got["plane"] * canvas.height + got["row"]) * canvas.width
                + got["col"])
        keep = _resolve(keys, got["depth"], got["index"], less_equal)
        plane, row = got["plane"][keep], got["row"][keep]
        col, depth = got["col"][keep], got["depth"][keep]
        index = got["index"][keep]
        if shade is None:
            paint = colour[index]
        else:
            # the colour of a winning sample was shaded with its primitive's own fragment,
            # in the chunk that found the sample, and is only gathered here
            paint = got["paint"][keep]
        canvas.planes[plane, row, col, :3] = paint
        canvas.planes[plane, row, col, 3] = 1.0
        canvas.depths[plane, row, col] = depth
        for rows in won.values():
            rows.clear()
        held = 0

    if flat:
        # A flat stack of a single colour, swept each primitive over a window of its own box.
        # The tile is the wrong shape for a line pass: a ribbon a pixel wide is swept over
        # the tile of the canvas it lies in, two orders of magnitude more arithmetic than the
        # pixel asks for, and a wireframe is nothing but those.  A window costs what its box
        # costs, and a primitive is cut into as many windows as it needs and swept in whatever
        # order they come out, neither of which a stack of one colour can see: see
        # :func:`_flat_windows`.
        depths = canvas.depths.transpose(1, 2, 0)
        for h, wd, window_r, window_c, prims in _flat_windows(canvas, i0, i1, j0, j1):
            step = max(1, _STACK_SAMPLES // (h * wd * planes))
            for a in range(0, prims.size, step):
                here = prims[a:a + step]
                wr, wc = window_r[a:a + step], window_c[a:a + step]
                tx, ty, tw, tz = x[here], y[here], w[here], z[here]
                # _sweep's two slopes per barycentric, vertex 2 the origin of the window
                den_inv = 1.0 / den[here]
                b0 = (ty[:, 1] - ty[:, 2]) * den_inv
                c0 = (tx[:, 2] - tx[:, 1]) * den_inv
                b1 = (ty[:, 2] - ty[:, 0]) * den_inv
                c1 = (tx[:, 0] - tx[:, 2]) * den_inv
                gx = (wc[:, None] + np.arange(wd)) + 0.5 - tx[:, 2][:, None]
                gy = (wr[:, None] + np.arange(h)) + 0.5 - ty[:, 2][:, None]
                l0 = (b0[:, None] * gx)[:, None, :] + (c0[:, None] * gy)[:, :, None]
                l1 = (b1[:, None] * gx)[:, None, :] + (c1[:, None] * gy)[:, :, None]
                iw = 1.0 / tw
                zn = tz * iw

                # all four sample planes along the last axis, as over a tile
                s0 = l0[..., None] + b0[:, None, None, None] * dx + c0[:, None, None, None] * dy
                s1 = l1[..., None] + b1[:, None, None, None] * dx + c1[:, None, None, None] * dy
                s2 = 1.0 - s0 - s1
                corner = (len(here), 1, 1, 1)
                depth = ((s0 * zn[:, 0].reshape(corner) + s1 * zn[:, 1].reshape(corner)
                          + s2 * zn[:, 2].reshape(corner))
                         / (s0 * iw[:, 0].reshape(corner) + s1 * iw[:, 1].reshape(corner)
                            + s2 * iw[:, 2].reshape(corner)))
                keep = (s0 >= 0.0) & (s1 >= 0.0) & (s2 >= 0.0)
                del s0, s1, s2
                # the window's rows and columns, which lie inside the canvas by construction
                rows_at = wr[:, None] + np.arange(h)
                cols_at = wc[:, None] + np.arange(wd)
                old = depths[rows_at[:, :, None], cols_at[:, None, :]]
                hit = keep & (depth <= old if less_equal else depth < old)
                taken = np.nonzero(hit)
                if not taken[0].size:
                    continue
                won["plane"].append(taken[3])
                won["row"].append(rows_at[taken[0], taken[1]])
                won["col"].append(cols_at[taken[0], taken[2]])
                won["depth"].append(depth[taken])
                won["index"].append(here[taken[0]])
                held += taken[0].size
                if held >= _STACK_WINNERS:
                    flush()
        flush()
        return canvas

    for (ra, rb), (ca, cb), tile in _stack_tiles(canvas, i0, i1, j0, j1):
        # primitives of this tile at a time, few enough to keep the arrays small
        step = max(1, _STACK_SAMPLES // ((rb - ra) * (cb - ca) * planes))

        for a in range(0, tile.size, step):
            here = tile[a:a + step]
            tx, ty, tw, tz = x[here], y[here], w[here], z[here]
            # _sweep's two slopes per barycentric, vertex 2 the origin of the tile
            den_inv = 1.0 / den[here]
            b0 = (ty[:, 1] - ty[:, 2]) * den_inv
            c0 = (tx[:, 2] - tx[:, 1]) * den_inv
            b1 = (ty[:, 2] - ty[:, 0]) * den_inv
            c1 = (tx[:, 0] - tx[:, 2]) * den_inv
            gx = np.arange(ca, cb) + 0.5 - tx[:, 2][:, None]
            gy = np.arange(ra, rb) + 0.5 - ty[:, 2][:, None]
            l0 = (b0[:, None] * gx)[:, None, :] + (c0[:, None] * gy)[:, :, None]
            l1 = (b1[:, None] * gx)[:, None, :] + (c1[:, None] * gy)[:, :, None]
            iw = 1.0 / tw
            zn = tz * iw

            if shade is not None:
                # the pixel centre's own barycentrics and share of 1/w, which is where the
                # fragment program runs: once per pixel, not once per sample
                l2 = 1.0 - l0 - l1
                wc = 1.0 / (l0 * iw[:, 0][:, None, None] + l1 * iw[:, 1][:, None, None]
                            + l2 * iw[:, 2][:, None, None])

            # all four sample planes along the last axis: a sample's barycentrics are the
            # pixel centre's, swept by these same slopes, plus the plane's own offset
            s0 = l0[..., None] + b0[:, None, None, None] * dx + c0[:, None, None, None] * dy
            s1 = l1[..., None] + b1[:, None, None, None] * dx + c1[:, None, None, None] * dy
            s2 = 1.0 - s0 - s1
            corner = (len(here), 1, 1, 1)
            depth = ((s0 * zn[:, 0].reshape(corner) + s1 * zn[:, 1].reshape(corner)
                      + s2 * zn[:, 2].reshape(corner))
                     / (s0 * iw[:, 0].reshape(corner) + s1 * iw[:, 1].reshape(corner)
                        + s2 * iw[:, 2].reshape(corner)))
            keep = (s0 >= 0.0) & (s1 >= 0.0) & (s2 >= 0.0)
            del s0, s1, s2
            old = canvas.depths[:, ra:rb, ca:cb].transpose(1, 2, 0)[None]
            hit = keep & (depth <= old if less_equal else depth < old)
            taken = np.nonzero(hit)
            if not taken[0].size:
                continue
            won["plane"].append(taken[3])
            won["row"].append(taken[1] + ra)
            won["col"].append(taken[2] + ca)
            won["depth"].append(depth[taken])
            won["index"].append(np.broadcast_to(here[:, None, None, None],
                                                hit.shape)[taken])
            if shade is not None:
                # one fragment per primitive and pixel, wherever the primitive wins at least
                # one of its four samples: the fragment program runs for the pixel, at its
                # centre, so the samples of one fragment are one lighting call and not four
                whole = np.nonzero(np.any(hit, axis=3))
                place = np.full(hit.shape[:3], -1, dtype=np.int64)
                place[whole] = np.arange(whole[0].size)
                at = here[whole[0]]
                fields = tuple(_interpolate(f[at], l0[whole], l1[whole], l2[whole], wc[whole],
                                            iw[whole[0]]) for f in varyings)
                paint = shade(fields, tuple(a[at] for a in attrs))
                won["paint"].append(paint[place[taken[:3]]])
            held += taken[0].size
            if held >= _STACK_WINNERS:
                flush()
    flush()
    return canvas


def _samples(order, ctrl, t, *, vectors=None, deformation=0.0, mode="none",
             function_mode=0.0, complex_scale=(1.0, 0.0)):
    """Position and value of ``ctrl``'s instances at the parameter values ``t``."""
    kw = dict(vectors=vectors, deformation=deformation, mode=mode,
              function_mode=function_mode, complex_scale=complex_scale)
    return bezier.curve(order, np.asarray(ctrl, dtype=np.float64)[:, None],
                        np.asarray(t, dtype=np.float64)[None, :], **kw)


def spline_samples(order, ctrl, n_segments=N_SEGMENTS, *, vectors=None, deformation=0.0,
                   mode="none", function_mode=0.0, complex_scale=(1.0, 0.0)):
    """Position and value at the ``n_segments + 1`` samples of each curve instance.

    ``splines.vert:61`` takes its parameter as ``position.x / float(n_segments)``, and
    both line objects draw the samples ``0..n_segments`` of it: ``three/edges.ts:107``
    asks for the ``6 * subdivision`` triangles of the first ``subdivision`` sub-segments,
    ``three/mesh.ts:237`` for ``subdivision + 1`` vertices of a line strip.  The curve is
    therefore walked in equal steps from ``t = 0`` to ``t = 1``, and ``n_segments = 1``,
    which the renderer switches to while the camera is being dragged
    (``camera.ts:345-359``), draws the chord between the instance's ends.

    A NaN control point spreads over the whole instance, which :func:`ngsplot.bezier.curve`
    passes on rather than dropping: that is what the ``isNan`` guard at
    ``splines.vert:66-69`` tests, and it culls every vertex of the instance at once.
    """
    n = int(n_segments)
    if n < 1:
        raise ValueError("n_segments must be at least 1")
    return _samples(order, ctrl, np.arange(n + 1, dtype=np.float64) / n, vectors=vectors,
                    deformation=deformation, mode=mode, function_mode=function_mode,
                    complex_scale=complex_scale)


def _project(canvas, cam, points):
    """Window space of object-space points: ``(x, y, depth, w)``, one row per point."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return camera.to_pixels(camera.clip_points(points, cam.model_view_projection),
                            canvas.width, canvas.height)


def _projects(x, y, depth, w, axis=-1):
    """Which instances project at all, along ``axis`` of the window space of their samples.

    The same test as :func:`_usable`, left standing per instance instead of folded into one
    answer, because a batch drops the broken rows and keeps the rest; see there.
    """
    return np.all(np.isfinite(x) & np.isfinite(y) & np.isfinite(depth) & np.isfinite(w)
                  & (w > 0.0), axis=axis)


def _usable(x, y, depth, w):
    """Whether a whole instance projects: the vertex program keeps none of a broken one."""
    return bool(_projects(x, y, depth, w))


def _rail(px, py, qx, qy, half_width):
    """The sideways offset ``(nx, ny)`` of ``half_width`` pixels at sample ``(px, py)``.

    ``splines.vert:96-101`` normalises the forward step of the sample in aspect-corrected
    screen space, gives it a quarter turn, and spends half the ``line_thickness`` uniform
    on the result; the ribbon is what lies between the sample plus and minus it.  A
    forward step of zero length has no direction to turn, and the NaNs that follow drop
    the sub-segment, which is what ``normalize`` of a zero vector does on the GPU.
    """
    dx, dy = np.asarray(qx) - np.asarray(px), np.asarray(qy) - np.asarray(py)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = float(half_width) / np.hypot(dx, dy)
    return -dy * scale, dx * scale


def _ribbon(canvas, x, y, w, z, nx, ny, rgb, *, less_equal=True):
    """Paint the ribbon the rails of a curve bound, one pair of triangles per segment.

    ``three/edges.ts:50-66`` lists the quad between two samples as the triangles
    ``(a_i, b_i, b_i+1)`` and ``(b_i+1, a_i+1, a_i)``, ``a`` and ``b`` being a sample
    plus and minus its own normal.  The offset rides on ``gl_Position.xy`` multiplied by
    ``gl_Position.w`` (``splines.vert:103``), which is a shift in window space: it moves
    a sample sideways and leaves its depth and its weight exactly as they were.

    Kept as the word for what :func:`_ribbon_triangles` builds in bulk, and as the painter
    a test can hold the bulk against.
    """
    for i in range(len(x) - 1):
        j = i + 1
        # each corner of the two triangles: which sample it comes from, and which rail
        for corners in (((i, 1.0), (i, -1.0), (j, -1.0)), ((j, -1.0), (j, 1.0), (i, 1.0))):
            k = np.array([c[0] for c in corners])
            s = np.array([c[1] for c in corners])
            _paint_flat(canvas, x[k] + s * nx[k], y[k] + s * ny[k], w[k], z[k], rgb,
                        less_equal=less_equal)


# One segment of a ribbon is two triangles, and a corner of one is a sample of the segment
# and a side of that sample (three/edges.ts:50-66): the first row is (a_i, b_i, b_i+1) and
# the second (b_i+1, a_i+1, a_i), with a the sample plus its rail and b the sample minus it.
_RIBBON_CORNERS = np.array(((0, 0, 1), (1, 1, 0)))
_RIBBON_RAILS = np.array(((1.0, -1.0, -1.0), (-1.0, 1.0, 1.0)))


def _ribbon_triangles(x, y, w, z, nx, ny, *, rails="sample"):
    """The triangles the rails of every instance of a curve bound, in the order drawn.

    ``x``, ``y``, ``w`` and ``z`` are the ``(n, m)`` window space of the ``m`` samples of
    each of ``n`` instances, and ``nx``, ``ny`` are their rails: ``(n, m)``, the rail of the
    sample a corner stands on, when ``rails`` is ``"sample"``, which is what ``THICK_LINES``
    does, since ``splines.vert:96-101`` offsets every sample along its *own* look-ahead; or
    ``(n, m - 1)``, the rail of the segment between two samples, when it is ``"segment"``,
    which is what a strip does -- ``three/mesh.ts:160-240`` builds the wireframe without the
    define, so it has no thickness to spread and offsets a segment along its own chord
    instead.  Which of the two it is decides whether a turn in a curve joins across the
    corner or meets at it, so it is a thing the caller states and not a thing that is guessed
    from a shape.

    The answer is four ``(n * (m - 1) * 2, 3)`` arrays in the order the loops drew them:
    instance, then segment, then the two triangles of :func:`_ribbon`.  The order is not
    incidental but the depth test's whole opinion, since a ribbon drawn later takes the pixels
    of one drawn earlier on a tie.  The corners are ``x[k] + s * nx[k]``, term for term as
    :func:`_ribbon` evaluated them, and the depth and the weight ride along untouched, which
    is what a shift of ``gl_Position.xy`` by its ``w`` (``splines.vert:103``) really means.
    """
    if rails not in ("sample", "segment"):
        raise ValueError(f"rails is either 'sample' or 'segment', not {rails!r}")
    x, y, w, z = (np.asarray(a, dtype=np.float64) for a in (x, y, w, z))
    nx, ny = (np.asarray(a, dtype=np.float64) for a in (nx, ny))
    n = x.shape[-1] - 1
    if n < 1:
        empty = np.zeros((0, 3))
        return empty, empty, empty, empty
    corner = np.arange(n)[:, None, None] + _RIBBON_CORNERS
    x, y, w, z = (a[:, corner] for a in (x, y, w, z))
    if rails == "sample":
        nx, ny = (a[:, corner] for a in (nx, ny))
    else:
        nx, ny = (a[:, :, None, None] for a in (nx, ny))
    return (np.reshape(x + _RIBBON_RAILS * nx, (-1, 3)),
            np.reshape(y + _RIBBON_RAILS * ny, (-1, 3)),
            np.reshape(w, (-1, 3)), np.reshape(z, (-1, 3)))


def _line_modes(function_mode):
    """Refuse the two ``splines.frag`` modes that colour by the value instead of painting.

    Mode 4 writes the value as a colour and mode 7 discards it outside the range
    (:41-52); neither has a use outside the renderer's own debugging, and neither is what
    the picture is made of, since the Python side hands the line objects a value of 0.
    """
    if float(function_mode) in (4.0, 7.0):
        raise NotImplementedError(
            f"function_mode {function_mode} colours the line by its value "
            "(splines.frag:41-52); only the flat black of the default is painted")


def paint_edges(canvas, cam, order, ctrl, *, n_segments=N_SEGMENTS, colour=(0.0, 0.0, 0.0),
                line_thickness=None, deformation=0.0, mode="none", function_mode=0.0,
                complex_scale=(1.0, 0.0), less_equal=True):
    """Draw the thick lines of one curve object -- the boundary, or a wireframe in 1d.

    ``ctrl`` holds the ``(n, order + 1, 4)`` control points of
    :func:`ngsplot.geometry.edge_points`, one instance per element edge, and what they
    become is the ``THICK_LINES`` pass of ``splines.vert``: every sample of the instance
    is pushed sideways by half the ``line_thickness`` uniform, that uniform being the
    slider divided by the canvas height (``scene.ts:776``), which works out at
    ``line_thickness * canvas.height / 4`` pixels of half width (:100-103) -- 2.5 pixels
    for the shipped slider.  The direction of the push is the sample's own look-ahead to
    ``t + 0.1 / n_segments`` (:88), extrapolated past the last sample, so the ribbon
    never mitres at a sample the way a joined polyline would: a corner is two ribbons
    crossing, with the wedge between them unpainted.

    The colour is black and stays black.  ``splines.frag:54-58`` paints ``edge_colors``
    only under ``HAVE_COLORS``, and nothing compiles that, because the Python side never
    sends the attribute; what the boundary really is, whatever the colormap says, is
    ``vec4(0, 0, 0, 1)``.  Lines ride on top of the surface because they are drawn after
    it and win a tie, ``less_equal`` being the standing substitute for the surface's
    ``polygonOffset``.
    """
    _line_modes(function_mode)
    if line_thickness is None:
        line_thickness = LINE_THICKNESS / canvas.height
    half = float(line_thickness) * canvas.height / 4.0
    n = int(n_segments)
    rgb = np.asarray(colour, dtype=np.float64)
    kw = dict(deformation=deformation, mode=mode, function_mode=function_mode,
              complex_scale=complex_scale)
    pos, _ = _samples(order, ctrl, np.arange(n + 1, dtype=np.float64) / n, **kw)
    # the look-ahead of splines.vert:88, at t + 0.1 / n_segments, for every sample
    ahead, _ = _samples(order, ctrl, (np.arange(n + 1, dtype=np.float64) + 0.1) / n, **kw)
    x, y, depth, w = _project(canvas, cam, pos)
    ax, ay, _, _ = _project(canvas, cam, ahead)
    shape = pos.shape[:-1]
    x, y, depth, w, ax, ay = (a.reshape(shape) for a in (x, y, depth, w, ax, ay))
    # splines.vert:66-69 culls the instance and not the segment, so a curve with one broken
    # control point is gone whole -- every one of its ribbons -- before the rasterizer
    keep = _projects(x, y, depth, w)
    nx, ny = _rail(x, y, ax, ay, half)
    X, Y, W, Z = _ribbon_triangles(x[keep], y[keep], w[keep], depth[keep], nx[keep], ny[keep])
    return _paint_stack(canvas, X, Y, W, Z, rgb=rgb, less_equal=less_equal)


def paint_wireframe(canvas, cam, order, ctrl, *, n_segments=N_SEGMENTS,
                    colour=(0.0, 0.0, 0.0), width=LINE_WIDTH, deformation=0.0, mode="none",
                    function_mode=0.0, complex_scale=(1.0, 0.0), less_equal=True):
    """Draw the mesh wireframe, which is a line and not a ribbon, ``width`` pixels wide.

    ``three/mesh.ts:160-240`` builds it as a ``THREE.Line`` running the same vertex
    program *without* ``THICK_LINES``: the same instances sampled at the same
    ``t = i / n_segments``, joined into a strip of ``n_segments + 1`` vertices.  A line
    has no thickness of its own -- ``LineBasicMaterial.linewidth`` is one of the
    properties WebGL ignores -- so the rasterizer gives it one device pixel, half on
    either side, and the ``line_thickness`` slider that widens the boundary has nothing
    to do with it.  Each segment is offset along its own direction, so the segments of a
    turn meet rather than join.
    """
    _line_modes(function_mode)
    half = float(width) / 2.0
    n = int(n_segments)
    rgb = np.asarray(colour, dtype=np.float64)
    pos, _ = spline_samples(order, ctrl, n, deformation=deformation, mode=mode,
                            function_mode=function_mode, complex_scale=complex_scale)
    x, y, depth, w = _project(canvas, cam, pos)
    shape = pos.shape[:-1]
    x, y, depth, w = (a.reshape(shape) for a in (x, y, depth, w))
    keep = _projects(x, y, depth, w)
    # the rail of a strip is the direction of the segment itself, so the two corners at
    # either end of it are pushed along the same line: see _ribbon_triangles
    nx, ny = _rail(x[:, :-1], y[:, :-1], x[:, 1:], y[:, 1:], half)
    X, Y, W, Z = _ribbon_triangles(x[keep], y[keep], w[keep], depth[keep], nx[keep], ny[keep],
                                   rails="segment")
    return _paint_stack(canvas, X, Y, W, Z, rgb=rgb, less_equal=less_equal)
