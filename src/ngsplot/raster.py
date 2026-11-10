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


def lattice_points(order, ctrl, *, vectors=None, deformation=0.0, mode="none",
                   function_mode=0.0, complex_scale=(1.0, 0.0), n_segments=N_SEGMENTS):
    """Where every lattice vertex of the surface is, and the value there: ``(ne, n_lattice, 3)``
    of deformed object-space position and ``(ne, n_lattice)`` of the blended fourth control
    component, the first half of :func:`lattice` for whoever has no use for the normals."""
    u, v = bezier.lattice(n_segments)
    return bezier.evaluate(int(order), np.asarray(ctrl, dtype=np.float64)[:, None], u, v,
                           vectors=vectors, deformation=deformation, mode=mode,
                           function_mode=function_mode, complex_scale=complex_scale)


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
    pos, w = lattice_points(order, ctrl, n_segments=n_segments, **kw)
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
#: sweep of :func:`_paint_triangle` is handed the stack instead of :func:`_paint_stack`.  The
#: batch lights a pixel by gathering the corners of the primitive that holds it, row by row;
#: the loop lights the whole box of one primitive by broadcasting its three corners over it,
#: and pays a dozen numpy calls per primitive for that.  The boxes of a mesh that fills the
#: frame add up to about one canvas whatever its fineness, so the batch's cost is set by the
#: picture and the loop's by the number of primitives in it, and the two cross at a fixed
#: number of canvas pixels per primitive.  Measured in one process, CPU time, best of five, on
#: order-1 meshes of the unit square at 900x600: the loop takes 0.97 of the batch's time at
#: 5625 pixels per primitive and 1.03 of it at 4500; on ``mesh_only`` -- 52 triangles, 10212
#: pixels each -- it takes 0.79.
_STACK_LOOP_AREA = 5000


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

    The sheet goes to :func:`_paint_stack` unless it is too coarse for a batch to pay for
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
    usable = np.isfinite(px) & np.isfinite(py) & np.isfinite(depth) & np.isfinite(w) & (w > 0.0)
    # component by component, as the corners below: numpy reduces along a short axis per row
    for k in range(value.shape[-1]):
        usable &= ~np.isnan(value[..., k])

    # The whole sheet is one stack, in the order an instanced mesh draws it: instance by
    # instance, and within an instance in the order subtriangles returns.  That order is not
    # a detail -- it is what settles every tie between two sub-triangles that meet on an edge,
    # which for a surface this fine is most of the ink on the sheet.
    tris = bezier.subtriangles(n_segments)
    e, ti = np.nonzero(usable[:, tris[:, 0]] & usable[:, tris[:, 1]] & usable[:, tris[:, 2]])
    # the three lattice corners of each, as places in the lattice of the whole sheet: the
    # vertex buffer an indexed draw fetches its varyings from, which is what the stack does too
    corner = e[:, None] * value.shape[1] + tris[ti]
    x, y, z, w = (a.reshape(-1)[corner] for a in (px, py, depth, w))
    p_view, n_view, val = (a.reshape(-1, a.shape[-1]) for a in (p_view, n_view, value))

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
        # -normal on the back faces, negated where it stands rather than chosen between copies
        normal = np.array(normal)
        np.negative(normal, out=normal, where=back[:, None])
        return _fragment_colour(val, normal, pos,
                                back & face_check, function_mode=function_mode,
                                complex_scale=complex_scale, function_values=function_values,
                                vmin=vmin, vmax=vmax, pal=pal, dark_backside=dark_backside)

    if x.shape[0] * _STACK_LOOP_AREA < canvas.width * canvas.height:
        # A stack too coarse to amortise a batch: one triangle at a time, which is what
        # :func:`_paint_triangle` is, and the reason it is kept.  The order is the stack's, so a
        # pixel two triangles both want is tested twice instead of sorted once -- against what
        # the earlier one left, which gives the later one the tie under LessEqualDepth and the
        # earlier one under LessDepth, the same two outcomes _resolve picks from a stack.
        for here in range(x.shape[0]):
            at = corner[here]
            _paint_triangle(canvas, x[here], y[here], w[here], z[here],
                            p_view[at], n_view[at], val[at], vmin=vmin, vmax=vmax, pal=pal,
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
    _paint_stack(canvas, x, y, w, z, varyings=(val, p_view, n_view), corners=corner,
                 attrs=(back,), shade=shade, less_equal=less_equal, offset=True)
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


#: how much wider than its corners the box of a triangle is taken, as a share of the pixel
#: coordinates of those corners.  The arithmetic that decides whether a sample is covered rounds
#: by a few units in the last place of the coordinates it is fed -- about 1e-13 of a pixel on a
#: canvas -- and a sample that close outside the corners can come out covered; this is a
#: millionth of a pixel there, which no rounding reaches and no sample position notices.
_BOX_SLACK = 2.0 ** -30


def _box(canvas, x, y):
    """The pixels whose samples a triangle may cover: rows ``i0:i1`` and columns ``j0:j1``.

    ``x`` and ``y`` hold the pixel coordinates of the corners along their last axis, of one
    triangle or of a stack of them, and the four answers are integers of the same shape less
    that axis, clipped to the canvas.  A triangle is the hull of its corners, so a sample it
    covers lies between the least and the greatest corner on either axis -- which is all a GPU
    asks too, the box of its snapped vertices -- and a pixel holds its samples at its centre
    plus the canvas's offsets.  So the box runs from the first pixel with a sample at or past
    the least corner to the last with one at or short of the greatest, give or take
    :data:`_BOX_SLACK`, and is empty on an axis where no sample position falls in between.

    That is not the box of whole pixels the corners touch, a half pixel and an offset wider on
    every side: for a sub-triangle a pixel across, the size of most of a fine mesh, that box is
    four times the samples, and every one it adds is a sample the arithmetic finds outside.
    """
    offsets = np.asarray(canvas.offsets(), dtype=np.float64)
    x, y = np.asarray(x), np.asarray(y)
    # corner by corner rather than a reduction along an axis of three, which numpy pays for
    # per triangle
    x_lo = np.minimum(np.minimum(x[..., 0], x[..., 1]), x[..., 2])
    x_hi = np.maximum(np.maximum(x[..., 0], x[..., 1]), x[..., 2])
    y_lo = np.minimum(np.minimum(y[..., 0], y[..., 1]), y[..., 2])
    y_hi = np.maximum(np.maximum(y[..., 0], y[..., 1]), y[..., 2])
    with np.errstate(invalid="ignore"):
        slack = _BOX_SLACK * (1.0 + np.maximum(abs(x_lo), abs(x_hi))
                              + np.maximum(abs(y_lo), abs(y_hi)))
        # a pixel's samples sit half a pixel and an offset past its own top left corner
        i0 = np.maximum(np.ceil(y_lo - slack - 0.5 - offsets[:, 1].max()), 0)
        i1 = np.minimum(np.floor(y_hi + slack - 0.5 - offsets[:, 1].min()) + 1, canvas.height)
        j0 = np.maximum(np.ceil(x_lo - slack - 0.5 - offsets[:, 0].max()), 0)
        j1 = np.minimum(np.floor(x_hi + slack - 0.5 - offsets[:, 0].min()) + 1, canvas.width)
        return tuple(np.asarray(a).astype(np.int64) for a in (i0, i1, j0, j1))


def _sweep(canvas, x, y, w, z, *, less_equal=True, shade=None):
    """Walk a triangle in window space over the sample planes of ``canvas``.

    ``x`` and ``y`` are the pixel coordinates of the three vertices with the first row on
    top, ``w`` their clip weight and ``z`` their window depth.  One yield per sample
    plane, giving ``(plane, rows, cols, hit, depth, rgb)`` over the :func:`_box` of the
    triangle: the absolute index of every pixel in it, the samples this plane still wins
    here, the depth of those samples, and the colour -- which ``shade``, if given, is
    asked for once, with the interpolator of the varyings as its argument.  A triangle
    that is degenerate or holds no sample position of the canvas yields nothing, which is
    how both are culled.
    """
    den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
    if den == 0.0 or not np.isfinite(den):
        return

    i0, i1, j0, j1 = (int(a) for a in _box(canvas, x, y))
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


# ------------------------------------------------------------ the stack, binned into windows

#: how many samples one chunk -- the primitives of one window shape, swept together -- is
#: walked over.  A chunk is a single array of ``(primitives, rows, columns, sample planes)``,
#: walked over a dozen times by the barycentrics, the depth and the depth test, so what
#: matters is that one of its temporaries is small enough that the next one finds it in
#: cache: a megabyte each, which on the meshes of :mod:`tools.bench_paint` is the fastest of
#: the sizes tried, by a few percent over either a sixteenth or a sixteenth of a gigabyte.
_STACK_SAMPLES = 1 << 16

#: the largest window :func:`_windows` sweeps a primitive over at a time, in pixels.  A box
#: coarser than that is swept as several windows, which is what keeps the array of a chunk to a
#: few tens of megabytes however far a primitive reaches.  A side of :data:`_WINDOW_SIDES`.
_STACK_WINDOW = 64

#: the sides a window is rounded up to: the powers of two and the steps half way between them,
#: so that no side of a box is swept more than half as far again as it reaches.  Powers of two
#: alone sweep a box of nine pixels over sixteen, three times the box; on a P2 cube at 900x600,
#: whose sub-triangles are about ten pixels across, the steps between them paint the surface
#: 6% and the wireframe 9% faster (one process, CPU time, best of five)
_WINDOW_SIDES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])

#: how many candidate samples :func:`_paint_stack` collects before it settles them against
#: the depth buffer.  Settling is a few scatters, which want rows to chew on, and what it holds
#: until then is four arrays of this length.
_STACK_WINNERS = 1 << 17

#: how many samples the colouring of a stack takes at a time, in bands of whole rows of what
#: the stack reaches.  Every one of them is a few rows of fragment arithmetic, so this bounds
#: the temporaries of the lighting rather than its cost, which is one call per band.
_STACK_FRAGMENTS = 1 << 18

#: how far below zero a barycentric has to be at all four corners of a window before the
#: window is dropped as lying outside its triangle, as a share of the terms it is summed
#: from: the arithmetic that finds a sample rounds it by a few times ``1e-16`` of those.
_STACK_CULL = 1e-9


def _resolve(keys, depth, index, less_equal, nearest, place):
    """Which of the candidates for one sample the depth test keeps.

    ``keys`` names the sample every candidate wants to write, ``depth`` is its window
    depth and ``index`` its place in the stack, the three of them the same length; ``nearest``
    and ``place`` are scratch, a float and an integer for every sample a key can name.  The
    test keeps the nearer fragment, and on an exact depth it keeps the fragment written
    last where the test is ``LessEqualDepth`` -- what the renderer sets,
    ``three/mesh.ts:57`` -- and the one written first where it is ``LessDepth``.  Two
    scatters do both, and no sort: the least depth every sample is wanted at, then the
    latest or the earliest place in the stack among the candidates at that depth.  Nothing
    with a depth that cannot be ordered against another reaches here, the comparison having
    already failed it.

    A primitive swept twice over a sample -- which a window pulled back onto its neighbour
    does -- is two candidates of one depth and one place, and both are kept: they write the
    same bits.
    """
    nearest[keys] = np.inf
    np.minimum.at(nearest, keys, depth)
    tie = depth == nearest.take(keys)
    place[keys] = -1 if less_equal else np.iinfo(np.int64).max
    (np.maximum if less_equal else np.minimum).at(place, keys[tie], index[tie])
    return np.flatnonzero(tie & (index == place.take(keys)))


def _windows(canvas, i0, i1, j0, j1):
    """Yield ``(rows, columns, window rows, window columns, primitives)``, a class at a time.

    ``i0``, ``i1``, ``j0`` and ``j1`` are the pixel rows and columns of each primitive's
    bounding box, clipped to the canvas and half-open.  The sweep is handed the primitives
    that share a window shape, so that one array walks all of them: a window is a rectangle
    of the primitive's own box, its two sides rounded up separately to a side of
    :data:`_WINDOW_SIDES`, so that a ribbon one pixel long is swept over a pixel and a
    sub-triangle of a few pixels over a few -- not over a tile of the canvas, which makes a
    primitive pay for the square of the coarsest box that shares it.

    Windows of one shape are yielded together, which is what lets their arrays share a
    length.  A box coarser than :data:`_STACK_WINDOW` is handed out as several windows, and
    the last window of an axis is pulled back to the edge of the canvas so that every window
    lies inside the canvas: the sweep of a window asks no pixel whether it is off it.  Where
    a window is pulled back onto its neighbour a pixel of the box is swept twice, and the two
    answers are the same bits, so it is :func:`_resolve` that is left to pick one.

    The windows come out in no particular order, and the sweep owes nothing to it:
    :func:`_paint_stack` settles a tie by the place in the stack of the primitives that made
    it, not by the order the windows came in.
    """
    # the rung of the ladder each side of a box is rounded up to, below the longest side a
    # window takes and the longest the canvas holds
    sides = _WINDOW_SIDES[_WINDOW_SIDES <= _STACK_WINDOW]
    rung_i = np.searchsorted(sides, np.minimum(i1 - i0, sides[sides <= canvas.height][-1]))
    rung_j = np.searchsorted(sides, np.minimum(j1 - j0, sides[sides <= canvas.width][-1]))
    # a key of a few bits, which numpy sorts by counting rather than comparing
    key = (rung_i * sides.size + rung_j).astype(np.int16)
    order = np.argsort(key, kind="stable")
    keys = key[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    for a, b in zip(starts, np.r_[starts[1:], keys.size]):
        part, number = order[a:b], int(keys[a])
        h, wd = (int(sides[rung]) for rung in divmod(number, sides.size))
        side_i, side_j = i1.take(part) - i0.take(part), j1.take(part) - j0.take(part)
        if side_i.max() <= h and side_j.max() <= wd:
            # every box of the class is one window, the most of a fine mesh
            yield (h, wd, np.minimum(i0.take(part), canvas.height - h),
                   np.minimum(j0.take(part), canvas.width - wd), part)
            continue
        span_i = -(side_i // -h)
        span_j = -(side_j // -wd)
        rows = np.minimum(i0[part][:, None] + h * np.arange(span_i.max()),
                          canvas.height - h)
        columns = np.minimum(j0[part][:, None] + wd * np.arange(span_j.max()),
                             canvas.width - wd)
        grid = ((np.arange(rows.shape[1])[None, :, None] < span_i[:, None, None])
                & (np.arange(columns.shape[1])[None, None, :] < span_j[:, None, None]))
        k, u, v = np.nonzero(grid)
        yield h, wd, rows[k, u], columns[k, v], part[k]


def _interpolate(ends, l0, l1, l2, wc, iw):
    """The value of a varying at the pixel centre of every row of a list of winners.

    ``ends`` are three ``(n, k)`` arrays, the values of the varying at corner 0, 1 and 2 of
    each primitive that won; ``l0``, ``l1``, ``l2`` and ``wc`` are the barycentrics and the
    share of ``1/w`` of the pixel centre it won, and ``iw`` the corners' ``1/w``.  This is
    :func:`_sweep`'s ``interp`` with the primitive as the row instead of the pixel, term for
    term, so that a colour comes out of it the same as out of the fragment program -- summed
    in place, which spares the copies and not a single rounding.
    """
    f0, f1, f2 = ends
    out = (l0 * iw[:, 0])[:, None] * f0
    out += (l1 * iw[:, 1])[:, None] * f1
    out += (l2 * iw[:, 2])[:, None] * f2
    out *= wc[:, None]
    return out


def _paint_stack(canvas, x, y, w, z, *, rgb=None, varyings=(), corners=None, attrs=(),
                 shade=None, less_equal=True, offset=False):
    """Sweep a whole stack of window-space triangles over the canvas, a batch at a time.

    ``x``, ``y``, ``w`` and ``z`` are ``(n, 3)``: row ``k`` holds primitive ``k``, in the
    order it is drawn, and its three corners' pixel columns, pixel rows, clip weights and
    window depths.  The depth test is the ``less_equal`` of :func:`_sweep`.

    A flat stack paints in the colour ``rgb``, one for all of it or one per primitive.  A
    shaded stack paints with ``shade``, which is given the ``varyings`` and the ``attrs`` of
    the primitives that won the pixels and answers ``(n, 3)`` of colour: ``varyings`` are
    ``(n, 3, k)`` per corner, interpolated perspective-correctly, and ``attrs`` ``(n, k)``
    per primitive, gathered.  One of the two, and only one, says which stack this is.  With
    ``corners``, the ``(n, 3)`` vertices of every primitive, the stack is an indexed one, as
    an indexed draw is: ``varyings`` are then ``(v, k)`` per vertex, and the corners of a
    primitive are fetched from them only once it has won a pixel.

    This is :func:`_paint_flat` and :func:`_paint_triangle` for a great many primitives at
    once, pixel for pixel.  The arithmetic of :func:`_sweep` is reproduced term for term so
    that the depths come out the same bits; what changes is only *when* it is evaluated: over
    windows of the primitives' own boxes (:func:`_windows`), every window of one shape in one
    array, instead of over one primitive at a time.  The candidates a sweep finds are settled
    against the depth buffer in batches, and settling is where the draw order comes back in.
    The stack keeps, for every sample it reaches, the place of the primitive holding it, so a
    candidate at the depth of the sample it wants takes it exactly when a sequential painter
    would have let it -- under ``LessEqualDepth`` when it comes later in the stack than the
    holder, under ``LessDepth`` when it comes earlier, and never from something drawn before
    the stack under ``LessDepth``.  Which batch a candidate came in, and which window, decides
    nothing: the picture is that of the stack drawn in order, whatever order it was swept in.

    The colour comes last, once the stack is settled.  A fragment program that lost the depth
    test never ran, and a varying interpolated for it would be arithmetic thrown away -- ten
    times over, by the usual depth complexity -- as well as colouring a pixel with a triangle
    that is not the one visible.  So a shaded stack lights each pixel once for every primitive
    left holding a sample of it, at the pixel centre, in as few calls of ``shade`` as the
    picture allows, and the samples of that pixel the primitive holds take the one colour --
    the fragment program runs once for the pixel and not once for the sample.

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

    # The same primitives survive here as survive _sweep: a determinant of no area or of
    # no finite value, and a box that holds no sample position of the canvas.  A corner that
    # is not a number takes the determinant with it, so it is culled here as it is there.
    den = ((y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2])
           + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2]))
    i0, i1, j0, j1 = _box(canvas, x, y)
    (part,) = np.nonzero(np.isfinite(den) & (den != 0.0) & (i0 < i1) & (j0 < j1))
    if part.size == 0:
        return canvas
    if part.size < count:
        x, y, w, z, den = (a[part] for a in (x, y, w, z, den))
        i0, i1, j0, j1 = i0[part], i1[part], j0[part], j1[part]
        if colour is not None:
            colour = colour[part]
    if offset:
        # the push belongs to the plane, so the three corners taking it is the same as every
        # fragment of the triangle taking it -- and it is measured before it is applied, as
        # _paint_triangle measures it
        z = z + _polygon_offsets(x, y, z)[:, None]
    # the varyings and the attributes stay those of the whole stack, and a winner's are found
    # through ``part``: its place in the stack before the culling, which is what they are kept by
    varyings = tuple(np.asarray(f, dtype=np.float64) for f in varyings)
    attrs = tuple(np.asarray(a) for a in attrs)

    # the sample offsets, lined up with the last axis of the swept array
    offsets = np.asarray(canvas.offsets(), dtype=np.float64)
    dx = offsets[:, 0][None, None, None, :]
    dy = offsets[:, 1][None, None, None, :]
    planes = len(offsets)

    # the part of the canvas the stack reaches, and which of its primitives holds each sample
    # of it: none, until the stack has been settled against the depth buffer
    top, left = int(i0.min()), int(j0.min())
    tall, wide = int(i1.max()) - top, int(j1.max()) - left
    owner = np.full((planes, tall, wide), -1, dtype=np.int64)
    holders = owner.reshape(-1)
    # scratch for settle(): which candidate last claimed a sample, which samples have more than
    # one candidate in a batch, and _resolve's two
    claim = np.empty(owner.size, dtype=np.int64)
    crowd = np.zeros(owner.size, dtype=bool)
    nearest = np.empty(owner.size, dtype=np.float64)
    place = np.empty(owner.size, dtype=np.int64)
    # the depth buffer as one row of samples, which a flat index reaches in one step and numpy's
    # take reaches faster than three indices do; a view, or the writes would go nowhere
    depth_of = canvas.depths.reshape(-1, copy=False)
    found = {name: [] for name in ("key", "at", "depth", "index")}
    held = 0

    def settle():
        """Settle what the sweep has found so far against what the buffer already holds."""
        nonlocal held
        if not held:
            return
        key, at, depth, index = (np.concatenate(rows) for rows in found.values())
        # Most samples have one candidate in a batch -- the inside of a triangle -- and need no
        # sorting.  The candidates of a sample more than one wants are told apart by
        # _resolve, which is the depth test between them.
        order = np.arange(key.size)
        claim[key] = order
        crowded = claim.take(key) != order
        if crowded.any():
            crowd[key[crowded]] = True
            contested = crowd.take(key)
            crowd[key[crowded]] = False
            (rival,) = np.nonzero(contested)
            pick = np.concatenate([np.flatnonzero(~contested),
                                   rival[_resolve(key[rival], depth[rival], index[rival],
                                                  less_equal, nearest, place)]])
            key, at, depth, index = key[pick], at[pick], depth[pick], index[pick]
        old = depth_of.take(at)
        holder = holders.take(key)
        # an equal depth is a tie, and a tie is settled by the place in the stack: what a
        # sequential painter writes later wins it under LessEqual and loses it under Less,
        # and a sample held from before this stack loses nothing to it under Less
        later = (index > holder) if less_equal else (holder >= 0) & (index < holder)
        won = (depth < old) | ((depth == old) & later)
        depth_of[at[won]] = depth[won]
        holders[key[won]] = index[won]
        for rows in found.values():
            rows.clear()
        held = 0

    for h, wd, window_r, window_c, prims in _windows(canvas, i0, i1, j0, j1):
        if ((i1[prims] - i0[prims]) > h).any() or ((j1[prims] - j0[prims]) > wd).any():
            # A window wholly outside one edge of its triangle holds none of its samples, and
            # a box swept as several windows is mostly such windows wherever the triangle is
            # long and thin across it -- the ribbon of a line, or a large element seen at a
            # slant.  A barycentric is affine on the screen, so it is below zero over the whole
            # window when it is at the four corners, which bound every sample in it.  How far
            # below is measured against the terms it was summed from, which is what the
            # arithmetic finding a sample rounds by.  A box swept as one window meets its
            # triangle, and is not asked.
            tx, ty = x[prims], y[prims]
            den_inv = 1.0 / den[prims]
            gx = window_c[:, None] + np.array([0.0, wd, 0.0, wd]) - tx[:, 2][:, None]
            gy = window_r[:, None] + np.array([0.0, 0.0, h, h]) - ty[:, 2][:, None]
            reach_x, reach_y = np.abs(gx).max(axis=1) + 1.0, np.abs(gy).max(axis=1) + 1.0
            meets = np.ones(prims.size, dtype=bool)
            sums = []
            for b, c in (((ty[:, 1] - ty[:, 2]) * den_inv, (tx[:, 2] - tx[:, 1]) * den_inv),
                         ((ty[:, 2] - ty[:, 0]) * den_inv, (tx[:, 0] - tx[:, 2]) * den_inv)):
                s = b[:, None] * gx + c[:, None] * gy
                scale = np.abs(b) * reach_x + np.abs(c) * reach_y
                meets &= s.max(axis=1) >= -_STACK_CULL * scale
                sums.append((s, scale))
            (s0, scale0), (s1, scale1) = sums
            meets &= (1.0 - s0 - s1).max(axis=1) >= -_STACK_CULL * (1.0 + scale0 + scale1)
            (meets,) = np.nonzero(meets)
            prims, window_r, window_c = prims[meets], window_r[meets], window_c[meets]
        size = h * wd * planes
        step = max(1, _STACK_SAMPLES // size)
        for a in range(0, prims.size, step):
            here = prims[a:a + step]
            wr, wc = window_r[a:a + step], window_c[a:a + step]
            tx, ty, tw, tz = (v.take(here, axis=0) for v in (x, y, w, z))
            # _sweep's two slopes per barycentric, vertex 2 the origin of the window
            den_inv = 1.0 / den.take(here)
            b0 = (ty[:, 1] - ty[:, 2]) * den_inv
            c0 = (tx[:, 2] - tx[:, 1]) * den_inv
            b1 = (ty[:, 2] - ty[:, 0]) * den_inv
            c1 = (tx[:, 0] - tx[:, 2]) * den_inv
            gx = (wc[:, None] + np.arange(wd)) + 0.5 - tx[:, 2][:, None]
            gy = (wr[:, None] + np.arange(h)) + 0.5 - ty[:, 2][:, None]
            l0 = (b0[:, None] * gx)[:, None, :] + (c0[:, None] * gy)[:, :, None]
            l1 = (b1[:, None] * gx)[:, None, :] + (c1[:, None] * gy)[:, :, None]

            # all four sample planes along the last axis: a sample's barycentrics are the
            # pixel centre's, swept by these same slopes, plus the plane's own offset -- added
            # in _sweep's order, the centre's sum first, in place to spare the copies
            s0 = l0[..., None] + b0[:, None, None, None] * dx
            s0 += c0[:, None, None, None] * dy
            s1 = l1[..., None] + b1[:, None, None, None] * dx
            s1 += c1[:, None, None, None] * dy
            s2 = 1.0 - s0
            s2 -= s1
            # covered: no barycentric below zero, which the least of the three answers for all
            # of them -- a NaN among them is the least, and is not at least zero either
            least = np.minimum(s0, s1)
            np.minimum(least, s2, out=least)
            (inside,) = np.nonzero(least.reshape(-1) >= 0.0)
            if not inside.size:
                continue
            # Everything after coverage is asked of the samples covered and no others, which on
            # a fine mesh is a fifth of what its windows hold or less.  A row of the window is a
            # sample here, and _sweep's arithmetic per sample is the same arithmetic per row.
            s0, s1, s2 = (s.reshape(-1).take(inside) for s in (s0, s1, s2))
            prim, at = np.divmod(inside, size)
            row, at = np.divmod(at, wd * planes)
            col, plane = np.divmod(at, planes)
            iw = 1.0 / tw
            zn = (tz * iw).take(prim, axis=0)
            iw = iw.take(prim, axis=0)
            depth = ((s0 * zn[:, 0] + s1 * zn[:, 1] + s2 * zn[:, 2])
                     / (s0 * iw[:, 0] + s1 * iw[:, 1] + s2 * iw[:, 2]))
            # the window's rows and columns, which lie inside the canvas by construction
            row += wr.take(prim)
            col += wc.take(prim)
            at = (plane * canvas.height + row) * canvas.width + col
            # a candidate as near as the sample it wants is kept for settle() to judge, which
            # knows who holds the sample; a farther one, or one of no depth, is lost already
            (near,) = np.nonzero(depth <= depth_of.take(at))
            if not near.size:
                continue
            found["key"].append((plane[near] * tall + (row[near] - top)) * wide
                                + (col[near] - left))
            found["at"].append(at[near])
            found["depth"].append(depth[near])
            found["index"].append(here.take(prim[near]))
            held += near.size
            if held >= _STACK_WINNERS:
                settle()
    settle()

    if shade is not None:
        # what the pixel centre of a primitive is lit from, per primitive and once: the slopes of
        # its barycentrics, its vertex 2 and its corners' 1/w, in _sweep's arithmetic with the
        # primitive as the row -- and the varyings side by side, so that the corners of a winner
        # are fetched and blended in one go
        den_inv = 1.0 / den
        slopes = ((y[:, 1] - y[:, 2]) * den_inv, (x[:, 2] - x[:, 1]) * den_inv,
                  (y[:, 2] - y[:, 0]) * den_inv, (x[:, 0] - x[:, 2]) * den_inv)
        with np.errstate(divide="ignore"):
            # a weight of zero is the caller's to cull, and a primitive with one wins nothing
            iw = 1.0 / w
        x2, y2 = x[:, 2].copy(), y[:, 2].copy()
        packed = np.concatenate(varyings, axis=-1)
        cuts = np.cumsum([f.shape[-1] for f in varyings])[:-1]

    # every sample the stack holds now takes the colour of the primitive holding it, a band of
    # rows of the reach at a time, written through the colour buffer as one row per sample
    colour_of = canvas.planes.reshape(-1, canvas.planes.shape[-1], copy=False)
    band = max(1, _STACK_FRAGMENTS // (planes * wide))
    for r0 in range(0, tall, band):
        hold = owner[:, r0:r0 + band]
        rows = top + r0
        if shade is None:
            k, r, c = np.nonzero(hold >= 0)
            into = (k * canvas.height + (rows + r)) * canvas.width + (left + c)
            colour_of[into, :3] = colour.take(hold[k, r, c], axis=0)
            colour_of[into, 3] = 1.0
            continue
        # the plane each sample takes its colour from: the first of its pixel's planes held by
        # the same primitive, which is where that primitive's fragment of the pixel is lit --
        # the fragment program runs once for the pixel and not once for the sample
        source = np.empty(hold.shape, dtype=np.int64)
        for k in range(planes):
            source[k] = k
            for j in range(k - 1, -1, -1):
                source[k][hold[k] == hold[j]] = j
        mine = hold >= 0
        fresh = (source == np.arange(planes)[:, None, None]) & mine
        k, r, c = np.nonzero(fresh)
        at = hold[k, r, c]
        # the pixel centre's own barycentrics and share of 1/w, which is where the fragment
        # program runs
        gx = (c + left) + 0.5 - x2.take(at)
        gy = (r + rows) + 0.5 - y2.take(at)
        b0, c0, b1, c1 = (s.take(at) for s in slopes)
        l0 = b0 * gx + c0 * gy
        l1 = b1 * gx + c1 * gy
        l2 = 1.0 - l0 - l1
        iw_at = iw.take(at, axis=0)
        wc = 1.0 / (l0 * iw_at[:, 0] + l1 * iw_at[:, 1] + l2 * iw_at[:, 2])
        # the winners' corners, a corner at a time: three rows of the varyings each
        drawn = part.take(at)
        if corners is None:
            ends = packed.take(drawn, axis=0)
            ends = (ends[:, 0], ends[:, 1], ends[:, 2])
        else:
            fetch = corners.take(drawn, axis=0)
            ends = tuple(packed.take(fetch[:, j], axis=0) for j in range(3))
        fields = np.split(_interpolate(ends, l0, l1, l2, wc, iw_at), cuts, axis=-1)
        lit = shade(tuple(fields), tuple(a.take(drawn, axis=0) for a in attrs))
        # A sample held takes the fragment lit for its source plane at its pixel.  The fragments
        # were lit in the order np.nonzero walks the fresh samples, so a running count of them
        # numbers each one by its row of ``lit``.
        number = np.cumsum(fresh).reshape(hold.shape) - 1
        k, r, c = np.nonzero(mine)
        into = (k * canvas.height + (rows + r)) * canvas.width + (left + c)
        colour_of[into, :3] = lit.take(number[source[k, r, c], r, c], axis=0)
        colour_of[into, 3] = 1.0
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
