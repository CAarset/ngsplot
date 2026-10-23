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


def _usable(x, y, depth, w):
    """Whether a whole instance projects: the vertex program keeps none of a broken one."""
    return bool(np.all(np.isfinite(x) & np.isfinite(y) & np.isfinite(depth) & np.isfinite(w)
                       & (w > 0.0)))


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
    """
    for i in range(len(x) - 1):
        j = i + 1
        # each corner of the two triangles: which sample it comes from, and which rail
        for corners in (((i, 1.0), (i, -1.0), (j, -1.0)), ((j, -1.0), (j, 1.0), (i, 1.0))):
            k = np.array([c[0] for c in corners])
            s = np.array([c[1] for c in corners])
            _paint_flat(canvas, x[k] + s * nx[k], y[k] + s * ny[k], w[k], z[k], rgb,
                        less_equal=less_equal)


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
    for e in range(pos.shape[0]):
        if not _usable(x[e], y[e], depth[e], w[e]):
            continue
        nx, ny = _rail(x[e], y[e], ax[e], ay[e], half)
        _ribbon(canvas, x[e], y[e], w[e], depth[e], nx, ny, rgb, less_equal=less_equal)
    return canvas


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
    for e in range(pos.shape[0]):
        if not _usable(x[e], y[e], depth[e], w[e]):
            continue
        for i in range(n):
            j = i + 1
            nx, ny = _rail(x[e, i], y[e, i], x[e, j], y[e, j], half)
            pick = slice(i, j + 1)
            _ribbon(canvas, x[e, pick], y[e, pick], w[e, pick], depth[e, pick],
                    np.repeat(nx, 2), np.repeat(ny, 2), rgb, less_equal=less_equal)
    return canvas
