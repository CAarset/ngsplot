"""Painting the mesh lines: the wireframe strips and the thick boundary edges.

Two objects paint lines and they are not the same primitive.  The wireframe is a
``THREE.Line`` -- one instance per triangle edge, ``LINE_STRIP`` over ``n_segments + 1``
samples, and no ``THICK_LINES`` in its defines (three/mesh.ts:160-240) -- so nothing in
the vertex program moves its vertices sideways and it comes out one device pixel wide.
The boundary edges are a ``THREE.Mesh`` whose material does define ``THICK_LINES``
(three/edges.ts:78), and its vertex program spreads each sample by ``line_thickness / 2``
in aspect-corrected clip units (shader/splines.vert:86-104), which for the default
uniform is 2.5 device pixels.  Both are black: ``HAVE_COLORS`` never compiles because the
Python side never sends ``edge_colors`` -- it appears in three/edges.ts:72,92,93 and
nowhere else -- so ``splines.frag`` takes the ``#else`` branch and writes
``vec4(0, 0, 0, 1)`` (shader/splines.frag:54-58).

The canvas is the 601 by 401 one of ``test_raster.py``, with an identity object matrix.
Both dimensions are odd, so the object origin lands on the corner between pixel rows 200
and 201 and columns 300 and 301: a line drawn through it is decided exactly, occupying
pixel row 200 and nothing beside it.  Coverage is resolved on the four sample positions of
4x multisampling, which sit +-0.125 and +-0.375 inside the pixel, so it comes in quarters
and a rail that lands on a quarter of a pixel is 0.125 px clear of the nearest sample: the
tests below put their lines on exact half-pixels, where that holds for both rails.  The
rails of a bent curve land wherever their own tangent puts them, though, so those tests
assert total ink and mean rows where a single pixel would be too sharp a thing to pin.
"""

import numpy as np
import pytest

from ngsplot import camera, raster

# The recorded buffer is 894 by 594 with the thickness slider at its default 5, which is
# what raster.LINE_THICKNESS holds; these tests divide the same 5 by their own height of
# 401 because scene.ts:776 divides the slider by the height of the canvas in use.
SLIDER = 5.0


def canvas_for(**kwargs):
    return raster.Canvas(601, 401, **kwargs)


def camera_for(**kwargs):
    cam = camera.Camera([0, 0, 0], 1.0, width=601, height=401, **kwargs)
    np.testing.assert_allclose(cam.object, np.eye(4), atol=0)
    return cam


def rows_of(cam, points):
    """Pixel rows that the given object points land on."""
    return np.asarray(cam.project(points), dtype=np.float64)[..., 1]


def row_of(cam, y):
    """Pixel row of the object point ``(0, y, 0)``, which the identity object maps affinely."""
    return rows_of(cam, [[0.0, y, 0.0]])[0]


def y_for_row(cam, row):
    """The inverse: the object height whose point lands on the given pixel row."""
    return (1.0 - 2.0 * row / cam.height) * cam.position[2] / cam.projection[1, 1]


def x_for_col(cam, col):
    """The object width whose point lands on the given pixel column."""
    return (2.0 * col / cam.width - 1.0) * cam.position[2] / cam.projection[0, 0]


def segment(a, b, order=1, value=0.0):
    """One instance of a straight line from ``a`` to ``b``, in the shader's own order.

    Order 1 evaluates ``t * p0 + (1 - t) * p1`` (shader/splines.vert:28), which runs from
    ``p1`` to ``p0``, so ``p0`` holds the end; the higher orders list the ends first and
    last and sit the intermediate control points on the chord, which reproduces the
    straight segment exactly with the ordinary parametrization.
    """
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    ctrl = np.zeros((1, order + 1, 4))
    if order == 1:
        ctrl[0] = np.array([(*b, value), (*a, value)])
    else:
        for i in range(order + 1):
            ctrl[0, i, :3] = a + i / order * (b - a)
            ctrl[0, i, 3] = value
    return ctrl


def ink(canvas, column=None, row=None):
    """How much paint a scanline carries, 1.0 for a fully painted pixel.

    The canvas resolves coverage, so a half painted pixel counts 0.5, and the sum along a
    scanline across a straight strip is its width in pixels whatever the sub-pixel phase:
    the samples of a row are spread over the whole pixel, so moving the strip sideways only
    moves coverage from one pixel to the next.  Per-pixel values do depend on the phase,
    which is why the tests below pin those on exact half-pixels.
    """
    if column is not None:
        return float(np.sum(1.0 - canvas.color[:, column, 0]))
    return float(np.sum(1.0 - canvas.color[row, :, 0]))


def painted_rows(canvas, column=300):
    return np.flatnonzero(canvas.color[:, column, 0] < 1.0)


def paint_edges(canvas, cam, ctrl, order=1, **kwargs):
    kwargs.setdefault("line_thickness", SLIDER / 401)
    raster.paint_edges(canvas, cam, order, ctrl, **kwargs)
    return canvas


# ---------------------------------------------------------------- sampling the curves

def test_a_first_order_curve_runs_from_p1_to_p0():
    # shader/splines.vert:28 -- `ps = t * p0 + (1.0 - t) * p1;`
    ctrl = np.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    pos, _ = raster.spline_samples(1, ctrl, n_segments=4)
    assert pos[0, 0, 0] == 0.0
    assert pos[0, -1, 0] == 1.0


def test_the_curve_basis_is_the_one_the_vertex_program_writes():
    # shader/splines.vert:35-48, orders 2 and 3
    ctrl = np.zeros((1, 3, 4))
    ctrl[0, 1] = [3.0, 0.0, 0.0, 0.0]
    pos, _ = raster.spline_samples(2, ctrl, n_segments=2)
    assert np.allclose(pos[0, :, 0], [0.0, 1.5, 0.0])

    ctrl = np.zeros((1, 4, 4))
    ctrl[0, 1] = [9.0, 0.0, 0.0, 0.0]
    pos, _ = raster.spline_samples(3, ctrl, n_segments=3)
    # b1 = 3 t (1-t)^2 of splines.vert:42 is 4/9 at t = 1/3 and 2/9 at t = 2/3, and the
    # samples land on t = 0, 1/3, 2/3, 1; the curve is lopsided because the basis is not
    b1 = 3 * np.array([0.0, 1 / 3, 2 / 3, 1.0]) * (1 - np.array([0.0, 1 / 3, 2 / 3, 1.0])) ** 2
    assert np.allclose(b1, [0.0, 4 / 9, 2 / 9, 0.0])
    assert np.allclose(pos[0, :, 0], 9.0 * b1)


def test_the_curve_is_sampled_at_multiples_of_one_over_n_segments():
    # shader/splines.vert:61 -- `float t0 = (position.x) / float(n_segments);`, over the
    # n_segments + 1 vertices the draw range covers (three/edges.ts:105-108)
    ctrl = np.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    for n in (1, 2, 5, 8):
        pos, _ = raster.spline_samples(1, ctrl, n_segments=n)
        assert pos.shape == (1, n + 1, 3)
        assert np.allclose(pos[0, :, 0], np.arange(n + 1) / n)


def test_the_value_slot_is_the_fourth_component_plus_the_vector_attributes():
    # shader/splines.vert:62-63 take value.yz from v0..v3 and :54 takes value.x from ps.w
    ctrl = np.array([[[0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 7.0]]])
    pos, value = raster.spline_samples(1, ctrl, n_segments=4)
    t = np.arange(5) / 4
    assert np.allclose(value[0, :, 0], t * 4.0 + (1 - t) * 7.0)
    assert np.all(value[0, :, 1:] == 0.0)

    vectors = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    pos, value = raster.spline_samples(1, ctrl, n_segments=4, vectors=vectors)
    assert np.allclose(value[0, :, 1], t * 1.0 + (1 - t) * 3.0)
    assert np.allclose(value[0, :, 2], t * 2.0 + (1 - t) * 4.0)


def test_a_scalar_on_a_curve_lifts_z_and_a_vector_pushes_the_point_about():
    # shader/splines.vert:73-77: DEFORMATION adds deformation * value to the position,
    # DEFORMATION_2D adds GetValue(deformation * value) to z only
    ctrl = np.array([[[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
    pos, _ = raster.spline_samples(1, ctrl, n_segments=2, deformation=0.5, mode="2d")
    assert np.allclose(pos[0, :, 2], 0.5)
    assert np.allclose(pos[0, :, :2], [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])

    vectors = np.array([[[2.0, 3.0], [2.0, 3.0]]])
    pos, value = raster.spline_samples(1, ctrl, n_segments=2, deformation=0.5, mode="3d",
                                      vectors=vectors)
    assert np.allclose(value[0, 0], [1.0, 2.0, 3.0])
    assert np.allclose(pos[0, :, 0], [0.5, 1.0, 1.5])
    assert np.allclose(pos[0, :, 1], 1.0)
    assert np.allclose(pos[0, :, 2], 1.5)


def test_a_nan_control_point_removes_the_whole_curve():
    # Both line objects pack a hard zero as the value slot -- ngsolve/webgui.py:343 builds
    # their coefficient function as (xyz, 0.0) -- so the shader's isNan(value) guard at
    # shader/splines.vert:66-69 never fires for them.  What removes a bad instance is
    # rarer: its NaN reaches gl_Position at :82-83 and then no fragment test passes on any
    # quad built from that instance.  The NaN reaches every sample of its own coordinate,
    # because every t mixes every control point and 0.0 * nan is nan, so even the two ends
    # where the middle control point has no weight; the other coordinates blend their own
    # control values and stay clean.
    ctrl = np.zeros((2, 3, 4))
    ctrl[1, :, :3] = [[-0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]
    ctrl[0, :, :3] = ctrl[1, :, :3]
    ctrl[0, 1, 1] = np.nan
    pos, value = raster.spline_samples(2, ctrl, n_segments=5)
    assert np.isnan(pos[0, :, 1]).all()
    assert np.isfinite(pos[0, :, [0, 2]]).all()
    assert np.all(value[0] == 0.0), "the line payload carries no value to poison"
    assert np.isfinite(pos[1]).all()

    canvas, cam = canvas_for(), camera_for()
    paint_edges(canvas, cam, ctrl, order=2)
    assert painted_rows(canvas).size > 0
    assert np.all(canvas.color[:, :200] == 1.0), "the poisoned instance painted anyway"


# ----------------------------------------------------------------------- thick edges

def test_the_default_edge_is_half_the_slider_width_in_pixels():
    # scene.ts:776 divides the slider (gui.ts:98, default 5) by the canvas height,
    # splines.vert:100 spends half of the result on each side, and a perspective camera
    # covers height / 2 pixels per clip unit, so the painted width is slider / 2 pixels.
    # The recorded picture agrees: the left border of square_scalar.png reads 128, 0, 0
    # on a scanline, half a pixel of ink and then two whole ones, 2.5 in all.
    canvas, cam = canvas_for(), camera_for()
    y = y_for_row(cam, 200.5)
    paint_edges(canvas, cam, segment([-0.3, y, 0.0], [0.3, y, 0.0]))
    # the strip is 2.5 px wide about the boundary between rows 200 and 201, so its rails
    # fall on the quarters at 199.25 and 201.75: rows 199 and 201 hold three of their four
    # samples, row 200 all four, and the greys read 64, 0, 64
    assert np.allclose(canvas.color[197:205, 300, 0], [1, 1, 0.25, 0.0, 0.25, 1, 1, 1])
    assert np.isclose(ink(canvas, column=300), 2.5)


def test_the_default_thickness_follows_the_canvas_height():
    # the uniform is the slider divided by the height of the canvas in use, and the width
    # it buys comes back out in pixels of that same canvas, so the painted strip does not
    # notice the resize: leave the default alone on a taller canvas and it is still half
    # the slider.  Passing line_thickness by hand is the opposite -- a fixed number of
    # clip units, which grows with the canvas.
    for height in (401, 894):
        canvas = raster.Canvas(601, height)
        cam = camera.Camera([0, 0, 0], 1.0, width=601, height=height)
        y = y_for_row(cam, height / 2 + 0.07)
        raster.paint_edges(canvas, cam, 1, segment([-0.3, y, 0.0], [0.3, y, 0.0]))
        assert np.isclose(ink(canvas, column=300), SLIDER / 2), height

    canvas = raster.Canvas(601, 894)
    cam = camera.Camera([0, 0, 0], 1.0, width=601, height=894)
    y = y_for_row(cam, 447.07)
    raster.paint_edges(canvas, cam, 1, segment([-0.3, y, 0.0], [0.3, y, 0.0]),
                       line_thickness=SLIDER / 401)
    assert ink(canvas, column=300) == pytest.approx(SLIDER / 2 * 894 / 401, abs=0.5)


def test_the_edge_is_as_thick_upwards_as_sideways():
    # splines.vert:94-95 multiply the direction by the aspect and :101 divides the offset
    # back by it, which is what makes the width isotropic on the screen; drop that and a
    # vertical line on this 1.4987 aspect canvas comes out half again as wide.
    canvas, cam = canvas_for(), camera_for()
    x = x_for_col(cam, 300.5)
    paint_edges(canvas, cam, segment([x, -0.2, 0.0], [x, 0.2, 0.0]))
    assert np.allclose(canvas.color[200, 297:305, 0], [1, 1, 0.25, 0.0, 0.25, 1, 1, 1])
    assert np.isclose(ink(canvas, row=200), 2.5)


def test_the_width_follows_the_slider():
    # scene.ts:776 divides gui.ts:98's slider by the canvas height, so a slider of 1 is
    # half a pixel wide and the recorded square_no_wireframe.png does show a single pixel
    # reading 128 where a boundary crosses a scanline.  Four samples on the rotated 4x grid
    # make every rail a quarter-pixel decision, so the profiles below are exact: a slider
    # of 20 paints nine rows whole and the two that touch its rails half.
    profiles = {1.0: (200, [0.5]), 5.0: (199, [0.25, 0.0, 0.25]),
                10.0: (198, [0.0] * 5),
                20.0: (195, [0.5] + [0.0] * 9 + [0.5])}
    for slider, (first, expect) in profiles.items():
        canvas, cam = canvas_for(), camera_for()
        y = y_for_row(cam, 200.5)
        paint_edges(canvas, cam, segment([-0.3, y, 0.0], [0.3, y, 0.0]),
                    line_thickness=slider / 401)
        last = first + len(expect)
        assert np.allclose(canvas.color[first:last, 300, 0], expect), slider
        assert np.all(canvas.color[:first, 300, 0] == 1.0), slider
        assert np.all(canvas.color[last:, 300, 0] == 1.0), slider
        assert np.isclose(ink(canvas, column=300), slider / 2), slider


def test_the_strip_has_no_joins_where_two_curves_meet():
    # every vertex is offset along its own forward tangent (shader/splines.vert:96-100) and
    # the two rails are joined by nothing, so each instance stops dead at its own end: the
    # outside quadrant of the corner keeps the white of the background, where a miter would
    # have painted.  The two strips are 2.5 pixels wide about the corner at (300.5, 200.5),
    # one running up the screen and one in from the left, and the pixels beside the corner
    # keep only the half that lies behind the end of the strip that owns them.  The pixel
    # the corner itself falls in is covered by both strips, but the rotated sample grid puts
    # its two negative offsets in x and y together, so both rails keep the very same two
    # samples and the pixel stays at half -- which is still half short of what a join would
    # have made of it.
    canvas, cam = canvas_for(), camera_for()
    corner = np.array([x_for_col(cam, 300.5), y_for_row(cam, 200.5), 0.0])
    ctrl = np.concatenate([
        segment(corner - [0.4, 0.0, 0.0], corner),
        segment(corner, corner + [0.0, 0.4, 0.0]),
    ])
    paint_edges(canvas, cam, ctrl)
    cover = 1.0 - canvas.color[:, :, 0]
    assert cover[201, 301] == 0.0, "a join filled the outside of the corner"
    assert np.isclose(cover[200, 301], 0.5), "the up-going strip did not stop at its own end"
    assert np.isclose(cover[201, 300], 0.5), "the incoming strip did not stop at its own end"
    assert np.isclose(cover[200, 300], 0.5), "the inside of the corner is overpainted"


BENT = np.array([[[-0.3, -0.1, 0.0, 0.0], [0.0, 0.25, 0.0, 0.0], [0.0, 0.25, 0.0, 0.0],
                  [0.3, -0.1, 0.0, 0.0]]])


def test_the_strip_stops_where_the_draw_range_does():
    # three/edges.ts:105-108 sets the range to 6 * subdivision vertices out of the 6 per
    # sub-segment it builds, so subdivision 1 draws the straight chord between the ends and
    # never visits the middle of the curve; three/mesh.ts:236-240 does the same with
    # sd + 1 vertices for the wireframe
    cam = camera_for()
    # the drawn polyline crosses column 300 on the segment between its samples at t = 0.4
    # and t = 0.6, which sit at y = 0.75 t (1-t) - 0.1 (t^3 + (1-t)^3) = 0.152 on either
    # side and straddle it at x = -+ 0.0456; subdivision 1 draws the chord instead, which
    # only ever touches the ends at y = -0.1
    lift, chord = row_of(cam, 0.152), row_of(cam, -0.1)
    assert abs(lift - chord) > 40, "the test curve is not bent enough to tell apart"

    canvas = canvas_for()
    paint_edges(canvas, cam, BENT, order=3, n_segments=5)
    assert abs(painted_rows(canvas).mean() - lift) < 1.5

    for painter, kwargs in ((raster.paint_edges, {"line_thickness": SLIDER / 401}),
                            (raster.paint_wireframe, {})):
        canvas = canvas_for()
        painter(canvas, cam, 3, BENT, n_segments=1, **kwargs)
        assert abs(painted_rows(canvas).mean() - chord) < 1.5, painter


def test_the_deformation_moves_the_edge_before_it_is_projected():
    # shader/splines.vert:73-77 happens on the object position, before the matrices at :82
    cam = camera_for()
    ctrl = segment([-0.3, 0.2, 0.0], [0.3, 0.2, 0.0], value=1.0)
    lifted = rows_of(cam, [[0.0, 0.2, 0.4]])[0]
    assert abs(lifted - row_of(cam, 0.2)) > 4.0, "the test depth change is invisible"

    canvas = canvas_for()
    paint_edges(canvas, cam, ctrl, deformation=0.4, mode="2d")
    rows = painted_rows(canvas)
    assert abs(rows.mean() - lifted) < 1.0, rows
    assert abs(rows.mean() - row_of(cam, 0.2)) > 1.0, "the lift was applied too late"


# ------------------------------------------------------------------------- wireframe

def test_the_wireframe_is_one_pixel_wide():
    # three/mesh.ts:160-240 builds a THREE.Line with no THICK_LINES define, so no vertex is
    # moved sideways and the strip is as wide as a line is on this rasterizer: one device
    # pixel, modelled here as a one-pixel quad straddling the path
    canvas, cam = canvas_for(), camera_for()
    raster.paint_wireframe(canvas, cam, 1, segment([-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]))
    # the line runs along row 200.5, so its half-pixel rails land on 200.0 and 201.0, which
    # is pixel row 200 and precisely nothing else
    assert np.allclose(canvas.color[197:205, 300, 0], [1, 1, 1, 0.0, 1, 1, 1, 1])
    assert np.isclose(ink(canvas, column=300), 1.0)


def test_the_wireframe_honours_its_width():
    canvas, cam = canvas_for(), camera_for()
    ctrl = segment([-0.3, 0.0, 0.0], [0.3, 0.0, 0.0])
    raster.paint_wireframe(canvas, cam, 1, ctrl, width=5.0)
    assert np.isclose(ink(canvas, column=300), 5.0)


def test_the_deformation_moves_the_wireframe_as_early_as_it_moves_the_edge():
    # three/mesh.ts:200-230 hands the wireframe the same p0..p3 the edge gets and
    # shader/splines.vert:73-77 lifts it in the same place, and it is the wireframe that has
    # to follow a deformed surface across the screen -- an edge lifted without the
    # wireframe that crosses the surface is a line drawn over the wrong picture
    cam = camera_for()
    ctrl = segment([-0.3, 0.2, 0.0], [0.3, 0.2, 0.0], value=1.0)
    lifted = rows_of(cam, [[0.0, 0.2, 0.4]])[0]

    canvas = canvas_for()
    raster.paint_wireframe(canvas, cam, 1, ctrl, deformation=0.4, mode="2d")
    rows = painted_rows(canvas)
    assert abs(rows.mean() - lifted) < 1.0, rows
    assert abs(rows.mean() - row_of(cam, 0.2)) > 1.0, "the lift was applied too late"


def test_the_deformation_of_a_bool_differs_from_that_of_a_function():
    """The payload decides whether the line vertices are already lifted, and the flag says
    which case it is: ngsolve/webgui.py:297-299 leaves ``deformation=True`` geometry flat
    for the shader to lift, while a CoefficientFunction is applied to the points in Python
    (:336-343) before they are packed as ``(xyz, 0.0)``.  Passing the lift to the second
    case happens to change nothing, for the same reason the shader's own lift cannot hurt a
    line: the displacement is scaled by ``value_``, which :343 sets to zero.
    """

    cam = camera_for()
    lifted = rows_of(cam, [[0.0, 0.2, 0.4]])[0]

    flat = segment([-0.3, 0.2, 0.0], [0.3, 0.2, 0.0], value=1.0)     # the bool's payload
    baked = segment([-0.3, 0.2, 0.4], [0.3, 0.2, 0.4])               # the function's, value 0

    canvas = canvas_for()
    raster.paint_wireframe(canvas, cam, 1, flat, deformation=0.4, mode="2d")
    assert abs(painted_rows(canvas).mean() - lifted) < 1.0
    canvas = canvas_for()
    raster.paint_wireframe(canvas, cam, 1, baked)                    # nothing to lift
    assert abs(painted_rows(canvas).mean() - lifted) < 1.0

    canvas = canvas_for()                             # a second lift has nothing to scale
    raster.paint_wireframe(canvas, cam, 1, baked, deformation=0.4, mode="2d")
    assert abs(painted_rows(canvas).mean() - lifted) < 1.0

# --------------------------------------------------------------- paint, depth, colour

def test_the_lines_are_flat_black_whatever_the_function_is():
    # shader/splines.frag:54-58: without HAVE_COLORS the colour is vec4(0, 0, 0, 1), and
    # nothing before it reads a colormap, a normal, or a light
    canvas, cam = canvas_for(), camera_for()
    paint_edges(canvas, cam, segment([-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]))
    # the strip straddles the boundary of rows 200 and 201 and only row 200 holds all of its
    # samples; the rows beside it carry the same black at a quarter of the coverage, which
    # is a neutral grey and not, say, a blue one
    assert np.all(canvas.color[200, 300, :3] == 0.0)
    grey = canvas.color[199, 300, :3]
    assert grey[0] < 1.0 and np.allclose(grey, grey[0]), "partial coverage took a colour"

    canvas = canvas_for()
    raster.paint_wireframe(canvas, cam, 1, segment([-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]),
                           colour=(0.25, 0.5, 0.75))
    assert np.allclose(canvas.color[200, 300, :3], [0.25, 0.5, 0.75])


def test_a_line_takes_the_pixels_it_wins_the_depth_test_on():
    # the surface is drawn with polygonOffset(1, 1) (three/mesh.ts:65-67), which pushes it
    # away from the eye so that a line in the same plane survives; the offset is a constant
    # here, so the test asks about a line a few pixels in front of and behind the surface
    ctrl = np.zeros((1, 3, 4))
    ctrl[0, :, :3] = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]
    cam = camera_for()

    canvas = canvas_for()
    raster.paint_surface(canvas, cam, 1, ctrl, vmin=0.0, vmax=1.0)
    surface = canvas.color[200, 300].copy()
    assert surface[0] < 1.0, "the surface did not even cover the pixel"
    raster.paint_wireframe(canvas, cam, 1, segment([-0.3, 0.0, 0.02], [0.3, 0.0, 0.02]))
    assert canvas.color[200, 300, 0] == 0.0, "a line in front did not paint"

    canvas = canvas_for()
    raster.paint_surface(canvas, cam, 1, ctrl, vmin=0.0, vmax=1.0)
    raster.paint_wireframe(canvas, cam, 1, segment([-0.3, 0.0, -0.02], [0.3, 0.0, -0.02]))
    assert np.allclose(canvas.color[200, 300], surface), "a line behind painted through"


def test_a_deformed_surface_keeps_the_lines_deformed_with_it():
    """The bug this pins was found as missing ink, not as a wrong pixel.

    A ``deformation`` of ``True`` in the render data lifts the surface in the shader
    (shader/trigsplines.vert:25-30) and lifts the lines by the same rule at :73-77, so both
    painters have to be told.  A wireframe left at ``z = 0`` under a surface pushed towards
    the camera ends up behind it, and the depth test erases it: on ``square_deformed`` that
    cost 84% of all the line ink in the picture while leaving every surface pixel correct.
    """
    tri = np.zeros((1, 3, 4))
    tri[0, :, :3] = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]
    tri[0, :, 3] = 1.0                      # the value the lift is scaled by
    line = segment([-0.3, 0.0, 0.0], [0.3, 0.0, 0.0], value=1.0)
    cam = camera_for()

    def painted_over(lift_line):
        # the ramp ends red, whose own green channel is nearly black, so the surface is told
        # to be green: the one channel a black line has to darken and cannot otherwise fake
        canvas = canvas_for()
        raster.paint_surface(canvas, cam, 1, tri, vmin=0.0, vmax=1.0, deformation=0.4,
                             mode="2d", pal=np.array([[0.0, 1.0, 0.0]]))
        assert (canvas.color[200, 280:320, 1] > 0.8).all(), "the surface is not green there"
        raster.paint_wireframe(canvas, cam, 1, line, deformation=lift_line,
                               mode="2d" if lift_line else "none")
        return float(np.mean(canvas.color[200, 280:320, 1] < 0.2))

    assert painted_over(0.4) == 1.0         # riding on top of the surface it belongs to
    assert painted_over(0.0) == 0.0         # flat, it is behind that surface and gone


def test_a_line_of_the_same_depth_wins_only_on_a_tie_it_is_allowed():
    # three.js defaults to LessEqualDepth, which is why the lines are painted after the
    # surface: the second line below is at exactly the depth of the first one
    canvas, cam = canvas_for(), camera_for()
    ctrl = segment([-0.3, 0.0, 0.0], [0.3, 0.0, 0.0])
    raster.paint_wireframe(canvas, cam, 1, ctrl)
    raster.paint_wireframe(canvas, cam, 1, ctrl, colour=(1.0, 0.0, 0.0))
    assert np.allclose(canvas.color[200, 300, :3], [1.0, 0.0, 0.0])

    canvas = canvas_for()
    raster.paint_wireframe(canvas, cam, 1, ctrl)
    raster.paint_wireframe(canvas, cam, 1, ctrl, colour=(1.0, 0.0, 0.0), less_equal=False)
    assert np.allclose(canvas.color[200, 300, :3], [0.0, 0.0, 0.0])


@pytest.mark.parametrize("mode", [4.0, 7.0])
def test_the_debug_colour_modes_of_the_line_shader_are_not_painted(mode):
    # shader/splines.frag:41-52 paints the raw value in those two modes; nothing on the
    # Python side sets them, so the rasterizer refuses instead of inventing a colour
    canvas, cam = canvas_for(), camera_for()
    ctrl = segment([-0.3, 0.0, 0.0], [0.3, 0.0, 0.0])
    with pytest.raises(NotImplementedError):
        raster.paint_wireframe(canvas, cam, 1, ctrl, function_mode=mode)
    with pytest.raises(NotImplementedError):
        paint_edges(canvas, cam, ctrl, function_mode=mode)
