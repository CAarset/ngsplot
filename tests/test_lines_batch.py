"""The batched line painter is the same stack of ribbons the loops built, pixel for pixel.

:func:`ngsplot.raster.paint_edges` and :func:`ngsplot.raster.paint_wireframe` used to ask
for one triangle at a time: for every instance, for every segment, twice over, a call that
swept a two-triangle ribbon and wrote its result to the canvas.  The batched version hands
the whole set of ribbons to :func:`ngsplot.raster._paint_stack` at once, which is what
:func:`ngsplot.raster.paint_surface` already does to the surface, and it is only the same
thing if three things survive the change.

The *order* survives because the depth test is a sequential thing: instance after instance,
segment after segment, and the two triangles of a segment in the order
``three/edges.ts:50-66`` lists them, ``(a_i, b_i, b_i+1)`` then ``(b_i+1, a_i+1, a_i)``,
``a`` and ``b`` being a sample plus and minus its own normal.  Where two ribbons tie, the
later one takes the pixels of the earlier on a tie, and a batch that enumerated its
triangles in another order would leave the crossing painted the other way.  One call paints
one colour, so within a call nothing ties with anything: the tie is made by drawing the same
set of curves twice over, in two colours, which is also what a boundary drawn over a surface
looks like from the depth buffer's side.

The *rails* survive because they are not one convention but two.  A boundary sample is
pushed along the direction to its own look-ahead at ``t + 0.1 / n_segments``
(``splines.vert:88``), so the two corners of a segment carry the rails of the samples they
come from and a bow turns the ribbon along its length; a wireframe segment is pushed along
its own chord, so both of its corners carry the same rail and a turn is two ribbons that
meet rather than join.  Both are visible as ink, and ``tests/test_raster_lines.py`` pins
the widths and the rows of the two exactly, at half-pixel positions; what is pinned here is
that batching them changed nothing.

The *filter* survives because a curve that does not project is dropped whole, before any of
its segments are drawn: ``splines.vert:66-69`` culls the instance, not the sub-segment.  A
batch that sorted the broken instance into its tiles anyway would sweep a mirrored ribbon of
some enormous extent, and where the mirror happens to fold back inside the window it would
paint black across the picture.  So the payload carries one curve with a control point of no
coordinate and one behind the eye, placed so that its mirror lands *inside* the frame, and
both of them have to leave the canvas exactly as it was -- and both, made live, have to
change it.

The canvas is the 601 by 401 of ``tests/test_raster_lines.py`` and the camera looks at the
unit cube from ``elev=20, azim=12``, far enough off axis that the rails of a bow turn
through a good part of a circle across the picture.
"""

import numpy as np
import pytest

from ngsplot import camera, raster

WIDTH, HEIGHT = 601, 401

CAMERA = dict(elev=20.0, azim=12.0)

# The three rows curves() appends to every payload, and where the two unprojectable ones
# go.  Measured with this camera: the chord below at z = 5 projects to columns 203..438 and
# rows 76..274, well inside the frame, on w of -2.9; the other one crosses the eye plane,
# its first six samples in front of it and the seventh behind.
CROSSING, NA, BEHIND = -3, -2, -1
BEHIND_CHORD = ((-1.3, -1.9, 5.0), (-0.7, -1.5, 5.0))
EYE_CROSSING = ((0.55, 0.35, 0.1), (-2.4, -2.4, 2.0))
POISON = (CROSSING, NA, BEHIND)

# Chords for the offedge payload, measured on the camera below: the frame of the 601 by 401
# canvas takes the unit sphere, so a world coordinate past about 1.1 is outside it.  The
# first six leave through the right, the left, the bottom, the top and through two corners;
# the last two are in front of the eye but wholly beside the frame, which is the case a
# window that starts at the clipped box has to drop rather than sweep.
OUTSIDE = (
    ((0.2, 0.0, 0.0), (3.0, 0.2, 0.0)),
    ((-0.1, -0.3, 0.0), (-3.0, -0.5, 0.0)),
    ((0.0, -0.1, 0.0), (0.15, -3.0, 0.0)),
    ((0.05, 0.1, 0.0), (-0.1, 3.0, 0.0)),
    ((-2.6, -2.2, 0.0), (2.6, 2.2, 0.0)),
    ((2.6, -2.2, 0.3), (-0.4, 0.6, -0.3)),
    ((2.3, 1.2, 0.0), (2.9, 0.4, 0.0)),
    ((-2.9, -2.6, 0.0), (-2.4, -2.9, 0.0)),
)


def canvas_for():
    return raster.Canvas(WIDTH, HEIGHT)


def camera_for(**kwargs):
    return camera.Camera([0, 0, 0], 1.0, width=WIDTH, height=HEIGHT, **CAMERA, **kwargs)


# ------------------------------------------------------------------------ the payload

def instance(a, b, order=1, bow=0.0, value=0.0):
    """One curve instance from ``a`` to ``b``, in the order the vertex program wants.

    Order 1 evaluates ``t * p0 + (1 - t) * p1`` (``splines.vert:28``), which runs from
    ``p1`` to ``p0``, so its ends go in backwards; higher orders sit their control points on
    the chord, which stays a straight line, and ``bow`` then pulls the interior ones
    sideways until the curve and its rails actually turn.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ctrl = np.zeros((1, order + 1, 4))
    if order == 1:
        ctrl[0] = np.array([(*b, value), (*a, value)])
    else:
        for i in range(order + 1):
            ctrl[0, i, :3] = b + (a - b) * (i / order)
            ctrl[0, i, 3] = value
        if bow:
            chord = a - b
            side = np.array([-chord[1], chord[0]])
            side /= np.hypot(*side)
            for i in range(1, order):
                ctrl[0, i, :2] += bow * np.sin(np.pi * i / order) * side
    return ctrl


def curves(kind="grid"):
    """Control points ``(n, order + 1, 4)`` of a set of curves, plus rows that do not project.

    The last three rows are the rows the usability filter exists for, and they are last so
    that the rows before them are the picture: a copy of the first row with a coordinate
    erased, the chord of ``BEHIND_CHORD``, and the chord of ``EYE_CROSSING``, which is in
    front of the eye for all but its last sample and so is the row that tells an instance
    filter from a segment one.
    """
    rows = []
    if kind == "bent":
        for i in range(6):
            y = -0.7 + 0.28 * i
            rows.append(instance([0.85, y, 0.05 * i], [-0.85, -y, -0.05 * i],
                                 order=3, bow=0.34 - 0.09 * i, value=0.3))
    elif kind == "dense":
        for i in range(24):
            t = -0.9 + 0.078 * i
            rows.append(instance([0.95, t, 0.2 * np.sin(3.0 * t)],
                                 [-0.95, -t, -0.2 * np.sin(3.0 * t)], value=0.3))
    elif kind == "offedge":
        # chords that leave the frame through each of its four sides and through a corner,
        # plus two that miss it altogether while still being in front of the eye
        for a, b in OUTSIDE:
            rows.append(instance(a, b))
    elif kind == "ragged":
        # a chord across the whole frame next to a stub of a few pixels, so that one stack
        # holds ribbons whose boxes differ by two orders of magnitude
        for i in range(7):
            t = -0.6 + 0.2 * i
            rows.append(instance([-2.2, t, 0.05 * i], [2.2, -t, -0.05 * i], order=2,
                                 value=0.3))
            rows.append(instance([0.4 * t, 0.5, 0.1], [0.4 * t + 0.003, 0.5 + 0.002, 0.1],
                                 order=2))
    else:
        for i in range(12):
            t = -0.85 + 0.155 * i
            rows.append(instance([0.9, t, 0.0], [-0.9, t * t, 0.0]))
            rows.append(instance([t, 0.9, 0.0], [t * t, -0.9, -0.0]))
    order = rows[0].shape[1] - 1
    rows.append(instance(*EYE_CROSSING, order=order))
    broken = rows[0].copy()
    broken[0, 0, 0] = np.nan
    rows.append(broken)
    behind = np.array(BEHIND_CHORD, dtype=np.float64)
    rows.append(instance(behind[0], behind[-1], order=order))
    return np.concatenate(rows, axis=0)


# ------------------------------------------------------------------------ the oracles

def reference_edges(canvas, cam, order, ctrl, *, n_segments=raster.N_SEGMENTS,
                    colour=(0.0, 0.0, 0.0), line_thickness=None, deformation=0.0,
                    mode="none", function_mode=0.0, complex_scale=(1.0, 0.0),
                    less_equal=True):
    """``paint_edges`` as it was written: a ribbon at a time, in the order of the loops."""
    raster._line_modes(function_mode)
    if line_thickness is None:
        line_thickness = raster.LINE_THICKNESS / canvas.height
    half = float(line_thickness) * canvas.height / 4.0
    n = int(n_segments)
    rgb = np.asarray(colour, dtype=np.float64)
    kw = dict(deformation=deformation, mode=mode, function_mode=function_mode,
              complex_scale=complex_scale)
    pos, _ = raster._samples(order, ctrl, np.arange(n + 1, dtype=np.float64) / n, **kw)
    ahead, _ = raster._samples(order, ctrl, (np.arange(n + 1, dtype=np.float64) + 0.1) / n,
                               **kw)
    x, y, depth, w = raster._project(canvas, cam, pos)
    ax, ay, _, _ = raster._project(canvas, cam, ahead)
    shape = pos.shape[:-1]
    x, y, depth, w, ax, ay = (a.reshape(shape) for a in (x, y, depth, w, ax, ay))
    for e in range(pos.shape[0]):
        if not raster._usable(x[e], y[e], depth[e], w[e]):
            continue
        nx, ny = raster._rail(x[e], y[e], ax[e], ay[e], half)
        raster._ribbon(canvas, x[e], y[e], w[e], depth[e], nx, ny, rgb, less_equal=less_equal)
    return canvas


def reference_wireframe(canvas, cam, order, ctrl, *, n_segments=raster.N_SEGMENTS,
                        colour=(0.0, 0.0, 0.0), width=raster.LINE_WIDTH, deformation=0.0,
                        mode="none", function_mode=0.0, complex_scale=(1.0, 0.0),
                        less_equal=True):
    """``paint_wireframe`` as it was written: one segment's ribbon at a time."""
    raster._line_modes(function_mode)
    half = float(width) / 2.0
    n = int(n_segments)
    rgb = np.asarray(colour, dtype=np.float64)
    pos, _ = raster.spline_samples(order, ctrl, n, deformation=deformation, mode=mode,
                                   function_mode=function_mode, complex_scale=complex_scale)
    x, y, depth, w = raster._project(canvas, cam, pos)
    shape = pos.shape[:-1]
    x, y, depth, w = (a.reshape(shape) for a in (x, y, depth, w))
    for e in range(pos.shape[0]):
        if not raster._usable(x[e], y[e], depth[e], w[e]):
            continue
        for i in range(n):
            j = i + 1
            nx, ny = raster._rail(x[e, i], y[e, i], x[e, j], y[e, j], half)
            pick = slice(i, j + 1)
            raster._ribbon(canvas, x[e, pick], y[e, pick], w[e, pick], depth[e, pick],
                           np.repeat(nx, 2), np.repeat(ny, 2), rgb, less_equal=less_equal)
    return canvas


CASES = dict(
    grid=dict(kind="grid", n_segments=4),
    bent=dict(kind="bent", n_segments=8),
    dense=dict(kind="dense", n_segments=12),
    hair=dict(kind="bent", n_segments=6, px=0.5),
    wide=dict(kind="bent", n_segments=6, px=18.0),
    offedge=dict(kind="offedge", n_segments=8),
    ragged=dict(kind="ragged", n_segments=6),
    depth_32=dict(kind="bent", n_segments=6, less_equal=False),
    lifted=dict(kind="bent", n_segments=6, deformation=0.4, mode="2d"),
)
PAINTERS = dict(edges=(raster.paint_edges, reference_edges),
                wireframe=(raster.paint_wireframe, reference_wireframe))


def payload(name, painter, ctrl=None, **extra):
    """The same call to the batched painter and to the loops, each on a fresh canvas."""
    kw = dict(CASES[name], **extra)
    ctrl = curves(kw.pop("kind")) if ctrl is None else np.asarray(ctrl, dtype=np.float64)
    # a case names a line one px wide at the sample; the two objects spell that differently,
    # the boundary in canvas fractions of the slider and the wireframe in device pixels
    px = kw.pop("px", None)
    if px is not None:
        kw.update(dict(line_thickness=4.0 * px / HEIGHT) if painter == "edges"
                  else dict(width=px))
    batched, oracle = PAINTERS[painter]
    cam = camera_for()
    return (batched(canvas_for(), cam, ctrl.shape[1] - 1, ctrl, **kw),
            oracle(canvas_for(), cam, ctrl.shape[1] - 1, ctrl, **kw))


# -------------------------------------------------------------------------- equivalence

@pytest.mark.parametrize("painter", sorted(PAINTERS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_batched_lines_are_the_lines_of_the_loops(case, painter):
    """Every pixel of the batch is a pixel the loops painted, at the same depth."""
    canvas, oracle = payload(case, painter)
    assert np.any(canvas.color[..., :3] != 1.0), "the payload has to put ink down"
    np.testing.assert_array_equal(canvas.planes, oracle.planes)
    np.testing.assert_array_equal(canvas.depths, oracle.depths)
    np.testing.assert_array_equal(canvas.color, oracle.color)


@pytest.mark.parametrize("painter", sorted(PAINTERS))
def test_the_lines_are_drawn_in_the_order_they_are_drawn(painter):
    """A second draw takes the pixels of the first only on the tie it is owed.

    ``LessEqualDepth`` lets the later call cover the earlier one and ``Less`` leaves the
    earlier one holding its pixels, and both of those have to be what the loops do, which is
    the only way to tell from the picture that the batch enumerates its ribbons instance by
    instance and segment by segment instead of in the order its tiles happened to come out.
    """
    batched, oracle = PAINTERS[painter]
    ctrl = curves("bent")
    order, cam, kw = ctrl.shape[1] - 1, camera_for(), dict(n_segments=8)

    for less_equal in (True, False):
        one, two = canvas_for(), canvas_for()
        for canvas, paint in ((one, batched), (two, oracle)):
            for colour in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)):
                paint(canvas, cam, order, ctrl, colour=colour, less_equal=less_equal, **kw)
        np.testing.assert_array_equal(one.planes, two.planes)
        np.testing.assert_array_equal(one.depths, two.depths)
        np.testing.assert_array_equal(one.color, two.color)

    over, under = canvas_for(), canvas_for()
    for canvas, less in ((over, True), (under, False)):
        for colour in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)):
            batched(canvas, cam, order, ctrl, colour=colour, less_equal=less, **kw)
    won = np.any(over.color[..., :3] != under.color[..., :3], axis=-1)
    assert won.sum() > 0.01 * won.size, "the second call has to win some pixels on a tie"
    assert not np.any(over.depths != under.depths), "and lose none of them on depth"


@pytest.mark.parametrize("painter", sorted(PAINTERS))
def test_what_does_not_project_reaches_no_canvas(painter):
    """A curve with a coordinate of no value, or behind the eye, is not drawn at all.

    Both are thrown away before a single one of their segments is swept, and the batch has to
    throw them away in the same place: it is the one filter that cannot be recovered
    afterwards, because a swept ribbon of a mirrored instance writes its colour over whatever
    it reaches.  The behind-the-eye row is placed so that its corners fold back inside the
    window -- a row that merely landed off the edge would be dropped by the tiling alone and
    the filter would never be asked -- and both rows, made live, change the picture.
    """
    batched, _ = PAINTERS[painter]
    cam = camera_for()
    order, kw = 3, dict(n_segments=6)

    ctrl = curves("bent")
    assert np.isnan(ctrl[NA, 0, 0]) and np.isfinite(ctrl[BEHIND]).all()
    whole = batched(canvas_for(), cam, order, ctrl, **kw)
    assert np.any(whole.color[..., :3] != 1.0)
    for name in ("color", "depths"):
        other = batched(canvas_for(), cam, order, np.delete(ctrl, POISON, axis=0), **kw)
        np.testing.assert_array_equal(getattr(whole, name), getattr(other, name))

    px, py, _, w = raster._project(canvas_for(), cam, ctrl[BEHIND, :, :3])
    assert np.all(w < 0.0), "the row is meant to be behind the eye, not merely off the edge"
    assert np.all((px > 0) & (px < WIDTH) & (py > 0) & (py < HEIGHT)), (px, py)

    live = ctrl.copy()
    live[NA, 0, 0] = 0.25
    live[BEHIND, :, 2] = 0.0
    assert np.any(batched(canvas_for(), cam, order, live,
                          **kw).color[..., :3] != whole.color[..., :3])


@pytest.mark.parametrize("painter", sorted(PAINTERS))
def test_a_curve_is_culled_as_a_curve_and_not_as_a_set_of_segments(painter):
    """A curve half behind the eye is not half drawn.

    ``splines.vert:66-69`` culls the vertex of the *instance*, and the ``discard`` that feeds
    the fragment program takes the whole curve out of the draw.  It is the one filter whose
    work a tile cannot do afterwards, and so the one a batch could move to the wrong place:
    the behind-the-eye curve has no corner left anywhere the canvas is, but a curve crossing
    the eye plane has six of its seven samples in front of it and well inside the frame.  A
    batch that tested the three corners of each ribbon would draw five of its six segments
    while the loops draw none, and those five are 1800 pixels of black -- which is what makes
    the assertion that nothing is painted say something.
    """
    batched, _ = PAINTERS[painter]
    cam = camera_for()
    ctrl = curves("bent")
    order, kw = ctrl.shape[1] - 1, dict(n_segments=6)

    pos, _ = raster.spline_samples(order, ctrl[[CROSSING]], kw["n_segments"])
    _, _, _, w = raster._project(canvas_for(), cam, pos)
    assert (w > 0).sum() == 6 and (w < 0).sum() == 1, w

    for row in POISON:
        alone = batched(canvas_for(), cam, order, ctrl[[row]], **kw)
        np.testing.assert_array_equal(alone.color, np.full_like(alone.color, 1.0))

    # and the row is not simply invisible: the same curve, with its far end brought in front
    # of the eye, paints the diagonal it is
    live = ctrl[[CROSSING]].copy()
    live[:, :, 2] = 0.0
    assert np.any(batched(canvas_for(), cam, order, live, **kw).color[..., :3] != 1.0)

    whole = batched(canvas_for(), cam, order, ctrl, **kw)
    without = batched(canvas_for(), cam, order, np.delete(ctrl, POISON, axis=0), **kw)
    assert np.any(whole.color[..., :3] != 1.0)
    np.testing.assert_array_equal(whole.color, without.color)
    np.testing.assert_array_equal(whole.depths, without.depths)


@pytest.mark.parametrize("painter", sorted(PAINTERS))
def test_a_broken_curve_takes_its_whole_curve_with_it(painter):
    """One bad control point is not a bad segment: the instance is gone, every ribbon of it.

    ``splines.vert:66-69`` culls the instance out of the draw, which for a curve of twelve
    segments is twenty-four triangles the loops never see.  A batch that tested each ribbon on
    its own three corners would draw the eleven good segments of the broken curve and leave a
    nearly complete picture where the loops leave a whole curve missing.
    """
    batched, _ = PAINTERS[painter]
    cam = camera_for()
    ctrl = curves("dense")
    order, kw = ctrl.shape[1] - 1, dict(n_segments=12)

    for row in POISON:
        alone = batched(canvas_for(), cam, order, ctrl[[row]], **kw)
        np.testing.assert_array_equal(alone.color, np.full_like(alone.color, 1.0))

    broken = batched(canvas_for(), cam, order, ctrl, **kw)
    without = batched(canvas_for(), cam, order, np.delete(ctrl, POISON, axis=0), **kw)
    assert np.any(broken.color[..., :3] != 1.0)
    np.testing.assert_array_equal(broken.color, without.color)

    # the good rows are drawn by both and a broken row alone is nothing, so the only thing
    # that could put the missing curve back is a filter that works ribbon by ribbon
    mended = ctrl.copy()
    mended[NA, 0, 0] = 0.25
    assert np.any(batched(canvas_for(), cam, order, mended, **kw).color[..., :3]
                  != broken.color[..., :3])


@pytest.mark.parametrize("painter", sorted(PAINTERS))
def test_a_ribbon_running_off_the_frame_is_cut_at_the_edge(painter):
    """A box the canvas edge clips is cut where the loops cut it, on all four sides.

    The chords of ``OUTSIDE`` leave the frame through every side and through two corners, so
    their boxes hang past the last row and the last column of the canvas: measured on this
    camera, four of the eight usable chords reach row 401 or column 601 and their boxes span
    29 by 402 to 928 by 1071 pixels.  A painter that swept a box of the *unclipped* extent
    would spend its time off canvas, and one that clipped the box without clipping the sweep
    would index outside the planes.  Both are caught by the parity with the loops, which is
    what is asserted here; the ink at the frame edge is asserted so that the parity is about
    ribbons that actually reach it.
    """
    batched, oracle = PAINTERS[painter]
    kw = dict(CASES["offedge"])
    kind = kw.pop("kind")
    px = kw.pop("px", None)
    if px is not None:
        kw.update(dict(line_thickness=4.0 * px / HEIGHT) if painter == "edges"
                  else dict(width=px))
    ctrl = curves(kind)
    cam, order = camera_for(), ctrl.shape[1] - 1
    canvas, match = (f(canvas_for(), cam, order, ctrl, **kw) for f in (batched, oracle))
    np.testing.assert_array_equal(canvas.planes, match.planes)
    np.testing.assert_array_equal(canvas.depths, match.depths)
    np.testing.assert_array_equal(canvas.color, match.color)

    ink = np.any(canvas.color[..., :3] != 1.0, axis=-1)
    assert ink[-1].any() and ink[:, -1].any(), "the sweep has to reach the last row and column"
    assert ink[0].any() and ink[:, 0].any(), "and the first ones"

    # and the two chords that miss the frame altogether, in front of the eye at columns
    # 607..715 and -656..-470, leave nothing behind
    beside = batched(canvas_for(), cam, order, ctrl[[6, 7]], **kw)
    np.testing.assert_array_equal(beside.color, np.full_like(beside.color, 1.0))


@pytest.mark.parametrize("painter", sorted(PAINTERS))
def test_ribbons_of_very_different_boxes_share_one_stack(painter):
    """A stub of a pixel and a chord across the frame are drawn by one call, and both right.

    ``ragged`` lays a chord from x = -2.2 to x = 2.2 beside a chord 0.003 long, so one stack
    holds ribbons whose boxes are 1 by 2 and 254 by 811 pixels -- two orders of magnitude,
    and the short ones are the picture as much as the long ones: measured on this camera the
    fourteen usable instances of the payload are seven of each, subdivided into six.  A batch
    that sized its sweep to the widest box of a call would sweep a million pixels for a stub,
    and one that sized it to the narrowest would miss the chord.  The loops ask each ribbon
    for its own box, so the parity here is the assertion that a batch may group ribbons by
    box without moving a pixel.
    """
    batched, oracle = PAINTERS[painter]
    kw = dict(CASES["ragged"])
    kind = kw.pop("kind")
    px = kw.pop("px", None)
    if px is not None:
        kw.update(dict(line_thickness=4.0 * px / HEIGHT) if painter == "edges"
                  else dict(width=px))
    ctrl = curves(kind)
    cam, order = camera_for(), ctrl.shape[1] - 1
    canvas, match = (f(canvas_for(), cam, order, ctrl, **kw) for f in (batched, oracle))
    np.testing.assert_array_equal(canvas.planes, match.planes)
    np.testing.assert_array_equal(canvas.depths, match.depths)
    np.testing.assert_array_equal(canvas.color, match.color)

    long_only = batched(canvas_for(), cam, order, ctrl[list(range(0, 14, 2))], **kw)
    stub_only = batched(canvas_for(), cam, order, ctrl[list(range(1, 14, 2))], **kw)
    assert np.any(long_only.color[..., :3] != stub_only.color[..., :3])
    ink = lambda c: int(np.any(c.color[..., :3] != 1.0, axis=-1).sum())
    # Every pixel of either half is in the whole, which a window cut to the box of a chord
    # could easily fail to do for a stub lying beside it; and the whole has no pixel that
    # neither half has, which is what a window rounded up past a box -- or pulled back to the
    # edge of the canvas, where two windows of one box overlap -- could invent.  Measured on
    # this camera the halves are 13171 pixels of ink and 22 for the edge painter, 7221 and 11
    # for the wireframe, with no pixel in common between them, so both bounds are met exactly.
    assert max(ink(long_only), ink(stub_only)) <= ink(canvas) <= ink(long_only) + ink(stub_only)


def test_every_window_lies_on_the_canvas_and_covers_its_box():
    """A window is swept without asking whether a pixel is off canvas, so it has to fit.

    :func:`ngsplot.raster._flat_windows` trades the tile of :func:`_stack_tiles` for a window
    sized to each primitive's own box, which is what lets a ribbon of a pixel be swept over a
    pixel.  Two things follow from the box being clipped to the canvas and the window being
    rounded up outwards: the window of a box against the edge hangs past it, and a box wider
    than one window is handed out as several.  The first is answered by pulling the last
    window back to the edge -- which makes two windows of one box overlap, and asks that the
    overlap be swept twice rather than leaving a pixel unswept -- and both are answered here
    over boxes that touch every edge and corner of a small canvas.
    """
    canvas = raster.Canvas(37, 23)
    rng = np.random.default_rng(20240)
    rows, cols = np.arange(canvas.height), np.arange(canvas.width)
    for _ in range(40):
        n = int(rng.integers(1, 30))
        lo_r, lo_c = rng.integers(-4, 30, n), rng.integers(-4, 46, n)
        i0, j0 = np.maximum(lo_r, 0), np.maximum(lo_c, 0)
        i1 = np.minimum(lo_r + rng.integers(1, 40, n), canvas.height)
        j1 = np.minimum(lo_c + rng.integers(1, 60, n), canvas.width)
        keep = (i0 < i1) & (j0 < j1)
        i0, i1, j0, j1 = (a[keep] for a in (i0, i1, j0, j1))
        if i0.size == 0:
            continue
        # what no window has swept yet, starting from the box itself
        left = ((rows[None, :, None] >= i0[:, None, None])
                & (rows[None, :, None] < i1[:, None, None])
                & (cols[None, None, :] >= j0[:, None, None])
                & (cols[None, None, :] < j1[:, None, None]))
        handed = np.zeros(i0.size, dtype=bool)
        for h, wd, window_r, window_c, prims in raster._flat_windows(canvas, i0, i1, j0, j1):
            assert 1 <= h <= min(raster._STACK_WINDOW, canvas.height)
            assert 1 <= wd <= min(raster._STACK_WINDOW, canvas.width)
            assert window_r.min() >= 0 and window_c.min() >= 0
            assert (window_r + h).max() <= canvas.height, "a window hangs off the last row"
            assert (window_c + wd).max() <= canvas.width, "and off the last column"
            for k, r, c in zip(prims, window_r, window_c):
                handed[k] = True
                left[k, r:r + h, c:c + wd] = False
        assert handed.all(), "every box is handed out"
        assert not left.any(), "every pixel of a box is swept by some window of it"
