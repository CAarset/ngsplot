"""The batched sweep has to give the pixels the per-primitive definitions give.

``_paint_triangle`` and ``_paint_flat`` above are the rasterizer as the shaders describe
it: one triangle at a time, depth test against the running buffer, last writer winning a
tie.  That is worth keeping as the definition, but it does not scale -- the interpreter
runs its inner arithmetic once per triangle, so a 942-element mesh at 900 by 600 costs
17 seconds of it.  ``_paint_stack`` is the same rasterizer with the loop lifted out: it
takes the whole stack of a pass at once, in window space, and spends its numpy calls on
screen tiles instead of on primitives.  Lifting the loop out is only allowed where it
cannot change a pixel, so these tests hold the batched sweep against the per-primitive one
on the cases where the two could disagree:

* two primitives reaching the same sample with the same depth, where the order of the
  writes decides the colour and the order of the *tiles* must not;
* primitives that leave the canvas, whose bounding box is clipped by its edge;
* degenerate and non-finite primitives, which are culled rather than painted;
* the varyings, which belong to the triangle that won the sample and to no other.
"""

import numpy as np
import pytest

from ngsplot import raster
from ngsplot.raster import _STACK_TILE_MAX

WIDTH, HEIGHT = 64, 48


def canvas_for(**kwargs):
    return raster.Canvas(WIDTH, HEIGHT, **kwargs)


def soup(rng, count, *, width=WIDTH, height=HEIGHT, ties=0, culls=0):
    """A stack of ``(count)`` triangles in window space, as ``(x, y, w, z, rgb)``.

    Overlapping triangles of random shape, ``ties`` of them duplicated so that a sample is
    won by two primitives at exactly the same depth, and ``culls`` of them made degenerate,
    non-finite or wholly off the canvas.  ``w`` is the clip weight, which the vertex
    program leaves positive for anything it passes on, and ``z`` the window depth.
    """
    # an anchor plus a spread of a few pixels: small triangles, so the stack conflicts
    x = rng.uniform(-4.0, width + 4.0, (count, 1)) + rng.uniform(0.0, 6.0, (count, 3))
    y = rng.uniform(-4.0, height + 4.0, (count, 1)) + rng.uniform(0.0, 6.0, (count, 3))
    w = rng.uniform(1.0, 3.0, (count, 3))
    z = rng.uniform(0.2, 0.9, (count, 3))
    rgb = rng.uniform(0.0, 1.0, (count, 3))

    for t in range(ties):
        # a copy at the same depth: every sample both reaches is a tie
        s = t % max(count - ties, 1)
        x[count - ties + t], y[count - ties + t] = x[s], y[s]
        z[count - ties + t], w[count - ties + t] = z[s], w[s]
    for t in range(culls):
        # the three ways a primitive never reaches a fragment program at all
        case = t % 3
        s = count - ties - culls + t
        if case == 0:
            x[s] = x[s, 0]
            y[s] = y[s, 0]
        elif case == 1:
            x[s, 1] = np.nan
        else:
            x[s] += 4000.0
    return x, y, w, z, rgb


def reference_flat(canvas, x, y, w, z, rgb, *, less_equal=True, offset=False):
    """Paint the stack one primitive at a time, as the rasterizer is written.

    ``offset`` asks each primitive for the polygon offset of its own plane first, which is
    what the surface material does (:func:`ngsplot.raster.polygon_offset`).
    """
    for t in range(x.shape[0]):
        depth = z[t]
        if offset:
            depth = depth + raster.polygon_offset(x[t], y[t], depth)
        raster._paint_flat(canvas, x[t], y[t], w[t], depth, rgb[t], less_equal=less_equal)
    return canvas


# ------------------------------------------------------------------ flat triangle stacks

@pytest.mark.parametrize("samples", [1, raster.MSAA_4X])
@pytest.mark.parametrize("less_equal", [True, False])
@pytest.mark.parametrize("seed", [7, 8])
def test_a_stack_paints_what_the_primitives_paint(seed, samples, less_equal):
    rng = np.random.default_rng(seed)
    x, y, w, z, rgb = soup(rng, 40, ties=6, culls=6)

    batch = raster.Canvas(WIDTH, HEIGHT, samples=samples)
    raster._paint_stack(batch, x, y, w, z, rgb=rgb, less_equal=less_equal)
    reference = reference_flat(canvas_for(samples=samples), x, y, w, z, rgb,
                               less_equal=less_equal)

    np.testing.assert_array_equal(batch.planes, reference.planes)
    np.testing.assert_array_equal(batch.depths, reference.depths)


# ------------------------------------------------------------------------------ depth ties

def stacked(rgb, depth):
    """One triangle, drawn twice, in the two colours and at the two depths given.

    The triangle is the same both times and covers a good part of the small canvas, so
    every sample one of the two reaches the other reaches too, and what a sample comes out
    as says only which of the two the depth test kept.
    """
    x = np.array([[8.0, 34.0, 20.0], [8.0, 34.0, 20.0]])
    y = np.array([[6.0, 9.0, 36.0], [6.0, 9.0, 36.0]])
    w = np.full((2, 3), 2.0)
    z = np.array([[v, v, v] for v in depth])
    return x, y, w, z, rgb


def covered(canvas):
    """The samples something was written to, as one boolean over the sample planes.

    The canvas begins the colour it is cleared to, opaque, so what the painter touched is
    what has come away from it.
    """
    return np.any(canvas.planes[..., :3] != np.asarray(raster.BACKGROUND[:3]), -1)


def test_a_tie_goes_to_the_primitive_drawn_last():
    """Under ``LessEqualDepth`` -- what the renderer asks for, ``three/mesh.ts:57`` -- the
    fragment at an equal depth passes the test, so of two coincident primitives the later
    is the one seen.  This is the whole of why a wireframe drawn over a surface is visible
    at all: their depths are equal by construction, and only the drawing order separates
    them."""
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    x, y, w, z, _ = stacked(rgb, (0.5, 0.5))

    batch = raster.Canvas(WIDTH, HEIGHT)
    raster._paint_stack(batch, x, y, w, z, rgb=rgb)

    assert covered(batch).any()
    np.testing.assert_array_equal(batch.planes[..., 0][covered(batch)], 0.0)
    np.testing.assert_array_equal(batch.planes[..., 1][covered(batch)], 1.0)
    reference = reference_flat(canvas_for(), x, y, w, z, rgb)
    np.testing.assert_array_equal(batch.planes, reference.planes)


def test_a_tie_goes_to_the_primitive_drawn_first_when_the_test_is_strict():
    """With ``Less``, an equal depth fails the test and the first writer keeps the sample:
    the batched sweep has to follow the test it is given, not its own notion of order."""
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    x, y, w, z, _ = stacked(rgb, (0.5, 0.5))

    batch = raster.Canvas(WIDTH, HEIGHT)
    raster._paint_stack(batch, x, y, w, z, rgb=rgb, less_equal=False)

    assert covered(batch).any()
    np.testing.assert_array_equal(batch.planes[..., 0][covered(batch)], 1.0)
    np.testing.assert_array_equal(batch.planes[..., 1][covered(batch)], 0.0)
    reference = reference_flat(canvas_for(), x, y, w, z, rgb, less_equal=False)
    np.testing.assert_array_equal(batch.planes, reference.planes)


@pytest.mark.parametrize("less_equal", [True, False])
def test_the_nearer_primitive_keeps_the_sample_whichever_of_them_is_drawn_first(
        less_equal):
    """Drawing order decides a tie and nothing else: the nearer of two overlapping
    triangles is the one seen whichever of the two the painter met first, under either
    test."""
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    x, y, w, z, _ = stacked(rgb, (0.4, 0.6))

    for order in (slice(None), slice(None, None, -1)):
        batch = raster.Canvas(WIDTH, HEIGHT)
        raster._paint_stack(batch, x[order], y[order], w[order], z[order], rgb=rgb[order],
                            less_equal=less_equal)
        won = covered(batch)
        assert won.any()
        np.testing.assert_array_equal(batch.planes[..., 0][won], 1.0)
        np.testing.assert_array_equal(batch.planes[..., 1][won], 0.0)
        # the depth of the near one, 0.4, and not that of the far one: the perspective
        # divide is not exact, so the bound is what says which of the two won
        assert np.all(batch.depths[won] < 0.5)


# ---------------------------------------------------------------- what does not get here

def test_a_triangle_wholly_outside_the_canvas_paints_nothing():
    """Six triangles, beyond each edge and each corner.  Their bounding boxes are clipped
    away by the canvas edge before there is anything to sweep, which is the same test the
    per-primitive painter makes and the only one a primitive wholly off the screen can
    pass."""
    for dx, dy in ((-60, 0), (WIDTH, 0), (0, -50), (0, HEIGHT), (-30, -25),
                   (WIDTH - 4, HEIGHT - 4)):
        x, y, w, z, rgb = stacked(np.array([[1.0, 0.0, 0.0]]), (0.5,))
        x, y = x[:1] + dx, y[:1] + dy
        batch = raster.Canvas(WIDTH, HEIGHT)
        raster._paint_stack(batch, x, y, w[:1], z[:1], rgb=rgb[0])
        reference = reference_flat(canvas_for(), x, y, w[:1], z[:1], rgb)

        assert not covered(batch).any()
        np.testing.assert_array_equal(batch.planes, reference.planes)
        np.testing.assert_array_equal(batch.depths, reference.depths)


def test_a_triangle_across_the_canvas_edge_paints_the_inside_only():
    """One triangle over the top-left corner, its samples clipped by the edge it crosses.
    The box the sweep runs over is clipped to the canvas, so this is where an off-by-one in
    the clipping would put a row of pixels in or take one out."""
    x, y, w, z, rgb = stacked(np.array([[1.0, 0.0, 0.0]]), (0.5,))
    x, y = x[:1] - 12.0, y[:1] - 9.0
    batch = raster.Canvas(WIDTH, HEIGHT)
    raster._paint_stack(batch, x, y, w[:1], z[:1], rgb=rgb[0])
    reference = reference_flat(canvas_for(), x, y, w[:1], z[:1], rgb)

    np.testing.assert_array_equal(batch.planes, reference.planes)
    np.testing.assert_array_equal(batch.depths, reference.depths)
    won = covered(batch)
    assert won.any() and not won.all()
    assert not won[:, HEIGHT - 8:, WIDTH - 8:].any()


@pytest.mark.parametrize("corner", [0.0, np.nan, np.inf, -np.inf])
def test_a_triangle_of_no_area_or_no_number_paints_nothing(corner):
    """A determinant of nothing -- three corners in a line, or one that is not a number --
    is no triangle at all.  ``_sweep`` returns before it measures anything in that case and
    the stack drops the primitive on the same test; a corner that is not a number has also
    to poison no bounding box on its way past."""
    x, y, w, z, rgb = stacked(np.array([[1.0, 0.0, 0.0]]), (0.5,))
    x, y = x[:1].copy(), y[:1].copy()
    if corner == 0.0:
        y[:, 2] = y[:, 0] + (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]) / (x[:, 1] - x[:, 0])
    else:
        x[0, 1] = corner
    batch = raster.Canvas(WIDTH, HEIGHT)
    raster._paint_stack(batch, x, y, w[:1], z[:1], rgb=rgb[0])
    reference = reference_flat(canvas_for(), x, y, w[:1], z[:1], rgb)

    assert not covered(batch).any()
    np.testing.assert_array_equal(batch.planes, reference.planes)
    np.testing.assert_array_equal(batch.depths, reference.depths)


@pytest.mark.parametrize("weight", [-2.0, 0.0, np.nan])
def test_a_triangle_behind_the_camera_is_the_caller_to_cull(weight):
    """A clip weight that is no good is not the rasterizer's business: the vertex program
    hands on whatever it is and ``ngsplot.raster.paint_surface`` is what screens it out,
    before painting.  The stack therefore has to give a weight like this the same pixels the
    per-primitive sweep gives it, whatever those are, rather than tidy them away."""
    x, y, w, z, rgb = stacked(np.array([[1.0, 0.0, 0.0]]), (0.5,))
    w = np.where(np.arange(3) == 1, weight, w[0])[None]
    batch = raster.Canvas(WIDTH, HEIGHT)
    raster._paint_stack(batch, x[:1], y[:1], w, z[:1], rgb=rgb[0])
    reference = reference_flat(canvas_for(), x[:1], y[:1], w, z[:1], rgb)

    np.testing.assert_array_equal(batch.planes, reference.planes)
    np.testing.assert_array_equal(batch.depths, reference.depths)


def test_a_triangle_over_more_than_a_tile_of_its_own_size_spans_the_tiles():
    """A triangle far bigger than the widest tile the stack will use, so that it is listed
    in several tiles: a sample in any one of them is won by it alone, and the picture is
    that of the per-primitive sweep.  The canvas is a strip whose box crosses the cap of
    :data:`_STACK_TILE_MAX` several times over, and wide enough that the corners of the
    strip are outside the triangle, whatever the cap is made."""
    width, height = 4 * _STACK_TILE_MAX + 7, 60
    assert width > 3 * _STACK_TILE_MAX
    x = np.array([[width / 4.0, 3.0 * width / 4.0, width / 2.0]])
    y = np.array([[-8.0, -8.0, height + 8.0]])
    w = np.ones((1, 3))
    z = np.full((1, 3), 0.5)
    rgb = np.array([[1.0, 0.0, 0.0]])

    batch = raster.Canvas(width, height)
    raster._paint_stack(batch, x, y, w, z, rgb=rgb)
    reference = reference_flat(raster.Canvas(width, height), x, y, w, z, rgb)

    np.testing.assert_array_equal(batch.planes, reference.planes)
    np.testing.assert_array_equal(batch.depths, reference.depths)
    won = covered(batch)
    assert won[:, height - 1, width // 2].any()
    assert not won[:, 0, 0].any()


def test_every_box_is_offered_to_every_tile_of_its_own_size_class():
    """The binning is the only place that decides where a primitive is swept, and nothing
    below it can notice a tile it was never offered.

    So what it promises is checked on its own terms: the side of a tile is the power of two
    the box fits into, kept between the floor and the cap, which is what holds a primitive
    to two tiles per axis; and a primitive is listed once in every tile its box touches, in
    stack order, and in no tile it does not.  A soup of random triangles only happens to
    straddle a tile edge, which is how a missed tile hid before.
    """
    rng = np.random.default_rng(23)
    canvas = canvas_for()
    lo, hi = raster._STACK_TILE_MIN, raster._STACK_TILE_MAX

    def side(box):
        assert box > 0
        return min(max(1 << (box - 1).bit_length(), lo), hi)

    for _ in range(30):
        count = int(rng.integers(1, 7))
        i0 = rng.integers(0, canvas.height, count)
        j0 = rng.integers(0, canvas.width, count)
        i1 = np.minimum(i0 + rng.integers(1, 60, count), canvas.height)
        j1 = np.minimum(j0 + rng.integers(1, 60, count), canvas.width)
        boxes = [int(max(i1[p] - i0[p], j1[p] - j0[p])) for p in range(count)]
        want = []
        for p in range(count):
            s = side(boxes[p])
            for r in range(int(i0[p]) // s, (int(i1[p]) - 1) // s + 1):
                for c in range(int(j0[p]) // s, (int(j1[p]) - 1) // s + 1):
                    want.append((p, r, c))

        got = []
        for (ra, rb), (ca, cb), prims in raster._stack_tiles(canvas, i0, i1, j0, j1):
            assert 0 <= ra < rb <= canvas.height and 0 <= ca < cb <= canvas.width
            assert np.all(np.diff(prims) > 0), "a tile's primitives are not in stack order"
            for p in prims:
                s = side(boxes[int(p)])
                assert (ra % s, ca % s) == (0, 0), "a tile does not start on its own grid"
                assert rb - ra <= s and cb - ca <= s, "a tile is wider than its size class"
                got.append((int(p), ra // s, ca // s))
        assert sorted(got) == sorted(want), "the tiles a box touches are not the tiles it got"


# ---------------------------------------------------------------------------- the varyings

def shaded(fields, constants):
    """What ``function.frag`` makes of the varyings, in miniature: a flip for the back
    face, a light out towards the camera, the value as the brightness of the band.

    Written for either shape a fragment programme is asked about -- one flat row per winner,
    or the whole bounding box of one primitive -- which is the difference between the two
    painters this compares.
    """
    value, normal, position = fields
    back, = constants
    # a back face is the front face with its normal negated, which is what the shader does
    # -- and a select between a value and its negation is what keeps the arithmetic exact
    normal = np.where(np.asarray(back)[..., None], -normal, normal)
    lit = np.clip(0.5 + (normal + position) / 8.0, 0.0, 1.0)
    return lit * (0.3 + 0.7 * np.abs(value))


def test_a_fragment_is_shaded_once_for_its_pixel_and_not_once_per_sample():
    """Four sample planes of one pixel and one primitive are four coverages and one colour.

    The fragment programme runs at the centre of the pixel, which is where the batch takes
    its barycentrics too, so the four samples of one fragment have one colour between them
    and one lighting call.  Shading them separately gives the same picture, which is why the
    parity above cannot see the difference; this is what says the saving is really there.
    """
    rng = np.random.default_rng(17)
    x, y, w, z, _ = soup(rng, 12)
    count = x.shape[0]
    value = rng.uniform(-1.0, 1.0, (count, 3, 1))
    normal = rng.uniform(-1.0, 1.0, (count, 3, 3))
    position = rng.uniform(-2.0, 2.0, (count, 3, 3))
    back = ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
            - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])) > 0.0

    shaded_as = []

    def counting(fields, constants):
        shaded_as.append(fields[0].shape[0])
        return shaded(fields, constants)

    batch = canvas_for(samples=raster.MSAA_4X)
    raster._paint_stack(batch, x, y, w, z, varyings=(value, normal, position), attrs=(back,),
                        shade=counting)
    samples = int(np.count_nonzero(covered(batch)))
    fragments = sum(shaded_as)
    assert 4 * fragments >= samples, "a fragment was shaded for one sample and not its pixel"
    assert fragments < samples, "every sample of a fragment was lit on its own"


def reference_shade(canvas, x, y, w, z, fields, constants, *, less_equal=True):
    """Paint the stack one primitive at a time, shading each fragment as ``_paint_triangle``
    does: the varyings of the primitive that won the sample, interpolated once at the pixel
    centre, whatever the sample plane the colour is written to."""
    for t in range(x.shape[0]):
        def shade_at(interp):
            return shaded(tuple(interp(f[t]) for f in fields),
                          tuple(a[t] for a in constants))

        for plane, rows, cols, hit, depth, rgb in raster._sweep(
                canvas, x[t], y[t], w[t], z[t], less_equal=less_equal, shade=shade_at):
            raster._write(canvas, plane, rows[hit], cols[hit], rgb[hit], depth[hit])
    return canvas


@pytest.mark.parametrize("samples", [1, raster.MSAA_4X])
@pytest.mark.parametrize("less_equal", [True, False])
def test_a_stack_shades_the_varyings_of_the_triangle_that_won(samples, less_equal):
    """Every pixel of a shaded stack belongs to one triangle, and its colour comes from that
    triangle's own varyings and its own facing: not from the nearer triangle that lost the
    depth test, and not from a neighbouring triangle whose varyings would have bled in had
    the shading been done over the tile instead of over the winners."""
    rng = np.random.default_rng(11)
    x, y, w, z, _ = soup(rng, 30, ties=4, culls=3)
    count = x.shape[0]
    value = rng.uniform(-1.0, 1.0, (count, 3, 1))
    normal = rng.uniform(-1.0, 1.0, (count, 3, 3))
    position = rng.uniform(-2.0, 2.0, (count, 3, 3))
    # gl_FrontFacing in window space, y down, as _paint_triangle measures it
    back = ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
            - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])) > 0.0
    assert back.any() and not back.all()

    batch = raster.Canvas(WIDTH, HEIGHT, samples=samples)
    raster._paint_stack(batch, x, y, w, z, varyings=(value, normal, position), attrs=(back,),
                        shade=shaded, less_equal=less_equal)
    reference = reference_shade(canvas_for(samples=samples), x, y, w, z,
                                (value, normal, position), (back,), less_equal=less_equal)

    np.testing.assert_array_equal(batch.planes, reference.planes)
    np.testing.assert_array_equal(batch.depths, reference.depths)
    assert covered(batch).any() and not covered(batch).all()


def test_a_stack_is_either_flat_or_shaded():
    """A colour given as ``rgb`` and a colour asked for as ``shade`` are the two ways to
    paint, and one call cannot mean both: neither is the default, so leaving one out says so."""
    x, y, w, z, rgb = stacked(np.array([[1.0, 0.0, 0.0]]), (0.5,))
    with pytest.raises(ValueError):
        raster._paint_stack(raster.Canvas(WIDTH, HEIGHT), x, y, w, z)
    with pytest.raises(ValueError):
        raster._paint_stack(raster.Canvas(WIDTH, HEIGHT), x, y, w, z, rgb=rgb[0],
                            shade=shaded)


# ---------------------------------------------------------------------- the polygon offset

def test_the_surface_offset_lags_a_surface_behind_what_was_drawn_before_it():
    """A surface is pushed away from the camera by its own slope, and by a constant besides,
    so a primitive that was already drawn at its depth keeps the sample -- which is the whole
    reason the wireframe, drawn over a surface of its own geometry, needs the surface to ask
    for the offset and itself not to."""
    red, blue = np.array([[0.9, 0.1, 0.1]]), np.array([[0.1, 0.1, 0.9]])
    x, y, w, z, _ = stacked(np.array([red[0], blue[0]]), (0.5, 0.5))
    plain, lifted = canvas_for(), canvas_for()
    for canvas, offset in ((plain, False), (lifted, True)):
        raster._paint_stack(canvas, x[:1], y[:1], w[:1], z[:1], rgb=red[0])
        raster._paint_stack(canvas, x[1:], y[1:], w[1:], z[1:], rgb=blue[0], offset=offset)

    hit = covered(lifted)                       # every sample the triangle reaches
    assert hit.any()
    assert np.all(plain.planes[hit][:, :3] == blue)     # an exact tie, and the later one wins
    assert np.all(lifted.planes[hit][:, :3] == red)     # the offset has broken the tie

    # what it was pushed by: this triangle faces the camera straight on, so it has no slope to
    # add and takes only the unit part of the offset, once.  The depth is the corner's own
    # value back out of the interpolation, which round-trips to the last bit here and to a few
    # units in the last place there, so the push is what is measured, not the depth.
    alone = canvas_for()
    raster._paint_stack(alone, x[1:], y[1:], w[1:], z[1:], rgb=blue[0], offset=True)
    step = alone.depths[hit] - plain.depths[hit]
    assert np.all(step > 0.0)                                    # away from the camera
    assert np.allclose(step, raster.DEPTH_RESOLUTION)             # by one unit of it, once


@pytest.mark.parametrize("samples", [1, raster.MSAA_4X])
def test_the_offset_of_a_stack_is_the_offset_of_every_primitive_in_it(samples):
    """The offset of a plane is decided by the slope of that plane, and by the rounding of
    its own arithmetic: a stack that offset its planes as one would differ from the surface
    pass in exactly the pixels where two of its triangles are within an offset of each other,
    which is where the wireframe of one triangle meets the surface of its neighbour."""
    rng = np.random.default_rng(13)
    x, y, w, z, rgb = soup(rng, 24, ties=3)
    # slopes that an offset of one part in sixteen million can actually decide
    z = z * (1.0 + rng.uniform(-2e-7, 2e-7, z.shape))

    batch = raster.Canvas(WIDTH, HEIGHT, samples=samples)
    raster._paint_stack(batch, x, y, w, z, rgb=rgb, offset=True)
    reference = reference_flat(canvas_for(samples=samples), x, y, w, z, rgb, offset=True)

    np.testing.assert_array_equal(batch.planes, reference.planes)
    np.testing.assert_array_equal(batch.depths, reference.depths)


def test_the_offset_of_a_stack_is_the_offset_of_each_primitive_row_by_row():
    """The batched formula and the one primitive at a time one have to answer the same number
    for the same triangle, including for the triangles that answer nothing usable: an offset
    that is no number takes a surface out of the depth test altogether, and one that is a
    number smaller than the depth buffer's unit does not move it at all."""
    rng = np.random.default_rng(17)
    x = rng.uniform(0.0, 40.0, (60, 3))
    y = rng.uniform(0.0, 40.0, (60, 3))
    z = rng.uniform(0.1, 0.9, (60, 3))
    nasty = np.array([[[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],      # no area
                     [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],      # a line
                     [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [np.inf, 1.0, 2.0]],   # overflow
                     [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [1.0, np.nan, 2.0]],   # no number
                     [[0.0, 1e154, 2.0], [3.0, 4.0, 5e154], [1.0, 2.0, 3.0]]])  # steep
    x, y, z = np.r_[x, nasty[:, 0]], np.r_[y, nasty[:, 1]], np.r_[z, nasty[:, 2]]

    stacked = raster._polygon_offsets(x, y, z)
    assert np.array_equal(stacked, np.array([raster.polygon_offset(*t) for t in zip(x, y, z)]))
    assert (stacked[:60] > raster.DEPTH_RESOLUTION).all()
