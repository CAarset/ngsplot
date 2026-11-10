"""Painting a surface: which pixels, which colour, how much light.

The known answers here come from the shaders the renderer compiles -- the band choice of
``getColor`` and the light of ``calcLight``, the lattice of ``trigsplines.vert`` -- and not
from what this module happens to do.  The canvas is 601 by 401 for a mesh centred on the
origin with radius 1, which makes the object matrix the identity and drops the object
origin on the exact centre of pixel (200, 300), so the pixel arithmetic comes out whole.
"""

import numpy as np
import pytest

from ngsplot import camera, colormap, lighting, raster

# one flat triangle in the z = 0 plane, big enough to measure, whole inside the canvas
A = [-0.4, -0.25, 0.0]
B = [0.4, -0.25, 0.0]
C = [0.0, 0.35, 0.0]
CORNERS = np.array([A, B, C])

UP = [0.0, 0.0, 1.0]


def canvas_for(**kwargs):
    return raster.Canvas(601, 401, **kwargs)


def camera_for(**kwargs):
    cam = camera.Camera([0, 0, 0], 1.0, width=601, height=401, **kwargs)
    # the colour tests read object points straight off the screen, which needs no scaling
    np.testing.assert_allclose(cam.object, np.eye(4), atol=0)
    return cam


def element(corners=CORNERS, values=0.5):
    """One order-1 element: three corners as control points, ``values`` as the scalar.

    The Bernstein slots of an order-1 triangle hold the three corners in an order fixed by
    the parametrization alone, so nothing below cares which is which: a constant scalar is
    constant everywhere either way, and a scalar linear in ``x`` is reproduced exactly.
    """
    ctrl = np.zeros((1, 3, 4))
    ctrl[0, :, :3] = np.asarray(corners, dtype=np.float64)
    ctrl[0, :, 3] = np.asarray(values, dtype=np.float64) * np.ones(3)
    return ctrl


def paint(canvas, cam, ctrl, *, order=1, vmin=-1.0, vmax=1.0, **kwargs):
    raster.paint_surface(canvas, cam, order, ctrl, vmin=vmin, vmax=vmax, **kwargs)
    return canvas


def painted(canvas):
    """Pixels the surface wrote into, i.e. those no longer the white background."""
    return np.any(canvas.color[..., :3] != 1.0, axis=-1)


def object_points(rows, cols, cam):
    """View and object points of the z = 0 plane behind the given pixel centres.

    The view points hold for any camera -- a pixel centre of a flat element facing the
    eye sits on the ray that hits ``Z = -3`` in view space, and the eye is at ``(0, 0, 3)``
    looking down ``-z``, which makes that ray easy to invert.  The object points read the
    view points as object points, so they need ``camera_for``'s identity object matrix.
    """
    x = (cols + 0.5) / cam.width * 2.0 - 1.0
    y = 1.0 - (rows + 0.5) / cam.height * 2.0
    back = cam.position[2]
    view = np.stack([x * back / cam.projection[0, 0], y * back / cam.projection[1, 1],
                     np.full_like(x, -back)], axis=-1)
    return view + np.array([0.0, 0.0, back]), view


def assert_between(got, bounds):
    """Fail with the count of pixels that escaped the band, rather than a bare False."""
    lo, hi = bounds
    assert not (lo & ~got).any(), f"{(lo & ~got).sum()} pixels inside left unpainted"
    assert not (got & ~hi).any(), f"{(got & ~hi).sum()} pixels outside painted"


def side(px, py, p, q):
    return (q[0] - p[0]) * (py - p[1]) - (q[1] - p[1]) * (px - p[0])


def inside_band(rows, cols, tri, tol=1e-6):
    """Two masks for a triangle: certainly inside, and not certainly outside.

    A pixel belongs to the triangle when its centre, ``(col + 0.5, row + 0.5)``, lies on
    the inside of all three edges, which is what the rasterizer asks of its sub-triangles.
    ``tol`` widens that test by pixels in both directions: the rasterizer places a
    lattice vertex at a slightly different float than :meth:`Camera.project` places the
    corner of the element, since its own coordinates come out of ``j / n_segments``, so
    a centre that lands exactly on an edge can fall either way.
    """
    turn = np.sign(side(tri[2][0], tri[2][1], tri[0], tri[1]))
    lo = np.ones(np.shape(rows), dtype=bool)
    hi = np.ones(np.shape(rows), dtype=bool)
    for p, q in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
        length = np.hypot(q[0] - p[0], q[1] - p[1])
        near = turn * side(cols + 0.5, rows + 0.5, p, q) / length
        lo &= near > tol
        hi &= near > -tol
    return lo, hi


def shade_row(colour, p_view, inside=False, normal=UP):
    """``calcLight`` for a row of view-space points lit by a flat surface normal."""
    n = len(p_view)
    return lighting.shade(np.tile(np.asarray(colour, dtype=np.float64), (n, 1)),
                          np.tile(np.asarray(normal, dtype=np.float64), (n, 1)),
                          p_view, inside=inside)


def resolve(rgb, p_view, colours):
    """Per pixel, which of ``colours`` -- seen from the front or the back -- it is.

    Returns the index into ``colours``, the ``inside`` flag and the distance to that
    candidate.  The tests below assert both the band and the light of every single pixel
    this way, without hard-coding which way the canvas happens to face.
    """
    stack, tags = [], []
    for k, colour in enumerate(colours):
        for inside in (False, True):
            stack.append(shade_row(colour, p_view, inside=inside))
            tags.append((k, inside))
    distance = np.abs(np.stack(stack) - rgb[None]).max(axis=-1)
    best = np.argmin(distance, axis=0)
    return (np.array([tags[b][0] for b in best]),
            np.array([tags[b][1] for b in best]),
            distance[best, np.arange(len(p_view))])


# ----------------------------------------------------------------- the lattice the vertex program reads


def test_the_lattice_lifts_the_surface_by_the_deformation_it_is_given():
    ctrl = element(values=[0.0, 1.0, 2.0])
    pos, value, nrm = raster.lattice(1, ctrl, deformation=0.5, mode="2d")
    assert pos.shape == value.shape == nrm.shape == (1, 21, 3)
    # the average of an affine field over the lattice is its value at the centroid, which
    # for the n = 5 lattice is the mean of the three corners
    assert pos[..., 2].mean() == pytest.approx(0.5 * np.mean([0.0, 1.0, 2.0]), abs=1e-12)
    assert value[..., 0] == pytest.approx(2.0 * pos[..., 2], abs=1e-12)
    assert np.all(np.isfinite(nrm))


# --------------------------------------------------------------------------------- coverage


def test_an_element_covers_exactly_its_own_projection():
    cam = camera_for()
    canvas = paint(canvas_for(samples=1), cam, element())
    tri = cam.project(CORNERS)[:, :2]
    rows, cols = np.indices((canvas.height, canvas.width))
    assert_between(painted(canvas), inside_band(rows, cols, tri))


@pytest.mark.parametrize("euler", [(24.0, 0.0, 0.0), (0.0, -18.0, 0.0), (15.0, 30.0, -10.0)])
def test_a_turntable_view_covers_exactly_its_own_projection(euler):
    """Turning the mesh does not break the rasterizer, and the plane stays affine.

    The renderer never moves its camera: it spins the object, which for the z = 0 plane is
    a similarity, so the projection of a triangle is still exactly a triangle.
    """
    cam = camera.Camera([0, 0, 0], 1.0, width=601, height=401,
                        euler_angles=np.asarray(euler) * camera.DEG)
    canvas = paint(canvas_for(samples=1), cam, element())
    tri = cam.project(CORNERS)[:, :2]
    rows, cols = np.indices((canvas.height, canvas.width))
    assert_between(painted(canvas), inside_band(rows, cols, tri))


def test_the_painted_area_and_centroid_are_the_projected_triangle():
    """As many pixels as the triangle holds, centred where the triangle is centred.

    A rasterizer that drops a strip, mirrors the picture or reads the projection
    the wrong way round moves the centroid by pixels, so bracketing it between the
    two masks of :func:`inside_band` -- which differ by the handful of pixels that sit
    on an edge -- is tight to a hundredth of a pixel.  The shoelace area of the
    projected triangle enters as a coarse check on the projection itself.
    """
    cam = camera_for()
    canvas = paint(canvas_for(samples=1), cam, element())
    tri = cam.project(CORNERS)[:, :2]
    area = 0.5 * abs((tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1])
                     - (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1]))
    got = painted(canvas)
    assert got.sum() == pytest.approx(area, rel=0.03)

    rows, cols = np.indices((canvas.height, canvas.width))
    lo, hi = inside_band(rows, cols, tri)
    assert lo.sum() <= got.sum() <= hi.sum()
    centres = np.stack([cols + 0.5, rows + 0.5], axis=-1)
    mean = lambda mask: centres[mask].mean(axis=0)
    low, high, centre = mean(lo), mean(hi), mean(got)
    assert np.all(centre >= np.minimum(low, high) - 0.02)
    assert np.all(centre <= np.maximum(low, high) + 0.02)


def test_nothing_is_painted_outside_the_element_bounding_box():
    cam = camera_for()
    canvas = paint(canvas_for(), cam, element())
    rows, cols = np.nonzero(painted(canvas))
    lo, hi = cam.project(CORNERS)[:, :2].min(axis=0), cam.project(CORNERS)[:, :2].max(axis=0)
    assert cols.min() >= np.floor(lo[0]) - 1 and cols.max() <= np.ceil(hi[0]) + 1
    assert rows.min() >= np.floor(lo[1]) - 1 and rows.max() <= np.ceil(hi[1]) + 1


# ------------------------------------------------------------------------ colour and light


def test_every_pixel_of_a_constant_element_is_the_lit_band_colour():
    cam = camera_for()
    canvas = paint(canvas_for(samples=1), cam, element(values=0.5))
    rows, cols = np.nonzero(painted(canvas))
    _, view = object_points(rows, cols, cam)
    pal = colormap.palette()
    band = colormap.band_index(np.array([0.5]), -1.0, 1.0)[0]
    got, inside, distance = resolve(canvas.color[rows, cols, :3], view, [pal[band]])
    assert distance.max() == pytest.approx(0.0, abs=1e-12)
    assert np.all(inside == inside[0])
    assert not np.all(np.isclose(canvas.color[rows, cols, :3], 1.0))


def test_the_lighting_reads_the_point_as_the_camera_sees_it():
    """``vPos_`` is ``modelViewMatrix * pos``, not ``viewMatrix * pos``.

    The renderer fits a mesh to the view by scaling ``1 / mesh_radius`` about its centre,
    and the specular term of ``calcLight`` is the angle between the reflected light and
    the direction to the eye -- so the same flat element lit at the same pixel is
    brighter the more the mesh is scaled up.  Only a camera whose object matrix is not
    the identity can tell the two apart, which is why the colour tests above use radius 1
    at the origin and this one takes the ground-truth square's own centre and radius.
    """
    cam = camera.Camera([0.5, 0.5, 0.0], 0.7071067811865476, width=601, height=401)
    assert not np.allclose(cam.object, np.eye(4))
    canvas = paint(canvas_for(samples=1), cam,
                   element(corners=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], values=0.5))
    rows, cols = np.nonzero(painted(canvas))
    _, view = object_points(rows, cols, cam)
    assert np.abs(view[:, 2] + 3.0).max() == pytest.approx(0.0, abs=1e-12)
    pal = colormap.palette()
    band = colormap.band_index(np.array([0.5]), -1.0, 1.0)[0]
    got = canvas.color[rows, cols, :3]
    distance = np.abs(got - shade_row(pal[band], view)).max()
    assert distance == pytest.approx(0.0, abs=1e-12)


def test_the_back_side_gets_the_flipped_normal_and_half_the_light():
    """``gl_FrontFacing`` and ``inside`` are two statements that always agree.

    The two windings of one flat triangle are the same geometry seen from its two
    sides.  The orientation the window counts as front keeps its normal and its full
    light; the other has its normal negated -- the flip is the only reason the back
    side is lit at all -- and is then multiplied by ``dimm = 0.5`` from
    ``dark_backside``: ``function.frag:74-79`` sets both in the same branch, under
    separate conditions.  Which real winding counts as front is a mirroring convention
    between ngsolve and three.js that the ground-truth picture settles, so this test
    pins the pairing of the two effects rather than a name for the winding.
    """
    cam = camera_for()
    first = paint(canvas_for(samples=1), cam, element())
    second = paint(canvas_for(samples=1), cam, element(corners=[B, A, C]))
    assert np.array_equal(painted(first), painted(second)), "the two sides cover one triangle"
    rows, cols = np.nonzero(painted(first))
    _, view = object_points(rows, cols, cam)
    pal = colormap.palette()
    band = colormap.band_index(np.array([0.5]), -1.0, 1.0)[0]
    plain = shade_row(pal[band], view)
    front = first.color[rows[0], cols[0], 0] > second.color[rows[0], cols[0], 0]
    bright, dim = (first, second) if front else (second, first)
    assert np.abs(bright.color[rows, cols, :3] - plain).max() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(dim.color[rows, cols, :3] - 0.5 * plain).max() == pytest.approx(0.0, abs=1e-12)
    assert (dim.color[rows, cols, :3] > 0.25 * pal[band]).all(), "the back side is not unlit"


def test_a_surface_dimmed_for_back_faces_can_be_left_bright():
    cam = camera_for()
    pal = colormap.palette()
    band = colormap.band_index(np.array([0.5]), -1.0, 1.0)[0]
    for corners in (CORNERS, [B, A, C]):
        plain = paint(canvas_for(samples=1), cam, element(corners=corners),
                      dark_backside=False)
        rows, cols = np.nonzero(painted(plain))
        _, view = object_points(rows, cols, cam)
        assert np.abs(plain.color[rows, cols, :3] - shade_row(pal[band], view)).max() < 1e-12

        dimmed = paint(canvas_for(samples=1), cam, element(corners=corners))
        other = dimmed.color[rows, cols, :3]
        assert (np.abs(other - plain.color[rows, cols, :3]).max() < 1e-12
                or np.abs(other - 0.5 * plain.color[rows, cols, :3]).max() < 1e-12)


def test_a_value_linear_across_the_element_shows_every_band():
    cam = camera_for()
    values = (CORNERS[:, 0] + 0.4) / 0.8      # 0 on the left corner, 1 on the right
    canvas = paint(canvas_for(samples=1), cam, element(values=values), vmin=0.0, vmax=1.0)
    rows, cols = np.nonzero(painted(canvas))
    obj, view = object_points(rows, cols, cam)
    got, inside, distance = resolve(canvas.color[rows, cols, :3], view, colormap.palette())
    assert distance.max() < 1e-9, "no palette colour explains this pixel"
    assert np.all(inside == inside[0])
    scaled = (obj[:, 0] + 0.4) / 0.8 * colormap.NCOLORS
    away = np.abs(scaled - np.round(scaled)) > 1e-6      # a centre exactly on a boundary
    assert np.array_equal(got[away], colormap.band_index(scaled[away] / colormap.NCOLORS,
                                                         0.0, 1.0))
    assert len(set(got.tolist())) == colormap.NCOLORS


def test_values_beyond_the_range_stick_to_the_two_end_colours():
    cam = camera_for()
    canvas = paint(canvas_for(samples=1), cam, element(values=4.0 * CORNERS[:, 0]))
    rows, cols = np.nonzero(painted(canvas))
    obj, view = object_points(rows, cols, cam)
    got, _, distance = resolve(canvas.color[rows, cols, :3], view, colormap.palette())
    assert distance.max() < 1e-9
    assert np.array_equal(got, colormap.band_index(4.0 * obj[:, 0], -1.0, 1.0))
    assert {0, colormap.NCOLORS - 1} <= set(got.tolist())


def test_a_coarser_palette_gives_fewer_bands():
    cam = camera_for()
    values = (CORNERS[:, 0] + 0.4) / 0.8
    canvas = paint(canvas_for(samples=1), cam, element(values=values), vmin=0.0, vmax=1.0,
                   pal=colormap.palette(2))
    rows, cols = np.nonzero(painted(canvas))
    _, view = object_points(rows, cols, cam)
    got, _, distance = resolve(canvas.color[rows, cols, :3], view, colormap.palette(2))
    assert distance.max() < 1e-9
    assert set(got.tolist()) == {0, 1}


def test_a_surface_without_function_values_is_grey_and_lit():
    cam = camera_for()
    canvas = paint(canvas_for(samples=1), cam, element(), function_values=False)
    rows, cols = np.nonzero(painted(canvas))
    _, view = object_points(rows, cols, cam)
    _, _, distance = resolve(canvas.color[rows, cols, :3], view,
                             [np.full(3, raster.MESH_GREY)])
    assert distance.max() < 1e-9
    # grey stays grey: no channel leads another
    assert np.ptp(canvas.color[rows, cols, :3], axis=-1).max() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------------- depth


def two_elements():
    """One element in front of another, one behind, both centred on the same direction."""
    return (element(corners=CORNERS + [0.0, 0.0, 0.3], values=0.9),
            element(corners=CORNERS + [0.0, 0.0, -0.3], values=-0.9))


def test_the_nearer_element_wins_where_they_overlap():
    cam = camera_for()
    near, far = two_elements()
    canvas = paint(canvas_for(samples=1), cam, np.vstack([near, far]))
    rows, cols = np.nonzero(painted(canvas))
    _, view = object_points(rows, cols, cam)
    got, _, distance = resolve(canvas.color[rows, cols, :3], view, colormap.palette())
    assert distance.max() < 1e-9
    centre = np.argmin((rows - 200.0) ** 2 + (cols - 300.0) ** 2)
    assert got[centre] == colormap.band_index(np.array([0.9]), -1.0, 1.0)[0]


def test_a_farther_element_painted_afterwards_cannot_take_a_pixel_back():
    cam = camera_for()
    near, far = two_elements()
    canvas = paint(canvas_for(samples=1), cam, near)
    color, depth = canvas.color.copy(), canvas.depth.copy()

    paint(canvas, cam, far)
    assert np.array_equal(canvas.color, color)
    assert np.array_equal(canvas.depth, depth)
    assert painted(paint(canvas_for(samples=1), cam, far)).any(), "vacuous: far paints nothing"


def test_equal_depth_overwrites_by_default_because_three_uses_less_equal():
    cam = camera_for()
    pal = colormap.palette()

    canvas = paint(canvas_for(samples=1), cam, element(values=-0.9))
    rows, cols = np.nonzero(painted(canvas))
    _, view = object_points(rows, cols, cam)
    before, _, _ = resolve(canvas.color[rows, cols, :3], view, pal)

    paint(canvas, cam, element(values=0.9))
    after, _, _ = resolve(canvas.color[rows, cols, :3], view, pal)
    assert np.all(after != before), "LessEqualDepth should let the second element through"

    strict = paint(canvas_for(samples=1), cam, element(values=-0.9))
    alone = canvas_for(samples=1)
    paint(alone, cam, element(values=-0.9))
    raster.paint_surface(strict, cam, 1, element(values=0.9), vmin=-1.0, vmax=1.0,
                         less_equal=False)
    assert np.array_equal(strict.color, alone.color)
    assert np.array_equal(strict.depth, alone.depth)


# ----------------------------------------------------------------------- degenerate input


def test_a_nan_value_removes_the_element_it_belongs_to():
    cam = camera_for()
    left = element(corners=[[-0.8, -0.25, 0.0], [-0.1, -0.25, 0.0], [-0.4, 0.3, 0.0]])
    right = element(corners=[[0.1, -0.25, 0.0], [0.8, -0.25, 0.0], [0.4, 0.3, 0.0]])
    right[0, 1, 3] = np.nan
    # one NaN control point poisons every lattice vertex of that element, so the vertex
    # program zeroes the position of all of them and the whole element disappears
    canvas = paint(canvas_for(), cam, np.vstack([left, right]))
    assert painted(canvas).any()
    assert np.array_equal(painted(canvas), painted(paint(canvas_for(), cam, left)))


def test_an_element_with_a_nan_position_paints_nothing():
    cam = camera_for()
    broken = element()
    broken[0, 2, 0] = np.nan
    assert not painted(paint(canvas_for(), cam, broken)).any()


def test_a_degenerate_element_paints_nothing():
    cam = camera_for()
    assert not painted(paint(canvas_for(), cam, element(corners=[A, B, [0.0, -0.25, 0.0]]))).any()


def test_an_element_behind_the_camera_paints_nothing():
    cam = camera_for()
    assert not painted(paint(canvas_for(), cam, element(corners=CORNERS + [0.0, 0.0, 3.5]))).any()


def test_an_element_off_the_side_paints_nothing():
    cam = camera_for()
    assert not painted(paint(canvas_for(), cam, element(corners=CORNERS + [8.0, 0.0, 0.0]))).any()


def test_an_element_only_half_on_screen_is_clipped_to_the_canvas():
    cam = camera_for()
    corners = CORNERS + [1.05, 0.0, 0.0]
    canvas = paint(canvas_for(samples=1), cam, element(corners=corners))
    assert painted(canvas).any()
    tri = cam.project(corners)[:, :2]
    rows, cols = np.indices((canvas.height, canvas.width))
    assert_between(painted(canvas), inside_band(rows, cols, tri))


def test_colouring_by_depth_is_refused():
    cam = camera_for()
    with pytest.raises(NotImplementedError, match="depth"):
        paint(canvas_for(), cam, element(), function_mode=8)


# -------------------------------------------------------------------------- antialiasing


def test_the_sample_grid_softens_the_silhouette():
    cam = camera_for()
    sharp = paint(canvas_for(samples=1), cam, element())
    soft = paint(canvas_for(samples=2), cam, element())
    assert np.all(painted(soft)[painted(sharp)]), "samples=2 must cover all of samples=1"
    assert painted(soft).sum() > painted(sharp).sum()

    band = colormap.band_index(np.array([0.5]), -1.0, 1.0)[0]
    rows, cols = np.nonzero(painted(soft) & ~painted(sharp))
    _, view = object_points(rows, cols, cam)
    full = shade_row(colormap.palette()[band], view)
    for i in range(len(rows)):
        here, there = soft.color[rows[i], cols[i], :3] - 1.0, full[i] - 1.0
        cov = here[np.argmax(np.abs(there))] / there[np.argmax(np.abs(there))]
        assert 0.0 < cov < 1.0
        assert np.abs(here - cov * there).max() < 1e-9
        assert abs(cov * 4 - round(cov * 4)) < 1e-9, "coverage comes in quarters"


# ----------------------------------------------------------------------- the framebuffer


def test_an_untouched_canvas_is_white():
    canvas = canvas_for()
    assert canvas.color.shape == (401, 601, 4)
    assert np.array_equal(canvas.color, np.ones((401, 601, 4)))
    assert np.all(np.isinf(canvas.depth))


def test_to_uint8_clamps_before_it_rounds():
    canvas = canvas_for(samples=1)
    canvas.planes[0, 0, 0] = (1.4, 0.5, -0.2, 1.0)
    canvas.planes[0, 0, 1] = (0.999, 0.0019, 0.5, 1.0)
    got = canvas.to_uint8()
    assert got.dtype == np.uint8
    assert got[0, 0].tolist() == [255, 128, 0, 255]
    assert got[0, 1].tolist() == [255, 0, 128, 255]


def test_a_canvas_needs_a_size_and_some_samples():
    with pytest.raises(ValueError, match="size"):
        raster.Canvas(0, 10)
    with pytest.raises(ValueError, match="samples"):
        raster.Canvas(10, 10, samples=0)
