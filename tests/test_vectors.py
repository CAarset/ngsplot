"""The arrow glyphs of a vector field: the recorded pictures of the real viewer, and the pieces of
``ClippingVectorsObject`` (``mesh.ts``) and ``vector_function.vert`` one at a time.

``tests/gt/vectors.png`` draws ``(0.8 - y, x - 0.5)`` on the unit square with the Objects folder's
Vectors box ticked, on the panel's grid of 10; ``vectors_grid.png`` the same with ``Draw``'s own
``vectors={"grid_size": 17, "offset": 0.4}``.  The arrows are compared over the picture we paint,
which leaves out the page's colour bar and axes, and arrow by arrow at their centres, where the
colour band of the arrow shows whatever lines lie under it.
"""

import numpy as np
import pytest

import gtdata
from ngsplot import camera, geometry, lighting, raster, scene

SCENES = ["vectors", "vectors_grid"]
MARGIN = 5          # pixels around what we painted, where the page painted it too
BAND_STEP = 40      # the least step in a channel between two neighbouring bands of the scenes


def painted(name, **kwargs):
    render_data = gtdata.render_data_of(name)
    width, height = gtdata.framebuffer_of(name)
    canvas = scene.paint(render_data, width=width, height=height, **kwargs)
    picture = gtdata.png_of(name)[:height, :width, :3].astype(int)
    return render_data, canvas, canvas.to_uint8()[..., :3].astype(int), picture


def arrows_of(render_data):
    """The arrows a recorded payload gets, as :func:`ngsplot.raster.paint_vectors` places them."""
    settings = scene.settings_of(render_data)
    ctrl, vectors = geometry.trig_points(render_data)
    return raster._arrow_instances(
        geometry.order_of(render_data), ctrl, center=render_data["mesh_center"],
        radius=render_data["mesh_radius"], grid_size=settings["vectors_grid_size"],
        offset=settings["vectors_offset"], n_segments=settings["subdivision"], vectors=vectors,
        deformation=0.0, mode="none", function_mode=settings["function_mode"],
        complex_scale=settings["complex_scale"], clipping_normal=settings["clipping_normal"],
        clipping_dist=settings["clipping_dist"])


@pytest.mark.parametrize("name", SCENES)
def test_the_arrows_are_the_recorded_picture(name):
    _, _, ours, picture = painted(name)
    rows, cols = np.nonzero(ours.min(axis=2) < 255)
    box = (slice(rows.min() - MARGIN, rows.max() + MARGIN + 1),
           slice(cols.min() - MARGIN, cols.max() + MARGIN + 1))
    difference = np.abs(ours[box] - picture[box]).max(axis=2)
    assert np.median(difference) == 0
    assert difference.mean() < 1.0          # about a quarter of a grey level, measured
    # what is left is the outline of a cone, a sample of four on the other side of it
    assert np.count_nonzero(difference > 120) < 5e-4 * difference.size   # 34 px of 168,000


@pytest.mark.parametrize("name", SCENES)
def test_every_arrow_has_the_recorded_colour(name):
    render_data, canvas, ours, picture = painted(name)
    centre = arrows_of(render_data)[0]
    view = camera.Camera.from_render_data(render_data, width=canvas.width, height=canvas.height)
    for x, y in np.rint(view.project(centre)[:, :2]).astype(int):
        here = (slice(y - 1, y + 2), slice(x - 1, x + 2))
        a = np.median(ours[here].reshape(-1, 3), axis=0)
        b = np.median(picture[here].reshape(-1, 3), axis=0)
        assert np.abs(a - b).max() < BAND_STEP / 2, (x, y, a, b)


def test_the_grid_is_the_panels_or_draws():
    assert scene.settings_of(gtdata.render_data_of("vectors"))["vectors_grid_size"] == 10
    settings = scene.settings_of(gtdata.render_data_of("vectors_grid"))
    assert (settings["vectors_grid_size"], settings["vectors_offset"]) == (17, 0.4)
    assert len(arrows_of(gtdata.render_data_of("vectors"))[0]) == 8 * 8


def test_cells_on_the_snapped_boundary_follow_the_top_left_rule():
    """On the grid of 17 the unit square's edges snap onto the centres of the cells next to them:
    the page keeps the cells on its left and bottom edges and leaves those on its right and top."""
    render_data = gtdata.render_data_of("vectors_grid")
    settings = scene.settings_of(render_data)
    ctrl, vectors = geometry.trig_points(render_data)
    pos, value, _ = raster.lattice(geometry.order_of(render_data), ctrl, vectors=vectors,
                                   n_segments=settings["subdivision"], function_mode=3.0)
    grid = raster.arrow_grid(render_data["mesh_center"], render_data["mesh_radius"], 17)
    _, hit = raster.vector_samples(pos, value, settings["subdivision"], grid, 17)
    assert hit[2, 2:14].all() and hit[2:14, 2].all()
    assert not hit[14].any() and not hit[:, 14].any()
    assert hit[2:14, 2:14].all() and hit.sum() == 12 * 12


def test_a_sample_is_the_field_at_the_centre_of_its_cell():
    render_data = gtdata.render_data_of("vectors")
    ctrl, vectors = geometry.trig_points(render_data)
    pos, value, _ = raster.lattice(geometry.order_of(render_data), ctrl, vectors=vectors,
                                   n_segments=5, function_mode=3.0)
    grid = raster.arrow_grid(render_data["mesh_center"], render_data["mesh_radius"], 10)
    samples, hit = raster.vector_samples(pos, value, 5, grid, 10, subpixel_bits=None)
    c, t1, t2, _ = grid
    i, j = np.nonzero(hit)
    p = c + (j[:, None] + 0.5 - 5) * t1 + (i[:, None] + 0.5 - 5) * t2
    inside = (p[:, :2] > 0).all(axis=1) & (p[:, :2] < 1).all(axis=1)
    assert inside.all() and hit.sum() == 64
    exact = np.stack([0.8 - p[:, 1], p[:, 0] - 0.5, 0 * p[:, 0]], axis=1)
    np.testing.assert_allclose(samples[i, j], exact, atol=1e-6)     # the payload is float32


def test_the_cone_is_threes():
    positions, normals, triangles = raster.cone()
    assert positions.shape == (43, 3) and triangles.shape == (20, 3)
    np.testing.assert_allclose(positions[:11], np.tile([0.0, 0.75, 0.0], (11, 1)))
    np.testing.assert_allclose(np.hypot(positions[11:22, 0], positions[11:22, 2]), 0.3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    np.testing.assert_allclose(normals[:22, 1], 0.2 / np.sqrt(1.04))
    a, b, c = (positions[triangles[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    middle = (a + b + c) / 3
    outward = np.where(np.arange(20)[:, None] < 10, middle * [1, 0, 1], [0.0, -1.0, 0.0])
    assert ((face * outward).sum(axis=1) > 0).all()   # counter-clockwise from outside: front faces


def test_an_arrow_turns_up_to_its_value():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3))
    q = raster.arrow_quaternion(v)
    up = raster.rotate(np.array([0.0, 1.0, 0.0]), q)
    np.testing.assert_allclose(up, v / np.linalg.norm(v, axis=1, keepdims=True), atol=1e-12)
    down = raster.rotate(np.array([0.0, 1.0, 0.0]), raster.arrow_quaternion([[0.0, -2.0, 0.0]]))
    np.testing.assert_allclose(down, [[0.0, -1.0, 0.0]], atol=1e-12)


def test_side_lights_add_to_the_diffuse_term_only():
    n = np.array([[0.2, -0.5, 0.8]])
    p = np.array([[0.1, 0.2, -3.0]])
    colour = np.array([[0.5, 0.25, 1.0]])
    plain = lighting.shade(colour, n, p)
    lit = lighting.shade(colour, n, p, side_lights=True)
    u = n[0] / np.linalg.norm(n[0])
    main = max(float(u @ lighting.LIGHT_DIR), 0.0)
    sides = sum(max(float(u @ s), 0.0) for s in lighting.SIDE_LIGHT_DIRS)
    extra = lighting.DIFFUSE * (-0.2 * main + 0.4 * sides)
    np.testing.assert_allclose(lit - plain, colour * extra, atol=1e-15)


def test_the_box_is_the_payloads_unless_told():
    render_data = gtdata.render_data_of("vectors")
    width, height = gtdata.framebuffer_of("vectors")
    _, _, ours, _ = painted("vectors")
    off = scene.paint(render_data, width=width, height=height, vectors=False).to_uint8()[..., :3]
    plain = scene.paint(gtdata.render_data_of("square_scalar"), width=width, height=height)
    assert (off != ours).any()
    unticked = dict(render_data, gui_settings={})
    np.testing.assert_array_equal(
        scene.paint(unticked, width=width, height=height).to_uint8(),
        scene.paint(render_data, width=width, height=height, vectors=False).to_uint8())
    finer = scene.paint(unticked, width=width, height=height, vectors={"grid_size": 17,
                                                                        "offset": 0.4})
    _, _, recorded_grid, _ = painted("vectors_grid")
    np.testing.assert_array_equal(finer.to_uint8()[..., :3], recorded_grid)
    assert plain is not None


def test_arrows_need_a_field_of_two_components_on_a_2d_mesh():
    scalar = gtdata.render_data_of("square_scalar")
    with pytest.raises(ValueError, match="two components"):
        scene.paint(scalar, width=90, height=60, vectors=True)
    scene.paint(dict(scalar, gui_settings={"Objects": {"Vectors": True}}), width=90, height=60)
    with pytest.raises(TypeError, match="grid_size, offset and colour"):
        scene.paint(gtdata.render_data_of("vectors"), width=90, height=60,
                    vectors={"grid": 3})


def test_a_box_fit_holds_the_arrows():
    render_data = gtdata.render_data_of("vectors")
    view = scene.frame(render_data, width=400, height=None, fit="box")
    settings = scene.settings_of(render_data)
    ctrl, vectors = geometry.trig_points(render_data)
    points = raster.vector_points(
        geometry.order_of(render_data), ctrl, center=render_data["mesh_center"],
        radius=render_data["mesh_radius"], grid_size=10, offset=0.0, n_segments=5,
        vectors=vectors, deformation=0.0, mode="none", function_mode=settings["function_mode"],
        complex_scale=settings["complex_scale"], clipping_normal=(0.0, 0.0, 1.0),
        clipping_dist=0.0)
    x, y = view.project(points)[:, :2].T
    assert (x >= 0).all() and (x <= view.width).all() and (y >= 0).all() and (y <= view.height).all()
    bare = scene.frame(dict(render_data, gui_settings={}), width=400, height=None, fit="box")
    assert bare.height <= view.height


def test_arrows_of_one_colour_are_that_colour_lit():
    """vectors={"colour": ...} paints every cone in that colour under the arrows' light, in place of its value's."""
    render_data = gtdata.render_data_of("vectors")
    width, height = gtdata.framebuffer_of("vectors")
    dark = scene.paint(render_data, width=width, height=height, vectors={"colour": "0.1"}).to_uint8()[..., :3]
    centre = arrows_of(render_data)[0]
    view = camera.Camera.from_render_data(render_data, width=width, height=height)
    for x, y in np.rint(view.project(centre)[:, :2]).astype(int):
        pixel = np.median(dark[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3), axis=0)
        assert pixel.max() - pixel.min() <= 2 and pixel.max() < 128, (x, y, pixel)  # grey: 0.1 lit, the highlight white
