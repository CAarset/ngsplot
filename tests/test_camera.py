"""Tests for the camera: the object matrix, the view, the projection, and pixel mapping.

Known answers come from the real renderer:

* ``tests/gt/camera.json`` — object, view and projection matrices of the ground-truth
  scenes, written by ``tools/probe_camera.py``.
* ``tests/gt/euler_orders.json`` — the object matrix for all six Euler orders, written by
  ``tools/probe_euler_orders.py``.
* hand-written numbers for the cases that need no renderer.

The matrices are compared as three.js sends them: ``Matrix4.elements`` is column-major, so
the flattened numpy matrix transposed.
"""

import json
import os

import numpy as np
import pytest

from ngsplot import camera

from gtdata import GT, render_data_of, settings_of

# the mesh of the ground-truth square: centre and radius as BuildRenderData reports them
CENTER = [0.5, 0.5, 1.1125369292536007e-308]
RADIUS = 0.7071067811865476
# 1 / mesh_radius of that mesh, exactly as the renderer computes it
SCALE = 1 / RADIUS


def _json(name):
    with open(os.path.join(GT, name)) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scenes():
    return _json("camera.json")


@pytest.fixture(scope="module")
def eulers():
    return _json("euler_orders.json")


def as_matrix(elements):
    return np.asarray(elements, dtype=np.float64).reshape(4, 4).T


# --------------------------------------------------------------------------- conversions


def test_three_element_order_is_column_major(scenes):
    elements = scenes["square_rotated"]["controls_mat"]
    m = camera.matrix_from_three(elements)
    assert camera.three_from_matrix(m).tolist() == elements
    # m[i, j] is elements[j * 4 + i]
    assert m[1, 2] == elements[2 * 4 + 1]


def test_chain_applies_arguments_left_to_right():
    p = np.array([1.0, 0.0, 0.0])
    turn = camera.rotation_from_euler([0, 0, 90])
    move = camera.translation([1, 0, 0])
    turned = camera.transform_points(p[None, :], turn)[0]
    want = camera.transform_points(turned[None, :], move)[0]
    got = camera.transform_points(p[None, :], camera.chain(turn, move))[0]
    np.testing.assert_allclose(got, want, atol=1e-15)
    np.testing.assert_allclose(got, [1.0, 1.0, 0.0], atol=1e-15)


# ---------------------------------------------------------------------------- rotations


def test_euler_xyz_is_the_product_of_the_elementary_rotations():
    angles = [10.0, 20.0, 30.0]
    expected = (camera.rotation_about_axis([1, 0, 0], np.deg2rad(10))
                @ camera.rotation_about_axis([0, 1, 0], np.deg2rad(20))
                @ camera.rotation_about_axis([0, 0, 1], np.deg2rad(30)))
    np.testing.assert_allclose(camera.rotation_from_euler(angles, "XYZ"), expected,
                               atol=1e-15)


@pytest.mark.parametrize("order", ["XYZ", "YXZ", "XZY", "ZXY", "YZX", "ZYX"])
def test_euler_orders_match_the_renderer(order, eulers):
    scene = eulers[order]
    got = camera.matrix_from_three(scene["controls_mat"])
    want = camera.object_matrix(
        scene["mesh_center"], scene["mesh_radius"],
        euler_angles=[*scene["euler_angles_requested"]])
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_rotation_about_an_axis_cycles_the_axes():
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    m = camera.rotation_about_axis(axis, 2 * np.pi / 3)
    np.testing.assert_allclose(camera.transform_points(np.eye(3), m),
                               np.roll(np.eye(3), 1, axis=1), atol=1e-15)


def test_quaternion_matches_the_same_rotation():
    angle = np.deg2rad(90)
    q = [0, 0, np.sin(angle / 2), np.cos(angle / 2)]
    np.testing.assert_allclose(camera.rotation_from_quaternion(q),
                               camera.rotation_from_euler([0, 0, 90]), atol=1e-15)
    # three.js reads the quaternion as (x, y, z, w)
    np.testing.assert_allclose(camera.rotation_from_quaternion([0, 0, 0, 1]),
                               np.eye(4), atol=1e-15)


# -------------------------------------------------------------------------- object matrix


@pytest.mark.parametrize("scene", [
    "mesh_only", "square_scalar", "square_order1", "square_deformed", "square_nc2",
    "square_nc32", "square_fixed_range", "square_rotated", "square_no_wireframe"])
def test_object_matrix_matches_the_renderer(scene, scenes):
    probe = scenes[scene]
    got = camera.object_matrix(
        probe["mesh_center"], probe["mesh_radius"],
        euler_angles=settings_of(scene).get("camera", {}).get("euler_angles"))
    np.testing.assert_allclose(got, camera.matrix_from_three(probe["controls_mat"]),
                               atol=1e-12)
    # every visible mesh object draws through that matrix; an invisible one is skipped by
    # the renderer and keeps a stale identity matrix, and Axes / Colorbar are not the mesh
    for obj in probe["objects"]:
        if obj["name"] in ("Surface", "Wireframe", "Edges") and obj["visible"]:
            np.testing.assert_allclose(camera.matrix_from_three(obj["matrixWorld"]),
                                       got, atol=1e-15)


def test_object_matrix_puts_the_mesh_into_the_unit_sphere():
    m = camera.object_matrix(CENTER, RADIUS)
    corners = camera.transform_points(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]), m)
    np.testing.assert_allclose(corners, SCALE * np.array([[-.5, -.5, 0], [.5, -.5, 0],
                                                          [.5, .5, 0], [-.5, .5, 0]]),
                               atol=1e-15)
    assert np.abs(corners).max() == pytest.approx(1 / np.sqrt(2), abs=1e-15)


def test_object_matrix_rotates_about_the_mesh_centre():
    m = camera.object_matrix(CENTER, RADIUS, euler_angles=[0, 0, 90])
    centre = camera.transform_points(np.array([CENTER]), m)[0]
    np.testing.assert_allclose(centre, [0, 0, 0], atol=1e-15)
    corner = camera.transform_points(np.array([[1.0, 0.0, 0.0]]), m)[0]
    np.testing.assert_allclose(corner, SCALE * np.array([0.5, 0.5, 0.0]), atol=1e-9)


def test_euler_angles_may_name_their_order():
    plain = camera.object_matrix(CENTER, RADIUS, euler_angles=[10, 20, 30])
    named = camera.object_matrix(CENTER, RADIUS, euler_angles=[10, 20, 30, "XYZ"])
    other = camera.object_matrix(CENTER, RADIUS, euler_angles=[10, 20, 30, "ZYX"])
    np.testing.assert_allclose(named, plain, atol=1e-15)
    assert not np.allclose(other, plain)


def test_transformations_are_applied_in_the_listed_order():
    m = camera.object_matrix(CENTER, RADIUS, transformations=[
        {"type": "rotateZ", "angle": 90},
        {"type": "move", "dir": [0, 0, 1], "dist": 0.5},
    ])
    want = camera.chain(
        camera.translation(-np.asarray(CENTER)),
        camera.scaling(SCALE),
        camera.rotation_from_euler([0, 0, 90]),
        camera.translation([0, 0, 0.5]),
    )
    np.testing.assert_allclose(m, want, atol=1e-12)


def test_a_rotation_after_a_move_turns_about_the_moved_centre():
    after_move = camera.chain(camera.translation(-np.asarray(CENTER)), camera.scaling(SCALE),
                              camera.translation([1, 1, 1]))
    want = camera.chain(
        after_move,
        camera.wrap_about(camera.rotation_from_euler([25, 0, 0]),
                          camera.transform_points(np.array([CENTER]), after_move)[0]))
    m = camera.object_matrix(CENTER, RADIUS, transformations=[
        {"type": "move", "dir": [1, 1, 1], "dist": 1.0},
        {"type": "rotateX", "angle": 25},
    ])
    np.testing.assert_allclose(m, want, atol=1e-12)
    # ... and the turn really keeps the centre of the mesh where the move had put it
    centre = camera.transform_points(np.array([CENTER]), after_move)[0]
    np.testing.assert_allclose(camera.transform_points(np.array([CENTER]), m)[0], centre,
                               atol=1e-12)


def test_wrap_about_keeps_the_point_fixed():
    point = np.array([0.3, -1.2, 0.7])
    m = camera.wrap_about(camera.rotation_from_euler([15, 35, 55]), point)
    np.testing.assert_allclose(camera.transform_points(point[None, :], m)[0], point,
                               atol=1e-14)


def test_move_translates_in_normalized_object_units():
    # panObject premultiplies dist * dir onto the already normalized matrix: the direction
    # is not normalized and the mesh scale does not apply to it
    m = camera.object_matrix(CENTER, RADIUS, transformations=[
        {"type": "move", "dir": [1, 0, 0], "dist": 0.25}])
    np.testing.assert_allclose(m[:3, 3], [0.25 - SCALE * CENTER[0], -SCALE * CENTER[1],
                                          -SCALE * CENTER[2]], atol=1e-15)
    m = camera.object_matrix(CENTER, RADIUS, transformations=[
        {"type": "move", "dir": [2, 0, 0]}])
    assert m[0, 3] == pytest.approx(2.0 - SCALE * CENTER[0], abs=1e-15)


# ------------------------------------------------------------------------------- view


def test_view_matrix_matches_the_renderer(scenes):
    probe = scenes["square_scalar"]["camera"]
    got = camera.view_matrix(probe["position"], probe["quaternion"])
    np.testing.assert_allclose(got, camera.matrix_from_three(probe["view"]), atol=1e-15)
    np.testing.assert_allclose(got, camera.translation([0, 0, -3]), atol=1e-15)


def test_view_matrix_inverts_the_camera_transform():
    q = [0.0, np.sin(np.deg2rad(20)), 0.0, np.cos(np.deg2rad(20))]
    position = [1.0, -2.0, 3.5]
    got = camera.view_matrix(position, q)
    world = camera.chain(camera.rotation_from_quaternion(q), camera.translation(position))
    np.testing.assert_allclose(got @ world, np.eye(4), atol=1e-12)


# --------------------------------------------------------------------------- projection


def test_perspective_matrix_matches_the_renderer(scenes):
    probe = scenes["square_rotated"]["camera"]
    got = camera.perspective_matrix(probe["aspect"], probe["fov"], probe["near"],
                                    probe["far"], probe["zoom"])
    np.testing.assert_allclose(got, camera.matrix_from_three(probe["projection"]),
                               atol=1e-12)


def test_perspective_matrix_by_hand():
    aspect, near, far = 2.0, 1.0, 100.0
    m = camera.perspective_matrix(aspect, 40.0, near, far)
    tan = np.tan(np.deg2rad(20.0))
    np.testing.assert_allclose(m, np.array([
        [1 / (aspect * tan), 0, 0, 0],
        [0, 1 / tan, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]]), atol=1e-15)
    assert m[1, 1] == pytest.approx(2.7474774194546225)


def test_zoom_multiplies_the_projection():
    one = camera.perspective_matrix(1.5, 40.0)
    two = camera.perspective_matrix(1.5, 40.0, zoom=2.0)
    np.testing.assert_allclose(two[:2, :2], 2 * one[:2, :2], atol=1e-15)
    np.testing.assert_allclose(two[2], one[2], atol=1e-15)


# ------------------------------------------------------------------------ to the screen


def test_the_mesh_centre_lands_in_the_middle_of_the_canvas(scenes):
    probe = scenes["square_scalar"]
    cam = camera.Camera(mesh_center=probe["mesh_center"], mesh_radius=probe["mesh_radius"],
                        width=probe["canvas"]["width"], height=probe["canvas"]["height"])
    clip = camera.clip_points(np.array([probe["mesh_center"]]), cam.model_view_projection)
    x, y, z, w = clip[0]
    near, far = probe["camera"]["near"], probe["camera"]["far"]
    assert w == pytest.approx(3.0)
    assert (x, y) == pytest.approx((0.0, 0.0))
    assert z / w == pytest.approx(
        (-(far + near) / (far - near) * -3 - 2 * far * near / (far - near)) / 3)
    px, py, depth, w = camera.to_pixels(clip, probe["canvas"]["width"],
                                        probe["canvas"]["height"])
    np.testing.assert_allclose([px[0], py[0]], [447, 297], atol=1e-9)
    assert depth[0] == pytest.approx(0.5 * (z / w) + 0.5)


def test_pixels_are_measured_from_the_top_left():
    clip = np.array([[-1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0]])
    x, y, _, _ = camera.to_pixels(clip, 100, 50)
    np.testing.assert_allclose(x, [0, 100], atol=1e-12)
    np.testing.assert_allclose(y, [0, 50], atol=1e-12)


def test_points_behind_the_camera_have_negative_w():
    cam = camera.Camera(CENTER, RADIUS, width=894, height=594)
    clip = camera.clip_points(np.array([[0.5, 0.5, 4.0]]), cam.model_view_projection)
    assert clip[0, 3] < 0


def test_camera_reproduces_the_renderer_matrices(scenes):
    probe = scenes["square_rotated"]
    cam = camera.Camera(mesh_center=probe["mesh_center"], mesh_radius=probe["mesh_radius"],
                        width=probe["canvas"]["width"], height=probe["canvas"]["height"],
                        euler_angles=settings_of("square_rotated")["camera"]["euler_angles"])
    np.testing.assert_allclose(cam.object, camera.matrix_from_three(probe["controls_mat"]),
                               atol=1e-12)
    np.testing.assert_allclose(cam.view, camera.matrix_from_three(probe["camera"]["view"]),
                               atol=1e-15)
    np.testing.assert_allclose(cam.projection, camera.matrix_from_three(probe["camera"]["projection"]),
                               atol=1e-12)
    np.testing.assert_allclose(cam.model_view_projection,
                               cam.projection @ cam.view @ cam.object, atol=1e-15)
    assert cam.aspect == pytest.approx(probe["uniforms"]["aspect"])


def test_a_quarter_turn_swaps_the_mesh_diagonals_on_screen():
    # a square canvas, because the renderer stretches x by 1 / aspect
    size = 594
    flat = camera.Camera(CENTER, RADIUS, width=size, height=size)
    turned = camera.Camera(CENTER, RADIUS, width=size, height=size, euler_angles=[0, 0, 90])
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    a = np.stack(camera.to_pixels(camera.clip_points(corners, flat.model_view_projection),
                                  size, size)[:2], axis=1)
    b = np.stack(camera.to_pixels(camera.clip_points(corners, turned.model_view_projection),
                                  size, size)[:2], axis=1)
    np.testing.assert_allclose(np.sort(b[:, 0]), np.sort(a[:, 1]), atol=1e-9)
    np.testing.assert_allclose(np.sort(b[:, 1]), np.sort(a[:, 0]), atol=1e-9)


def test_camera_reads_the_render_data_of_a_scene():
    data = render_data_of("square_rotated")
    probe = _json("camera.json")["square_rotated"]
    cam = camera.Camera.from_render_data(data, width=894, height=594)
    np.testing.assert_allclose(cam.object, camera.matrix_from_three(probe["controls_mat"]),
                               atol=1e-12)
    assert cam.aspect == pytest.approx(probe["camera"]["aspect"])


def test_a_scene_without_camera_settings_just_centers_and_scales():
    data = render_data_of("square_scalar")
    probe = _json("camera.json")["square_scalar"]
    assert "camera" not in data.get("gui_settings", {})
    cam = camera.Camera.from_render_data(data, width=894, height=594)
    np.testing.assert_allclose(cam.object, camera.matrix_from_three(probe["controls_mat"]),
                               atol=1e-15)


def test_model_view_is_the_object_matrix_as_the_camera_sees_it(scenes):
    probe = scenes["square_rotated"]
    cam = camera.Camera.from_render_data(render_data_of("square_rotated"), width=894,
                                         height=594)
    np.testing.assert_allclose(cam.model_view, cam.view @ cam.object, atol=0)
    # and the two matrices the renderer itself held, multiplied out
    world = camera.matrix_from_three(probe["controls_mat"])
    view = camera.matrix_from_three(probe["camera"]["view"])
    np.testing.assert_allclose(cam.model_view, view @ world, atol=1e-12)


@pytest.mark.parametrize("order", ["XY", "XXY", "WXY", "XYXZ"])
def test_an_unknown_euler_order_is_refused(order):
    with pytest.raises(ValueError, match="Euler order"):
        camera.rotation_from_euler([10, 20, 30], order)


def test_euler_order_is_spelled_case_insensitively():
    np.testing.assert_allclose(camera.rotation_from_euler([10, 20, 30], "zyx"),
                               camera.rotation_from_euler([10, 20, 30], "ZYX"), atol=0)


def test_a_mesh_without_extent_is_refused():
    with pytest.raises(ValueError, match="mesh_radius"):
        camera.object_matrix([0.0, 0.0, 0.0], 0.0)
