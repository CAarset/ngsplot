"""Bézier-triangle evaluation, checked against the webgui shaders.

The expected values below are literal transcriptions of the GLSL in
shader/utils.h (GetPositionAndScalar, GetNormal, GetVectorValues, GetValue) and
of the subdivision lattice in three/mesh.ts, so that an implementation that
departs from the renderer fails here rather than in a screenshot diff.
"""

import numpy as np
import pytest

from ngsplot import bezier

# attribute name lists, three/mesh.ts:100-131 (surface); the wireframe names are
# p0..p_order and v0..v_order (three/mesh.ts:214-220)
NAMES = {
    1: ["p0", "p1", "p2"],
    2: ["p00", "p01", "p02", "p10", "p11", "p20"],
    3: ["p00", "p01", "p02", "p03", "p10", "p11", "p12", "p20", "p21", "p30"],
}
VEC_NAMES = {
    1: ["v0", "v1", "v2"],
    2: ["vec00_01", "vec02_10", "vec11_20"],
    3: ["vec00_01", "vec02_03", "vec10_11", "vec12_20", "vec21_30"],
}
BASIS = {
    1: ["u", "v", "w"],
    2: ["u*u", "2*u*v", "v*v", "2*u*w", "2*v*w", "w*w"],
    3: ["u*u*u", "3*u*u*v", "3*u*v*v", "v*v*v", "3*u*u*w", "6*u*v*w",
        "3*v*v*w", "3*u*w*w", "3*v*w*w", "w*w*w"],
}


def bary(n, seed=0, margin=0.0):
    """n random (u, v) pairs inside the reference triangle"""
    r = np.random.default_rng(seed).uniform(0, 1, (3, n))
    r /= r.sum(axis=0)
    u, v = r[0], r[1]
    if margin:
        u, v = u * (1 - margin), v * (1 - margin)
    return u, v


def glsl_normal_2(P, u, v, mode="none", deformation=0.0, D=None):
    """GetNormal, ORDER==2, utils.h:134-190."""
    w = 1.0 - u - v
    p00, p01, p02, p10, p11, p20 = (P[:, i] for i in range(6))
    B00, B01, B10 = 2.0 * u, 2.0 * v, 2.0 * w
    du = B00[:, None] * (p00 - p10) + B01[:, None] * (p01 - p11) + B10[:, None] * (p10 - p20)
    dv = B00[:, None] * (p01 - p10) + B01[:, None] * (p02 - p11) + B10[:, None] * (p11 - p20)
    if mode == "3d":
        du[:, 0] += deformation * du[:, 3]
        dv[:, 0] += deformation * dv[:, 3]
        d00, d01, d02, d10, d11, d20 = (D[:, i] for i in range(6))
        du[:, 1:3] += deformation * (B00[:, None] * (d00 - d10)
                                     + B01[:, None] * (d01 - d11)
                                     + B10[:, None] * (d10 - d20))
        dv[:, 1:3] += deformation * (B00[:, None] * (d01 - d10)
                                     + B01[:, None] * (d02 - d11)
                                     + B10[:, None] * (d11 - d20))
    if mode == "2d":
        du[:, 2] += deformation * du[:, 3]
        dv[:, 2] += deformation * dv[:, 3]
    n = np.cross(du[:, :3], dv[:, :3])
    return n / np.linalg.norm(n, axis=1)[:, None]


def glsl_normal_3(P, u, v, mode="none", deformation=0.0, D=None):
    """GetNormal, ORDER==3, utils.h:260-338."""
    w = 1.0 - u - v
    p00, p01, p02, p03, p10, p11, p12, p20, p21, p30 = (P[:, i] for i in range(10))
    B00, B01, B02 = 3.0 * u * u, 6.0 * u * v, 3.0 * v * v
    B10, B11, B20 = 6.0 * u * w, 6.0 * v * w, 3.0 * w * w
    B = (B00, B01, B02, B10, B11, B20)
    du = (B00[:, None] * (p00 - p10) + B01[:, None] * (p01 - p11)
          + B02[:, None] * (p02 - p12) + B10[:, None] * (p10 - p20)
          + B11[:, None] * (p11 - p21) + B20[:, None] * (p20 - p30))
    dv = (B00[:, None] * (p01 - p10) + B01[:, None] * (p02 - p11)
          + B02[:, None] * (p03 - p12) + B10[:, None] * (p11 - p20)
          + B11[:, None] * (p12 - p21) + B20[:, None] * (p21 - p30))
    if mode == "3d":
        du[:, 0] += deformation * du[:, 3]
        dv[:, 0] += deformation * dv[:, 3]
        d00, d01, d02, d03, d10, d11, d12, d20, d21, d30 = (D[:, i] for i in range(10))
        du[:, 1:3] += deformation * (B00[:, None] * (d00 - d10) + B01[:, None] * (d01 - d11)
                                     + B02[:, None] * (d02 - d12) + B10[:, None] * (d10 - d20)
                                     + B11[:, None] * (d11 - d21) + B20[:, None] * (d20 - d30))
        dv[:, 1:3] += deformation * (B00[:, None] * (d01 - d10) + B01[:, None] * (d02 - d11)
                                     + B02[:, None] * (d03 - d12) + B10[:, None] * (d11 - d20)
                                     + B11[:, None] * (d12 - d21) + B20[:, None] * (d21 - d30))
    if mode == "2d":
        du[:, 2] += deformation * du[:, 3]
        dv[:, 2] += deformation * dv[:, 3]
    n = np.cross(du[:, :3], dv[:, :3])
    return n / np.linalg.norm(n, axis=1)[:, None]


def phase_shift(P, D, function_mode, scale):
    """utils.h:300-311: replace p.w by the requested phase of the complex value."""
    s_re, s_im = (scale[0], -scale[1]) if function_mode == 5.0 else (scale[1], scale[0])
    out = P.copy()
    out[..., 3] = out[..., 3] * s_re + D[..., 0] * s_im
    return out


@pytest.mark.parametrize("order", [1, 2, 3])
def test_names_follow_the_renderer_attribute_lists(order):
    assert bezier.names(order) == NAMES[order]
    assert bezier.vec_names(order) == VEC_NAMES[order]
    assert len(bezier.index_pairs(order)) == (order + 1) * (order + 2) // 2


@pytest.mark.parametrize("order", [1, 2, 3])
def test_bernstein_basis_is_the_glsl_basis(order):
    u, v = bary(64, 1)
    w = 1.0 - u - v
    got = bezier.bernstein(order, u, v)
    assert got.shape == (64, (order + 1) * (order + 2) // 2)
    for i, expr in enumerate(BASIS[order]):
        want = eval(expr, {"u": u, "v": v, "w": w})     # noqa: S307 - fixed literals
        assert got[..., i] == pytest.approx(want, rel=1e-15)
    assert got.sum(axis=-1) == pytest.approx(1.0, rel=1e-14)     # partition of unity


def test_evaluate_is_exact_for_a_quadratic_field():
    # z = u^2 + v^2 has Bernstein coefficients 1 at p00 and p02, 0 elsewhere: this
    # pins both the basis and the ordering of the control points. The positions are
    # the Greville abscissae, for which the patch is exactly xy = (u, v).
    ctrl = np.zeros((6, 4))
    ctrl[:, :2] = [[1, 0], [0.5, 0.5], [0, 1], [0.5, 0], [0, 0.5], [0, 0]]
    ctrl[0, 3] = ctrl[2, 3] = 1.0
    u, v = bary(40, 2)
    xyz, value = bezier.evaluate(2, ctrl, u, v)
    assert value == pytest.approx(u**2 + v**2, abs=1e-14)
    assert xyz[:, 0] == pytest.approx(u, abs=1e-14)
    assert xyz[:, 1] == pytest.approx(v, abs=1e-14)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_evaluate_at_a_corner_returns_the_corner_control_point(order):
    ctrl = np.random.default_rng(3).normal(size=(len(NAMES[order]), 4))
    corners = {(0, 0): (1, 0), (0, order): (0, 1), (order, 0): (0, 0)}
    for i, ab in enumerate(bezier.index_pairs(order)):
        if ab not in corners:
            continue
        _xyz, value = bezier.evaluate(order, ctrl, np.array([float(corners[ab][0])]),
                                      np.array([float(corners[ab][1])]))
        assert value[0] == pytest.approx(ctrl[i, 3], rel=1e-15)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_derivatives_are_the_gradient_of_the_evaluation(order):
    ctrl = np.random.default_rng(4).normal(size=(len(NAMES[order]), 4))
    u, v = bary(30, 5, margin=0.2)
    h = 1e-6
    du, dv = bezier.derivatives(order, ctrl, u, v)
    for d, param in ((du, 0), (dv, 1)):
        eps = np.where(np.arange(2) == param, h, 0.0)
        up = bezier.evaluate(order, ctrl, u + eps[0], v + eps[1])
        um = bezier.evaluate(order, ctrl, u - eps[0], v - eps[1])
        want = np.hstack([up[0] - um[0], (up[1] - um[1])[:, None]]) / (2 * h)
        assert d == pytest.approx(want, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("order, glsl", [(2, glsl_normal_2), (3, glsl_normal_3)])
def test_normal_matches_the_glsl_transcription(order, glsl):
    P = np.random.default_rng(6).normal(size=(60, len(NAMES[order]), 4))
    u, v = bary(60, 7, margin=0.1)
    assert bezier.normal(order, P, u, v) == pytest.approx(glsl(P, u, v), rel=1e-12)


@pytest.mark.parametrize("order, glsl", [(2, glsl_normal_2), (3, glsl_normal_3)])
@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_deformed_normal_matches_the_glsl_transcription(order, glsl, mode):
    k = len(NAMES[order])
    P = np.random.default_rng(8).normal(size=(40, k, 4))
    D = np.random.default_rng(9).normal(size=(40, k, 2))
    u, v = bary(40, 10, margin=0.1)
    want = glsl(P, u, v, mode=mode, deformation=0.7, D=D)
    got = bezier.normal(order, P, u, v, vectors=D, deformation=0.7, mode=mode)
    assert got == pytest.approx(want, rel=1e-12)


def test_order1_normal_is_the_constant_cross_product():
    P = np.random.default_rng(11).normal(size=(20, 3, 4))
    u, v = bary(20, 12)
    du, dv = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]         # vec4 du = p1 - p0
    want = np.cross(du[:, :3], dv[:, :3])
    want /= np.linalg.norm(want, axis=1)[:, None]
    assert bezier.normal(1, P, u, v) == pytest.approx(want, rel=1e-13)


def test_generic_derivative_formula_agrees_with_the_order1_cross_product():
    P = np.random.default_rng(13).normal(size=(10, 3, 4))
    du, dv = bezier.derivatives(1, P, np.array([0.3]), np.array([0.2]))
    got = np.cross(du[0, :3], dv[0, :3])
    want = np.cross((P[:, 1] - P[:, 0])[0, :3], (P[:, 2] - P[:, 0])[0, :3])
    assert got / np.linalg.norm(got) == pytest.approx(want / np.linalg.norm(want), rel=1e-13)


def test_order1_with_vertex_normals_blends_them():
    # HAVE_NORMALS (nodal p1 data): normalize(u*n0 + v*n1 + w*n2)
    P = np.random.default_rng(14).normal(size=(5, 3, 4))
    N = np.random.default_rng(15).normal(size=(5, 3, 3))
    u, v = np.array([0.2]), np.array([0.3])
    w = 1 - u - v
    want = u[:, None] * N[:, 0] + v[:, None] * N[:, 1] + w[:, None] * N[:, 2]
    want /= np.linalg.norm(want, axis=1)[:, None]
    assert bezier.normal(1, P, u, v, vertex_normals=N) == pytest.approx(want, rel=1e-14)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_normal_of_a_flat_element_is_the_positive_z_axis(order):
    # the Greville embedding puts u along +x and v along +y, so a 2d element is
    # front facing when seen from +z and is therefore never dimmed
    P = np.zeros((len(NAMES[order]), 4))
    for i, (a, b) in enumerate(bezier.index_pairs(order)):
        P[i, :2] = [(order - a - b) / order, b / order]
    u, v = bary(25, 16)
    assert bezier.normal(order, P, u, v) == pytest.approx(
        np.tile([0.0, 0.0, 1.0], (25, 1)), abs=1e-12)


def test_deformed_position_matches_the_vertex_shader():
    # trigsplines.vert:25-30, DEFORMATION -> pv.xyz += deformation*value_
    #                    DEFORMATION_2D -> pv.z += GetValue(deformation*value_)
    P = np.random.default_rng(17).normal(size=(6, 4))
    D = np.random.default_rng(18).normal(size=(6, 2))
    u, v = np.array([0.3]), np.array([0.25])
    xyz0, value = bezier.evaluate(2, P, u, v)
    vv = bezier.vector_values(2, D, u, v)
    v3 = np.stack([value, vv[:, 0], vv[:, 1]], axis=-1)
    got3, _ = bezier.evaluate(2, P, u, v, vectors=D, deformation=0.5, mode="3d")
    assert got3 == pytest.approx(xyz0 + 0.5 * v3, rel=1e-14)
    got2, _ = bezier.evaluate(2, P, u, v, deformation=0.5, mode="2d")
    assert got2[:, 2] == pytest.approx(xyz0[:, 2] + 0.5 * value, rel=1e-14)
    assert got2[:, :2] == pytest.approx(xyz0[:, :2], rel=1e-15)


def test_deformed_2d_position_follows_the_selected_function_mode():
    P = np.random.default_rng(19).normal(size=(6, 4))
    D = np.random.default_rng(20).normal(size=(6, 2))
    u, v = np.array([0.3]), np.array([0.25])
    xyz0, value = bezier.evaluate(2, P, u, v)
    vv = bezier.vector_values(2, D, u, v)[0]
    got, _ = bezier.evaluate(2, P, u, v, vectors=D, deformation=0.5, mode="2d",
                             function_mode=1.0)
    assert got[0, 2] == pytest.approx(xyz0[0, 2] + 0.5 * vv[0], rel=1e-14)


def test_get_value_implements_the_function_modes():
    v3 = np.array([[3.0, 4.0, 12.0]])
    assert bezier.get_value(v3, 0.0) == pytest.approx([3.0])
    assert bezier.get_value(v3, 1.0) == pytest.approx([4.0])
    assert bezier.get_value(v3, 2.0) == pytest.approx([12.0])
    assert bezier.get_value(v3, 3.0) == pytest.approx([13.0])            # length
    assert bezier.get_value(v3, 5.0, (0.6, 0.8)) == pytest.approx([3 * 0.6 - 4 * 0.8])
    assert bezier.get_value(v3, 6.0, (0.6, 0.8)) == pytest.approx([3 * 0.8 + 4 * 0.6])
    assert bezier.get_value(v3, 4.0) == pytest.approx([0.0])             # unused mode


@pytest.mark.parametrize("order, glsl", [(2, glsl_normal_2), (3, glsl_normal_3)])
def test_complex_phase_shift_in_the_normal(order, glsl):
    # under DEFORMATION_2D the shader replaces p.w by the requested phase of the
    # complex value, using only the real vector component (the .x / .z halves of
    # each vec pair), before differencing.
    k = len(NAMES[order])
    P = np.random.default_rng(21).normal(size=(40, k, 4))
    D = np.random.default_rng(22).normal(size=(40, k, 2))
    u, v = bary(40, 23, margin=0.1)
    scale = (0.6, 0.8)
    for mode in (5.0, 6.0):
        got = bezier.normal(order, P, u, v, vectors=D, deformation=0.4, mode="2d",
                            function_mode=mode, complex_scale=scale)
        want = glsl(phase_shift(P, D, mode, scale), u, v, mode="2d", deformation=0.4)
        assert got == pytest.approx(want, rel=1e-12), f"order {order}, mode {mode}"


@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_lattice_has_the_renderer_shape(n):
    u, v = bezier.lattice(n)
    assert len(u) == (n + 1) * (n + 2) // 2
    assert bezier.subtriangles(n).shape == (n * n, 3)
    assert np.all(u + v <= 1.0 + 1e-12)
    assert set(zip(np.round(u * n).astype(int), np.round(v * n).astype(int))) == {
        (x, s - x) for s in range(n + 1) for x in range(s + 1)
    }


def test_subtriangle_sequence_is_the_mesh_ts_lattice():
    # three/mesh.ts:25-42, truncated to the draw range 3*sd*sd with sd = 3, written
    # out from the two loops by hand
    want = []
    for i in range(3):
        for j in range(i + 1):
            want += [(j, i - j), (j + 1, i - j), (j, i - j + 1)]
        for j in range(i):
            want += [(j + 1, i - j - 1), (j + 1, i - j), (j, i - j)]
    u, v = bezier.lattice(3)
    lattice = np.stack([np.round(u * 3).astype(int), np.round(v * 3).astype(int)], -1)
    assert [tuple(p) for p in lattice[bezier.subtriangles(3)].reshape(-1, 2)] == want


def test_subtriangles_are_wound_counter_clockwise():
    # winding decides gl_FrontFacing, which decides the dark-backside dimming
    u, v = bezier.lattice(6)
    t = bezier.subtriangles(6)
    signed = ((u[t[:, 1]] - u[t[:, 0]]) * (v[t[:, 2]] - v[t[:, 0]])
              - (v[t[:, 1]] - v[t[:, 0]]) * (u[t[:, 2]] - u[t[:, 0]]))
    assert np.all(signed > 0)


def test_subtriangles_span_consecutive_lattice_rows():
    for n in (1, 4, 7):
        u, v = bezier.lattice(n)
        row = np.round(u * n).astype(int) + np.round(v * n).astype(int)
        rows = np.sort(row[bezier.subtriangles(n)], axis=1)
        assert (rows.min(), rows.max()) == (0, n)
        assert np.all(rows[:, 2] == rows[:, 0] + 1)          # never skips a row
        upward = np.sum(rows[:, 1:] == rows[:, :1] + 1, axis=1)
        assert np.count_nonzero(upward == 2) == n * (n + 1) // 2      # one vertex on row i
        assert np.count_nonzero(upward == 1) == n * (n - 1) // 2      # two vertices there


def test_vector_values_use_the_same_basis():
    D = np.random.default_rng(24).normal(size=(10, 2))
    u, v = bary(30, 25)
    assert bezier.vector_values(3, D, u, v) == pytest.approx(bezier.bernstein(3, u, v) @ D,
                                                             rel=1e-15)


def test_degrees_outside_one_to_three_are_rejected():
    with pytest.raises(ValueError):
        bezier.names(4)
    with pytest.raises(ValueError):
        bezier.bernstein(0, np.array([0.5]), np.array([0.2]))
