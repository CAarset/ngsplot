"""Decoding of ``BuildRenderData`` payloads.

The tests that need ngsolve ask for it through the ``payloads`` fixture, so the
pure-numpy guarantees of :mod:`ngsplot.geometry` are checked even without the
optional dependency.  Known answers come from the element geometry itself: for a
mesh of straight triangles with an affine function the control points are exact,
so the decoded arrays can be compared to hand-written numbers instead of to
another implementation.
"""

import base64
import math

import numpy as np
import pytest

from ngsplot import bezier, geometry

TOL = 1e-6


# --------------------------------------------------------------------------- #
# pure numpy
# --------------------------------------------------------------------------- #


def test_decode_array_roundtrips_float32():
    values = np.array([1.0, -2.5, 1e-30, np.inf], dtype=np.float32)
    payload = base64.b64encode(values).decode("ascii")
    decoded = geometry.decode_array(payload)
    assert decoded.dtype == np.float32
    assert decoded.tobytes() == values.tobytes()


def test_decode_array_accepts_binary_encoding_and_arrays():
    values = np.arange(6, dtype=np.float32)
    assert geometry.decode_array(values.tobytes()).tobytes() == values.tobytes()
    assert geometry.decode_array(bytearray(values.tobytes())).tobytes() == values.tobytes()
    assert geometry.decode_array(values).dtype == np.float32
    assert geometry.decode_array(values * 2).tobytes() == (values * 2).astype(np.float32).tobytes()


def test_decode_array_rejects_truncated_payload():
    with pytest.raises(ValueError, match="float32"):
        geometry.decode_array(base64.b64encode(b"\x00\x01\x02").decode("ascii"))


def test_greville_points_are_the_control_point_abscissae():
    for order in (1, 2, 3):
        pairs = bezier.index_pairs(order)
        u, v = geometry.greville(order)
        assert u.shape == v.shape == (len(pairs),)
        assert np.allclose(u, [(order - a - b) / order for a, b in pairs])
        assert np.allclose(v, [b / order for a, b in pairs])
        # corners of the reference triangle are hit by the corner names only
        assert tuple(zip(u, v))[pairs.index((0, 0))] == (1.0, 0.0)
        assert tuple(zip(u, v))[pairs.index((0, order))] == (0.0, 1.0)
        assert tuple(zip(u, v))[pairs.index((order, 0))] == (0.0, 0.0)


def test_sample_values_inverts_the_payload_fit():
    """Fitting samples of a polynomial and sampling it back is the identity."""
    for order in (1, 2, 3):
        u, v = geometry.greville(order)
        # a degree-`order` polynomial in the reference coordinates
        samples = 3.0 * u * u - u * v + 0.5 * v
        basis = bezier.bernstein(order, u, v)  # (K, K), rows are sample points
        ctrl = samples @ np.linalg.inv(basis).T  # control points of that polynomial
        full = np.zeros((1, len(u), 4))
        full[..., 3] = ctrl
        out = geometry.sample_values(order, full)
        assert out.shape == (1, len(u))
        assert np.abs(out[0] - samples).max() < TOL
        # the same field handed over without the vec4 wrapper
        np.testing.assert_allclose(geometry.sample_values(order, ctrl[None, :]), out)
    with pytest.raises(ValueError, match="control points"):
        geometry.sample_values(2, np.zeros((1, 5)))


def test_minmax_reproduces_ngsolve_vector_initialisation():
    # ngs Vector::MinMax starts from numeric_limits max()/min(), not from +-inf,
    # so a never-positive function reports DBL_MIN as its maximum.
    assert geometry.minmax([-1.0, -2.0]) == (-2.0, geometry.DBL_MIN)
    assert geometry.minmax([0.0, 0.0]) == (0.0, geometry.DBL_MIN)
    assert geometry.minmax([1.0, 2.0, 3.0]) == (1.0, 3.0)
    assert geometry.minmax([]) == (geometry.DBL_MAX, geometry.DBL_MIN)


def test_minmax_ignores_infinities_only_on_request():
    values = [np.inf, -np.inf, 0.5]
    assert geometry.minmax(values) == (0.5, 0.5)
    assert geometry.minmax(values, ignore_inf=False) == (-np.inf, np.inf)
    # NaN never takes part: neither bound is updated by it
    assert geometry.minmax([np.nan, 1.0, 2.0]) == (1.0, 2.0)
    assert geometry.minmax([np.nan, 1.0, 2.0], ignore_inf=False) == (1.0, 2.0)


# --------------------------------------------------------------------------- #
# against live BuildRenderData output
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def payloads():
    """Unit square (maxh=0.5) with a scalar and a two-component function."""
    pytest.importorskip("ngsolve.webgui")
    import ngsolve as ngs
    from netgen.geom2d import unit_square

    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    regions = {vb: vb for vb in (ngs.VOL, ngs.BND, ngs.BBND)}

    scalar = ngs.GridFunction(ngs.H1(mesh, order=2))
    scalar.Set(ngs.x - 2 * ngs.y)
    vector = ngs.GridFunction(ngs.VectorH1(mesh, order=2))
    vector.Set(ngs.CF((ngs.x, 2 * ngs.y)))

    vertices = np.asarray(mesh.ngmesh._getVertices(), dtype=np.float64).reshape(-1, 3)
    trigs = np.asarray(mesh.ngmesh._get2dElementsAsTriangles(), dtype=np.int32).reshape(-1, 3)
    return {
        "ngs": ngs,
        "mesh": mesh,
        "regions": regions,
        "scalar": scalar,
        "vector": vector,
        # element corners, listed as (V0, V1, V2); the reference map of the payload
        # is x -> V0, y -> V1, 1-x-y -> V2
        "V": vertices[trigs],
        # boundary segments, in the order the payload lists them; each one is
        # drawn from its second to its first vertex
        "segments": np.asarray(mesh.ngmesh._getSegments(), dtype=np.int32).reshape(-1, 2),
        "vertices": vertices,
    }


def slot_position(order, a, b, corner):
    """Physical position of control point (a, b) on a straight triangle.

    ``bezier.index_pairs`` lists the degree-``order`` names ``p{a}{b}``; the
    payload stores them so that ``a`` is the exponent of the reference ``y`` and
    ``b`` the one of ``x``, which is what the shader's ``u, v, w`` reading of the
    same slots mirrors (``u`` is ``1-x-y``, ``v`` is ``x``, ``w`` is ``y``).
    """
    return (b * corner[0] + a * corner[1] + (order - a - b) * corner[2]) / order


@pytest.mark.parametrize("order", [1, 2, 3])
def test_trig_ctrl_describes_the_element_geometry(payloads, order):
    data = payloads["ngs"].webgui.BuildRenderData(
        payloads["mesh"], payloads["scalar"], order=order, regions=payloads["regions"]
    )
    ctrl, vecs = geometry.trig_points(data)
    V = payloads["V"]
    assert ctrl.shape == (len(V), len(bezier.names(order)), 4)
    assert ctrl.dtype == np.float32
    assert vecs is None
    for j, (a, b) in enumerate(bezier.index_pairs(order)):
        expected = np.array([slot_position(order, a, b, e) for e in V])
        np.testing.assert_allclose(ctrl[:, j, :2], expected[:, :2], atol=TOL)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_trig_value_component_is_the_function_at_the_control_points(payloads, order):
    data = payloads["ngs"].webgui.BuildRenderData(
        payloads["mesh"], payloads["scalar"], order=order, regions=payloads["regions"]
    )
    ctrl, _ = geometry.trig_points(data)
    V = payloads["V"]
    for j, (a, b) in enumerate(bezier.index_pairs(order)):
        expected = np.array(
            [pos[0] - 2.0 * pos[1] for pos in (slot_position(order, a, b, e) for e in V)]
        )
        np.testing.assert_allclose(ctrl[:, j, 3], expected, atol=TOL)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_vector_components_come_from_the_vector_attributes(payloads, order):
    data = payloads["ngs"].webgui.BuildRenderData(
        payloads["mesh"], payloads["vector"], order=order, regions=payloads["regions"]
    )
    assert data["funcdim"] == 2
    ctrl, vecs = geometry.trig_points(data)
    assert vecs is not None and vecs.shape == ctrl.shape[:2] + (2,)
    V = payloads["V"]
    for j, (a, b) in enumerate(bezier.index_pairs(order)):
        expected = np.array(
            [2.0 * pos[1] for pos in (slot_position(order, a, b, e) for e in V)]
        )
        np.testing.assert_allclose(vecs[:, j, 0], expected, atol=TOL)
        # a two-component function leaves the second vector component empty
        assert np.all(vecs[:, j, 1] == 0.0)


def test_trig_points_reports_the_payload_element_count(payloads):
    data = payloads["ngs"].webgui.BuildRenderData(
        payloads["mesh"], payloads["scalar"], order=2, regions=payloads["regions"]
    )
    ctrl, _ = geometry.trig_points(data)
    assert ctrl.shape[0] == payloads["mesh"].ne


def test_each_array_of_the_payload_is_one_control_point(payloads):
    data = payloads["ngs"].webgui.BuildRenderData(
        payloads["mesh"], payloads["scalar"], order=2, regions=payloads["regions"]
    )
    ctrl, _ = geometry.trig_points(data)
    for j, payload in enumerate(data["Bezier_trig_points"]):
        # array j of the payload is control point j of every element, bit for bit
        assert ctrl[:, j].tobytes() == geometry.decode_array(payload).tobytes()


@pytest.mark.parametrize("order", [1, 2, 3])
def test_wireframe_rows_are_elements_times_edges(payloads, order):
    data = payloads["ngs"].webgui.BuildRenderData(
        payloads["mesh"], payloads["scalar"], order=order, regions=payloads["regions"]
    )
    pts, vecs = geometry.wireframe_points(data)
    V = payloads["V"]
    assert pts.shape == (3 * len(V), order + 1, 4)
    assert vecs is None
    # row element * 3 + edge, edge k of the reference triangle being
    # (t, 0) -> (0, t) -> (t, 1-t); for order 1 the fit is the identity
    corners = {0: (2, 0), 1: (2, 1), 2: (1, 0)}
    if order == 1:
        for edge, (start, end) in corners.items():
            rows = np.arange(len(V)) * 3 + edge
            np.testing.assert_allclose(pts[rows, 0, :2], V[:, start, :2], atol=TOL)
            np.testing.assert_allclose(pts[rows, 1, :2], V[:, end, :2], atol=TOL)
    # the wireframe carries the function values, sampled along the edges
    for j, t in enumerate(np.linspace(0.0, 1.0, order + 1)):
        for edge, (start, end) in corners.items():
            rows = np.arange(len(V)) * 3 + edge
            p = (1 - t) * V[:, start] + t * V[:, end]
            np.testing.assert_allclose(pts[rows, j, 3], p[:, 0] - 2.0 * p[:, 1], atol=TOL)


def test_wireframe_extras_are_per_point_pairs(payloads):
    data = payloads["ngs"].webgui.BuildRenderData(
        payloads["mesh"], payloads["vector"], order=2, regions=payloads["regions"]
    )
    pts, vecs = geometry.wireframe_points(data)
    assert pts.shape[1] == 3 and vecs.shape == pts.shape[:2] + (2,)
    V = payloads["V"]
    corners = {0: (2, 0), 1: (2, 1), 2: (1, 0)}
    for edge, (start, end) in corners.items():
        rows = np.arange(len(V)) * 3 + edge
        for j, t in enumerate(np.linspace(0.0, 1.0, 3)):
            p = (1 - t) * V[:, start] + t * V[:, end]
            np.testing.assert_allclose(vecs[rows, j, 0], 2.0 * p[:, 1], atol=TOL)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_edge_payload_is_the_boundary_skeleton_with_zero_values(payloads, order):
    data = payloads["ngs"].webgui.BuildRenderData(
        payloads["mesh"], payloads["scalar"], order=order, regions=payloads["regions"]
    )
    pts, vecs = geometry.edge_points(data)
    segments = payloads["segments"]
    vertices = payloads["vertices"][segments]
    assert pts.shape == (len(segments), order + 1, 4)
    assert vecs is None
    # netgen.webgui builds the edge function as CF((xyz, 0.0)), so the value
    # component that drives the colour of the thick edges is identically zero
    assert np.all(pts[:, :, 3] == 0.0)
    # one row per boundary segment, walked from its second to its first vertex
    np.testing.assert_allclose(pts[:, 0, :2], vertices[:, 1, :2], atol=TOL)
    np.testing.assert_allclose(pts[:, -1, :2], vertices[:, 0, :2], atol=TOL)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_sample_values_recover_the_colour_range(payloads, order):
    data = payloads["ngs"].webgui.BuildRenderData(
        payloads["mesh"], payloads["scalar"], order=order, regions=payloads["regions"]
    )
    ctrl, _ = geometry.trig_points(data)
    samples = geometry.sample_values(data["order2d"], ctrl)
    assert samples.shape == ctrl.shape[:2]
    assert np.isfinite(samples).all()
    # f = x - 2y on the unit square spans [-2, 1]; the payload range is fitted
    # from samples at the same points, so both agree to the float32 rounding
    assert geometry.minmax(samples) == pytest.approx((-2.0, 1.0), abs=TOL)
    assert data["funcmin"] == pytest.approx(-2.0, abs=TOL)
    assert data["funcmax"] == pytest.approx(1.0, abs=TOL)


def test_trig_points_rejects_a_payload_of_the_wrong_degree(payloads):
    data = payloads["ngs"].webgui.BuildRenderData(
        payloads["mesh"], payloads["scalar"], order=2, regions=payloads["regions"]
    )
    with pytest.raises(ValueError, match="order"):
        geometry.trig_points(data, order=4)
    broken = dict(data)
    broken["Bezier_trig_points"] = data["Bezier_trig_points"][:2]
    with pytest.raises(ValueError, match="at least"):
        geometry.trig_points(broken)
    broken = dict(data)
    broken["Bezier_trig_points"] = [base64.b64encode(b"\0" * 8).decode()] + list(
        data["Bezier_trig_points"][1:]
    )
    with pytest.raises(ValueError, match="vec4"):
        geometry.trig_points(broken)
    broken["Bezier_trig_points"] = [base64.b64encode(b"\0" * 6).decode()] + list(
        data["Bezier_trig_points"][1:]
    )
    with pytest.raises(ValueError, match="float32"):
        geometry.trig_points(broken)


def test_order_comes_from_the_payload_when_not_given(payloads):
    data = payloads["ngs"].webgui.BuildRenderData(
        payloads["mesh"], payloads["scalar"], order=1, regions=payloads["regions"]
    )
    assert geometry.order_of(data) == 1 == data["order2d"]
    assert geometry.trig_points(data, order=1)[0].shape[1] == 3
    missing = {k: v for k, v in data.items() if k != "order2d"}
    with pytest.raises(KeyError, match="order2d"):
        geometry.order_of(missing)
