"""The seam to ngsolve: a ``GridFunction`` or a ``CoefficientFunction`` in, a payload out.

``tools/capture_dims.py`` holds the scene definitions the captures under ``tests/gt/dims``
were recorded from, and a scene is a triple of what to hand ``Draw``.  The seam builds the
same payload with no widget in sight -- ``WebGLScene`` and the registered draw types, which
is what ``ngsolve.webgui.Draw`` is made of -- so the equality asserted here is a claim: the
arrays and the numbers that decide the picture come out the same, key by key, for every mesh
dimension and every kind of input the ground truth holds.

Worth pinning beside it is the reason the seam exists.  The keywords of ``Draw`` have to land
where the page reads them, because they are how a script says ``vmin``, ``vmax`` or ``ncolors``:
``min``/``max`` become ``funcmin``/``funcmax``, ``autoscale=False`` becomes the flag that stops
the renderer from re-reading the function, and the flat names inside ``settings`` are the very
ones ``scene.settings_of`` folds into the panel.  And a bare ``CoefficientFunction`` still
hears what ngsolve says about it: without a mesh there is nothing to draw.  And since the
payload is built by a widget that this package has no use for once the payload is in hand, the
file answers for that widget as well: built, read, closed.
"""

import base64
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))

ngs = pytest.importorskip("ngsolve")

import capture_dims  # noqa: E402  (imports ngsolve, which the skip above has just settled)
import gtdata  # noqa: E402
from ngsplot import _ngsolve, geometry, scene  # noqa: E402

#: every dims scene with a page recorded against it; ``cf_nomesh`` and ``volume_cuts`` are
#: the two the capture refused to render, and they get their own test below
SCENES = ("line_scalar", "line_deformed", "numberspace", "numberspace_l2pwc",
          "surface_scalar", "volume_scalar", "cf_mesh", "cf_vector", "vec_hcurl",
          "complex_h1")


def as_json(value):
    """A payload value as the page held it, so that two of them can be compared.

    A payload that reached ``GenerateHTML`` went through ``json.dumps`` with no default
    handler, so anything in it is already a number, a string, a flag, a list or a dict; a
    payload that holds anything else is one the page could not have been built from.
    """
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    raise AssertionError(f"a payload must not hold a {type(value).__name__}")


def differs(mine, recorded):
    """The keys the seam did not reproduce, each with what the seam sent instead."""
    out = {}
    mine, recorded = as_json(mine), as_json(recorded)
    for key in sorted(set(mine) | set(recorded)):
        if mine.get(key, "<missing>") != recorded.get(key, "<missing>"):
            out[key] = (mine.get(key, "<missing>"), recorded.get(key, "<missing>"))
    return out


@pytest.mark.parametrize("name", SCENES)
def test_the_seam_builds_the_payload_the_recording_was_drawn_from(name):
    """No widget, no browser: the payload of ``Draw`` again, byte for byte."""
    obj, args, keywords = capture_dims.SCENES[name]()
    mine = _ngsolve.render_data(obj, *args, **keywords)
    assert differs(mine, capture_dims.draw(name)) == {}
    assert differs(mine, gtdata.render_data_of("dims/" + name)) == {}


def scalar_2d():
    """A unit square and the simplest thing to draw on it."""
    mesh = capture_dims.mesh_square()
    gfu = ngs.GridFunction(ngs.H1(mesh, order=2))
    gfu.Set(ngs.x - ngs.y)
    return mesh, gfu


def test_the_range_of_a_still_draw_is_the_range_it_was_given():
    mesh, gfu = scalar_2d()
    payload = _ngsolve.render_data(gfu, min=-0.5, max=2.0, autoscale=False)
    assert payload["funcmin"] == -0.5 and payload["funcmax"] == 2.0
    assert payload["autoscale"] is False
    assert scene.view_range(payload) == (-0.5, 2.0)
    # and left to itself the same function reports the ends it has
    running = _ngsolve.render_data(gfu)
    assert running["autoscale"] is True
    assert running["funcmin"] == pytest.approx(-1.0) and running["funcmax"] == pytest.approx(1.0)
    assert scene.view_range(running) == (running["funcmin"], running["funcmax"])


def test_the_number_of_bands_and_the_lines_come_in_through_the_settings():
    """The shortcuts ``Draw`` advertises are the names the panel folds, and so ours."""
    mesh, gfu = scalar_2d()
    payload = _ngsolve.render_data(gfu, settings={"colormap_ncolors": 3, "mesh": False,
                                                 "line_thickness": 1, "elements": False})
    settings = scene.settings_of(payload)
    assert settings["ncolors"] == 3
    assert settings["wireframe"] is False and settings["surface"] is False
    assert settings["line_width"] == 1


def test_a_part_of_a_value_is_asked_for_the_way_the_panel_holds_it():
    """``Colormap.eval`` survives onto the panel for a scalar, where nothing overwrites it."""
    mesh, gfu = scalar_2d()
    for given, mode in ((0, 0.0), (1, 1.0), (3, 3.0)):
        payload = _ngsolve.render_data(gfu, settings={"Colormap": {"eval": given}})
        assert scene.settings_of(payload)["function_mode"] == mode, given
    # a vector is not so obliging: initComplex writes the mode of its own over whatever the
    # payload asked for, and only a top-level eval outranks it (gui.ts:552-571)
    vector = _ngsolve.render_data(ngs.CF((-ngs.y, ngs.x)), capture_dims.mesh_square(),
                                  settings={"Colormap": {"eval": 1}})
    assert scene.settings_of(vector)["function_mode"] == 3.0


def test_the_eval_keyword_of_draw_reaches_the_payload_nowhere():
    """``netgen/webgui.py:564`` asks for ``eval_`` and then reads ``eval``, so both die."""
    mesh, gfu = scalar_2d()
    assert "eval" not in _ngsolve.render_data(gfu, eval=1)
    with pytest.raises(KeyError, match="eval"):
        _ngsolve.render_data(gfu, eval_=1)


def test_the_view_the_page_would_open_on_is_the_view_of_the_payload():
    mesh, gfu = scalar_2d()
    payload = _ngsolve.render_data(gfu, euler_angles=[20.0, 30.0, 0.0])
    camera = payload["gui_settings"]["camera"]
    assert camera["euler_angles"] == [20.0, 30.0, 0.0]
    from ngsplot import camera as camera_module
    # the angles belong to the object matrix, which turns the scene about its centre; the view
    # is the fixed eye at (0, 0, 3) and hears nothing of them (camera.ts:9-31)
    turned = camera_module.Camera.from_render_data(payload, width=601, height=401)
    straight = camera_module.Camera.from_render_data(
        _ngsolve.render_data(gfu), width=601, height=401)
    assert not np.allclose(turned.object, straight.object)
    assert np.allclose(turned.view, straight.view)
    assert turned.view[2, 3] == -3.0


def test_a_gridfunction_and_the_field_it_was_set_from_agree_about_the_picture():
    """``gfu.Set(cf)`` is an interpolation, so the mesh arrives identical and the values nearly.

    The control points are positions and values in one array, and the positions are the same
    patches either way; only what ``Set`` solved for the nodes differs, which is to say in the
    last bits of a double.  The edges of the mesh are compared byte for byte, the control
    points as numbers.
    """
    mesh = capture_dims.mesh_square()
    gfu = ngs.GridFunction(ngs.H1(mesh, order=2))
    gfu.Set(ngs.x * ngs.y)
    from_seam = _ngsolve.render_data(gfu)
    from_function = _ngsolve.render_data(ngs.x * ngs.y, mesh)
    assert set(from_seam) == set(from_function)
    for key in ("edges", "mesh_center", "mesh_radius", "mesh_dim",
                "funcdim", "order2d", "order3d", "draw_vol", "draw_surf", "deformation",
                "autoscale", "show_mesh", "show_wireframe"):
        assert from_seam[key] == from_function[key], key

    def control_points(payload):  # encode(..., dtype=np.float32), three arrays per set
        return [np.frombuffer(base64.b64decode(b), dtype="<f4")
                for b in payload["Bezier_points"] + payload["Bezier_trig_points"]]

    points, other = control_points(from_seam), control_points(from_function)
    assert len(points) == len(other)
    for mine, theirs in zip(points, other):
        assert mine.shape == theirs.shape
        # measured worst slot 1.5e-16: the patches agree, and what is left is a residue
        # the interpolation lands on zero
        assert np.max(np.abs(mine - theirs)) < 1e-12
    assert from_seam["funcmin"] == pytest.approx(from_function["funcmin"], abs=1e-12)
    assert from_seam["funcmax"] == pytest.approx(from_function["funcmax"], abs=1e-12)


def test_a_solid_from_ngsolve_paints_a_solid():
    mesh = capture_dims.mesh_3d()
    gfu = ngs.GridFunction(ngs.H1(mesh, order=2))
    gfu.Set(ngs.x + 2 * ngs.y + 3 * ngs.z)
    payload = _ngsolve.render_data(gfu)
    assert payload["mesh_dim"] == 3 and payload["draw_vol"] is True
    canvas = scene.paint(payload, width=601, height=401)
    painted = canvas.to_uint8()[..., :3]
    assert (painted < 250).any(), "a volume that paints nothing paints no volume"
    assert int((painted.min(axis=-1) < 200).sum()) > 1000


def test_a_line_from_ngsolve_is_drawn_flat_and_in_black():
    """The 1D payload has no range, and so nothing to colour a function with (docs/DIMS.md)."""
    mesh = capture_dims.mesh_1d()
    gfu = ngs.GridFunction(ngs.H1(mesh, order=2))
    gfu.Set(ngs.x * ngs.x)
    payload = _ngsolve.render_data(gfu)
    assert payload["mesh_dim"] == 1
    assert (payload["funcmin"], payload["funcmax"]) == (geometry.DBL_MAX, geometry.DBL_MIN)
    canvas = scene.paint(payload, width=601, height=401)
    ink = canvas.to_uint8()[..., :3]
    assert int((ink.min(axis=-1) < 250).sum()) > 1000  # the mesh is drawn
    assert (ink.max(axis=-1) == ink.min(axis=-1)).all(), "a line without a range has no hue"


def test_a_constant_field_reports_the_bounds_of_a_field_that_never_spoke():
    """The sentinel bounds of ``ngs.Vector.MinMax``, from the library and not from a file."""
    payload = _ngsolve.render_data(ngs.GridFunction(ngs.NumberSpace(capture_dims.mesh_square())))
    assert payload["funcdim"] == 1
    assert scene.view_range(payload) == (0.0, 2.2250738585072014e-308)


def test_a_coefficient_function_without_a_mesh_says_what_ngsolve_says():
    with pytest.raises(RuntimeError, match="without mesh"):
        _ngsolve.render_data(ngs.x * ngs.y)


def test_the_widget_the_scene_built_is_closed_once_the_payload_is_read(monkeypatch):
    """A payload costs a widget to build, and should not cost one to keep.

    ``Draw`` reads ``show`` only after ``WebGLScene`` has constructed its widget, so the scene
    arrives holding a widget whoever that widget is -- an ``anywidget`` in a kernel, the plain
    stand-in where ``netgen`` could not import one -- and closing it is left to the caller.  The
    seam is that caller.  A subclass that counts closes is what makes the call visible whichever
    of the two the installed ``netgen`` builds, and the order is worth as much as the count: the
    payload has to be in hand before the widget goes, and has to be the one the widget was
    handed rather than a second assembly of it.
    """
    import netgen.webgui as netgen_webgui

    seen = []

    class Recording(netgen_webgui.WebGuiWidget):
        def close(self):
            seen.append(self.value)  # what the widget still held at the moment it was closed
            inherited = getattr(super(), "close", None)
            if inherited is not None:  # the stand-in of a netgen without anywidget has none
                inherited()

    monkeypatch.setattr(netgen_webgui, "WebGuiWidget", Recording)
    mesh, gfu = scalar_2d()
    first, second = _ngsolve.render_data(gfu), _ngsolve.render_data(gfu)

    assert len(seen) == 2, "one widget closed per payload, no more and none left open"
    assert seen[0] is first and seen[1] is second, (
        "the payload was read out of the widget before it went, and is the very dict the widget "
        "was handed rather than the arrays assembled a second time")


def test_a_run_of_plots_leaves_no_widget_registered():
    """The accumulation a notebook reports, measured where it can happen at all.

    A widget is a kernel object only when ``anywidget`` was importable: it opens a comm and
    files itself in a module-level registry that holds it for the life of the process, which is
    why dropping the scene frees nothing and ``__del__`` never runs to close it.  The run that
    reported this had 438 of them and some 23 MB of ``widget_state`` behind them, for scenes
    that were never displayed.  Where ``anywidget`` is absent the widget is a stand-in with no
    comm and nothing to count, and there is nothing here to measure.
    """
    pytest.importorskip("anywidget")
    ipywidgets = pytest.importorskip("ipywidgets")

    def registered():
        widgets = getattr(ipywidgets.Widget, "_objects", None)  # ipywidgets 7 kept it here
        if widgets is None:  # ipywidgets 8 moved it out of the class
            from ipywidgets.widgets.widget import _instances as widgets
        return len(widgets)

    mesh, gfu = scalar_2d()
    before = registered()
    for _ in range(5):
        _ngsolve.render_data(gfu)
    assert registered() == before, "a payload per call, and a widget the kernel carries per call"
