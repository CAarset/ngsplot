"""The pyplot layer: an NGSolve picture that matplotlib owns.

Nothing here paints.  :func:`ngsplot.pyplot.Plot` decides what it was handed, asks
:func:`ngsplot.scene.paint` for the pixels, and lays those pixels on an axis -- which is what
lets a colourbar, a title, a second subplot or a ``savefig`` sit
around an NGSolve draw without a second renderer being written.

The tests therefore ask two things of every knob.  That the picture is *the* picture: a plot of
a payload is the renderer's array, byte for byte, at the size the page would have used.  And
that a knob does what the panel's does, which is checked against the renderer call the knob is
meant to become rather than against a number: if the renderer changes, the plot follows, and
that is the point of the layer.
"""

import io
import sys

import matplotlib

matplotlib.use("Agg")  # the tests never open a window

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import image as mplimage  # noqa: E402
from matplotlib import pyplot as mpl  # noqa: E402

from ngsplot import colormap, pyplot, scene  # noqa: E402

import gtdata  # noqa: E402

#: a scalar on a surface, a vector, a solid and a line: one payload per dimension the
#: renderer claims to serve, plus a mesh with nothing drawn on it
SCALARS = ("dims/cf_mesh", "dims/cf_vector", "dims/volume_scalar", "dims/line_scalar")


@pytest.fixture(autouse=True)
def _fresh_figure():
    """Every test gets its own figure, and none of them leaks into the next."""
    pyplot.close("all")
    mpl.close("all")
    yield
    pyplot.close("all")
    mpl.close("all")


def ngsolve_objects():
    """``(ngsolve, capture_dims)``, skipped when there is no ngsolve to ask."""
    ngs = pytest.importorskip("ngsolve")
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "tools"))
    import capture_dims
    return ngs, capture_dims


def painted(payload, **kw):
    width = int(kw.pop("width", pyplot.WIDTH))
    height = int(kw.pop("height", pyplot.HEIGHT))
    return scene.paint(payload, width=width, height=height, **kw)


def test_plot_lays_the_renderers_picture_on_a_matplotlib_canvas():
    """The image a plot returns is the renderer's array, not a rendering of it."""
    from matplotlib.image import AxesImage

    payload = gtdata.render_data_of("dims/cf_mesh")
    image = pyplot.Plot(payload)
    assert isinstance(image, AxesImage)
    assert np.array_equal(image.get_array(), painted(payload).to_uint8())


@pytest.mark.parametrize("name", SCALARS)
def test_every_dimension_arrives_at_the_same_size(name):
    """A line, a surface and a solid are all drawn on the page's 900x600 canvas."""
    image = pyplot.Plot(gtdata.render_data_of(name))
    height, width, channels = image.get_array().shape
    assert (width, height, channels) == (pyplot.WIDTH, pyplot.HEIGHT, 4)
    assert tuple(image.get_extent()) == (0.0, float(pyplot.WIDTH), float(pyplot.HEIGHT), 0.0)


def test_the_figure_holds_the_picture_and_nothing_else():
    """The axes fills the figure and its frame is off: the picture is the plot, as in Draw."""
    image = pyplot.Plot(gtdata.render_data_of("dims/cf_mesh"))
    figure, axes = image.axes.get_figure(), image.axes
    assert axes.axison is False
    assert tuple(np.round(axes.get_position().extents, 9)) == (0.0, 0.0, 1.0, 1.0)
    assert figure.get_size_inches() == pytest.approx((pyplot.WIDTH / pyplot.DPI,
                                                      pyplot.HEIGHT / pyplot.DPI))


def test_a_saved_figure_is_the_picture_at_its_own_size(tmp_path):
    """What savefig writes is what the renderer painted, so frames chain into an animation."""
    payload = gtdata.render_data_of("dims/cf_mesh")
    pyplot.Plot(payload)
    pyplot.savefig(tmp_path / "frame.png")
    written = mplimage.imread(tmp_path / "frame.png")
    assert written.shape == (pyplot.HEIGHT, pyplot.WIDTH, 4)
    # the renderer's array is unclamped and its uint8 form is what a plot carries, so this is a
    # statement about matplotlib losing nothing on the way to a file, which is what frames need
    assert np.array_equal((written * 255).round().astype(np.uint8), painted(payload).to_uint8())


def test_an_axis_someone_else_made_is_painted_on_as_well():
    """A plot is an image on an axis, so gridspec, titles and labels all still work."""
    figure, axes = mpl.subplots(figsize=(4, 3))
    image = pyplot.Plot(gtdata.render_data_of("dims/cf_mesh"), ax=axes, width=400, height=300)
    assert image.axes is axes
    assert np.array_equal(image.get_array(),
                          painted(gtdata.render_data_of("dims/cf_mesh"),
                                  width=400, height=300).to_uint8())
    axes.set_title("u = xy")
    assert figure.get_size_inches() == pytest.approx((4, 3))


def test_a_picture_needs_no_rulers_and_no_stretching():
    """A plot takes the frame off a user's axis and keeps the shape the picture was painted in."""
    figure, axes = mpl.subplots(figsize=(4, 3))
    before = axes.get_position().width, axes.get_position().height
    image = pyplot.Plot(gtdata.render_data_of("dims/cf_mesh"), ax=axes)
    assert axes.axison is False
    axes.set_axis_on()
    assert axes.axison is True
    box, size = axes.get_position(), figure.get_size_inches()
    height, width, _ = image.get_array().shape
    # the axes may give room away to keep that shape, in inches and at one dpi, but takes none
    assert (box.width * size[0]) / (box.height * size[1]) == pytest.approx(width / height)
    assert box.width <= before[0] + 1e-12 and box.height <= before[1] + 1e-12
    assert figure.get_size_inches() == pytest.approx((4, 3))


@pytest.mark.parametrize("kwargs", [
    {"vmin": -0.5, "vmax": 2.0},
    {"autoscale": False},
    {"ncolors": 3},
    {"cmap": "viridis"},
    {"wireframe": False},
    {"edges": False, "line_width": 1.0},
    {"surface": False},
    {"elev": 20.0, "azim": 30.0, "roll": 0.0},
    {"n_segments": 3},
])
def test_a_keyword_of_the_plot_is_a_keyword_of_the_renderer(kwargs):
    """Nothing is translated on the way: the knob changes the picture exactly as it would there."""
    payload = gtdata.render_data_of("dims/surface_scalar")
    assert np.array_equal(pyplot.Plot(payload, **kwargs).get_array(),
                          painted(payload, **kwargs).to_uint8())


def test_the_view_is_named_the_way_a_viewer_names_it():
    """``elev``, ``azim`` and ``roll``: matplotlib's three words for a point of view.

    ``Axes3D.view_init`` takes them, in degrees, and its defaults are what every other 3D
    viewer opens with, so a picture of a solid is turned by the same numbers a reader of
    matplotlib would reach for.
    """
    payload = gtdata.render_data_of("dims/volume_scalar")
    image = pyplot.Plot(payload, elev=30, azim=-60, width=300, height=200)
    assert np.array_equal(image.get_array(),
                          painted(payload, elev=30, azim=-60, width=300,
                                  height=200).to_uint8())
    # and the picture really is turned: a flat view of the same solid looks different
    assert (image.get_array() != painted(payload, width=300, height=200).to_uint8()).any()


def test_the_triple_a_page_drag_leaves_behind_is_not_a_plot_keyword():
    """One spelling of a point of view, and it is the one a reader knows (docs/SPEC.md)."""
    payload = gtdata.render_data_of("dims/cf_mesh")
    with pytest.raises(TypeError, match="elev, azim and roll"):
        pyplot.Plot(payload, euler_angles=[20.0, 30.0, 0.0])
    # the renderer still reads it, because that is how a saved page replays
    assert painted(payload, euler_angles=[20.0, 30.0, 0.0]) is not None


def test_a_thing_that_is_not_a_knob_is_refused_rather_than_ignored():
    payload = gtdata.render_data_of("dims/cf_mesh")
    with pytest.raises((TypeError, ValueError), match="nonsense"):
        pyplot.Plot(payload, nonsense=1)


def test_the_part_of_a_value_asked_for_is_named_either_way():
    """``eval`` is the panel's name for it and ``function_mode`` the renderer's, and Draw's own
    spelling reaches nothing (docs/SPEC.md, "Why Draw is clunky"), so this layer answers for it."""
    payload = gtdata.render_data_of("dims/complex_h1")
    for name, mode in [("real", 5.0), ("imag", 6.0), ("norm", 3.0), ("x", 0.0), (6, 6.0)]:
        assert np.array_equal(pyplot.Plot(payload, eval=name).get_array(),
                              painted(payload, function_mode=mode).to_uint8())
    assert np.array_equal(pyplot.Plot(payload, function_mode=6).get_array(),
                          painted(payload, function_mode=6).to_uint8())


def test_a_plot_can_ask_for_a_colourbar_and_makes_room_for_it():
    """The picture is the plot, so a bar is the one thing that may move it."""
    payload = gtdata.render_data_of("dims/surface_scalar")
    image = pyplot.Plot(payload, colorbar=True)
    figure = image.axes.get_figure()
    assert len(figure.axes) == 2
    assert figure.axes[0].get_position().width < 1.0
    assert figure.axes[1].collections or figure.axes[1].images


def test_a_colourbar_shares_the_colours_and_the_ends_of_the_picture():
    """The bar is the texture of the draw and the range of the draw, which is what makes it a
    colourbar rather than a decoration: bands, not a gradient, and the ends the colours used."""
    payload = gtdata.render_data_of("dims/surface_scalar")
    image = pyplot.Plot(payload, ncolors=5)
    bar = pyplot.colorbar(image)
    assert bar.ax.get_figure() is image.axes.get_figure()
    assert bar.cmap.N == 5
    bands = scene.colormap_of(None, 5)
    assert np.max(np.abs(np.asarray(bar.cmap.colors, dtype=float)[..., :3] * 255
                         - bands * 255)) < 1
    low, high = scene.view_range(payload, function_mode=scene.settings_of(payload)["function_mode"])
    assert (bar.norm.vmin, bar.norm.vmax) == pytest.approx((low, high))


def test_a_colourbar_follows_the_colormap_and_the_range_of_the_picture():
    """A bar asked for after the fact is still the picture's bar, knobs and all."""
    payload = gtdata.render_data_of("dims/surface_scalar")
    image = pyplot.Plot(payload, cmap="viridis", vmin=-0.5, vmax=2.0, ncolors=32)
    bar = pyplot.colorbar(image)
    assert np.max(np.abs(np.asarray(bar.cmap.colors, dtype=float)[..., :3] * 255
                         - scene.colormap_of("viridis", 32) * 255)) < 1
    assert (bar.norm.vmin, bar.norm.vmax) == pytest.approx((-0.5, 2.0))


def test_a_mesh_has_no_colourbar_to_give_it():
    """A mesh-only draw is indexed by region and painted one green; there is no scale to show."""
    image = pyplot.Plot(gtdata.render_data_of("mesh_only"))
    with pytest.raises(ValueError, match="mesh"):
        pyplot.colorbar(image)


def test_the_layer_registers_no_colormap_anywhere():
    """The webgui ramp is the picture's own texture; matplotlib's registry is left as found.

    A registered "ngsolve" would name one band count and leak into every other figure in the
    process, so the bar is handed a private :class:`~matplotlib.colors.ListedColormap` built
    from :func:`ngsplot.colormap.to_matplotlib` instead.
    """
    known = set(matplotlib.colormaps)
    image = pyplot.Plot(gtdata.render_data_of("dims/surface_scalar"), cmap="ngsolve")
    bar = pyplot.colorbar(image)
    assert set(matplotlib.colormaps) == known
    assert isinstance(bar.cmap, matplotlib.colors.ListedColormap)
    # "ngsolve" is a name this layer answers for, and the ramp it stands for is the page's
    assert np.max(np.abs(np.asarray(bar.cmap.colors, dtype=float)[..., :3] * 255
                         - colormap.palette(colormap.NCOLORS) * 255)) < 1


@pytest.mark.parametrize("name", ["numberspace", "line_scalar", "volume_scalar"])
def test_a_payload_that_carries_no_hope_of_a_range_still_plots(name):
    """A constant field and a line have no colour range; they plot flat rather than fail."""
    payload = gtdata.render_data_of("dims/" + name)
    image = pyplot.Plot(payload)
    assert np.array_equal(image.get_array(), painted(payload).to_uint8())
    assert np.isfinite(painted(payload).color).all()


# --------------------------------------------------------------------------- ngsolve in


def test_a_gridfunction_plots_as_its_payload_does():
    ngs, capture_dims = ngsolve_objects()
    from ngsplot import _ngsolve

    mesh = capture_dims.mesh_square()
    gfu = ngs.GridFunction(ngs.H1(mesh, order=2))
    gfu.Set(ngs.x - ngs.y)
    assert np.array_equal(pyplot.Plot(gfu).get_array(),
                          pyplot.Plot(_ngsolve.render_data(gfu)).get_array())


def test_a_coefficient_function_is_painted_on_the_mesh_it_is_given():
    ngs, capture_dims = ngsolve_objects()
    from ngsplot import _ngsolve

    mesh = capture_dims.mesh_square()
    assert np.array_equal(pyplot.Plot(ngs.x * ngs.y, mesh=mesh).get_array(),
                          pyplot.Plot(_ngsolve.render_data(ngs.x * ngs.y, mesh)).get_array())


def test_a_coefficient_function_without_a_mesh_hears_what_ngsolve_says():
    ngs, _ = ngsolve_objects()
    with pytest.raises(RuntimeError, match="without mesh"):
        pyplot.Plot(ngs.x * ngs.y)


def test_a_mesh_alone_plots_the_mesh_the_viewer_paints():
    _, capture_dims = ngsolve_objects()
    from ngsplot import _ngsolve

    mesh = capture_dims.mesh_square()
    image = pyplot.Plot(mesh)
    assert np.array_equal(image.get_array(),
                          pyplot.Plot(_ngsolve.render_data(mesh)).get_array())
    # the one texel a mesh-only draw is handed is green, and the shading of a mesh leaves it
    # green: gtdata.png_of("mesh_only") has the same green, at about a quarter brightness
    green = image.get_array()[..., :3]
    assert (green[..., 1] > 40).any()
    assert ((green[..., 1] > 4 * green[..., 0]) & (green[..., 1] > 4 * green[..., 2])).any()


def test_a_mesh_under_its_own_name_is_still_the_mesh_drawn():
    """``Plot(mesh=mesh)`` is ``Draw(mesh)``: the keyword alone says what to draw."""
    _, capture_dims = ngsolve_objects()
    from ngsplot import _ngsolve

    mesh = capture_dims.mesh_square()
    named = pyplot.Plot(mesh=mesh)
    assert np.array_equal(named.get_array(), pyplot.Plot(mesh).get_array())
    assert np.array_equal(named.get_array(),
                          painted(_ngsolve.render_data(mesh)).to_uint8())

    figure, axes = mpl.subplots(figsize=(3, 2))
    assert pyplot.Plot(ax=axes, mesh=mesh, wireframe=False).axes is axes


def test_nothing_to_draw_is_said_rather_than_guessed():
    """A Plot with nothing in it names what it takes instead of painting a mesh by accident."""
    with pytest.raises(TypeError, match="something to draw"):
        pyplot.Plot()


def test_the_name_of_the_plot_is_the_counterpart_of_the_name_of_draw():
    """``ngsolve.webgui.Draw`` has its ``ngsplot.Plot``, and no half-migrated second name."""
    import ngsplot

    assert not hasattr(pyplot, "plot")
    assert ngsplot.Plot is pyplot.Plot
    assert "Plot" in ngsplot.__all__


def test_a_deformation_is_understood_twice_and_understood_once():
    """The payload is told to lift its geometry, as a page is; the amount rides along as scale.

    ``Draw`` reads ``deformation`` as the flag the shaders switch on and ``scale`` as how far
    to push, and hands both to the page (``deformation``, ``deformation_scale``), which is what
    :func:`ngsplot.scene.settings_of` folds back into one amount.  Passing a float to ``Draw``
    is not an alternative: it is taken for a displacement field and ngsolve refuses it.
    """
    ngs, capture_dims = ngsolve_objects()
    from ngsplot import _ngsolve

    mesh = capture_dims.mesh_square()
    gfu = ngs.GridFunction(ngs.H1(mesh, order=2))
    gfu.Set(ngs.x * ngs.y)
    image = pyplot.Plot(gfu, deformation=0.7)
    assert np.array_equal(image.get_array(),
                          painted(_ngsolve.render_data(gfu, deformation=True,
                                                       scale=0.7)).to_uint8())
    assert not np.array_equal(image.get_array(),
                              painted(_ngsolve.render_data(gfu, deformation=True)).to_uint8())
    with pytest.raises(Exception, match="Dimensions"):
        _ngsolve.render_data(gfu, deformation=0.7)


# ------------------------------------------------------------------------- pyplot habits


def test_the_figure_and_the_axis_are_tracked_as_pyplot_tracks_them():
    """plot() draws into the current axis, sca() moves it, cla() empties it, close() forgets it."""
    figure = pyplot.gcf()
    assert pyplot.gcf() is figure
    first = pyplot.Plot(gtdata.render_data_of("dims/cf_mesh"))
    assert first.axes is pyplot.gca()
    assert pyplot.gca() is first.axes

    other, _ = mpl.subplots(figsize=(3, 2))
    axes, = other.axes
    pyplot.sca(axes)
    second = pyplot.Plot(gtdata.render_data_of("dims/cf_vector"))
    assert second.axes is axes
    assert len(axes.get_images()) == 1

    pyplot.cla()
    assert axes.get_images() == []
    assert pyplot.gca() is axes

    pyplot.close("all")
    assert pyplot.gcf() is not figure


def test_a_plot_replaces_the_picture_rather_than_lying_under_it():
    """Plotting twice is one picture over the whole axis, with whatever else was on it intact."""
    image = pyplot.Plot(gtdata.render_data_of("dims/cf_mesh"))
    axes = image.axes
    axes.set_title("later")
    image = pyplot.Plot(gtdata.render_data_of("dims/cf_vector"))
    assert len(axes.get_images()) == 1
    assert axes.get_title() == "later"
