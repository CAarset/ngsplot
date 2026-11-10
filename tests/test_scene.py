"""The knobs: what the defaults are, and what every one of them does.

``scene.paint`` is the entry point a script calls, so this file asks two things of it.
The first is that with nothing given but render data it produces what the browser
produced -- :mod:`tests.test_parity` measures that picture by picture, and here the
settings behind it are checked one by one against the uniforms read out of the live
renderer: the number of colour bands, the range they cover, the thickness of a line and
the subdivision of a curve.

The second is that every one of those settings is a keyword, matplotlib-style, and that
giving it does what the same click on the panel did.  That half is answered with the
recorded scenes themselves: ``square_fixed_range``, ``square_nc2``, ``square_nc32``,
``square_no_wireframe`` and ``square_rotated`` are ``square_scalar`` redrawn after one
panel control was moved, so drawing ``square_scalar`` with the matching keyword has to
give the other scene's picture, and a wrong knob shows up as a wrong picture rather
than as a passing unit test.

The comparisons are the ones :mod:`tests.test_parity` sets up: inside the surface we
painted, shrunk by 15 pixels to stay off the silhouette, and away from any pixel the
recorded picture shows a line on, since a line and the surface under it come from
different payloads and land on a pixel at different phases.
"""

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

import gtdata
from ngsplot import colormap, scene

# the scenes captured from the plain page; the five variants of square_scalar are the
# known answers for the range, the band count, the lines and the angle
SCENES = ["square_scalar", "square_no_wireframe", "square_rotated", "square_deformed",
          "square_nc2", "square_nc32", "square_fixed_range", "mesh_only"]

INSET = 15       # pixels, to stay clear of the silhouette and of the Edges outline
FRINGE = 2       # pixels of antialiasing around a drawn line


def square():
    """``square_scalar``'s render data and the buffer it was drawn into."""
    render_data = gtdata.render_data_of("square_scalar")
    return render_data, *gtdata.framebuffer_of("square_scalar")


def levels(canvas):
    """Our canvas as grey levels, unrounded: the units the recorded picture is in."""
    return np.clip(canvas.color[..., :3] * 255.0, 0.0, 255.0)


def grown(mask, fringe=FRINGE):
    """`mask` widened by `fringe` pixels in every direction."""
    out = np.zeros_like(mask)
    window = 2 * fringe + 1
    out[fringe:-fringe, fringe:-fringe] = (
        sliding_window_view(mask, (window, window)).any(axis=(2, 3)))
    return out


def surface_box(canvas):
    """True inside the surface of ``canvas``, shrunk by :data:`INSET` pixels all round."""
    rows, cols = np.nonzero(canvas.color[..., 0] < 1)
    box = np.zeros(canvas.color.shape[:2], dtype=bool)
    box[rows.min() + INSET:rows.max() - INSET, cols.min() + INSET:cols.max() - INSET] = True
    return box


def off_lines(canvas, picture):
    """Our canvas against a recorded picture, on the surface and away from its lines."""
    ours = canvas.to_uint8()[..., :3].astype(int)
    difference = np.abs(picture - ours).max(axis=2)
    return difference[surface_box(canvas) & ~grown(picture.max(axis=2) < 90)]


def draw(name, **keywords):
    """A recorded scene through ``scene.paint``, next to the picture it should equal."""
    render_data = gtdata.render_data_of(name)
    width, height = gtdata.framebuffer_of(name)
    canvas = scene.paint(render_data, width=width, height=height, **keywords)
    picture = gtdata.png_of(name)[:height, :width, :3].astype(int)
    return canvas, picture


def drawn_like(name, other, **keywords):
    """One scene's payload drawn with a knob, against the scene a panel move recorded."""
    render_data = gtdata.render_data_of(name)
    width, height = gtdata.framebuffer_of(other)
    canvas = scene.paint(render_data, width=width, height=height, **keywords)
    return canvas, gtdata.png_of(other)[:height, :width, :3].astype(int)


def assert_picture(canvas, picture):
    """The comparison every knob ends in: the same surface, grey level by grey level."""
    difference = off_lines(canvas, picture)
    assert difference.size > 0.5 * surface_box(canvas).sum(), "the line mask ate the surface"
    assert np.median(difference) == 0
    assert difference.mean() < 0.5
    assert np.count_nonzero(difference > 8) < 0.02 * difference.size


# ---------------------------------------------------------------------- the defaults

def test_the_defaults_are_the_settings_the_page_had():
    """Nothing given, and the uniforms say the same thing anyway.

    These are the panel defaults of the webgui (``gui.ts:63-105``): eight colour bands,
    autoscaling on, a line thickness of five, a subdivision of five.  The last two are
    numbers of the panel and of the canvas both at once -- ``scene.ts:776`` divides the
    thickness by the height of the drawing buffer before the shader ever sees it.
    """
    assert scene.DEFAULTS["ncolors"] == 8
    assert scene.DEFAULTS["autoscale"] is True
    assert scene.DEFAULTS["line_width"] == 5.0
    assert scene.DEFAULTS["subdivision"] == 5
    assert scene.DEFAULTS["wireframe"] is True and scene.DEFAULTS["edges"] is True
    assert scene.DEFAULTS["dark_backside"] is True


@pytest.mark.parametrize("name", SCENES)
def test_the_settings_of_a_scene_are_the_uniforms_it_was_drawn_with(name):
    render_data = gtdata.render_data_of(name)
    uniforms = gtdata.uniforms_of(name)
    _, height = gtdata.framebuffer_of(name)
    settings = scene.settings_of(render_data)

    assert settings["subdivision"] == uniforms["n_segments"]
    assert settings["line_width"] / height == pytest.approx(uniforms["line_thickness"])
    assert settings["function_mode"] == uniforms["function_mode"]
    assert settings["dark_backside"] == uniforms["dark_backside"]
    assert scene.view_range(render_data, function_mode=settings["function_mode"]) == (
        uniforms["colormap_min"], uniforms["colormap_max"])
    palette = scene.colormap_of(None, settings["ncolors"], settings["mesh_only"])
    recorded = np.asarray(uniforms["tex_colormap"]["data"]).reshape(-1, 4)[:, :3]
    assert len(palette) == len(recorded)
    assert np.abs(palette - recorded).max() == 0.0


def test_a_linear_mesh_is_not_subdivided_and_a_fast_draw_is_not_either():
    """``gui.ts:229-231`` and ``camera.ts:345-359``: no curve to follow, no samples.

    A scene of linear elements copies the subdivision of the panel as 1, and fast_draw
    forces 1 on whatever the panel says; both save the very work the renderer does worst.
    """
    linear = {"funcdim": 1, "order2d": 1, "order3d": 1}
    assert scene.settings_of(linear)["subdivision"] == 1
    curved = {"funcdim": 1, "order2d": 2, "order3d": 1}
    assert scene.settings_of(curved)["subdivision"] == 5
    assert scene.settings_of({**curved, "gui_settings": {"Misc": {"fast_draw": True}}})[
        "subdivision"] == 1
    assert scene.settings_of({**curved, "gui_settings": {"Misc": {"subdivision": 3}}})[
        "subdivision"] == 3


def test_a_flat_payload_lifts_by_the_scale_it_carries():
    """``gui.ts:582-587``: the deformation is the scale when the geometry came flat."""
    assert scene.settings_of({"funcdim": 1, "deformation": True, "deformation_scale": 2.0})[
        "deformation"] == 2.0
    assert scene.settings_of({"funcdim": 1, "deformation": True})["deformation"] == 1.0
    assert scene.settings_of({"funcdim": 1, "deformation": False, "deformation_scale": 2.0})[
        "deformation"] == 0.0


def test_a_payload_may_name_its_knobs_the_way_draw_does():
    """The flat names a payload carries outrank the sections they belong to.

    ``updateSettings`` (``gui.ts:146-175``) copies every flat name of the payload's
    settings into the section it belongs to -- ``colormap_min`` into ``Colormap.min``,
    ``mesh`` into ``Objects.Wireframe``, ``elements`` into ``Objects.Surface`` -- over
    whatever that section said as well.  They are the names ``Draw`` itself writes, so
    they are the first ones a script meets.
    """
    settings = scene.settings_of({
        "funcdim": 1, "order2d": 3, "order3d": 2,
        "gui_settings": {
            "Colormap": {"min": 0.0, "max": 1.0, "ncolors": 8},
            "colormap_min": -2.0, "colormap_max": 3.0, "colormap_ncolors": 4,
            "line_thickness": 2.0, "subdivision": 7,
            "mesh": False, "edges": False, "elements": False,
        },
    })
    assert settings["ncolors"] == 4
    assert settings["line_width"] == 2.0
    assert settings["subdivision"] == 7
    assert settings["wireframe"] is False and settings["edges"] is False
    assert settings["surface"] is False

    ends = {"funcdim": 1, "funcmin": -9.0, "funcmax": 9.0, "draw_surf": True,
            "autoscale": False,      # otherwise the ends of the function overwrite them
            "gui_settings": {"Colormap": {"min": 0.0, "max": 1.0},
                             "colormap_min": -2.0, "colormap_max": 3.0}}
    assert scene.view_range(ends) == (-2.0, 3.0)


def test_a_linear_mesh_goes_unsubdivided_only_until_the_payload_says_otherwise():
    """The page forces one segment for a linear mesh *before* the payload speaks.

    ``gui.ts:229-231`` writes the forcing into the defaults, and the settings of the
    payload are applied over the defaults afterwards, so a payload that names a
    subdivision of its own gets it.  ``Misc.reduce_subdivision`` is the older name of
    the ``fast_draw`` checkbox, which shortens it the other way.
    """
    linear = {"order2d": 1, "order3d": 1}
    assert scene.settings_of(linear)["subdivision"] == 1
    assert scene.settings_of({**linear,
                              "gui_settings": {"Misc": {"subdivision": 3}}})["subdivision"] == 3
    assert scene.settings_of({**linear,
                              "gui_settings": {"subdivision": 9}})["subdivision"] == 9

    curved = {"order2d": 3, "order3d": 2}
    assert scene.settings_of(curved)["subdivision"] == 5
    assert scene.settings_of({**curved, "gui_settings": {"Misc": {"reduce_subdivision": True}}})["subdivision"] == 1


# ------------------------------------------------------------------- the value range

def test_an_autoscaled_draw_follows_the_function_and_a_still_one_the_panel():
    """The panel's own minimum and maximum, versus the ends of the function."""
    moving = {"funcdim": 1, "funcmin": -3.0, "funcmax": 7.0, "draw_surf": True}
    assert scene.view_range(moving) == (-3.0, 7.0)
    still = {**moving, "gui_settings": {"Colormap": {"autoscale": False, "min": 0.0,
                                                     "max": 1.0}}}
    assert scene.view_range(still) == (0.0, 1.0)
    assert scene.view_range(still, autoscale=True) == (-3.0, 7.0)
    assert scene.view_range(moving, vmin=-1.0) == (-1.0, 7.0)
    assert scene.view_range(moving, vmax=-1.0) == (-3.0, -1.0)


def test_a_length_never_goes_below_zero():
    """``gui.ts:372-379``: drawn as a length, a vector runs from 0 to the further end."""
    vector = {"funcdim": 2, "funcmin": -3.0, "funcmax": 7.0, "draw_surf": True}
    assert scene.view_range(vector, function_mode=3.0) == (0.0, 7.0)
    assert scene.view_range(vector) == (-3.0, 7.0)      # as components, not length


def test_a_payload_that_says_it_is_still_is_left_alone():
    """``Draw(autoscale=False)`` leaves one flag at the top of the payload (``webgui.py:548``).

    The panel re-reads the ends of the function only when that flag lets it
    (``scene.ts:497``, through ``updateColormapToAutoscale``), so a vector drawn as a
    length otherwise keeps whatever ``funcmin`` and ``funcmax`` happen to say, a negative
    end and all -- which is the point of asking for a fixed range.
    """
    vector = {"funcdim": 2, "funcmin": -3.0, "funcmax": 7.0, "draw_surf": True}
    assert scene.settings_of(vector)["autoscale"] is True
    assert scene.view_range(vector, function_mode=3.0) == (0.0, 7.0)

    still = {**vector, "autoscale": False}
    assert scene.settings_of(still)["autoscale"] is False
    assert scene.view_range(still, function_mode=3.0) == (-3.0, 7.0)
    assert scene.view_range(still, function_mode=3.0, vmax=1.0) == (-3.0, 1.0)


def test_a_mesh_without_a_function_is_painted_by_region():
    """``gui.ts:328-334``: one colour per region, one band per region, no autoscaling."""
    regions = {"funcdim": 0, "mesh_regions_2d": 3}
    assert scene.view_range(regions) == (-0.5, 2.5)
    assert scene.view_range(regions, autoscale=False) == (-0.5, 2.5)
    assert scene.colormap_of(None, 8, mesh_only=True).tolist() == [[0.0, 1.0, 0.0]]


def test_a_payload_without_a_range_is_spread_over_the_unit_interval():
    """The sentinel pair of an accumulator that never ran is not a range.

    A field sampled nowhere -- or, as here, a one-dimensional one, whose minimum and
    maximum ngsolve only accumulates over surface samples (``webgui.py:475``) -- arrives
    as ``(DBL_MAX, DBL_MIN)``.  The page paints it as one flat colour and labels its
    colourbar ``1.80e+308``; a range of nothing is worth the 0 to 1 matplotlib gives.
    """
    from ngsplot.geometry import DBL_MAX, DBL_MIN
    sentinel = {"funcdim": 1, "funcmin": DBL_MAX, "funcmax": DBL_MIN, "draw_surf": True}
    assert scene.view_range(sentinel) == (0.0, 1.0)
    flat = {"funcdim": 1, "funcmin": 0.0, "funcmax": DBL_MIN, "draw_surf": True}
    assert scene.view_range(flat) == (0.0, DBL_MIN)     # a constant field, honestly
    one_value = {"funcdim": 1, "funcmin": 2.0, "funcmax": 2.0, "draw_surf": True}
    assert scene.view_range(one_value) == (0.0, 1.0)


def test_which_part_of_a_value_is_drawn_is_the_panels_choice():
    """``gui.ts:529-571``: a complex value by its real part, a vector by its length.

    ``BuildRenderData`` splits a complex function into two components and writes
    ``is_complex`` next to them (``ngsolve/webgui.py:303-305``), so the payload carries the
    distinction itself; the keyword is for a pair that came without it.  An ``eval`` at the
    top of the payload outranks both, since ``initComplex`` writes it once the panel is
    built (``gui.ts:568-571``).  A ``Colormap`` section that names a mode of its own is
    written *over* by the mode chosen for a complex or multi-component value, and so stands
    only for a scalar; and a flat ``eval`` beside the sections is a name the page never
    folds (``gui.ts:145-175``), which is to say it means nothing.
    """
    assert scene.settings_of({"funcdim": 1})["function_mode"] == 0.0
    assert scene.settings_of({"funcdim": 2})["function_mode"] == 3.0
    assert scene.settings_of({"funcdim": 3})["function_mode"] == 3.0
    assert scene.settings_of({"funcdim": 2, "is_complex": True})["function_mode"] == 5.0
    assert scene.settings_of({"funcdim": 2, "is_complex": True})["is_complex"] is True
    assert scene.settings_of({"funcdim": 2}, is_complex=True)["function_mode"] == 5.0
    assert scene.settings_of({"funcdim": 2, "is_complex": True}, is_complex=False)["function_mode"] == 3.0
    assert scene.settings_of({"funcdim": 2, "eval": 1})["function_mode"] == 1.0
    assert scene.settings_of({"funcdim": 1, "gui_settings": {"Colormap": {"eval": 1}}})["function_mode"] == 1.0
    assert scene.settings_of({"funcdim": 2, "gui_settings": {"Colormap": {"eval": 1}}})["function_mode"] == 3.0
    assert scene.settings_of(
        {"funcdim": 2, "is_complex": True, "gui_settings": {"Colormap": {"eval": 6}}})["function_mode"] == 5.0
    assert scene.settings_of({"funcdim": 1, "eval": 2,
                              "gui_settings": {"Colormap": {"eval": 1}}})["function_mode"] == 2.0
    assert scene.settings_of({"funcdim": 2, "gui_settings": {"eval": 0}})["function_mode"] == 3.0
    assert scene.settings_of({"funcdim": 1, "gui_settings": {"eval": 1}})["function_mode"] == 0.0


def test_a_missing_field_is_the_unit_interval_too():
    """A payload with no function on it at all still has to mean something."""
    assert scene.view_range({"funcdim": 1}) == (0.0, 1.0)


# ------------------------------------------------------------------------- colours

def test_the_matplotlib_round_trip_is_the_exact_palette():
    """``to_matplotlib`` out and ``cmap`` back in gives the same colours, band for band.

    A ``ListedColormap`` is read at the centres of its bands, which is where a shader
    reads a texture of ``ncolors`` texels, so no band moves.  Two things cost: a
    ``ListedColormap`` keeps bytes, so the round trip is exact to a grey level and no
    finer, and it carries its own band count with it -- a colormap of eight bands drawn
    with thirty-two asks for the eight, resampled, where a name is cut to the count.
    """
    for n in (2, 8, 32):
        assert np.abs(scene.colormap_of(colormap.to_matplotlib(colormap.palette(n)), n)
                      - colormap.palette(n)).max() < 1 / 255
    # the webgui ramp answers to a name of its own, at whatever band count is asked for
    assert np.array_equal(scene.colormap_of("ngsolve", 32), colormap.palette(32))


def test_a_named_colormap_is_sampled_at_the_centres_of_its_bands():
    """A matplotlib colormap of any length, cut into ``ncolors`` equal bands."""
    from matplotlib import colormaps
    wanted = np.asarray(colormaps["viridis"]((np.arange(8) + 0.5) / 8))[:, :3]
    assert np.array_equal(scene.colormap_of("viridis", 8), wanted)
    assert scene.colormap_of("viridis", 3).shape == (3, 3)


def test_a_colormap_of_one_band_is_refused_as_gently_as_one_colour_is_allowed():
    """Two texels is where a lookup starts to mean something; the panel slider says 2."""
    with pytest.raises(ValueError, match="at least 2"):
        scene.colormap_of("viridis", 1)
    assert scene.colormap_of(None, 1, mesh_only=True).shape == (1, 3)


def test_a_different_colormap_draws_a_different_picture():
    """The keyword reaches the picture, and not only the settings object."""
    render_data = gtdata.render_data_of("square_scalar")
    width, height = gtdata.framebuffer_of("square_scalar")
    plain = scene.paint(render_data, width=width, height=height)
    viridis = scene.paint(render_data, width=width, height=height, cmap="viridis")
    box = surface_box(plain)
    changed = np.abs(levels(plain) - levels(viridis)).max(axis=2)[box]
    assert (changed > 8).mean() > 0.9


# ---------------------------------------------------------------------------- lines

def test_the_wireframe_switch_leaves_no_ink_of_its_own():
    """``Objects.Wireframe`` off, and not a dark pixel left in the middle of the surface."""
    render_data, width, height = square()
    def dark(**keywords):
        canvas = scene.paint(render_data, width=width, height=height, **keywords)
        return int((levels(canvas).max(axis=2)[surface_box(canvas)] < 90).sum())

    assert dark() > 3000
    assert dark(wireframe=False) == 0        # the interior of square_scalar is all wireframe
    assert dark(edges=False) > 3000          # the outline is outside the box anyway


def test_the_surface_switch_leaves_the_lines_standing():
    """``Objects.Surface`` off: no colour of its own, and nothing but grey ink left."""
    render_data, width, height = square()
    canvas = scene.paint(render_data, width=width, height=height, surface=False)
    assert int((np.ptp(canvas.color[..., :3], axis=-1) > 0).sum()) == 0
    both = scene.paint(render_data, width=width, height=height)
    assert int((np.ptp(both.color[..., :3], axis=-1) > 0).sum()) > 1000


def test_the_recorded_range_is_the_scene_that_recorded_it():
    """``square_fixed_range`` is ``square_scalar`` with autoscaling off at -0.5 and 2."""
    canvas, picture = drawn_like("square_scalar", "square_fixed_range",
                                 vmin=-0.5, vmax=2.0)
    assert_picture(canvas, picture)


def test_the_recorded_band_count_is_the_scene_that_recorded_it():
    """The colour slider, at 2 and at 32."""
    for ncolors, other in ((2, "square_nc2"), (32, "square_nc32")):
        canvas, picture = drawn_like("square_scalar", other, ncolors=ncolors)
        assert_picture(canvas, picture)


def test_the_recorded_lines_are_the_scene_that_recorded_it():
    """``square_no_wireframe`` is ``square_scalar`` with the wireframe off and a line of 1."""
    canvas, picture = drawn_like("square_scalar", "square_no_wireframe",
                                 wireframe=False, line_width=1)
    assert_picture(canvas, picture)


def test_the_recorded_angle_is_the_scene_that_recorded_it():
    """The camera of the payload, and the same camera given by hand."""
    canvas, picture = draw("square_rotated")
    assert_picture(canvas, picture)
    canvas, picture = drawn_like("square_scalar", "square_rotated", euler_angles=(20, 30, 0))
    assert_picture(canvas, picture)


def test_a_deformed_scene_lifts_its_lines_with_its_surface():
    """The lines of ``square_deformed`` move when the deformation does, as they do there.

    A payload whose ``deformation`` is ``true`` says the geometry went out flat and the
    shader lifted it, so the lines have to be lifted by us as well; turning the scale to
    zero here flattens both together, and the picture is a plain square with its lines.
    """
    render_data = gtdata.render_data_of("square_deformed")
    width, height = gtdata.framebuffer_of("square_deformed")
    assert scene.settings_of(render_data)["deformation"] == 1.0
    assert gtdata.uniforms_of("square_deformed")["deformation"] == 1.0
    lifted = scene.paint(render_data, width=width, height=height)
    flat = scene.paint(render_data, width=width, height=height, deformation=0.0)
    moved = np.abs(levels(lifted) - levels(flat)).max(axis=2) > 8
    assert moved.sum() > 5000                      # a whole sheet of bent surface
    assert moved[surface_box(lifted)].sum() > 5000  # and not only along its silhouette


# ------------------------------------------------------------------------ the canvas

def test_the_picture_is_the_buffer_and_nothing_else():
    """Whatever size is asked for, that is the array, at four channels and in bytes."""
    render_data, _, _ = square()
    canvas = scene.paint(render_data, width=40, height=25)
    assert canvas.color.shape == (25, 40, 4)
    assert canvas.to_uint8().shape == (25, 40, 4)
    assert canvas.to_uint8().dtype == np.uint8
    assert (canvas.to_uint8()[0, 0] == (255, 255, 255, 255)).all()


def test_the_same_call_twice_is_the_same_picture():
    """A canvas belongs to one call: nothing is kept on the module between draws."""
    render_data, width, height = square()
    first = scene.paint(render_data, width=width, height=height, cmap="viridis")
    again = scene.paint(render_data, width=width, height=height, cmap="viridis")
    assert np.array_equal(first.color, again.color)
    third = scene.paint(render_data, width=width, height=height, ncolors=2)
    assert not np.array_equal(first.color, third.color)
