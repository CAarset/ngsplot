"""One call, every dimension: lines, surfaces, solids, spaces, and bare coefficient functions.

``tests/gt/dims`` holds the captures recorded for ``docs/DIMS.md`` -- a 1D mesh, a 2D surface
mesh, a 3D solid, a ``NumberSpace``, an ``L2`` of order 0, a ``CoefficientFunction`` drawn
with and without a mesh, a vector field, an ``H(curl)`` field and a complex ``H1`` one.  Four
findings of that document decide what a test may ask of each.

* The recorded screenshots are of a *page*, so they carry axes and a colourbar that ngsplot
  leaves to matplotlib.  The comparison is made inside the mesh, away from that furniture.
* A complex function arrives as two real components, but it says so while doing it, and
  that word -- not the caller's keyword -- is what decides whether the real part or the
  length of the pair is plotted.  It is painted from its payload like every other scene.
* A 1D draw never reaches the surface code, and that is where ngsolve accumulates its value
  range, so a 1D payload arrives with the sentinel pair ``(DBL_MAX, DBL_MIN)`` and the
  function itself is not drawn at all: what is recorded is a flat black line over the mesh.
  The line is matched exactly.  Colouring it is left as a deliberate improvement.
* ``surface_scalar`` is not a surface in 3D -- the gmsh import dropped the bump's ``z`` and
  left the mesh without boundary elements -- so only its outline is a parity target here.
"""

import os

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

import gtdata
from ngsplot import geometry, scene

#: rows around the mesh, well clear of the axes and the colourbar of the page
MIDDLE = (slice(240, 360), slice(None))


def draw(name, **keywords):
    """Paint a dims scene into a canvas the size of its own buffer, and return the recording."""
    render_data = gtdata.render_data_of(name)
    width, height = gtdata.framebuffer_of(name)
    canvas = scene.paint(render_data, width=width, height=height, **keywords)
    return canvas, gtdata.png_of(name)[:height, :width, :3].astype(int)


def interior(canvas, fringe=20):
    """The box of the picture our own surface covers, grown a little past its edge."""
    rows, cols = np.nonzero(canvas.color[..., 0] < 1)
    box = np.zeros(canvas.color.shape[:2], bool)
    box[rows.min() + fringe:rows.max() - fringe, cols.min() + fringe:cols.max() - fringe] = True
    return box


def grown(mask, fringe=2):
    """``mask`` spread into its neighbourhood, to cover the pixels a line bleeds into."""
    side = 2 * fringe + 1
    padded = np.pad(mask, fringe)
    return sliding_window_view(padded, (side, side)).any(axis=(2, 3))


def disagreement(canvas, picture, where):
    """``(median, mean, fraction)`` of the worst-channel difference over ``where``.

    ``where`` is a boolean mask or a tuple of slices; a mask of all ``True`` would compare
    the whole page, furniture included, so a region that is mostly background is a bug.
    """
    ours = canvas.to_uint8()[..., :3].astype(int)
    difference = np.abs(picture - ours).max(axis=2)
    region = np.zeros(difference.shape, bool)
    region[where] = True
    values = difference[region]
    assert values.size > 0.2 * region.sum(), "the compared region is mostly empty"
    return (float(np.median(values)), float(values.mean()), float((values > 8).mean()))


def assert_like_recording(canvas, picture, where):
    median, mean, fraction = disagreement(canvas, picture, where)
    assert median == 0, f"half the pixels differ, by {median} on average"
    assert mean < 0.5, f"the pixels differ by {mean:.3f} on average"
    assert fraction < 0.02, f"{100 * fraction:.2f}% of the pixels differ by more than 8"


def looks_like_recording(canvas, picture, where):
    """Whether ``assert_like_recording`` would pass -- for the assertions about difference."""
    median, mean, fraction = disagreement(canvas, picture, where)
    return median == 0 and mean < 0.5 and fraction < 0.02


SPANS = {
    # scene: (mesh_dim, funcdim, draw_vol) as the payload says
    "dims/line_scalar": (1, 1, False),
    "dims/line_deformed": (1, 1, False),
    "dims/cf_mesh": (2, 1, False),
    "dims/numberspace": (2, 1, False),
    "dims/numberspace_l2pwc": (2, 1, False),
    "dims/surface_scalar": (2, 1, False),
    "dims/cf_vector": (2, 2, False),
    "dims/vec_hcurl": (2, 2, False),
    "dims/complex_h1": (2, 2, False),
    "dims/volume_scalar": (3, 1, True),
}


def test_the_captures_span_every_mesh_dimension_and_kind_of_input():
    """The scenes below really are the 1D, 2D and 3D cases the API has to serve."""
    for name, span in SPANS.items():
        render_data = gtdata.render_data_of(name)
        got = (render_data["mesh_dim"], render_data["funcdim"], bool(render_data["draw_vol"]))
        assert got == span, f"{name} is {got}, not {span}"


SPACES = ["dims/numberspace", "dims/numberspace_l2pwc", "dims/cf_mesh", "dims/cf_vector",
          "dims/vec_hcurl", "dims/complex_h1", "dims/volume_scalar"]


@pytest.mark.parametrize("name", SPACES)
def test_a_dimension_draws_the_picture_that_was_recorded(name):
    """Defaults, no keywords: a space of any dimension comes out as the page drew it."""
    canvas, picture = draw(name)
    inside = interior(canvas) & ~grown(picture.max(axis=2) < 90)
    assert_like_recording(canvas, picture, inside)


@pytest.mark.parametrize("name", ["dims/line_scalar", "dims/line_deformed"])
def test_a_line_is_the_flat_line_the_page_recorded(name):
    """Draw plots no function along a 1D mesh, and neither does this, to the pixel.

    What the capture holds is a black line a couple of pixels thick along the mesh line,
    plus the furniture of the page.  Inside a band around the mesh the two pictures hold
    exactly the same ink -- same rows, same columns, same pixels.  Colouring the line by the
    function it carries is the one place ngsplot will differ, and it is documented as such.
    """
    canvas, picture = draw(name)
    ours = canvas.to_uint8()[..., :3].astype(int)
    ink = [mask[MIDDLE] for mask in (ours.max(axis=2) < 250, picture.max(axis=2) < 250)]
    rows = [np.nonzero(mask.any(axis=1))[0] for mask in ink]
    assert (rows[0].min(), rows[0].max()) == (rows[1].min(), rows[1].max())
    assert ink[0].sum() == ink[1].sum() == (ink[0] & ink[1]).sum() > 1000
    median, mean, fraction = disagreement(canvas, picture, MIDDLE)
    assert (median, fraction) == (0.0, 0.0)
    assert mean < 0.05


@pytest.mark.parametrize("name", ["dims/line_scalar", "dims/line_deformed"])
def test_the_range_of_a_one_dimensional_payload_is_no_range(name):
    """The sentinel pair of a 1D payload becomes the unit interval, not a division by zero.

    ``webgui.Draw`` adds to ``funcmin``/``funcmax`` only while it draws a surface, so a 1D
    payload leaves them at the ``DBL_MAX``/``DBL_MIN`` pair it started with.  Take that
    literally and every value lands in the same band by accident; the guard makes the range
    the unit interval, which is what a flat colour over it looks like either way.
    """
    render_data = gtdata.render_data_of(name)
    assert (render_data["funcmin"], render_data["funcmax"]) == (geometry.DBL_MAX, geometry.DBL_MIN)
    assert scene.view_range(render_data) == (0.0, 1.0)


def test_a_field_that_never_leaves_zero_keeps_the_range_it_was_given():
    """A ``NumberSpace`` holds one dof and this capture holds the value 0: not a missing range.

    ``(0.0, DBL_MIN)`` is a real range -- a field of zeros -- and only ``(DBL_MAX, DBL_MIN)``
    means nobody ever wrote to it.  Telling the two apart is the difference between a flat
    colour and a silently rescaled one.
    """
    render_data = gtdata.render_data_of("dims/numberspace")
    assert (render_data["funcmin"], render_data["funcmax"]) == (0.0, geometry.DBL_MIN)
    assert scene.view_range(render_data) == (0.0, geometry.DBL_MIN)


def test_a_number_space_is_the_one_flat_colour_the_page_showed():
    """One degree of freedom, one colour, shaded as the page shades a flat square."""
    canvas, picture = draw("dims/numberspace")
    inside = interior(canvas) & ~grown(picture.max(axis=2) < 90)
    ours, theirs = canvas.to_uint8()[..., :3], picture
    assert inside.sum() > 10000
    assert np.allclose(ours[inside].mean(axis=0), theirs[inside].mean(axis=0), atol=1.5)
    blue, green = ours[inside].mean(axis=0)[2], ours[inside].mean(axis=0)[1]
    assert blue > 100 and green < 40, "the zero band of the ramp is blue, not anything else"


def test_a_complex_payload_says_so_and_the_caller_can_overrule_it():
    """The payload names a complex function; the keyword is there to overrule it, not to supply it.

    ``BuildRenderData`` splits a complex function into real and imaginary parts and writes
    ``is_complex`` beside the two of them (``ngsolve/webgui.py:303-305``), and the panel answers
    that flag with mode 5, the real part, where a plain two-component field gets mode 3, the
    length.  So a complex scene draws itself correctly with no keywords at all -- and a caller
    who hands over a real/imaginary pair that the payload cannot identify, or who would rather
    see the length of a complex one, is what the keyword is for.
    """
    assert gtdata.render_data_of("dims/complex_h1")["is_complex"] is True
    canvas, picture = draw("dims/complex_h1")
    inside = interior(canvas) & ~grown(picture.max(axis=2) < 90)
    assert_like_recording(canvas, picture, inside)

    overruled, _ = draw("dims/complex_h1", is_complex=False)
    assert not looks_like_recording(overruled, picture, inside), "mode 3 came out like mode 5"

    # ... and the other way round: a vector field the caller insists on calling complex
    vector, recorded = draw("dims/cf_vector")
    inside = interior(vector) & ~grown(recorded.max(axis=2) < 90)
    assert looks_like_recording(vector, recorded, inside)
    assert not looks_like_recording(draw("dims/cf_vector", is_complex=True)[0], recorded, inside)


def test_the_surface_capture_agrees_with_the_page_in_outline_only():
    """``surface_scalar`` is a flat square wearing a bump's clothes, so it settles nothing.

    ``docs/DIMS.md``, "Capture limitations": the gmsh import kept 2 coordinates per
    point and dropped ``z = 0.15 sin(pi x) sin(pi y)``, and the written ``.msh`` declared a
    physical group without any 1D elements, so the mesh has no boundary edges either.  Its
    outline -- rows, columns, how much ink lies between them -- is reproduced exactly.  A
    couple of percent of its surface is not, brighter in some places and darker in others, so
    it is no single shading factor and no other scene shows it.  A capture of a mesh that lost
    its third coordinate at import cannot say which of the two pictures is wrong, so this one
    is a parity target for its geometry only, and a 2-in-3 mesh built by netgen itself is owed.
    """
    canvas, picture = draw("dims/surface_scalar")
    ours = canvas.to_uint8()[..., :3].astype(int)
    ink = [mask.max(axis=2) < 250 for mask in (ours, picture)]
    # the overlays of the page sit outside the mesh, so compare on the rows and columns the
    # surface itself covers
    rows, cols = np.nonzero(ink[0])
    box = np.zeros(ink[0].shape, bool)
    box[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = True
    theirs = ink[1] & box
    assert (rows.min(), rows.max()) == tuple(np.nonzero(theirs.any(axis=1))[0][[0, -1]])
    assert (cols.min(), cols.max()) == tuple(np.nonzero(theirs.any(axis=0))[0][[0, -1]])
    assert abs(ink[0].sum() - theirs.sum()) < 0.01 * ink[0].sum()
    # ... and the gap this capture cannot settle stays on record rather than being masked
    difference = np.abs(picture - ours).max(axis=2)
    assert (difference[interior(canvas)] > 8).mean() > 0.01, (
        "the shading of surface_scalar matches after all: DIMS limitation 3 is settled, this "
        "scene belongs in SPACES with the rest, and the capture that mangled it should be redone")


def test_a_coefficient_function_without_a_mesh_never_reaches_a_picture():
    """``cf_nomesh`` is in DIMS as a failure, not a scene: there is nothing to compare to.

    ``Draw(cf)`` with nothing to sample on raises ``RuntimeError: Cannot draw
    CoefficientFunction without mesh`` inside ``ngsolve/webgui.py:119`` before it writes any
    render data, so no PNG and no page exist.  ngsplot takes the same view: the mesh is the
    caller's to hand over, and a bare coefficient function is the caller's error to receive.
    """
    assert not os.path.exists(os.path.join(gtdata.GT, "dims", "cf_nomesh.png"))
    assert not os.path.exists(os.path.join(gtdata.GT, "dims", "cf_nomesh.png.html"))
