"""The whole path against the picture the real renderer drew.

``tools/capture_gt.py`` renders each scene in Chromium with the real ``ngsolve.webgui``
and records, next to the screenshot, the render data it handed the page and the uniforms
the materials ended up with.  A scene here therefore needs neither ngsolve nor a
browser: the recorded render data goes through ngsplot's own geometry, camera and
rasterizer, and the two pictures are compared byte for byte.

Where they are compared, and why not everywhere:

* The surface is compared inside its own bounding box, shrunk by 15 pixels, and away from
  any line: a line and the surface under it reach a pixel through different code, so the
  two are measured separately rather than added together.  A pixel of the silhouette is
  left out as well, since which side of the mesh's own edge a sample falls on is a question
  of half a pixel and the shader's float32 and this module's float64 answer it differently
  now and then.
* The lines are compared as ink, not as pixels.  A line is opaque black over a surface we
  have just drawn ourselves, so the ink in a pixel is one minus the fraction of that
  surface still visible in it -- measurable only in the channels bright enough in our own
  render to tell black from not-quite-black, which matters because the palette is
  saturated and a channel of it sits at zero.  Antialiasing here is 4x MSAA, four rotated
  samples per pixel, so ink arrives in quarters: the recorded border of ``square_scalar``
  is grey 128, never a gradient.
* The screenshot is 900 by 600 and the drawing buffer 894 by 594 -- the page keeps its
  scrollbar gutter -- and the buffer sits at the top left of the picture.  The second test
  below measures that offset instead of assuming it.

What remains of the surface comparison is float precision: a quarter of a percent of the
pixels, all of them on a colour band boundary, where the shader's float32 ``floor`` and
this module's float64 one pick neighbouring texels.  A median difference of zero grey
levels over a hundred thousand pixels is the point.  What remains of the lines is where
they land: the totals agree to a few percent and every pixel we ink is one the page inked,
but a wireframe line and the surface it lies on are drawn from separate payloads, and a
recorded 0.75 where we record 0.5 -- or the reverse -- is one sample of four on the wrong
side of a strip that is itself only a pixel wide.
"""

import functools

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

import gtdata
from ngsplot import camera, geometry, raster

# every scene in tests/gt that draws a surface; scenes without one are covered elsewhere
SCENES = ["square_scalar", "square_no_wireframe", "square_rotated", "square_deformed",
          "square_nc2", "square_nc32", "square_fixed_range", "mesh_only"]

INSET = 15       # pixels, to stay clear of the silhouette and of the Edges outline
FRINGE = 2       # pixels of antialiasing around a drawn line
LINE_EDGE = 0.15  # ink this thin is not a line, just the measure being imprecise

# every scene that draws a line through the middle of its surface; square_no_wireframe
# switches the wireframe off, which is what the last test of the file is about
WIRED = [scene for scene in SCENES if scene != "square_no_wireframe"]


def levels(canvas):
    """Our canvas as grey levels, unrounded: the units the recorded picture is in."""
    return np.clip(canvas.color[..., :3] * 255.0, 0.0, 255.0)


@functools.lru_cache(maxsize=None)
def render(scene, lines):
    """The scene through ngsplot: the canvas, the recorded picture, our surface alone.

    Cached, because a scene is rendered twice -- once for its own sake and once as the
    background the lines are measured against -- and nothing here writes to a canvas.  Pass
    ``lines=False`` for the surface on its own; the picture is always the whole recording.
    """
    render_data = gtdata.render_data_of(scene)
    uniforms = gtdata.uniforms_of(scene)
    width, height = gtdata.framebuffer_of(scene)
    view = camera.Camera.from_render_data(render_data, width=width, height=height)
    order = geometry.order_of(render_data)
    ctrl, vectors = geometry.trig_points(render_data)
    canvas = raster.Canvas(width, height)
    raster.paint_surface(
        canvas, view, order, ctrl,
        vmin=uniforms["colormap_min"], vmax=uniforms["colormap_max"],
        pal=np.array(uniforms["tex_colormap"]["data"], dtype=np.float64).reshape(-1, 4)[:, :3],
        n_segments=uniforms["n_segments"],
        function_mode=uniforms["function_mode"],
        dark_backside=uniforms["dark_backside"],
        vectors=vectors,
        # draw_surf reaches the page as null for a plain Draw, and only an explicit false
        # defines NO_FUNCTION_VALUES: null == false is false in JavaScript
        function_values=render_data.get("draw_surf") is not False,
        deformation=float(uniforms.get("deformation", 0.0)),
        mode=raster.deformation_mode(render_data["mesh_dim"], render_data["funcdim"]))
    surface = levels(canvas)          # taken before any line is painted: the background

    if lines:
        # a `deformation` of true in the render data means the geometry was sent flat and
        # lifted in the shader, so the lines have to be lifted by us the same way; a
        # CoefficientFunction deformation is already folded into the line vertices by
        # ngsolve itself (webgui.py:343) and must not be lifted twice
        lift = float(uniforms["deformation"]) if render_data["deformation"] is True else 0.0
        mode = raster.deformation_mode(render_data["mesh_dim"], render_data["funcdim"])
        objects = gtdata.settings_of(scene).get("Objects", {})
        if objects.get("Edges", True) and render_data.get("edges"):
            raster.paint_edges(
                canvas, view, order, geometry.edge_points(render_data)[0],
                n_segments=uniforms["n_segments"], line_thickness=uniforms["line_thickness"],
                deformation=lift, mode=mode)
        if objects.get("Wireframe", True) and render_data.get("Bezier_points"):
            raster.paint_wireframe(
                canvas, view, order, geometry.wireframe_points(render_data)[0],
                n_segments=uniforms["n_segments"], deformation=lift, mode=mode)
    picture = gtdata.png_of(scene)[:height, :width, :3].astype(int)
    return canvas, picture, surface


def rasterize(scene):
    """The scene's surface on its own, as a canvas next to its recorded picture."""
    canvas, picture, _ = render(scene, lines=False)
    return canvas, picture


def interior(canvas, shape):
    """True inside the surface we painted, shrunk by :data:`INSET` pixels all round."""
    rows, cols = np.nonzero(canvas.color[..., 0] < 1)
    box = np.zeros(shape, dtype=bool)
    box[rows.min() + INSET:rows.max() - INSET, cols.min() + INSET:cols.max() - INSET] = True
    return box


def grown(mask, fringe=FRINGE):
    """`mask` widened by `fringe` pixels in every direction."""
    window = 2 * fringe + 1
    out = np.zeros_like(mask)
    out[fringe:-fringe, fringe:-fringe] = (
        sliding_window_view(mask, (window, window)).any(axis=(2, 3)))
    return out


def near_lines(picture):
    """True where the recorded picture shows a line, fringe included."""
    return grown(picture.max(axis=2) < 90)


@pytest.mark.parametrize("scene", SCENES)
def test_the_surface_is_the_recorded_picture(scene):
    canvas, picture = rasterize(scene)
    ours = canvas.to_uint8()[..., :3].astype(int)
    difference = np.abs(picture - ours).max(axis=2)
    inside, lines = interior(canvas, picture.shape[:2]), near_lines(picture)
    surface = difference[inside & ~lines]
    assert surface.size > 0.5 * inside.sum(), "the line mask swallowed the picture"
    assert np.median(surface) == 0
    assert surface.mean() < 0.5         # under half a grey level over the surface
    assert np.count_nonzero(surface > 2) < 0.01 * surface.size


def test_the_drawing_buffer_starts_at_the_top_left_of_the_picture():
    """The page's 6-pixel gutter is on the right and bottom, not centred.

    Shifting the comparison by a single pixel costs an order of magnitude, because a
    colour band edge and the shading gradient both move; only the recorded offset is flat.
    """
    canvas, picture = rasterize("square_no_wireframe")
    ours = canvas.to_uint8()[..., :3].astype(int)
    inside = interior(canvas, picture.shape[:2])

    def mean_at(dy, dx):
        shifted = np.roll(np.roll(ours, dy, axis=0), dx, axis=1)
        return np.abs(picture - shifted).max(axis=2)[inside].mean()

    assert mean_at(0, 0) < 0.05
    for dy, dx in ((0, 1), (1, 0), (1, 1), (3, 3), (6, 6)):
        assert mean_at(dy, dx) > 10 * mean_at(0, 0)


def test_the_picture_holds_the_colour_as_the_shader_computed_it():
    """No linear-to-sRGB ramp sits between the fragment program and the PNG.

    three.js applies one in its ``colorspace_fragment`` include, which none of these
    hand-written programs contain, and the canvas is not colour-managed; the screenshot is
    the plain round and clamp of what calcLight returned.
    """
    canvas, picture = rasterize("square_no_wireframe")
    ours = np.clip(canvas.color[..., :3], 0.0, 1.0)
    ramp = np.where(ours <= 0.0031308, 12.92 * ours, 1.055 * ours ** (1 / 2.4) - 0.055)
    inside = interior(canvas, picture.shape[:2])
    direct = np.abs(picture - np.rint(ours * 255)).max(axis=2)[inside]
    encoded = np.abs(picture - np.rint(ramp * 255)).max(axis=2)[inside]
    assert direct.mean() < 0.05
    assert encoded.mean() > 40          # measured 68


# ----------------------------------------------------------------------------- the lines

def ink(pixels, surface):
    """How much of each pixel a black line drawn over ``surface`` covers.

    A line is opaque black, so what it leaves behind is the surface scaled towards zero,
    and one minus that scale is the coverage.  Only channels bright in our own render can
    carry the ratio: the palette is saturated, so a channel of the surface can sit at zero,
    and zero of anything is indistinguishable from black.  A pixel with nothing to measure
    -- page background, or nothing but zero channels -- reads zero.
    """
    bright = surface >= 40
    coverage = np.clip(1.0 - pixels / np.maximum(surface, 1.0), 0.0, 1.0)
    return (np.where(bright, coverage, 0.0).sum(axis=2) / np.maximum(bright.sum(axis=2), 1))


def central_box(canvas):
    """True over the middle half of the surface we painted, rows and then columns.

    The wireframe lives there.  The outline does not, and at the outline the line and the
    surface fight over the same samples, which is the phase question this module declines
    to settle pixel by pixel.
    """
    painted = np.isfinite(canvas.depths).any(axis=0)
    rows, cols = np.nonzero(painted)
    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1
    box = np.zeros(painted.shape, dtype=bool)
    box[top + (bottom - top) // 4:bottom - (bottom - top) // 4,
        left + (right - left) // 4:right - (right - left) // 4] = True
    return box


@pytest.mark.parametrize("scene", WIRED)
def test_the_lines_are_the_recorded_lines(scene):
    """The right amount of ink, in the right places, wherever a line crosses the surface."""
    bare, picture, surface = render(scene, lines=False)
    canvas, _, _ = render(scene, lines=True)
    box = central_box(bare)
    recorded = ink(picture.astype(float), surface) * box
    ours = ink(levels(canvas), surface) * box
    assert recorded.sum() > 500, "the box holds no line to compare"
    assert ours.sum() / recorded.sum() == pytest.approx(1.0, abs=0.1)
    # the amount is only worth stating together with the placement: two pixels is the
    # narrowest tolerance that forgives one sample of four landing on the wrong side of a
    # strip which is itself one pixel wide
    assert (ours * grown(recorded > LINE_EDGE)).sum() >= 0.99 * ours.sum()
    assert (recorded * grown(ours > LINE_EDGE)).sum() >= 0.95 * recorded.sum()


def test_a_slider_of_five_paints_two_and_a_half_pixels():
    """The known answer for the ``Edges`` outline: half a pixel of 128, then two of black.

    ``line_thickness`` reaches the material as the slider over the height of the drawing
    buffer (scene.ts:776) and the shader widens the strip by half of that in clip space
    (shader/splines.vert:97-103), which on the screen is a line half the slider wide: five
    here, so 2.5 pixels.  The left edge of the unit square lands at column 254.67, which
    gives 253 half of it and 254 and 255 all of it -- and the last two are pixels of the
    surface, where a coplanar line only wins the depth test because the surface pushed
    itself back by one depth step for exactly this (three/mesh.ts:65-67).
    """
    scene = "square_scalar"
    _, picture, surface = render(scene, lines=False)
    canvas, _, _ = render(scene, lines=True)
    height = gtdata.framebuffer_of(scene)[1]
    assert gtdata.uniforms_of(scene)["line_thickness"] * height == pytest.approx(5.0)
    assert [tuple(p) for p in picture[300, 253:256]] == [(128,) * 3, (0,) * 3, (0,) * 3]
    recorded, ours = ink(picture.astype(float), surface), ink(levels(canvas), surface)
    assert ours[300, 253] == pytest.approx(0.5, abs=0.01)      # 127.5 of a white 255
    rows, columns = slice(110, 481), slice(253, 256)           # the whole border, bar none
    assert np.abs(recorded[rows, columns] - ours[rows, columns]).max() < 0.01
    assert np.median(recorded[rows, columns].sum(axis=1)) == pytest.approx(2.5, abs=0.02)
    assert np.median(ours[rows, columns].sum(axis=1)) == pytest.approx(2.5, abs=0.02)


def test_a_slider_of_one_paints_half_a_pixel():
    """The same border with the slider at its minimum: one pixel, and it is exactly 128.

    ``square_no_wireframe`` was captured with ``line_thickness`` at 1, a 594th of the
    drawing buffer, so the strip is half a pixel wide.  It falls across the two middle
    samples of column 254 and across nothing else, and there the mesh has not reached a
    single sample of that pixel, so what it covers is the page: 128, 128, 128, to the last
    bit, against our 127.5.  Read as ink the same pixel says 0.41 rather than 0.5, because
    our own surface does reach a quarter of it and the measure divides by what it painted.
    """
    scene = "square_no_wireframe"
    _, picture, surface = render(scene, lines=False)
    canvas, _, _ = render(scene, lines=True)
    height = gtdata.framebuffer_of(scene)[1]
    assert gtdata.uniforms_of(scene)["line_thickness"] * height == pytest.approx(1.0)
    assert tuple(picture[300, 254]) == (128, 128, 128)
    assert np.allclose(levels(canvas)[300, 254], 127.5)
    recorded, ours = ink(picture.astype(float), surface), ink(levels(canvas), surface)
    assert ours[300, 253] == ours[300, 255] == 0.0             # one pixel, no fringe
    assert recorded[300, 253] < 0.01 and recorded[300, 255] < 0.01
    rows, columns = slice(110, 481), slice(252, 258)
    assert np.abs(recorded[rows, columns] - ours[rows, columns]).max() < 0.01
    assert np.median(recorded[rows, columns].sum(axis=1)) == pytest.approx(0.41, abs=0.01)


def test_no_ink_sits_where_no_line_is_drawn():
    """What the measure calls ink in a box with no line in it is the measure's own noise.

    ``square_no_wireframe`` draws nothing but the outline, which is outside the box, so
    this is the floor of the comparison above: two percent of what the same square records
    with its wireframe on, in the recording and in ours alike.  It is why that test asks
    for a tenth rather than for a byte-exact total.
    """
    def in_box(scene):
        bare, picture, surface = render(scene, lines=False)
        canvas, _, _ = render(scene, lines=True)
        box = central_box(bare)
        return (ink(picture.astype(float), surface)[box].sum(),
                ink(levels(canvas), surface)[box].sum())

    loud_recorded, loud_ours = in_box("square_scalar")
    quiet_recorded, quiet_ours = in_box("square_no_wireframe")
    assert loud_recorded > 500 and loud_ours > 500
    assert quiet_recorded < 0.03 * loud_recorded      # measured 28 against 1482
    assert quiet_ours < 0.03 * loud_ours              # we leak no ink of our own
