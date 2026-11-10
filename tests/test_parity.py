"""The whole path against the picture the real renderer drew.

``tools/capture_gt.py`` renders each scene in Chromium with the real ``ngsolve.webgui``
and records, next to the screenshot, the render data it handed the page and the uniforms
the materials ended up with.  A scene here therefore needs neither ngsolve nor a
browser: the recorded render data goes through ngsplot's own geometry, camera and
rasterizer, and the two pictures are compared byte for byte.

Where they are compared, and why not everywhere:

* Only the surface.  The lines -- ``Wireframe`` and ``Edges`` -- are not painted yet, so
  a pixel is dropped when the recorded picture is dark there or within two pixels of a
  dark one, which is a drawn line and its antialiased fringe.  That is a statement about
  the ground truth, not a licence: ``square_no_wireframe`` has no line inside the measured
  rectangle at all and is compared whole.
* Only inside the surface's own bounding box, shrunk by 15 pixels, to stay off the
  silhouette, where the two implementations may legitimately choose opposite coverage for
  the same boundary pixel, and off the ``Edges`` outline that follows the mesh boundary.
* The screenshot is 900 by 600 and the drawing buffer 894 by 594 -- the page keeps its
  scrollbar gutter -- and the buffer sits at the top left of the picture.  The second test
  below measures that offset instead of assuming it.

What remains is float precision: a quarter of a percent of the pixels, all of them on a
colour band boundary, where the shader's float32 ``floor`` and this module's float64 one
pick neighbouring texels.  A median difference of zero grey levels over a hundred thousand
pixels is the point.
"""

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


def rasterize(scene):
    """The scene through ngsplot, as a canvas, next to its recorded picture."""
    render_data = gtdata.render_data_of(scene)
    uniforms = gtdata.uniforms_of(scene)
    width, height = gtdata.framebuffer_of(scene)
    ctrl, vectors = geometry.trig_points(render_data)
    canvas = raster.Canvas(width, height)
    raster.paint_surface(
        canvas, camera.Camera.from_render_data(render_data, width=width, height=height),
        geometry.order_of(render_data), ctrl,
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
    picture = gtdata.png_of(scene)[:height, :width, :3].astype(int)
    return canvas, picture


def interior(canvas, shape):
    """True inside the surface we painted, shrunk by :data:`INSET` pixels all round."""
    rows, cols = np.nonzero(canvas.color[..., 0] < 1)
    box = np.zeros(shape, dtype=bool)
    box[rows.min() + INSET:rows.max() - INSET, cols.min() + INSET:cols.max() - INSET] = True
    return box


def near_lines(picture):
    """True where the recorded picture shows a line, fringe included."""
    dark = picture.max(axis=2) < 90
    near = np.zeros_like(dark)
    near[FRINGE:-FRINGE, FRINGE:-FRINGE] = (
        sliding_window_view(dark, (2 * FRINGE + 1,) * 2).any(axis=(2, 3)))
    return near


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
