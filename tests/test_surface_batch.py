"""The surface painter as one batched stack, against the same surface drawn one
triangle at a time.

``paint_surface`` walks ``ne`` elements and ``n_segments ** 2`` sub-triangles each, which on
a recorded picture is tens of thousands of calls into ``_paint_triangle``, every one of them
setting up its own bounding box and sweeping its own rows.  The batched painter hands the
whole sheet to :func:`ngsplot.raster._paint_stack` at once, which sweeps every triangle over
windows of its own box, all the windows of one shape in one array.  What may not change is
the picture, and the picture here is decided by three things the batch has to get exactly right:

* the draw order.  An instanced ``THREE.Mesh`` draws instance by instance, and within an
  instance in the order ``subtriangles`` returns (:mod:`ngsplot.bezier`), so the stack takes
  its primitives element-major and sub-triangle-minor and resolves its ties by it;
* the varying.  ``value_`` carries the value slot and, for a vector function, its two
  components, and ``vPos_``/``vNormal_`` are the model-view point and the normal through the
  matching normal matrix (utils.h:422).  The batch interpolates them only where a triangle
  has won a sample, at the pixel centre, which is where the fragment program would have run;
* the polygon offset, which the surface material alone asks for (three/mesh.ts:65-67) and
  which is what lets the wireframe drawn inside the surface pass the depth test.

The oracle is the loop, written out below exactly as the painter used to be written, on the
same ``lattice`` and the same projection.  It is compared sample plane by sample plane, so a
tie broken the other way shows up as a colour and not as a shrug.
"""

import numpy as np
import pytest

from ngsplot import bezier, camera, raster

WIDTH, HEIGHT = 240, 180

SHEET = 4 * 4 * 2           # rows of mesh(): the sub-triangles of a 4x4 sheet, then N_DUP
N_DUP = 8                   # copies of the first of them, then one row behind the eye
BEHIND = SHEET + N_DUP      # that row
POISONED = SHEET - 1        # the sheet row with a corner of no value


def mesh(n=4, span=0.8, flip=4):
    """``n`` by ``n`` quads over ``z = 0.4 * (x*x - y*y)``, as order-1 elements.

    The saddle is the payload a hand-built mesh should have: it curves away from a tilted
    camera at all four corners so the sub-triangles compete in depth across the ridge, its
    value is the height, so the colour bands follow the lighting instead of hiding it, and one
    element in ``flip`` is wound the other way, which is what puts back faces on the sheet at
    all -- a saddle seen from one side is one orientation throughout.

    Then eight rows of the sheet come back at the end of ``ctrl``: same three corners, same
    order, a value of their own.  They are there because a depth *test*, as opposed to a depth
    comparison, only has something to say when two primitives tie, and nothing else in the
    sheet ties -- two sub-triangles of a mesh either have different depths or do not overlap.
    A coincident row ties to the last bit, since it is the same arithmetic on the same numbers,
    so on its pixels the depth test alone decides whether the copy drawn later wins
    (``LessEqualDepth``, the three.js default) or the one drawn first keeps them (``Less``).

    The last row is a triangle behind the eye, and it is placed so that what falls on the
    canvas when ``w <= 0`` is *not* thrown away is a patch of empty background well away from
    the sheet, where its mirrored projection -- the corners divided by a negative ``w`` put it
    on the far side of the frame, and that is the point -- would be seen.  Two more rows
    poison the mesh: ``POISONED`` has a corner whose value is NaN, which is the other half of
    the usability filter the loop applies before it rasterizes anything.
    """
    grid = np.linspace(-span, span, n + 1)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    z = 0.4 * (x * x - y * y)
    corner = [(i, j) for i in range(n + 1) for j in range(n + 1)]
    where = {c: k for k, c in enumerate(corner)}
    tris = []
    for i in range(n):
        for j in range(n):
            tris.append([where[(i, j)], where[(i + 1, j)], where[(i + 1, j + 1)]])
            tris.append([where[(i, j)], where[(i + 1, j + 1)], where[(i, j + 1)]])
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    values = z.ravel()
    tris = np.asarray(tris)
    sheet = np.concatenate([pts[tris], values[tris][:, :, None]], axis=-1)
    # the eye of camera_for is at (0, 0, 3) and w only turns negative past it; this triangle
    # sits just behind, far enough off the view axis that its negative-w projection lands back
    # inside the window instead of flung outside it, where a dropped filter would go unseen
    behind = np.array([[0.0, 0.0, 3.6], [0.4, 0.0, 3.6], [0.4, 0.4, 3.6]])
    behind[:, :2] += [-1.9, -2.25]
    ctrl = np.zeros((len(sheet) + N_DUP + 1, 3, 4))
    ctrl[:len(sheet)] = sheet
    ctrl[len(sheet):len(sheet) + N_DUP, :, :3] = sheet[:N_DUP, :, :3]
    ctrl[len(sheet):len(sheet) + N_DUP, :, 3] = 0.35
    ctrl[BEHIND, :, :3] = behind
    ctrl[BEHIND, :, 3] = 0.3
    ctrl[POISONED, 1, 3] = np.nan                    # one corner of no value
    if flip:                                         # every flip-th row wound the other way
        at = np.arange(0, len(sheet), flip)
        turned = ctrl[at]
        ctrl[at] = turned[:, [0, 2, 1]]
    return ctrl


def vectors_for(ctrl, mode="xy"):
    """Two vector components per control point, varying so that mode 4 has something to show."""
    t = ctrl[:, :, 0] - ctrl[:, :, 1]
    return np.stack([t, 0.5 * t + 0.25 * ctrl[:, :, 2]], axis=-1)


def canvas_for(**kwargs):
    return raster.Canvas(WIDTH, HEIGHT, **kwargs)


def camera_for(**kwargs):
    return camera.Camera([0.0, 0.0, 0.0], 1.0, width=WIDTH, height=HEIGHT, **kwargs)


def reference_surface(canvas, cam, order, ctrl, *, vmin, vmax, pal=None, n_segments=raster.N_SEGMENTS,
                      vectors=None, deformation=0.0, mode="none", function_mode=0.0,
                      complex_scale=(1.0, 0.0), vertex_normals=None, function_values=True,
                      dark_backside=True, skip_face_check=False, less_equal=True):
    """``paint_surface`` as the two loops, kept as the oracle for the batched one.

    Every line of this is a line the painter had before it was batched, including the order
    of the loops and the point at which the usability filter is consulted.
    """
    function_mode = float(function_mode)
    order = int(order)
    pos, value, nrm = raster.lattice(order, ctrl, vectors=vectors, deformation=deformation,
                                     mode=mode, function_mode=function_mode,
                                     complex_scale=complex_scale,
                                     vertex_normals=vertex_normals, n_segments=n_segments)
    mv = cam.model_view
    ne = value.shape[0]
    points = pos.reshape(-1, 3)
    px, py, depth, w = camera.to_pixels(
        camera.clip_points(points, cam.model_view_projection), canvas.width, canvas.height)
    p_view = camera.transform_points(points, mv)
    n_view = np.einsum("ij,...j->...i", np.linalg.inv(mv[:3, :3]).T, nrm.reshape(-1, 3))
    shape = (ne,) + value.shape[1:-1]
    px, py, depth, w = (a.reshape(shape) for a in (px, py, depth, w))
    p_view, n_view = (a.reshape(value.shape) for a in (p_view, n_view))
    usable = (np.isfinite(px) & np.isfinite(py) & np.isfinite(depth) & np.isfinite(w) & (w > 0.0)
              & ~np.any(np.isnan(value), axis=-1))
    for e in range(ne):
        for tri in bezier.subtriangles(n_segments):
            if not np.all(usable[e, tri]):
                continue
            raster._paint_triangle(
                canvas, px[e, tri], py[e, tri], w[e, tri], depth[e, tri], p_view[e, tri],
                n_view[e, tri], value[e, tri], vmin=vmin, vmax=vmax, pal=pal,
                function_mode=function_mode, complex_scale=complex_scale,
                function_values=function_values, dark_backside=dark_backside,
                skip_face_check=skip_face_check, less_equal=less_equal)
    return canvas


CASES = dict(
    scalar=dict(vmin=-0.4, vmax=0.4),
    grey=dict(vmin=-0.4, vmax=0.4, function_values=False),
    flat_shading=dict(vmin=-0.4, vmax=0.4, dark_backside=False),
    unlit=dict(vmin=-0.4, vmax=0.4, function_mode=4.0, vectors=True),
    depth_32=dict(vmin=-0.4, vmax=0.4, less_equal=False),
    straight=dict(vmin=-0.4, vmax=0.4, skip_face_check=True),
)


def payload(name, **extra):
    ctrl = mesh()
    kw = dict(CASES[name], **extra)
    if kw.pop("vectors", False):
        kw["vectors"] = vectors_for(ctrl)
    return ctrl, kw


@pytest.mark.parametrize("samples", [raster.MSAA_4X, 1])
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_batched_surface_is_the_surface_of_the_loops(samples, case):
    ctrl, kw = payload(case)
    cam = camera_for(elev=25.0, azim=15.0)

    batched = raster.paint_surface(canvas_for(samples=samples), cam, 1, ctrl, n_segments=2, **kw)
    loops = reference_surface(canvas_for(samples=samples), cam, 1, ctrl, n_segments=2, **kw)

    ink = np.any(batched.color[..., :3] != 1.0, axis=-1)
    assert ink.sum() > 0.15 * ink.size, "a picture this small should be mostly surface"
    np.testing.assert_array_equal(batched.planes, loops.planes)
    np.testing.assert_array_equal(batched.depths, loops.depths)
    np.testing.assert_array_equal(batched.color, loops.color)


def test_a_stack_of_few_coarse_triangles_is_swept_one_at_a_time(monkeypatch):
    """Which of the two sweeps runs is a measured choice, and both draw the same picture.

    The batch lights a pixel by gathering the corners of the primitive that holds it, row by
    row; the per-primitive sweep lights the box of a primitive by broadcasting its corners over
    it, and pays a dozen numpy calls per primitive for that.  The boxes of a mesh that fills the
    frame add up to about one canvas however coarse the mesh is, so the first cost is set by the
    picture and the second by how many primitives it is cut into: a coarse mesh is the case the
    loop wins -- on the recorded ``mesh_only``, by a fifth -- and the loop is what the painter used
    to be.  At or above :data:`ngsplot.raster._STACK_LOOP_AREA` pixels of canvas per primitive, the
    batch runs.

    Both branches then have to give the loop's planes exactly, which is the point of keeping it:
    the choice may move arithmetic around, it may not decide a pixel.
    """
    kw = dict(vmin=-0.4, vmax=0.4)
    cam = camera_for(elev=25.0, azim=15.0)
    sheet = mesh()
    # the same saddle, the same boxes, a sixteenth of its elements: what changes is how many
    # primitives there are to pay a call for, which is what the choice is made on
    stacks = dict(coarse=sheet[:2], fine=sheet)

    # the loops first: reference_surface calls _paint_triangle, which is what gets counted
    wanted = {name: reference_surface(canvas_for(), cam, 1, ctrl, n_segments=2, **kw)
              for name, ctrl in stacks.items()}

    calls = []
    real = raster._paint_triangle
    monkeypatch.setattr(raster, "_paint_triangle",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    for name, ctrl in stacks.items():
        calls.clear()
        got = raster.paint_surface(canvas_for(), cam, 1, ctrl, n_segments=2, **kw)
        coarse = len(calls) > 0
        assert coarse == (name == "coarse"), f"the {name} stack took the wrong sweep"
        assert np.any(wanted[name].color[..., :3] != 1.0), "nothing drawn, nothing compared"
        np.testing.assert_array_equal(got.planes, wanted[name].planes)
        np.testing.assert_array_equal(got.depths, wanted[name].depths)

    # where the choice was measured to turn: at 4500 pixels of canvas per primitive the batch is
    # already ahead, at 5625 the loop still is.  A threshold outside that gap is a guess.
    assert 4500 < raster._STACK_LOOP_AREA < 5625


def test_the_sheet_shows_both_of_its_faces():
    """The payload has a back face in it, or the claims about back faces test nothing.

    Winding an element the other way moves no corner and changes no value.  The only thing
    that can then differ on the canvas is how a face seen from behind is treated: the negated
    normal of ``gl_FrontFacing`` and the dimming that goes with it.  Two identical pictures
    would mean no pixel ever took that path, and every equality above would have been an
    equality about a front face.
    """
    kw = dict(vmin=-0.4, vmax=0.4)
    cam = camera_for(elev=25.0, azim=15.0)
    turned = raster.paint_surface(canvas_for(), cam, 1, mesh(), n_segments=2, **kw)
    even = raster.paint_surface(canvas_for(), cam, 1, mesh(flip=0), n_segments=2, **kw)
    assert np.any(turned.color[..., :3] != even.color[..., :3])


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_surface_reaches_the_edges_it_should(case):
    """A surface that is drawn twice from the same camera is drawn the same twice."""
    ctrl, kw = payload(case)
    cam = camera_for(elev=25.0, azim=15.0)
    once = raster.paint_surface(canvas_for(), cam, 1, ctrl, n_segments=2, **kw)
    again = raster.paint_surface(canvas_for(), cam, 1, ctrl, n_segments=2, **kw)
    np.testing.assert_array_equal(once.color, again.color)
    np.testing.assert_array_equal(once.depths, again.depths)


def test_what_is_behind_the_camera_and_what_is_no_value_reaches_no_canvas():
    """The usability filter is a filter in the batch too, not a hint.

    A corner with ``w <= 0``, or a NaN in the value, takes its sub-triangle out of the draw in
    the loop, before anything is rasterized.  A batch that swept it anyway would sweep an
    inverted triangle, painting canvas a colour it has no right to, so a triangle lifted behind
    the eye has to leave the picture exactly as it was, and so does an element whose value is
    NaN at every corner.  An equality against nothing is worth nothing, so both, made live,
    have to change the sheet -- and the behind-the-eye row has to be behind it for the right
    reason, which is that its corners fold back to the *inside* of the window: a triangle that
    merely lands off the edge would be thrown away by the window binning alone, and the filter
    would never be asked.
    """
    ctrl = mesh()
    kw = dict(vmin=-0.4, vmax=0.4)
    cam = camera_for(elev=25.0, azim=15.0)
    without = np.delete(ctrl, BEHIND, axis=0)

    sheet = raster.paint_surface(canvas_for(), cam, 1, without, n_segments=2, **kw)
    whole = raster.paint_surface(canvas_for(), cam, 1, ctrl, n_segments=2, **kw)
    assert np.any(sheet.color[..., :3] != 1.0)
    np.testing.assert_array_equal(whole.color, sheet.color)
    np.testing.assert_array_equal(whole.depths, sheet.depths)

    px, py, _, w = raster._project(canvas_for(), cam, ctrl[BEHIND, :, :3])
    assert np.all(w < 0.0), "the row is meant to be behind the eye, not merely off the edge"
    assert np.all((px > 0) & (px < WIDTH) & (py > 0) & (py < HEIGHT)), (px, py)

    infront = ctrl.copy()                               # the row, back in front of the eye
    infront[BEHIND, :, :2] *= 0.2                       # far enough off axis to miss the sheet
    infront[BEHIND, :, 2] = 0.2
    assert np.any(raster.paint_surface(canvas_for(), cam, 1, infront, n_segments=2,
                                       **kw).color[..., :3] != sheet.color[..., :3])

    # an element with no value at any corner has no usable sub-triangle at all, which is the
    # same sheet with that element taken out of it -- and its value filled in is not
    poisoned = np.delete(ctrl, POISONED, axis=0)
    blank = ctrl.copy()
    blank[POISONED, :, 3] = np.nan
    np.testing.assert_array_equal(
        raster.paint_surface(canvas_for(), cam, 1, blank, n_segments=2, **kw).color,
        raster.paint_surface(canvas_for(), cam, 1, poisoned, n_segments=2, **kw).color)
    filled = blank.copy()
    filled[POISONED, :, 3] = 0.2
    assert np.any(raster.paint_surface(canvas_for(), cam, 1, filled, n_segments=2,
                                       **kw).color[..., :3] != sheet.color[..., :3])


def test_a_tie_between_two_copies_of_a_triangle_goes_to_the_depth_test():
    """Two coincident rows tie, and only ``LessEqualDepth`` says which of them is seen.

    Every other pair of sub-triangles in the sheet is decided by depth alone, so nothing else
    here could tell the two depth functions apart.  The duplicated rows of ``mesh`` sit on the
    corners of the rows before them and differ from them only in value, so at their samples the
    two depths are the same number and the test is all that is left: ``LessEqualDepth``
    (three/mesh.ts:57), which is what ``less_equal=True`` asks for, lets the copy drawn later
    take the pixel, and ``Less``, which is what the ``depth_32`` recording used, leaves the
    first one holding it.  A batch that resolved its winners in one ``lexsort`` key whatever
    the caller asked for would draw the same picture both ways, which is what this rules out.
    """
    ctrl = mesh()
    kw = dict(vmin=-0.4, vmax=0.4)
    cam = camera_for(elev=25.0, azim=15.0)

    for less_equal in (True, False):
        canvas = raster.paint_surface(canvas_for(), cam, 1, ctrl, n_segments=2,
                                      less_equal=less_equal, **kw)
        oracle = reference_surface(canvas_for(), cam, 1, ctrl, n_segments=2,
                                   less_equal=less_equal, **kw)
        np.testing.assert_array_equal(canvas.planes, oracle.planes)
        np.testing.assert_array_equal(canvas.depths, oracle.depths)
        np.testing.assert_array_equal(canvas.color, oracle.color)

    later = raster.paint_surface(canvas_for(), cam, 1, ctrl, n_segments=2, less_equal=True, **kw)
    earlier = raster.paint_surface(canvas_for(), cam, 1, ctrl, n_segments=2, less_equal=False,
                                   **kw)
    won = np.any(later.color[..., :3] != earlier.color[..., :3], axis=-1)
    assert won.sum() > 0.01 * won.size, "the two depth tests have to disagree somewhere"
    # and they disagree exactly where the copies are, which is the only place they can
    coincident = np.any(later.depths != earlier.depths, axis=0)
    assert not np.any(coincident)



