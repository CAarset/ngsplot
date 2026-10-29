"""The knobs: render data in, picture out, defaults exactly as the webgui has them.

Everything the live viewer keeps in its control panel -- the colour range, the number
of bands, which of the three kinds of geometry is drawn at all and how thick its lines
are, the subdivision of every curve -- is a keyword here, and every keyword that is left
out takes the value ``three/gui.ts`` gives the panel on a fresh draw.  A payload straight
from ``ngsolve.webgui`` therefore paints the picture ``ngsolve.webgui`` would have
painted, ``Draw`` included in the payload its own ``gui_settings``, be it in the sections
a panel snapshot has or in the flat names ``Draw`` itself writes.
See docs/SPEC.md, "Colour", and docs/DIMS.md.

The knobs deliberately mirror matplotlib's names: ``vmin``/``vmax`` switch off the
autoscaled range as passing them does to ``imshow``, ``cmap`` takes any matplotlib
colormap, and ``ncolors`` is the webgui's own band count.
"""

import numpy as np

from . import camera, colormap, geometry, raster

__all__ = ["DEFAULTS", "settings_of", "view_range", "colormap_of", "frame", "paint"]

#: ``three/gui.ts`` settings of a fresh panel: the values a first ``Draw`` renders with.
DEFAULTS = {
    "ncolors": 8,               # Colormap.ncolors
    "autoscale": True,          # Colormap.autoscale
    "line_width": 5.0,          # Misc.line_thickness, in gui pixels
    "subdivision": 5,           # Misc.subdivision
    "fast_draw": False,         # Misc.fast_draw
    "dark_backside": True,      # the shader's dark_backside uniform
    "surface": True,            # Objects.Surface
    "wireframe": True,          # Objects.Wireframe
    "edges": True,              # Objects.Edges
    "vectors": False,           # Vectors.show
}

#: ``gui.ts:151-165``: a payload may name a setting flat, at the top of ``gui_settings``.
_FLAT_NAMES = {
    "autoscale": ("Colormap", "autoscale"),
    "colormap_min": ("Colormap", "min"),
    "colormap_max": ("Colormap", "max"),
    "colormap_ncolors": ("Colormap", "ncolors"),
    "subdivision": ("Misc", "subdivision"),
    "line_thickness": ("Misc", "line_thickness"),
    "edges": ("Objects", "Edges"),
    "mesh": ("Objects", "Wireframe"),
    "elements": ("Objects", "Surface"),
}


def _section(settings, name):
    """One section of a payload's ``gui_settings``, which may be absent or null."""
    section = settings.get(name) or {}
    return section if isinstance(section, dict) else {}


def _sections_of(settings):
    """The three sections a payload's settings resolve to, flat names folded in.

    A payload either names a setting inside its section -- the shape a panel snapshot has,
    and the only one a live panel can produce -- or flat, which is what ``Draw`` writes
    into ``settings=``.  The flat name wins, because the page folds it in afterwards
    (``gui.ts:146-175``), and ``Misc.reduce_subdivision`` is the former name of the
    ``fast_draw`` checkbox.
    """
    folded = {name: dict(_section(settings, name)) for name in ("Colormap", "Misc", "Objects")}
    for flat, (section, name) in _FLAT_NAMES.items():
        if settings.get(flat) is not None:
            folded[section][name] = settings[flat]
    if folded["Misc"].get("reduce_subdivision") is not None:
        folded["Misc"]["fast_draw"] = folded["Misc"]["reduce_subdivision"]
    return folded


def settings_of(render_data, *, is_complex=None):
    """The effective settings of ``render_data``: the panel defaults, then its own.

    A complex function arrives split into its real and imaginary parts with ``is_complex``
    written beside them (``ngsolve/webgui.py:303-305``), and that word is what makes it draw
    the real part rather than the length of the pair and keep its geometry undeformed.  It is
    taken from the payload; the keyword is for a pair that arrived without one -- a complex
    coefficient function assembled by hand, say -- and for overruling a payload that has it.

    A payload records the panel in ``gui_settings``, either in the sections the lil-gui
    status object has (``Colormap``, ``Misc``, ``Objects``, ``camera``) or by the flat
    names ``Draw`` writes, and leaves out whatever was left at its default.  Two settings
    are not panel values at all but follow from the payload: the subdivision drops to 1
    when no element is curved
    (``gui.ts:229-231``) and the deformation defaults to ``deformation_scale`` when the
    payload says the geometry was sent flat (``gui.ts:582-587``).
    """
    settings = render_data.get("gui_settings") or {}
    colormap_section, misc, objects = (_sections_of(settings)[name]
                                       for name in ("Colormap", "Misc", "Objects"))

    out = dict(DEFAULTS)
    out["ncolors"] = int(colormap_section.get("ncolors", out["ncolors"]))
    out["autoscale"] = bool(colormap_section.get("autoscale", out["autoscale"]))
    out["line_width"] = float(misc.get("line_thickness", out["line_width"]))
    out["subdivision"] = int(misc.get("subdivision", out["subdivision"]))
    out["fast_draw"] = bool(misc.get("fast_draw", out["fast_draw"]))
    out["dark_backside"] = bool(misc.get("dark_backside", out["dark_backside"]))
    out["surface"] = bool(objects.get("Surface", out["surface"]))
    out["wireframe"] = bool(objects.get("Wireframe", out["wireframe"]))
    out["edges"] = bool(objects.get("Edges", out["edges"]))
    out["vectors"] = bool(objects.get("Vectors", out["vectors"]))

    # Nothing curved to subdivide, and the page asks for one segment -- but into the
    # defaults, before the payload's settings (gui.ts:229-231), so a payload that named a
    # subdivision of its own keeps it.  fast_draw shortens it the other way, always.
    order = max(int(render_data.get("order2d", 1)), int(render_data.get("order3d", 1)))
    if out["fast_draw"] or (order <= 1 and "subdivision" not in misc):
        out["subdivision"] = 1

    # ``Draw(autoscale=False)`` writes a flag of its own at the top of the payload
    # (``netgen/webgui.py:548``), and that, not the panel's checkbox alone, is what lets the
    # panel re-read the ends of the function on a render (``scene.ts:497``).
    out["autoscale"] = out["autoscale"] and bool(render_data.get("autoscale", True))
    scale = float(render_data.get("deformation_scale") or 1.0)
    out["deformation"] = scale if render_data.get("deformation") else 0.0

    # `eval` picks which part of a value to draw: x, y, z, length, real, imaginary.  A complex
    # function draws its real part and a vector its length, both as the panel does
    # (gui.ts:529-571), and for those two the panel writes that mode over whatever the payload
    # named in its Colormap section.  A scalar lets the mode it named stand, and an `eval` at
    # the top of the payload is written last of all (gui.ts:568-571).  A flat `eval` beside the
    # sections is not one of the names updateSettings folds, so nothing reads it.
    funcdim = int(render_data.get("funcdim", 1))
    if is_complex is None:
        is_complex = render_data.get("is_complex", False)
    is_complex = bool(is_complex)
    mode = 5.0 if is_complex else (3.0 if funcdim in (2, 3) else 0.0)
    if not is_complex and funcdim not in (2, 3):
        mode = colormap_section.get("eval", mode)
    mode = render_data.get("eval", mode)
    out["function_mode"] = float(mode)

    phase = float(_section(settings, "Complex").get("phase", 0.0))
    out["complex_scale"] = (float(np.cos(-phase)), float(np.sin(-phase)))
    out["mesh_only"] = int(render_data.get("funcdim", 1)) == 0
    out["is_complex"] = is_complex
    return out


def view_range(render_data, vmin=None, vmax=None, autoscale=None, function_mode=0.0):
    """The value range the colours are spread over, as the panel computes it.

    An autoscaled draw takes ``funcmin`` and ``funcmax`` of the payload, which ngsolve
    fills from the samples it sent (``ngsolve/webgui.py:475``); a mesh with no function
    on it is instead indexed by region, ``-0.5`` to ``mesh_regions_2d - 0.5``
    (``gui.ts:328-334``).  A vector drawn as a length runs ``0`` to the larger absolute
    end (``gui.ts:372-379``).  Giving ``vmin`` or ``vmax`` switches that end of the
    autoscaling off for that draw, which is what the panel's own minimum and maximum
    fields do.

    A payload may carry no range at all: the sentinel pair ``(DBL_MAX, DBL_MIN)`` of an
    accumulator that never ran, or equal ends from a constant field.  Only the first is
    a range that cannot be drawn -- the second is what a constant field really is, and
    the renderer paints it as one flat colour, so this does too.
    """
    settings = render_data.get("gui_settings") or {}
    panel = _sections_of(settings)["Colormap"]
    if autoscale is None:
        autoscale = bool(panel.get("autoscale", True)) and bool(render_data.get("autoscale", True))

    vmin_panel, vmax_panel = panel.get("min"), panel.get("max")
    if vmin_panel is not None and vmax_panel is not None:
        low, high = float(vmin_panel), float(vmax_panel)
    elif int(render_data.get("funcdim", 1)) == 0:
        regions = int(render_data.get("mesh_regions_2d") or 1)
        low, high = -0.5, regions - 0.5
    else:
        low = render_data.get("funcmin")
        high = render_data.get("funcmax")
        low = 0.0 if low is None else float(low)
        high = 1.0 if high is None else float(high)

    # updateColormapToAutoscale (gui.ts:366-390): the panel's ends follow the function
    # while autoscale is on and there is a function to follow, and a drawn length never
    # goes below zero.
    if autoscale and (render_data.get("draw_surf") or render_data.get("draw_vol")):
        if render_data.get("funcmin") is not None and render_data.get("funcmax") is not None:
            low, high = float(render_data["funcmin"]), float(render_data["funcmax"])
        if not high > low:
            low, high = 0.0, 1.0
        if float(function_mode) == 3.0:
            low, high = 0.0, max(abs(low), abs(high))
    if vmin is not None:
        low = float(vmin)
    if vmax is not None:
        high = float(vmax)
    if not high > low:
        # nothing left to spread over: matplotlib's answer to the same problem
        low, high = 0.0, 1.0
    return low, high


def colormap_of(cmap=None, ncolors=colormap.NCOLORS, mesh_only=False):
    """The ``(ncolors, 3)`` texture of a draw, the webgui ramp unless ``cmap`` says otherwise.

    ``mesh_only`` draws are not coloured by a function at all and get the single green
    texel the viewer hands a mesh (``colormap.ts:127-131``), whatever band count came with
    it.  ``cmap`` is a matplotlib colormap or the name of one, sampled at the centre of
    each band so that a ``ListedColormap`` of ``ncolors`` colours comes through unchanged
    -- which is also how :func:`ngsplot.colormap.to_matplotlib` turns the webgui ramp back
    into one.  The name ``"ngsolve"`` is the webgui ramp itself, cut to the band count
    asked for; matplotlib does not know it, and a registered copy of it would know one
    band count only.
    """
    if mesh_only:
        return np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    ncolors = int(ncolors)
    if ncolors < 2:
        # the panel's own slider starts at 2, and below it no value can reach a band
        raise ValueError("ncolors must be at least 2")
    if cmap is None or cmap == "ngsolve":
        return colormap.palette(ncolors).astype(np.float64)
    if isinstance(cmap, str):
        from matplotlib import colormaps

        cmap = colormaps[cmap]
    if not callable(cmap):
        raise TypeError(f"cmap must be a colormap or a name, not {type(cmap).__name__}")
    centres = (np.arange(ncolors) + 0.5) / ncolors
    return np.asarray(cmap(centres), dtype=np.float64)[..., :3]


def _placing(render_data, settings, n_segments, deformation, function_mode, complex_scale):
    """What decides where a picture lies, as :func:`paint` resolves it from its keywords and the
    payload: ``(order, n, lift, flat, mode, function_mode, complex_scale)`` -- the order of the
    geometry, its subdivision, how far the surface and the lines are lifted, which deformation
    the shaders compile, and the part of a value that lifts it."""
    order = geometry.order_of(render_data)
    function_mode = settings["function_mode"] if function_mode is None else float(function_mode)
    if complex_scale is None:
        complex_scale = settings["complex_scale"]
    n = settings["subdivision"] if n_segments is None else int(n_segments)
    lift = settings["deformation"] if deformation is None else float(deformation)
    mode = raster.deformation_mode(int(render_data["mesh_dim"]), int(render_data["funcdim"]),
                                  settings["is_complex"]) if lift else "none"
    flat = float(lift) if render_data.get("deformation") is True else 0.0
    return order, n, lift, flat, mode, function_mode, complex_scale


def _frame(render_data, width, height, fit, camera_kwargs, placing):
    """The camera of :func:`frame`, from the keywords already resolved by :func:`_placing`."""
    if fit not in (None, "sphere", "box"):
        raise ValueError(f"fit frames the bounding 'sphere' of the mesh, as the page does, or "
                         f"the 'box' of what is painted, not {fit!r}")
    if (width is None or height is None) and (fit != "box" or width is None and height is None):
        raise TypeError("a side of the canvas is left to the picture only with fit='box', "
                        "which sizes it from the other")
    view = camera.Camera.from_render_data(render_data, width=width or height,
                                         height=height or width, **camera_kwargs)
    if fit == "box":
        order, n, lift, flat, mode, function_mode, complex_scale = placing
        view = _box_view(view, _extent(render_data, order, n, lift, flat, mode, function_mode,
                                       complex_scale),
                         width is None, height is None)
    return view


def frame(render_data, *, width, height, fit=None, n_segments=None, deformation=None,
          function_mode=None, complex_scale=None, is_complex=None, **camera_kwargs):
    """The camera :func:`paint` would paint ``render_data`` with, and so its canvas, unpainted.

    The keywords are those of :func:`paint` that say where the picture lies -- the canvas, the
    framing, the turns, and what moves the geometry: the subdivision, the deformation, and the
    part of a value that lifts it -- and the answer is the :class:`ngsplot.camera.Camera`, whose
    ``width`` and ``height`` are the canvas.  It is what to ask when a side is left to
    ``fit="box"``: the size of a picture of a long domain, known before it is painted, so that a
    figure can be laid out around it.  Handed back as ``view=`` to :func:`paint` or
    :func:`ngsplot.pyplot.Plot`, it paints without framing again -- which a time loop over one
    mesh then does once and not every frame, as long as no deformation moves the mesh between
    the frames.
    """
    settings = settings_of(render_data, is_complex=is_complex)
    placing = _placing(render_data, settings, n_segments, deformation, function_mode,
                       complex_scale)
    return _frame(render_data, width, height, fit, camera_kwargs, placing)


def paint(render_data, *, width=None, height=None, view=None, canvas=None, fit=None,
          vmin=None, vmax=None, autoscale=None, cmap=None, ncolors=None,
          n_segments=None, line_width=None, surface=None, wireframe=None, edges=None,
          function_mode=None, complex_scale=None, dark_backside=None,
          deformation=None, is_complex=None, **camera_kwargs):
    """Draw ``render_data`` into a :class:`ngsplot.raster.Canvas` and return it.

    ``width`` and ``height`` are the drawing buffer, which is the size of the page's
    canvas rather than of the element holding it (docs/OVERLAYS.md, "The overlay
    layer").  Pass ``view`` to draw with a camera built elsewhere -- by :func:`frame`, say --
    which brings its canvas with it, and any other
    :class:`ngsplot.camera.Camera` keyword -- ``elev``, ``azim``, ``roll``, the page's
    own ``euler_angles`` or ``transformations`` -- to turn the default one; a payload
    that records a camera in ``gui_settings`` is followed without being asked.

    ``fit`` is what the camera frames.  ``"sphere"``, or nothing, is the page's own framing:
    the bounding sphere of the mesh fills :data:`ngsplot.camera.FILL` of the canvas height,
    whatever the angle, which leaves a mesh much longer than it is wide a strip across the
    middle.  ``"box"`` frames the box of what is painted instead, as seen from the angle asked
    for, with the same margin on its tighter side and its middle in the middle of the canvas
    (:meth:`ngsplot.camera.Camera.fit`).  With ``"box"`` one of ``width`` and ``height`` may
    be ``None``, and is then sized to the box, with the margin of the other side;
    :func:`frame` says which size that is without painting.

    Left alone, every keyword is the setting of a fresh panel (see
    :func:`settings_of`): the colours come from ``colormap_of()``, the range from
    :func:`view_range`, and the boundary carries ``line_width`` gui pixels of ink,
    which is half that many device pixels at any canvas height
    (``three/scene.ts:776``).  ``surface=False``, ``wireframe=False`` and ``edges=False``
    hide the three kinds of geometry the Objects folder holds, ``ncolors`` and ``cmap``
    change the bands, and ``deformation`` is the amount the geometry is pushed along the
    value, which a payload that says it sent flat geometry needs lifted for the lines as
    much as for the surface.

    ``is_complex`` is taken from the payload, which says so when the function it came from
    was complex; pass it to draw a real/imaginary pair as one complex value, or to ask for
    the length of a complex one.  ``complex_scale`` then turns the drawn part by an angle,
    as the phase slider of the panel does.
    """
    settings = settings_of(render_data, is_complex=is_complex)
    if settings["vectors"]:
        raise NotImplementedError(
            "the arrow glyphs of Vectors.show are not painted yet; "
            "draw them with matplotlib's quiver if a payload asks for them")
    placing = _placing(render_data, settings, n_segments, deformation, function_mode,
                       complex_scale)
    order, n, lift, flat, mode, function_mode, complex_scale = placing
    n_colors = settings["ncolors"] if ncolors is None else int(ncolors)
    pal = colormap_of(cmap, n_colors, settings["mesh_only"])
    low, high = view_range(render_data, vmin, vmax, autoscale, function_mode)

    if view is None:
        view = _frame(render_data, width, height, fit, camera_kwargs, placing)
    elif camera_kwargs:
        raise TypeError("camera keywords with an explicit view contradict it")
    elif fit == "box":
        raise TypeError("an explicit view is framed already; fit frames the one paint makes")
    elif (width, height) not in ((None, None), (view.width, view.height)):
        raise TypeError(f"the view brings a canvas of {view.width} x {view.height}; leave the "
                        f"width and height out, or name that one")
    if canvas is None:
        canvas = raster.Canvas(view.width, view.height)
    if surface is None:
        surface = settings["surface"]
    if surface:
        ctrl, vectors = geometry.trig_points(render_data)
        raster.paint_surface(
            canvas, view, order, ctrl,
            vmin=low, vmax=high, pal=pal, n_segments=n,
            vectors=vectors,
            deformation=lift, mode=mode,
            function_mode=function_mode,
            complex_scale=complex_scale,
            dark_backside=settings["dark_backside"] if dark_backside is None else bool(dark_backside),
            # a plain Draw reaches the page with draw_surf null, and only an explicit false
            # is NO_FUNCTION_VALUES
            function_values=render_data.get("draw_surf") is not False)

    if wireframe is None:
        wireframe = settings["wireframe"]
    if edges is None:
        edges = settings["edges"]
    width_px = settings["line_width"] if line_width is None else float(line_width)
    if edges and render_data.get("edges"):
        raster.paint_edges(canvas, view, order, geometry.edge_points(render_data)[0],
                           n_segments=n, line_thickness=width_px / canvas.height,
                           deformation=flat, mode=mode)
    if wireframe and render_data.get("Bezier_points"):
        raster.paint_wireframe(canvas, view, order, geometry.wireframe_points(render_data)[0],
                               n_segments=n, deformation=flat, mode=mode)
    return canvas


def _extent(render_data, order, n, lift, flat, mode, function_mode, complex_scale):
    """Every point the passes of :func:`paint` put on the canvas, as ``(k, 3)`` mesh points.

    The lattice of the surface and the samples of the two line objects, each deformed the way
    its own pass deforms it, so that a box fit around them is fit around the picture and not
    around the mesh it was painted from.
    """
    kw = dict(mode=mode, function_mode=function_mode, complex_scale=complex_scale)
    pieces = []
    ctrl, vectors = geometry.trig_points(render_data)
    if len(ctrl):
        pos, _ = raster.lattice_points(order, ctrl, vectors=vectors, deformation=lift,
                                       n_segments=n, **kw)
        pieces.append(pos.reshape(-1, 3))
    for key, points in (("edges", geometry.edge_points),
                        ("Bezier_points", geometry.wireframe_points)):
        if render_data.get(key):
            pos, _ = raster.spline_samples(order, points(render_data)[0], n,
                                           deformation=flat, **kw)
            pieces.append(pos.reshape(-1, 3))
    return np.concatenate(pieces) if pieces else np.zeros((0, 3))


#: the most a side of the canvas sized to a box is made, as a multiple of the side it was
#: sized from: a box this long for its width is a line, and a canvas around it is mostly margin
_BOX_SIDE = 8


def _box_view(view, points, size_width, size_height):
    """``view`` framed on the box of ``points``, on a canvas whose side ``size_width`` or
    ``size_height`` says is to be sized to that box first.

    A side sized to the box is the box at :data:`ngsplot.camera.FILL` of the other side, plus
    the margin that leaves on the other side, so the picture has one margin all round.  Which
    way the box runs does not depend on the canvas it is measured on, since a pixel is the same
    length of mesh across as down.
    """
    if size_width or size_height:
        seen = view.box(points)
        across, down = (1.0, 1.0) if seen is None else (seen[0][1] - seen[0][0],
                                                        seen[0][3] - seen[0][2])
        margin = 1.0 - camera.FILL
        if size_height:
            ratio = down / across if across > 0 else float(_BOX_SIDE)
            side = view.width * (camera.FILL * min(ratio, _BOX_SIDE) + margin)
            view = view.reframed(height=int(min(max(round(side), 1), _BOX_SIDE * view.width)))
        else:
            ratio = across / down if down > 0 else float(_BOX_SIDE)
            side = view.height * (camera.FILL * min(ratio, _BOX_SIDE) + margin)
            view = view.reframed(width=int(min(max(round(side), 1), _BOX_SIDE * view.height)))
    return view.fit(points)
