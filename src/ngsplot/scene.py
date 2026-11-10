"""The knobs: render data in, picture out, defaults exactly as the webgui has them.

Everything the live viewer keeps in its control panel -- the colour range, the number
of bands, the wireframe and its width, the subdivision of every curve -- is a keyword
here, and every keyword that is left out takes the value ``three/gui.ts`` gives the
panel on a fresh draw.  A payload straight from ``ngsolve.webgui`` therefore paints
the picture ``ngsolve.webgui`` would have painted, ``Draw`` included in the payload
its own ``gui_settings``.  See docs/SPEC.md, "Colour", and docs/DIMS.md.

The knobs deliberately mirror matplotlib's names: ``vmin``/``vmax`` switch off the
autoscaled range as passing them does to ``imshow``, ``cmap`` takes any matplotlib
colormap, and ``ncolors`` is the webgui's own band count.
"""

import numpy as np

from . import camera, colormap, geometry, raster

__all__ = ["DEFAULTS", "settings_of", "view_range", "colormap_of", "paint"]

#: ``three/gui.ts`` settings of a fresh panel: the values a first ``Draw`` renders with.
DEFAULTS = {
    "ncolors": 8,               # Colormap.ncolors
    "autoscale": True,          # Colormap.autoscale
    "line_width": 5.0,          # Misc.line_thickness, in gui pixels
    "subdivision": 5,           # Misc.subdivision
    "fast_draw": False,         # Misc.fast_draw
    "dark_backside": True,      # the shader's dark_backside uniform
    "wireframe": True,          # Objects.Wireframe
    "edges": True,              # Objects.Edges
    "vectors": False,           # Vectors.show
}


def _section(settings, name):
    """One section of a payload's ``gui_settings``, which may be absent or null."""
    section = settings.get(name) or {}
    return section if isinstance(section, dict) else {}


def settings_of(render_data, *, is_complex=False):
    """The effective settings of ``render_data``: the panel defaults, then its own.

    ``is_complex`` says what the payload cannot: a complex function reaches the page
    split into two components (``ngsolve/webgui.py:304-305``) and is otherwise
    indistinguishable from a vector one, but draws the real part rather than the
    length, and is not deformed along it.

    A payload records the panel in ``gui_settings``, in the same sections the lil-gui
    status object has (``Colormap``, ``Misc``, ``Objects``, ``camera``), and leaves out
    whatever was left at its default.  Two settings are not panel values at all but
    follow from the payload: the subdivision drops to 1 when no element is curved
    (``gui.ts:229-231``) and the deformation defaults to ``deformation_scale`` when the
    payload says the geometry was sent flat (``gui.ts:582-587``).
    """
    settings = render_data.get("gui_settings") or {}
    colormap_section = _section(settings, "Colormap")
    misc = _section(settings, "Misc")
    objects = _section(settings, "Objects")

    out = dict(DEFAULTS)
    out["ncolors"] = int(colormap_section.get("ncolors", out["ncolors"]))
    out["autoscale"] = bool(colormap_section.get("autoscale", out["autoscale"]))
    out["line_width"] = float(misc.get("line_thickness", out["line_width"]))
    out["subdivision"] = int(misc.get("subdivision", out["subdivision"]))
    out["fast_draw"] = bool(misc.get("fast_draw", out["fast_draw"]))
    out["dark_backside"] = bool(misc.get("dark_backside", out["dark_backside"]))
    out["wireframe"] = bool(objects.get("Wireframe", out["wireframe"]))
    out["edges"] = bool(objects.get("Edges", out["edges"]))
    out["vectors"] = bool(objects.get("Vectors", out["vectors"]))

    order = max(int(render_data.get("order2d", 1)), int(render_data.get("order3d", 1)))
    if order <= 1 or out["fast_draw"]:
        out["subdivision"] = 1

    scale = float(render_data.get("deformation_scale") or 1.0)
    out["deformation"] = scale if render_data.get("deformation") else 0.0

    # `eval` picks which part of a value to draw: x, y, z, length, real, imaginary.  A
    # complex function draws its real part and a vector its length, both as the panel does
    # (gui.ts:530-563); a payload may say `eval` outright, which outranks the panel.
    funcdim = int(render_data.get("funcdim", 1))
    mode = 5.0 if is_complex else (3.0 if funcdim in (2, 3) else 0.0)
    mode = settings.get("eval", mode)
    mode = render_data.get("eval", mode)
    out["function_mode"] = float(mode)

    phase = float(_section(settings, "Complex").get("phase", 0.0))
    out["complex_scale"] = (float(np.cos(-phase)), float(np.sin(-phase)))
    out["mesh_only"] = int(render_data.get("funcdim", 1)) == 0
    out["is_complex"] = bool(is_complex)
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
    panel = _section(settings, "Colormap")
    if autoscale is None:
        autoscale = bool(panel.get("autoscale", True))

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


def paint(render_data, *, width, height, view=None, canvas=None,
          vmin=None, vmax=None, autoscale=None, cmap=None, ncolors=None,
          n_segments=None, line_width=None, wireframe=None, edges=None,
          function_mode=None, complex_scale=None, dark_backside=None,
          deformation=None, is_complex=False, **camera_kwargs):
    """Draw ``render_data`` into a :class:`ngsplot.raster.Canvas` and return it.

    ``width`` and ``height`` are the drawing buffer, which is the size of the page's
    canvas rather than of the element holding it (docs/OVERLAYS.md, "The overlay
    layer").  Pass ``view`` to draw with a camera built elsewhere, and any other
    :class:`ngsplot.camera.Camera` keyword -- ``euler_angles``, ``transformations`` --
    to tilt the default one; a payload that records a camera in ``gui_settings`` is
    followed without being asked.

    Left alone, every keyword is the setting of a fresh panel (see
    :func:`settings_of`): the colours come from ``colormap_of()``, the range from
    :func:`view_range`, and the boundary carries ``line_width`` gui pixels of ink,
    which is half that many device pixels at any canvas height
    (``three/scene.ts:776``).  ``wireframe=False`` and ``edges=False`` hide the two
    kinds of line, ``ncolors`` and ``cmap`` change the bands, and ``deformation`` is
    the amount the geometry is pushed along the value, which a payload that says it
    sent flat geometry needs lifted for the lines as much as for the surface.

    ``is_complex`` says what no payload records: that the two components of the
    function are the real and the imaginary part of one complex value, which the viewer
    colours by real part rather than by length.  ``complex_scale`` then turns that part
    by an angle, as the phase slider of the panel does.
    """
    settings = settings_of(render_data, is_complex=is_complex)
    if settings["vectors"]:
        raise NotImplementedError(
            "the arrow glyphs of Vectors.show are not painted yet; "
            "draw them with matplotlib's quiver if a payload asks for them")
    if view is None:
        view = camera.Camera.from_render_data(render_data, width=width, height=height,
                                             **camera_kwargs)
    elif camera_kwargs:
        raise TypeError("camera keywords with an explicit view contradict it")
    if canvas is None:
        canvas = raster.Canvas(width, height)

    order = geometry.order_of(render_data)
    function_mode = settings["function_mode"] if function_mode is None else float(function_mode)
    n_colors = settings["ncolors"] if ncolors is None else int(ncolors)
    pal = colormap_of(cmap, n_colors, settings["mesh_only"])
    low, high = view_range(render_data, vmin, vmax, autoscale, function_mode)
    n = settings["subdivision"] if n_segments is None else int(n_segments)
    lift = settings["deformation"] if deformation is None else float(deformation)
    mode = raster.deformation_mode(int(render_data["mesh_dim"]), int(render_data["funcdim"]),
                                  settings["is_complex"]) if lift else "none"

    flat = float(lift) if render_data.get("deformation") is True else 0.0
    ctrl, vectors = geometry.trig_points(render_data)
    raster.paint_surface(
        canvas, view, order, ctrl,
        vmin=low, vmax=high, pal=pal, n_segments=n,
        vectors=vectors,
        deformation=lift, mode=mode,
        function_mode=function_mode,
        complex_scale=settings["complex_scale"] if complex_scale is None else complex_scale,
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
                           n_segments=n, line_thickness=width_px / height,
                           deformation=flat, mode=mode)
    if wireframe and render_data.get("Bezier_points"):
        raster.paint_wireframe(canvas, view, order, geometry.wireframe_points(render_data)[0],
                               n_segments=n, deformation=flat, mode=mode)
    return canvas
