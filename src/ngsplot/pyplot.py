"""Plotting that looks like NGSolve's Draw and reads like pyplot.

This is the layer a script or a notebook calls, and it has one command: :func:`Plot`, the
counterpart of ``ngsolve.webgui.Draw``, importable from the package itself as
``from ngsplot import Plot``.  It paints nothing: it decides what it was handed, asks ngsolve for
the payload ``Draw`` would have sent (see :mod:`ngsplot._ngsolve`), asks
:func:`ngsplot.scene.paint` for the pixels that payload paints, and lays those pixels on an axis.
From there matplotlib owns the figure, so a title, a colourbar, a second subplot, a ``savefig``
into a video and a widget in a notebook all work the way they already do, and nothing about the
picture has to be relearned.

What a picture is here: one :class:`~matplotlib.image.AxesImage` whose array is the renderer's
array, byte for byte, drawn with ``interpolation="nearest"`` because it is a finished raster and
not data to be resampled, with its extent in the page's own pixels.  The axis a plot makes for
itself covers the figure, because that is what Draw looks like: the picture *is* the plot.  A
plot into an axis someone else made keeps what was on it and takes its frame away -- rulers
counting pixels are not what a Draw wants, and :meth:`~matplotlib.axes.Axes.set_axis_on` puts
them back for whoever means them.  The picture is never stretched: it keeps the shape it was
painted in, and gives room away rather than taking any, exactly as any image does.

The keywords are split by what they are for, and the rule is one line long:

* the keywords of :func:`ngsplot.scene.paint` and of :class:`ngsplot.camera.Camera` go to the
  painter -- ``vmin``, ``vmax``, ``autoscale``, ``cmap``, ``ncolors``, ``n_segments``,
  ``line_width``, ``surface``, ``wireframe``, ``edges``, ``function_mode``, ``complex_scale``,
  ``dark_backside``, ``is_complex``, ``elev``, ``azim``, ``roll``, ``transformations``, ...
* everything else goes to ``Draw``, so the keywords that shape the payload -- ``order``,
  ``draw_vol``, ``draw_surf``, ``settings``, ``scale``, ``nodal_p1`` -- keep their
  meaning, and a payload that was already built refuses them with a named error instead.
* ``deformation`` is the panel's slider and belongs on both sides: a payload is told to lift its
  geometry by the flag the shaders switch on, with the amount in ``Draw``'s ``scale``, since a
  number under ``deformation`` is read as a displacement field and ngsolve refuses it.  A payload
  that is already built cannot be, so there the amount goes to the painter, which takes it as
  ``scene.paint(deformation=...)``.
* ``eval`` is the exception worth spelling out: ``Draw`` reads it under a name nobody writes
  (docs/SPEC.md, "Why Draw is clunky") so it reaches nothing.  Here it means what the panel's
  dropdown means -- ``"real"``, ``"imag"``, ``"norm"``, ``"x"`` -- or the number from
  ``function.frag``, and it is the same knob as ``function_mode``.
* the point of view is spelled the way a viewer spells it: ``elev``, ``azim`` and ``roll``, in
  degrees, the three of ``Axes3D.view_init``.  Left out, they are the view the page opens with,
  straight at ``xy``, which is what a flat picture wants; ``elev=30, azim=-60`` is the angle
  matplotlib's own 3D plots open with, and what a solid wants.  The ``euler_angles`` triple a
  panel drag leaves in ``settings['camera']`` is a payload's private business:
  :func:`ngsplot.scene.paint` still reads it, so a saved page replays with the camera it was
  saved with, while a plot refuses the name rather than keep two spellings of one knob.

Nothing is registered in matplotlib's registry: the webgui ramp exists per picture, handed to a
colourbar as a private ``ListedColormap`` of the band count that picture used.
"""

import inspect
import weakref

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as _pyplot
from matplotlib.cm import ScalarMappable

from . import _ngsolve
from . import camera as _camera
from . import colormap as _colormap
from . import scene as _scene

#: the canvas a page opens with, in device-independent pixels (``three/scene.ts`` sizes itself
#: to the element, and the recorded baselines were taken at this size)
WIDTH = 900
HEIGHT = 600
#: pixels per inch of it, so that a figure is the picture at one-to-one and savefig keeps it so
DPI = 100

# every keyword of the renderer, and of the camera it builds, is a keyword of a plot; the names
# come from the signatures rather than a list, so a knob added there arrives here by itself
_RENDERER_NAMES = ((set(inspect.signature(_scene.paint).parameters)
                    - {"render_data", "width", "height", "canvas", "camera_kwargs"})
                   | (set(inspect.signature(_camera.Camera.__init__).parameters)
                      - {"self", "mesh_center", "mesh_radius", "width", "height"}))

#: the parts of a value ``function.frag`` knows how to pick, under the names the panel uses
_PARTS = {"x": 0.0, "y": 1.0, "z": 2.0, "norm": 3.0, "length": 3.0, "abs": 3.0,
          "real": 5.0, "imag": 6.0, "imaginary": 6.0}

_bare_axes = weakref.WeakSet()
_pictures = weakref.WeakKeyDictionary()   # AxesImage -> the texture and ends it was painted with
_plotted = weakref.WeakKeyDictionary()    # Axes -> the pictures it is holding
_bars = weakref.WeakKeyDictionary()       # Axes -> the bar standing next to it, and what built it
_state = {"last": None}                   # the last picture plotted, as a weakref


def figure(**kwargs):
    """A figure the size of the page's canvas, which is the figure a plot wants by default."""
    kwargs.setdefault("figsize", (WIDTH / DPI, HEIGHT / DPI))
    kwargs.setdefault("dpi", DPI)
    return _pyplot.figure(**kwargs)


def gcf():
    """The current figure, made the size of the page's canvas if there is none yet."""
    if not _pyplot.get_fignums():
        return figure()
    return _pyplot.gcf()


def gca():
    """The axis a plot goes into: the current one, or a bare one covering the figure.

    Draw has no frame around its picture, so an empty figure gets an axis that covers it and
    shows nothing else.  An axis that is already there -- one from ``subplots``, from
    ``gridspec``, from ``sca`` -- is used as it is and keeps its place; a plot takes its frame
    off and nothing else.
    """
    figure_ = gcf()
    if not figure_.axes:
        return _bare(figure_.add_axes([0.0, 0.0, 1.0, 1.0]))
    return _pyplot.gca()


def sca(axes):
    """Make an axis the one plots go into, as ``matplotlib.pyplot.sca`` does."""
    return _pyplot.sca(axes)


def cla():
    """Empty the current axis, and make it a bare one again if it was one."""
    axes = gca()
    axes.cla()
    if axes in _bare_axes:
        _bare(axes)
    return axes


def close(*args, **kwargs):
    """Close figures, as ``matplotlib.pyplot.close`` does -- ``"all"`` included."""
    return _pyplot.close(*args, **kwargs)


def show(*args, **kwargs):
    """Show the figures, as ``matplotlib.pyplot.show`` does."""
    return _pyplot.show(*args, **kwargs)


def savefig(*args, **kwargs):
    """Save the current figure: the picture at its own size, which is what frames need."""
    return gcf().savefig(*args, **kwargs)


def Plot(obj=None, ax=None, *, mesh=None, colorbar=False, width=WIDTH, height=HEIGHT,
         eval=None, **kwargs):
    """Draw an NGSolve object, or a payload of one, as an image on an axis.

    The counterpart of ``ngsolve.webgui.Draw``, without the window: what ``Draw`` would have put
    in front of you is painted here and handed over as a matplotlib picture.

    ``obj`` is a ``GridFunction``, a ``Mesh``, a ``CoefficientFunction``, or the payload of any
    of them -- what :func:`ngsplot._ngsolve.render_data` returns, which is also what a saved
    ``render_data`` of a page holds.  A ``CoefficientFunction`` belongs to no mesh, so it is
    drawn on the one given as ``mesh``, and hears from ngsolve when that was left out.  A
    ``Mesh`` needs no second argument either way, so ``Plot(mesh)`` and ``Plot(mesh=mesh)`` are
    both the mesh on its own, as ``Draw(mesh)`` is.

    ``width`` and ``height`` are pixels of the picture, not inches of paper: the renderer paints
    a raster and matplotlib is told how big it is, which is what keeps a savefig at one figure
    dpi exactly the pixels the renderer made.  They default to the canvas a page opens with.

    The rest of the keywords are the panel's knobs, spelled as the renderer spells them; see the
    module docstring for which is which.  ``colorbar=True`` puts a bar next to the picture, in
    the same bands and between the same ends, and takes the room for it out of the picture --
    beside taking the frame off, the only two things a plot does to an axis it did not make.

    The point of view is ``elev``, ``azim`` and ``roll``, in degrees -- the three angles of
    ``Axes3D.view_init``, put on a camera that stays where the page put it.  Left out, they are
    the view the page opens with, straight at ``xy``, which is what a flat picture wants;
    ``elev=30, azim=-60`` is the angle matplotlib's own 3D plots open with, and what a solid
    wants to be seen from.

    Returns the :class:`~matplotlib.image.AxesImage`.  Plotting twice into one axis replaces the
    picture rather than laying a second under it, which is what an animation of a time loop
    needs; titles, labels and colourbars are left to whoever made them.
    """
    width, height = int(width), int(height)
    if "euler_angles" in kwargs:
        raise TypeError("the point of view is spelled elev, azim and roll here, the three "
                        "angles of matplotlib's view_init; euler_angles is what a page's camera "
                        "holds, and scene.paint still reads it off a payload")
    if obj is None:
        obj, mesh = mesh, None  # a mesh under its own name is what to draw, as it is for Draw
    if obj is None:
        raise TypeError("Plot needs something to draw: a GridFunction, a CoefficientFunction "
                        "with mesh=, a Mesh, or the payload of one")
    payload, renderer = _split(obj, mesh, kwargs)
    if eval is not None:
        if "function_mode" in renderer:
            raise TypeError("eval and function_mode are the same knob asked for twice")
        renderer["function_mode"] = _part_of_value(eval)

    canvas = _scene.paint(payload, width=width, height=height, **renderer)
    axes = gca() if ax is None else ax
    if axes not in _bare_axes:
        axes.set_axis_off()  # a picture needs no rulers in pixels; set_axis_on puts them back
    _retire(axes, keep_bar=colorbar)
    image = axes.imshow(canvas.to_uint8(), extent=(0.0, float(width), float(height), 0.0),
                        origin="upper", interpolation="nearest")
    _plotted.setdefault(axes, []).append(image)
    _pictures[image] = _painted_with(payload, renderer)
    _state["last"] = weakref.ref(image)
    if colorbar:
        _colorbar(image)
    return image


def colorbar(image=None, **kwargs):
    """A colourbar standing next to a picture, in that picture's bands and between its ends.

    The bar cannot be the picture itself: the picture is already coloured, its array holds
    colours rather than values, and a bar built on it would show the greys of a viridis nobody
    asked for.  So the bar is built on a private mappable carrying the texture and the range the
    picture was painted with, and the bands come out as bands -- a ``BoundaryNorm`` on the
    ``ListedColormap`` of the page's ramp -- with the edges of the bands ticked, which is the
    same scale the panel's two fields hold.

    With no argument it takes the last picture plotted here.  ``mesh`` has no colours to put a
    bar against and says so.

    One bar stands beside an axis.  A repaint hands that bar the bands and the ends of the new
    picture rather than building another one and carving the room for it out again, so a time
    loop keeps the bar it was handed and the picture keeps the room it gave up once.  The bar
    goes when a picture is plotted without asking for one, or when it is asked for in other
    words -- a bar built with ticks of its own is not talked into a different kind of bar.
    """
    image = _last_plotted() if image is None else image
    picture = _pictures.get(image)
    if picture is None:
        raise TypeError(f"a colourbar needs a picture plotted here, not a {type(image).__name__}")
    pal = picture["pal"]
    if len(pal) < 2:
        raise ValueError("a mesh-only draw is painted one colour and has no scale to show")
    boundaries = np.linspace(picture["low"], picture["high"], len(pal) + 1)
    cmap = _colormap.to_matplotlib(pal)
    norm = mcolors.BoundaryNorm(boundaries, len(pal))
    standing = _bars.get(image.axes)
    reuse = standing is not None and _stands(standing["bar"]) and _same_ask(standing["kwargs"],
                                                                           kwargs)
    if reuse:
        # matplotlib's own way of telling a bar that what it keys has changed: the mappable keeps
        # its place in the bar, and the bar reads the ramp and the scale off it and redraws
        mappable = standing["mappable"]
        mappable.set_cmap(cmap)
        mappable.set_norm(norm)
        standing["bar"].update_normal(mappable)
        return standing["bar"]
    if standing is not None and _stands(standing["bar"]):
        standing["bar"].remove()  # the room comes back before the new bar takes it
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    # a bar hands the room it took back to the axis it reads off its mappable; a private mappable
    # has no axis of its own unless we give it the one the picture sits in
    mappable.axes = image.axes
    bar = image.axes.get_figure().colorbar(mappable, ax=image.axes, **kwargs)
    _bars[image.axes] = {"bar": bar, "mappable": mappable, "kwargs": kwargs}
    return bar


#: ``Plot(colorbar=True)`` reaches the function above under this name, its own keyword shadowing it
_colorbar = colorbar


def _split(obj, mesh, kwargs):
    """The payload to paint and the keywords the painter should get, from one bag of keywords."""
    lift = kwargs.pop("deformation", None)
    renderer = {name: kwargs.pop(name) for name in list(kwargs) if name in _RENDERER_NAMES}
    if isinstance(obj, dict):
        if kwargs:
            raise TypeError("a payload is already the payload, so nothing is left to do with "
                            + ", ".join(sorted(kwargs)))
        if mesh is not None:
            raise TypeError("a payload carries the mesh it was built on")
        if lift is not None:
            renderer["deformation"] = float(lift)
        return obj, renderer

    draw_kwargs = dict(kwargs)
    if "autoscale" in renderer:
        # the same word means the same thing on both sides, and a payload that says otherwise
        # is the one worth keeping: it replays as the picture it was asked for
        draw_kwargs["autoscale"] = renderer["autoscale"]
    if lift is not None:
        if isinstance(lift, bool):
            draw_kwargs["deformation"] = lift
        else:
            # ``Draw`` takes a number here for a displacement field and ngsolve refuses it, so
            # the amount goes where the page keeps its own and settings_of reads it back out
            draw_kwargs["deformation"] = True
            draw_kwargs.setdefault("scale", float(lift))
    return _ngsolve.render_data(obj, *((mesh,) if mesh is not None else ()), **draw_kwargs), renderer


def _painted_with(payload, renderer):
    """The texture and the ends a payload was painted with, which a colourbar has to agree on.

    A restatement of the first lines of :func:`ngsplot.scene.paint` for the keywords that were
    given, which is why the keywords are passed through untouched rather than pre-resolved.
    """
    settings = _scene.settings_of(payload, is_complex=renderer.get("is_complex"))
    ncolors = settings["ncolors"] if renderer.get("ncolors") is None else int(renderer["ncolors"])
    pal = _scene.colormap_of(renderer.get("cmap"), ncolors, settings["mesh_only"])
    mode = (settings["function_mode"] if renderer.get("function_mode") is None
            else float(renderer["function_mode"]))
    low, high = _scene.view_range(payload, renderer.get("vmin"), renderer.get("vmax"),
                                 renderer.get("autoscale"), mode)
    return {"pal": pal, "low": low, "high": high}


def _part_of_value(which):
    """The number ``function.frag`` matches on, from the panel's name for a part of a value."""
    if isinstance(which, str):
        try:
            return _PARTS[which.lower()]
        except KeyError:
            raise ValueError(f"no part of a value called {which!r}; name one of "
                             + ", ".join(sorted(_PARTS))
                             + ", or give the number from function.frag") from None
    return float(which)


def _bare(axes):
    """An axis that is nothing but the picture: the whole figure, no frame, no distortion."""
    axes.set_position([0.0, 0.0, 1.0, 1.0])
    axes.set_axis_off()
    axes.set_aspect("equal")
    _bare_axes.add(axes)
    return axes


def _retire(axes, keep_bar=False):
    """Take the picture off an axis, and the bar beside it unless the next one will use it."""
    for image in _plotted.get(axes, ()):
        image.remove()
    if not keep_bar:
        standing = _bars.pop(axes, None)
        if standing is not None and _stands(standing["bar"]):
            standing["bar"].remove()
    _plotted[axes] = []


def _stands(bar):
    """Is this bar still installed in a figure?  Someone may have taken it away behind us.

    ``bar.ax.get_figure()`` keeps answering with the figure it came from after the bar is gone,
    so only the figure's own list says whether the bar is in it -- and a bar taken twice raises
    out of matplotlib's ``delaxes`` rather than doing nothing.
    """
    figure = bar.ax.get_figure()
    return figure is not None and bar.ax in figure.axes


def _same_ask(had, wants):
    """Was the bar standing asked for in the same words as this one would have been?

    Told by what the keywords spell: a bar built with ticks, a formatter or its own ``cax``
    cannot be talked into the different kind of bar those other words ask for, so a differently
    asked-for bar is built again and the room goes with the old one.
    """
    return repr(had) == repr(wants)


def _last_plotted():
    """The last picture plotted here, while someone still holds it."""
    image = None if _state["last"] is None else _state["last"]()
    if image is None:
        raise TypeError("nothing has been plotted here yet; Plot() hands back the picture "
                        "colourbar() takes")
    return image
