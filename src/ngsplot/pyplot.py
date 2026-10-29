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
import math
import weakref

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as _pyplot
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Bbox

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
#: the widest a colourbar gets, as a share of the width of the picture it keys
BAR_SHARE = 0.08
#: the least of its canvas a picture has to be painted across before a bar measures the painting
#: rather than the canvas it was painted on: a bar the length of a line is a key nobody can read.
#: A function on a 1D mesh is painted across 0.7% of the rows of its canvas, and a 6 x 1.2 pipe
#: across 21%, so the line is told from the pipe well short of either
BAR_MIN_PAINT = 0.05
#: how many times wider than tall a picture is painted before its bar goes underneath it: a bar
#: beside a picture is as long as the picture is tall, and that is too short to read on a pipe
BAR_TURN = 2.0
#: the longest a colourbar gets for its width, which is what :data:`BAR_SHARE` gives way to
#: under a picture much wider than it is tall: past this a bar is a line with labels on it
BAR_MAX_ASPECT = 30.0
#: the gap matplotlib leaves between a picture and its bar, per side, as a share of the room;
#: under an axis with its frame off, every side gets the gap the right-hand side gets
_BAR_PAD = {"right": 0.05, "left": 0.10, "top": 0.05, "bottom": 0.15}

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

    ``fit="box"`` frames the box of the picture instead of the bounding sphere the page frames,
    so a long mesh fills the canvas instead of a strip across it (see
    :func:`ngsplot.scene.paint`), and then one of ``width`` and ``height`` may be ``None``: it is
    sized to the box, which gives a canvas of the picture's own shape.  ``Draw``'s ``center``
    and ``radius`` are the other way to frame a picture, and are not asked for together with it.
    ``view`` is a camera framed beforehand -- :func:`ngsplot.frame` takes the same ``fit``,
    ``width`` and ``height`` and paints nothing -- and brings its canvas with it, so a figure can
    be laid out for the picture first and a time loop frames its mesh once.

    The rest of the keywords are the panel's knobs, spelled as the renderer spells them; see the
    module docstring for which is which.  ``colorbar=True`` puts a bar next to the picture, in
    the same bands and between the same ends, and takes the room for it out of the picture --
    beside taking the frame off, the only two things a plot does to an axis it did not make.
    ``colorbar`` may also be a dict of the keywords of :func:`colorbar`, which is the same bar
    asked for in other words: ``colorbar={"orientation": "horizontal"}``, or
    ``colorbar={"cax": slot}`` for a bar in a place of the caller's own.

    The point of view is ``elev``, ``azim`` and ``roll``, in degrees -- the three angles of
    ``Axes3D.view_init``, put on a camera that stays where the page put it.  Left out, they are
    the view the page opens with, straight at ``xy``, which is what a flat picture wants;
    ``elev=30, azim=-60`` is the angle matplotlib's own 3D plots open with, and what a solid
    wants to be seen from.

    Returns the :class:`~matplotlib.image.AxesImage`.  Plotting twice into one axis replaces the
    picture rather than laying a second under it, which is what an animation of a time loop
    needs; titles, labels and colourbars are left to whoever made them.  A bar the plot put up
    goes with a picture that no longer asks for it, and a bar on a ``cax`` of the caller's stays
    and keys the new picture instead, since the place it stands in is the caller's.
    """
    width = None if width is None else int(width)
    height = None if height is None else int(height)
    ask = _bar_ask(colorbar)
    if kwargs.get("fit") is not None and {"center", "radius"} & set(kwargs):
        raise TypeError("fit frames the picture, and so do center and radius; ask for one")
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
    view = renderer.get("view")
    if view is not None and (width, height) == (WIDTH, HEIGHT):
        # the page's canvas is what a plot gets when it names none, and a view names its own
        width = height = None
    if eval is not None:
        if "function_mode" in renderer:
            raise TypeError("eval and function_mode are the same knob asked for twice")
        renderer["function_mode"] = _part_of_value(eval)

    canvas = _scene.paint(payload, width=width, height=height, **renderer)
    axes = gca() if ax is None else ax
    if axes not in _bare_axes:
        axes.set_axis_off()  # a picture needs no rulers in pixels; set_axis_on puts them back
    _retire(axes, keep_bar=ask is not None)
    image = axes.imshow(canvas.to_uint8(),
                        extent=(0.0, float(canvas.width), float(canvas.height), 0.0),
                        origin="upper", interpolation="nearest")
    _plotted.setdefault(axes, []).append(image)
    _pictures[image] = _painted_with(payload, renderer)
    _state["last"] = weakref.ref(image)
    standing = _bars.get(axes)
    if ask is None and standing is not None:
        ask = standing["kwargs"]  # a bar on the caller's cax keys whatever is painted here
    if ask is not None:
        _colorbar(image, **ask)
    return image


def colorbar(image=None, **kwargs):
    """A colourbar standing next to a picture, in that picture's bands and between its ends.

    The bar cannot be the picture itself: the picture is already coloured, its array holds
    colours rather than values, and a bar built on it would show the greys of a viridis nobody
    asked for.  So the bar is built on a private mappable carrying the texture and the range the
    picture was painted with, and the bands come out as bands -- a ``BoundaryNorm`` on the
    ``ListedColormap`` of the page's ramp -- with the edges of the bands ticked, which is the
    same scale the panel's two fields hold.  Every edge is ticked, and as many of them labelled
    as the bar has room for at the size it is drawn: every second edge, every fourth, down to
    the two ends, so a label always sits on the end of a band and the two ends are always read.

    With no argument it takes the last picture plotted here.  ``mesh`` has no colours to put a
    bar against and says so.  Left to itself a bar is measured against the picture it keys: as
    long as that picture is drawn, and no wider than :data:`BAR_SHARE` of it -- nor thinner than
    its length over :data:`BAR_MAX_ASPECT`, which is where a picture far wider than tall gives
    way.  It stands beside the picture, or underneath one painted more than :data:`BAR_TURN`
    times as wide as it is tall, where a bar beside it would be too short to label.  The numbers
    matplotlib sizes a bar with -- ``shrink``, ``fraction``, ``aspect``, ``pad`` and the
    ``orientation`` or ``location`` that turns them -- are whoever asks for the bar's to name,
    and a bar given a ``cax`` of its own is sized by whoever made that.

    One bar stands beside an axis.  A repaint hands that bar the bands and the ends of the new
    picture rather than building another one and carving the room for it out again, so a time
    loop keeps the bar it was handed and the picture keeps the room it gave up once.  A bar the
    plot put up goes when a picture is plotted without asking for one; a bar on a ``cax`` of the
    caller's stays, and keys each picture painted on its axis after it.  Either goes when it is
    asked for in other words -- a bar built with ticks of its own is not talked into a different
    kind of bar, and a bar asked for on another ``cax`` takes the place of the one on the first.
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
    figure = image.axes.get_figure()
    standing = _bars.get(image.axes)
    live = standing is not None and _stands(standing["bar"])
    carved = live and "cax" not in standing["kwargs"]
    geometry = _bar_geometry(image, kwargs, _cell(image.axes, standing if carved else None))
    if (live and _same_ask(standing["kwargs"], kwargs)
            and _fits(standing["geometry"], geometry)):
        # matplotlib's own way of telling a bar that what it keys has changed: the mappable keeps
        # its place in the bar, and the bar reads the ramp and the scale off it and redraws
        bar, mappable = standing["bar"], standing["mappable"]
        mappable.set_cmap(cmap)
        mappable.set_norm(norm)
        bar.update_normal(mappable)
    else:
        if live:
            _take_down(image.axes, standing)  # the room comes back before the new bar takes it
        cax = kwargs.get("cax")
        if cax is not None and cax.get_figure() is figure and cax not in figure.axes:
            # the place the last bar stood in, asked for again in other words: the caller's
            # axis went with that bar, and comes back empty for this one
            cax.cla()
            figure.add_axes(cax)
        mappable = ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        room = None
        if cax is None:
            # a bar hands the room it took back to the axis it reads off its mappable; a private
            # mappable has no axis of its own unless we give it the one the picture sits in.  A
            # bar on the caller's cax took no room, and matplotlib would hand the picture the
            # cell of the cax's own grid instead
            mappable.axes = image.axes
            room = (image.axes.get_position(original=True).frozen(), image.axes.get_anchor())
        bar = figure.colorbar(mappable, ax=image.axes, **geometry)
        _bars[image.axes] = {"bar": bar, "mappable": mappable, "kwargs": kwargs,
                             "geometry": geometry, "room": room}
    if "ticks" not in kwargs:
        # set after every update_normal, which puts matplotlib's own locator back
        bar.locator = _BandEdges(boundaries)
    return bar


#: ``Plot(colorbar=True)`` reaches the function above under this name, its own keyword shadowing it
_colorbar = colorbar


def _bar_ask(colorbar):
    """The keywords a plot's ``colorbar`` asks for a bar in, or ``None`` for no bar asked."""
    if isinstance(colorbar, dict):
        return dict(colorbar)
    if colorbar is None or isinstance(colorbar, (bool, np.bool_)):
        return {} if colorbar else None
    raise TypeError("colorbar is True, False, or a dict of the keywords of pyplot.colorbar, "
                    f"not a {type(colorbar).__name__}")


class _BandEdges(ticker.Locator):
    """The edges of a picture's bands, as many of them as its bar has room to label.

    Every edge is a tick, which is what the page's own scale holds, but a label per edge is
    unreadable once the bands are denser than the font -- nine labels on a bar the height of a
    pipe.  So the edges are labelled every second, every fourth, every ``k``-th for a ``k`` that
    divides the band count, which keeps a label on both ends and an equal number of bands
    between two labels, and the step is the smallest whose labels fit.  What fits is read at
    draw time off the size the bar is drawn at and the size of its labels: a label along a
    vertical bar needs two font sizes, matplotlib's own spacing for a y axis, and one across a
    horizontal bar needs its own width, which is measured from the labels the bar's formatter
    makes, at three fifths of a font size a character, plus one font size between two of them.
    The two ends are labelled however short the bar is.
    """

    def __init__(self, edges):
        self.edges = np.asarray(edges, dtype=np.float64)

    def __call__(self):
        return self.tick_values(None, None)

    def tick_values(self, vmin, vmax):
        bands = len(self.edges) - 1
        room = self._room()
        for step in range(1, bands + 1):
            if bands % step == 0 and bands // step + 1 <= room:
                return self.edges[::step]
        return self.edges[[0, -1]]

    def _room(self):
        """How many labels fit along the axis, as it is drawn now."""
        axis = self.axis
        if axis is None:
            return len(self.edges)
        figure = axis.axes.figure
        box = axis.axes.bbox
        size = axis.get_major_ticks(1)[0].label1.get_size()
        if axis.axis_name == "y":
            length, spacing = box.height, 2.0 * size
        else:
            labels = axis.major.formatter.format_ticks(self.edges)
            length, spacing = box.width, (0.6 * max(map(len, labels), default=1) + 1.0) * size
        if not spacing > 0:
            return len(self.edges)
        # the labels at the two ends may hang over the ends of the bar, so n of them need n - 1
        # spacings and not n
        return int(length * 72.0 / figure.dpi // spacing) + 1


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


def _cell(axes, standing=None):
    """The room an axis was given, before the bar of ours ``standing`` beside it took any away.

    A bar carves its room out of the picture's cell and keeps the piece it carved, so the two
    pieces laid side by side are the cell again.  The picture's axis reports the narrowed one on
    its own, and a bar sized against that would be sized against what the last bar left behind.
    An axis placed by hand has no cell to read back, and its own record of where it was placed is
    overwritten by the bar (see :func:`_take_down`), so its room is the one the bar recorded.
    """
    figure = axes.get_figure()
    spec = axes.get_subplotspec()
    if figure is None or spec is None:
        if standing is not None and standing["room"] is not None:
            return standing["room"][0]
        return axes.get_position(original=True)
    cell = spec.get_position(figure)
    if standing is not None:
        taken = standing["bar"].ax.get_subplotspec()
        if taken is not None:
            cell = Bbox.union([cell, taken.get_position(figure)])
    return cell


def _take_down(axes, standing):
    """Remove a bar of ours, and give the axis beside it back the room the bar took.

    matplotlib does that for an axis in a grid, from the grid; an axis placed by hand has its
    position carved down when the bar is made, its original position included
    (``matplotlib/colorbar.py``, ``make_axes``, by ``_set_position``), so ``Colorbar.remove``
    has nothing left to put back, and a picture repainted with its bar would shrink by a bar's
    width every time.  The bar recorded the room and the anchor it found, and they go back here.
    """
    standing["bar"].remove()
    room = standing.get("room")
    if room is not None and axes.get_subplotspec() is None:
        position, anchor = room
        axes._set_position(position)
        axes.set_anchor(anchor)


def _painted(array):
    """The share of the canvas it was exported on that a picture was painted across, by rows and
    by columns.

    The renderer fits a mesh into the raster it is asked for, and the fit leaves the page's own
    colour around the mesh: a square mesh on a three-to-two canvas is painted over two thirds of its
    rows and under half of its columns.  A bar is a key to the mesh, so it is measured by what was
    painted rather than by the canvas around it.  Every eighth pixel answers this well enough -- the
    number sizes a bar, not a pixel -- and a picture painted over barely any of its canvas is a
    line, whose canvas is the picture as far as a reader is concerned: a bar the length of a line
    keys nothing, so there the shares are the whole of the canvas.
    """
    view = np.asarray(array)
    if view.ndim != 3 or view.shape[0] < 32 or view.shape[1] < 32:
        return 1.0, 1.0  # too small a canvas to read a fit out of
    view = view[::8, ::8, :3]
    corners = np.stack([view[0, 0], view[0, -1], view[-1, 0], view[-1, -1]])[:, None, None]
    painted = ((view != corners).any(axis=3)).all(axis=0)  # unlike every corner: painting, not page
    rows, columns = np.flatnonzero(painted.any(axis=1)), np.flatnonzero(painted.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return 1.0, 1.0  # a canvas with nothing painted on it is all canvas
    along = (rows[-1] - rows[0] + 1) / view.shape[0]
    across = (columns[-1] - columns[0] + 1) / view.shape[1]
    return (along, across) if min(along, across) >= BAR_MIN_PAINT else (1.0, 1.0)


def _bar_geometry(image, kwargs, cell):
    """Room for a bar beside a picture, taken out of the room the picture was given.

    A bar is sized by the room it shares with the picture, but the picture keeps the shape it was
    painted in and leaves the rest of that room empty, so the two disagree whenever their shapes
    do -- the bar then stands over the picture and takes room the picture could have used.  The
    rules here are caps: the bar spans no more of the cell than the picture spans of it, and it
    is no wider than :data:`BAR_SHARE` of the picture -- unless that would make it longer than
    :data:`BAR_MAX_ASPECT` times its width, which is a line and not a bar.

    Which side it stands on comes first.  A picture painted more than :data:`BAR_TURN` times
    as wide as it is tall gets its bar underneath, since a bar beside it would be as long as the
    picture is tall.  The gap is matplotlib's for the side, and under an axis with its frame off
    the right-hand side's for every side: the larger gaps below and on the left are the room
    matplotlib keeps for tick labels, which a picture does not have.

    ``shrink`` is the share of the cell the bar spans, and it is exact -- a bar is that share of
    the cell long.  ``fraction`` is the share of the axis the picture gives up, which is the room a
    bar stands in rather than its width: matplotlib makes a bar as wide as its length over
    ``aspect``, and under a tight or constrained layout makes the width out of that room instead.
    A room too narrow to hold the bar is a room the bar gets shortened into, because matplotlib
    gives a bar a row of ``shrink`` and an aspect of its own and cuts the bar to the width of its
    room times that aspect.  So the room is asked for as the bar's own width, which is what the
    second route makes of it too, and the two agree on one bar measured by its picture -- in
    either orientation, ``aspect`` being length over width in both.  What that picture is, is
    the mesh inside the raster rather than the page the raster was fit onto (:func:`_painted`).
    Whatever is named is left named, and a bar with a ``cax`` of its own is nobody's cell to
    size.
    """
    out = dict(kwargs)
    figure = image.axes.get_figure()
    if figure is None or "cax" in out:
        return out
    inches = figure.get_size_inches()
    x0, x1, y0, y1 = image.get_extent()  # the shape imshow letterboxes to, not the array's
    painted = _painted(image.get_array())
    tall, wide = painted[0] * abs(y1 - y0), painted[1] * abs(x1 - x0)
    if "orientation" not in out and "location" not in out:
        out["orientation"] = "horizontal" if wide > BAR_TURN * tall else "vertical"
    if "location" in out:
        side = out["location"]
    else:
        side = "bottom" if out["orientation"] == "horizontal" else "right"
    horizontal = side in ("top", "bottom")
    if "pad" not in out and not image.axes.axison:
        out["pad"] = _BAR_PAD["right"]
    pad = float(out.get("pad", _BAR_PAD.get(side, 0.05)))
    shape = abs(x1 - x0) / abs(y1 - y0) if not horizontal else abs(y1 - y0) / abs(x1 - x0)
    long_share, wide_share = painted if not horizontal else painted[::-1]
    along = cell.height * inches[1] if not horizontal else cell.width * inches[0]
    across = cell.width * inches[0] if not horizontal else cell.height * inches[1]
    if not (math.isfinite(shape) and shape > 0 and along > 0 and across > 0 and pad < 0.85):
        return out  # a shape, a cell or a pad to size a bar against there is none here

    def picture(fraction):  # what the axis shows of the picture, with that much room given up
        fit = min(along, across * (1 - fraction - pad) / shape)
        return fit * long_share, shape * fit * wide_share  # along the bar, across it

    fraction = float(out.get("fraction", 0.15))
    aspect = float(out.get("aspect", 20.0))
    for _ in range(2):  # the room a bar needs depends on the room it takes; twice settles it
        long_side, wide_side = picture(fraction)
        length = float(out.get("shrink", long_side / along)) * along
        if "aspect" not in out:
            # a bar is as wide as its length over ``aspect``: ask for the cap, but never for a
            # fatter bar than the 20 it would have stood on anyway, nor a thinner one than a line
            aspect = min(max(20.0, length / (BAR_SHARE * wide_side)), BAR_MAX_ASPECT)
        if "fraction" in out:
            break  # room named is room given, whatever width the bar takes in it
        # a room too narrow to hold the bar is a room the bar is cut short in, so the room a bar
        # asks for is the width it will be: its length over its aspect
        fraction = length / aspect / across

    out.setdefault("shrink", long_side / along)
    out.setdefault("fraction", fraction)
    out.setdefault("aspect", aspect)
    return out


def _fits(had, wants, tolerance=0.01):
    """Does the room a bar stands in still fit the picture beside it?

    A figure resized between two pictures wants other room, and the bar is built again rather
    than argued with; a repaint at the size it stands at must leave it standing, so the numbers
    are compared loose.  Only the numbers both sides have an opinion about are compared: a bar
    whose width a caller named keeps standing whatever the room says.  The side a bar stands on
    is compared as a word, and a picture that turned from tall to wide gets its bar turned.
    """
    return (all(math.isclose(float(had[name]), float(wants[name]), rel_tol=tolerance,
                             abs_tol=1e-3)
                for name in ("fraction", "shrink", "aspect", "pad")
                if name in had and name in wants)
            and all(had.get(name) == wants.get(name) for name in ("orientation", "location")))


def _retire(axes, keep_bar=False):
    """Take the picture off an axis, and the bar beside it unless something will use it.

    The next picture uses the bar when it asks for one, and a bar on a ``cax`` of the caller's
    is used whether it asks or not: that bar stands in a place the caller made, and is theirs to
    take down.  A bar someone already took down is forgotten rather than taken down twice.
    """
    for image in _plotted.get(axes, ()):
        image.remove()
    standing = _bars.get(axes)
    if standing is not None and not _stands(standing["bar"]):
        del _bars[axes]
    elif standing is not None and not keep_bar and "cax" not in standing["kwargs"]:
        del _bars[axes]
        _take_down(axes, standing)
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
    asked-for bar is built again and the room goes with the old one.  A ``cax`` is told by which
    axis it is, since two axes can spell the same.
    """
    return had.keys() == wants.keys() and all(
        had[name] is wants[name] if name == "cax" else repr(had[name]) == repr(wants[name])
        for name in had)


def _last_plotted():
    """The last picture plotted here, while someone still holds it."""
    image = None if _state["last"] is None else _state["last"]()
    if image is None:
        raise TypeError("nothing has been plotted here yet; Plot() hands back the picture "
                        "colourbar() takes")
    return image
