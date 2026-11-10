"""The seam to ngsolve: every import of ngsolve this package makes lives in this file.

``Draw`` is not a plotting call.  It is the constructor of a widget that, on its way to a
browser, assembles a dict of arrays -- the mesh as Bezier patches, the samples of the
function on them, the range, the flags, the camera -- and hands that dict to the page.  That
dict is all ngsplot wants, so the seam assembles it the way ``Draw`` does and stops there:
``Draw`` fills in its default keywords and builds a ``WebGLScene``, and the scene's
``GetData`` dispatches on the type of the object to whichever draw type was registered for it.
The scene is built here without its widget, so no widget is made, none is displayed and no
browser is asked.  Registration is the part worth remembering: ``ngsolve.webgui`` registers the
handlers for ``ngsolve.Mesh``, ``CoefficientFunction`` and ``GridFunction``, and merely
importing it is what makes them known -- which is why a ``Draw(gfu)`` written without it
complains ``Cannot draw object of type GridFunction``.

Two habits of ``Draw`` come along with it, and they are worth knowing rather than quietly
correcting.  A ``CoefficientFunction`` has no mesh of its own, so the mesh or region goes in
positionally and the handler raises ``Cannot draw CoefficientFunction without mesh``
(``ngsolve/webgui.py:119``) when it was left out.  And the keyword meant to pick which part of
a value to colour is spelled ``eval_`` where it is looked up and ``eval`` where it is read
(``netgen/webgui.py:564-573``), so neither spelling arrives anywhere; ngsplot asks the renderer
directly, with ``function_mode``.
"""

__all__ = ["render_data"]


def render_data(obj, *args, **kwargs):
    """The render data ``Draw`` would send for ``obj``, built without a widget or a browser.

    The keywords are the keywords of ``Draw`` -- ``order``, ``draw_vol``, ``draw_surf``,
    ``deformation``, ``scale``, ``min``, ``max``, ``autoscale``, ``vectors``, ``objects``,
    ``euler_angles``, ``center``, ``radius``, ``settings`` -- and they land where the page reads
    them, so ``scene.settings_of`` and ``scene.view_range`` say of the result what the live
    viewer would have done with it.  ``show`` and ``filename`` are not among them: this draws
    nothing, and says so rather than ignoring them.

    No widget is built, which is the point of not calling ``Draw``: ``WebGLScene.__init__``
    constructs a ``WebGuiWidget`` before it looks at ``show`` (``netgen/webgui.py:438``), and
    with ``anywidget`` installed that is an ``ipywidgets`` widget, which in a kernel opens a comm
    and ships the whole payload to the front end -- about 100 KB for a few hundred P3 elements
    -- and files itself in the registry the notebook's widget state is saved from.  Closing it
    afterwards stops the registry growing but not the traffic, and one reported live view still
    left 44 MB of widget state in its executed notebook.  A comm also belongs to the kernel's
    main thread, so a payload that opened one could not be built anywhere else.  The scene's
    ``GetData`` needs none of it: it reads the object, the positional arguments and the keywords
    off the scene, so the scene is made without its constructor and asked for the dict.  A
    ``netgen`` too old to name its defaults in ``_get_draw_default_args`` goes through ``Draw``
    instead, and the widget it builds is closed once the payload is read.
    """
    import ngsolve.webgui  # noqa: F401  registers the draw types of ngsolve's own objects
    from netgen import webgui

    refused = sorted({"show", "filename"} & set(kwargs))
    if refused:
        raise TypeError("render_data draws nothing, so it takes no " + " or ".join(refused))
    defaults = getattr(webgui, "_get_draw_default_args", None)
    if defaults is None:
        scene = webgui.Draw(obj, *args, show=False, **kwargs)
        value = scene.widget.value
        close = getattr(scene.widget, "close", None)
        if close is not None:  # the stand-in widget of a netgen without anywidget has none
            close()
        return value

    # what Draw does before it builds the scene: its defaults, overridden by what was asked
    options = defaults()
    options.update(kwargs)
    scene = webgui.WebGLScene.__new__(webgui.WebGLScene)
    scene.obj, scene.args, scene.kwargs = obj, args, options
    scene.encoding = "b64"  # what WebGLScene.__init__ sets, and what the page decodes
    scene.widget = None
    return scene.GetData()
