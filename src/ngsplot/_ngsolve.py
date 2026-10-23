"""The seam to ngsolve: every import of ngsolve this package makes lives in this file.

``Draw`` is not a plotting call.  It is the constructor of a widget that, on its way to a
browser, assembles a dict of arrays -- the mesh as Bezier patches, the samples of the
function on them, the range, the flags, the camera -- and hands that dict to the page.  That
dict is all ngsplot wants, so the seam assembles it the same way and stops there: ``Draw``
builds a ``WebGLScene``, which dispatches on the type of the object to whichever draw type was
registered for it, and the payload the widget is handed is read back out instead of being sent
anywhere.  No widget is displayed and no browser is asked.  Registration is the part worth
remembering: ``ngsolve.webgui`` registers the handlers for ``ngsolve.Mesh``,
``CoefficientFunction`` and ``GridFunction``, and merely importing it is what makes them known
-- which is why a ``Draw(gfu)`` written without it complains ``Cannot draw object of type
GridFunction``.

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
    nothing but a picture somewhere else.
    """
    import ngsolve.webgui  # noqa: F401  registers the draw types of ngsolve's own objects
    from netgen.webgui import Draw

    scene = Draw(obj, *args, show=False, **kwargs)
    # the payload the widget was handed, so that the arrays are assembled once and not twice
    return scene.widget.value
