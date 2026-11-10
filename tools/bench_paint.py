"""Measure what one paint costs, and what it costs per element.

The price of a picture grows with the mesh it is handed, and the constant of that
proportionality is what this tool exists to measure.  Measured once per sub-triangle of
every element in Python, it came out at roughly 17 ms an element: a mesh of 5000 elements
took a minute and a half to paint, where the browser the picture is copied from does the
same picture inside a frame.  This is how a change to the sweep is measured, before and
after.  It asserts nothing, so that it runs anywhere, and prints one row per case:

    case              insts   ms      ms/inst    ms/Mpx

``insts`` is the number of instances the pass draws -- elements for the surface, edges for
the boundary and the wireframe -- because that is what the cost is proportional to.  Any
case may be given as ``case/surface``, ``case/edges`` or ``case/wireframe`` to price one
pass of :func:`ngsplot.scene.paint` alone, which is where a regression is looked for first;
the bare cases paint all three, and are the default.

A machine does not run at one speed, and a whole run drifts together with the load on it, so
two timings taken in two runs say nothing about which setting is faster.  For a comparison,
name the settings to compare and the tool paints every case with each of them in turn, in the
same process, over several rounds, keeping the fastest of each:

    .venv/bin/python tools/bench_paint.py --tune before=_STACK_WINDOW=32 \\
        --tune after=_STACK_WINDOW=64 [case ...]

then prints one column of milliseconds per setting and the ratio between the first and the
last.  The names are attributes of :mod:`ngsplot.raster`, which are read from the module when
a paint runs, so a setting holds for that paint alone.

    .venv/bin/python tools/bench_paint.py [--width 900] [--height 600] [--rounds 3]
        [--tune name=key=value,...] [case ...]
"""

import argparse
import contextlib
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "tools")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import ngsolve as ngs  # noqa: E402
from netgen.geom2d import unit_square  # noqa: E402

from ngsplot import _ngsolve, geometry, raster, scene  # noqa: E402


def square(maxh):
    """A 2d mesh of the unit square with a scalar field on P2."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=maxh))
    gf = ngs.GridFunction(ngs.H1(mesh, order=2))
    gf.Set(ngs.x * ngs.y)
    return gf


def cube(maxh):
    """A 3d mesh of the unit cube, painted as the volume the page is shown with."""
    mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=maxh))
    gf = ngs.GridFunction(ngs.H1(mesh, order=2))
    gf.Set(ngs.x * ngs.y)
    return gf


BASE_PASS = ("surface", "edges", "wireframe")

BASE = {
    "square-0.4": lambda: square(0.4),
    "square-0.2": lambda: square(0.2),
    "square-0.1": lambda: square(0.1),
    "square-0.05": lambda: square(0.05),
    "cube-0.5": lambda: cube(0.5),
    "cube-0.3": lambda: cube(0.3),
}

#: one pass of a scene, for the case named ``case/pass``; the passes of a split left out are
#: the ones :func:`ngsplot.scene.paint` would otherwise run too
PASSES = {f"{name}/{which}": {other: other == which for other in BASE_PASS}
          for name in BASE for which in BASE_PASS}

CASES = {**BASE, **{case: BASE[case.split("/")[0]] for case in PASSES}}


def instances(payload, which):
    """How many instances one pass of this payload holds, 0 when it paints nothing.

    The keys and the calls are the ones :func:`ngsplot.scene.paint` uses, so the count is
    the one the pass really draws.
    """
    if which == "surface":
        return geometry.trig_points(payload)[0].shape[0]
    if not payload.get("edges" if which == "edges" else "Bezier_points"):
        return 0
    points = (geometry.edge_points if which == "edges"
              else geometry.wireframe_points)(payload)
    return points[0].shape[0]


def prepare(name):
    """The payload of a case, built once, with the passes it asks to be painted with."""
    payload = _ngsolve.render_data(CASES[name](), **({"draw_vol": True} if "cube" in name else {}))
    passes = PASSES.get(name, {})
    count = sum(instances(payload, which)
                for which in BASE_PASS if passes.get(which, True))
    return payload, passes, count


def paint(payload, passes, width, height):
    """Seconds of one whole paint, the canvas and the camera outside the clock."""
    start = time.perf_counter()
    scene.paint(payload, width=width, height=height, **passes)
    return time.perf_counter() - start


@contextlib.contextmanager
def tuned(settings):
    """Hold the module attributes of one setting while the block runs, then put back."""
    saved = {key: getattr(raster, key) for key in settings}
    try:
        for key, value in settings.items():
            setattr(raster, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(raster, key, value)


def parse_tune(text):
    """A tune as a label and the settings it names.

    ``name`` ends at the first ``:`` or ``=``, either of which may separate it from the
    settings, so both forms of the example above parse; a tune that names no setting is an
    error rather than a second measurement of the run it was meant to differ from.
    """
    cut = min((at for at in (text.find(":"), text.find("=")) if at >= 0), default=-1)
    label, rest = (text[:cut], text[cut + 1:]) if cut >= 0 else (text, "")
    settings = {}
    for item in filter(None, rest.split(",")):
        key, sep, value = item.partition("=")
        if not sep or not hasattr(raster, key.strip()):
            raise SystemExit(f"{item!r} is not a name of a setting of ngsplot.raster "
                             f"with a value, as in _STACK_WINDOW=64")
        settings[key.strip()] = int(value)
    if not settings:
        raise SystemExit(f"--tune {text!r} names no setting, so it would measure the same "
                         f"painting twice")
    return label, settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--rounds", type=int, default=3,
                        help="paintings of a case per setting, the fastest kept")
    parser.add_argument("--tune", action="append", default=[], metavar="name:key=value,...",
                        help="a setting to price, repeated to compare settings in one run")
    parser.add_argument("cases", nargs="*", choices=[*CASES], help="default: all of them")
    args = parser.parse_args(argv)

    tunes = [parse_tune(text) for text in args.tune] or [("ms", {})]
    wanted = args.cases or list(BASE)
    mpx = args.width * args.height / 1e6
    print(f"canvas {args.width} x {args.height} ({mpx:.2f} Mpx, {len(wanted)} cases, "
          f"{args.rounds} rounds)\n")
    head = f"{'case':<24}{'insts':>7}" + "".join(f"{label:>12}" for label, _ in tunes)
    if len(tunes) > 1:
        head += f"{'gain':>8}"
    print(head + f"{'ms/inst':>9}{'ms/Mpx':>9}")
    for name in wanted:
        payload, passes, count = prepare(name)
        best = [None] * len(tunes)
        for _ in range(max(1, args.rounds)):
            for place, (_, settings) in enumerate(tunes):
                with tuned(settings):
                    seconds = paint(payload, passes, args.width, args.height)
                if best[place] is None or seconds < best[place]:
                    best[place] = seconds
        row = f"{name:<24}{count:>7}" + "".join(f"{1e3 * s:>12.0f}" for s in best)
        if len(tunes) > 1:
            row += f"{best[0] / best[-1]:>7.2f}x"
        print(row + f"{1e3 * best[-1] / max(count, 1):>9.2f}{1e3 * best[-1] / mpx:>9.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
