"""Render the pictures of the README into ``docs/images``, with ngsplot itself.

Every picture is laid on an axis exactly the size of its canvas, so matplotlib copies the
renderer's pixels instead of resampling them -- a one-pixel wireframe line survives -- and any
scaling is left to the browser showing the README.  The comparison with the browser viewer
reads a recorded screenshot of ``ngsolve.webgui.Draw`` from ``tests/gt``; everything else
needs ngsolve to build its mesh.

    .venv/bin/python tools/readme_images.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import ngsolve as ngs  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib import pyplot as mpl  # noqa: E402
from netgen.geom2d import SplineGeometry  # noqa: E402
from netgen.occ import Box, Cylinder, OCCGeometry, Pnt, Sphere, Z  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "tests")]

import gtdata  # noqa: E402
import ngsplot  # noqa: E402
from ngsplot import pyplot, scene  # noqa: E402

OUT = os.path.join(_ROOT, "docs", "images")
DPI = 100
INK = {"light": "#1f2328", "dark": "#e6edf3"}      # GitHub's own text colours
MUTED = {"light": "#59636e", "dark": "#9198a1"}


def figure(width, height, transparent=False):
    """A figure of ``width`` x ``height`` pixels at :data:`DPI`."""
    fig = mpl.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    if transparent:
        fig.patch.set_alpha(0.0)
    return fig


def axes_at(fig, x, y, width, height):
    """An axis covering pixels ``x .. x + width`` and ``y .. y + height``, rows from the top."""
    W, H = fig.get_size_inches() * fig.dpi
    return fig.add_axes([x / W, 1.0 - (y + height) / H, width / W, height / H])


def save(fig, name, **kwargs):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, **kwargs)
    mpl.close(fig)
    print(f"{path}: {os.path.getsize(path) / 1024:.0f} KB")


def channel(length=6.0, width=1.4, maxh=0.12):
    """A channel with three holes and a field rippling along it: the long domain of the banner."""
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (length, width))
    for cx, cy, r in ((1.2, 0.7, 0.22), (2.9, 0.45, 0.16), (4.4, 0.85, 0.2)):
        geo.AddCircle((cx, cy), r=r, leftdomain=0, rightdomain=1)
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(3)
    u = ngs.GridFunction(ngs.H1(mesh, order=3))
    u.Set(ngs.sin(2.2 * ngs.x - 1.5 * ngs.y) * ngs.cos(4 * ngs.y + 0.6 * ngs.x)
          * ngs.exp(-0.08 * ngs.x))
    return u


def banner(payload):
    """The wordmark beside a long domain, on a transparent page, for light and for dark."""
    view = ngsplot.frame(payload, fit="box", width=1040, height=None)
    for theme in ("light", "dark"):
        fig = figure(1600, view.height + 40, transparent=True)
        ax = axes_at(fig, 556, 20, view.width, view.height)
        canvas = scene.paint(payload, view=view, cmap="RdBu_r", ncolors=16, wireframe=False,
                             canvas=_clear_canvas(view))
        ax.imshow(canvas.to_uint8(), interpolation="nearest")
        ax.set_axis_off()
        fig.text(0.03, 0.58, "ngsplot", fontsize=76, fontweight="bold", color=INK[theme],
                 va="center")
        fig.text(0.032, 0.30, "NGSolve's Draw, as a matplotlib figure", fontsize=19,
                 color=MUTED[theme], va="center")
        save(fig, f"banner-{theme}.png", transparent=True)


def _clear_canvas(view):
    """A canvas whose page is see-through, so the banner sits on either theme."""
    from ngsplot import raster

    return raster.Canvas(view.width, view.height, background=(1.0, 1.0, 1.0, 0.0))


def draw_vs_plot(scene_name="square_rotated", margin=14):
    """The browser's picture and ngsplot's, from one recorded payload, cropped to the mesh."""
    payload = gtdata.render_data_of(scene_name)
    width, height = gtdata.framebuffer_of(scene_name)
    ours = scene.paint(payload, width=width, height=height).to_uint8()[..., :3]
    theirs = gtdata.png_of(scene_name)[:height, :width, :3]
    rows, cols = np.nonzero((ours < 250).any(axis=2))
    crop = (slice(rows.min() - margin, rows.max() + margin),
            slice(cols.min() - margin, cols.max() + margin))
    left, right = theirs[crop], ours[crop]
    h, w = left.shape[:2]
    gap, title = 40, 44
    fig = figure(2 * w + gap, h + title)
    for x, picture, label in ((0, left, "ngsolve.webgui.Draw  ·  browser"),
                              (w + gap, right, "ngsplot.Plot  ·  matplotlib")):
        ax = axes_at(fig, x, title, w, h)
        ax.imshow(picture, interpolation="nearest")
        ax.set_axis_off()
        fig.text((x + w / 2) / (2 * w + gap), 1 - 22 / (h + title), label, ha="center",
                 va="center", fontsize=13, family="monospace", color="#1f2328")
    save(fig, "draw-vs-plot.png")


def waves(maxh=0.08):
    """One oscillating field on the unit square, between -1 and 1."""
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=maxh))
    u = ngs.GridFunction(ngs.H1(mesh, order=3))
    r = ngs.sqrt((ngs.x - 0.3) ** 2 + (ngs.y - 0.6) ** 2)
    u.Set(0.5 * (ngs.sin(4 * np.pi * ngs.x) * ngs.sin(3 * np.pi * ngs.y) + ngs.cos(14 * r)))
    return u


def colormaps(payload, names=("ngsolve", "viridis", "RdBu_r", "twilight")):
    """The same field in four colour maps, each over a bar of its sixteen bands."""
    view = ngsplot.frame(payload, fit="box", width=300, height=None)
    gap, title, bar = 24, 34, 64
    fig = figure(len(names) * view.width + (len(names) - 1) * gap, title + view.height + bar)
    for k, name in enumerate(names):
        x = k * (view.width + gap)
        image = pyplot.Plot(payload, ax=axes_at(fig, x, title, view.width, view.height),
                            view=view, cmap=name, vmin=-1, vmax=1, ncolors=16, wireframe=False)
        cax = axes_at(fig, x + 20, title + view.height + 10, view.width - 40, 12)
        pyplot.colorbar(image, cax=cax, orientation="horizontal")
        fig.text((x + view.width / 2) / fig.bbox.width, 1 - 16 / fig.bbox.height,
                 f'cmap="{name}"', ha="center", va="center", fontsize=12, family="monospace")
    save(fig, "colormaps.png")


def solid(width=520, bar=110):
    """A machined block, P2 on curved tetrahedra: a hole through it and a dimple in its top."""
    shape = (Box(Pnt(0, 0, 0), Pnt(2, 1, 1)) - Cylinder(Pnt(0.6, 0.5, 0), Z, r=0.3, h=1)
             - Sphere(Pnt(1.5, 0.5, 1), 0.35))
    mesh = ngs.Mesh(OCCGeometry(shape).GenerateMesh(maxh=0.15))
    mesh.Curve(2)
    u = ngs.GridFunction(ngs.H1(mesh, order=2))
    u.Set(ngs.sin(2 * ngs.x) * ngs.cos(2 * ngs.y) + ngs.z)
    payload = ngsplot.render_data(u)
    view = ngsplot.frame(payload, fit="box", width=width, height=None, elev=40, azim=-30)
    fig = figure(view.width + bar, view.height)
    image = pyplot.Plot(payload, ax=axes_at(fig, 0, 0, view.width, view.height), view=view,
                        ncolors=12)
    pyplot.colorbar(image, cax=axes_at(fig, view.width + 10, 30, 14, view.height - 60))
    save(fig, "solid.png")


def long_domain(payload):
    """The channel framed by its box, with its bar underneath where a long picture wants it."""
    view = ngsplot.frame(payload, fit="box", width=880, height=None)
    bar = 56
    fig = figure(view.width, view.height + bar)
    image = pyplot.Plot(payload, ax=axes_at(fig, 0, 0, view.width, view.height), view=view,
                        ncolors=12)
    pyplot.colorbar(image, cax=axes_at(fig, 140, view.height + 6, view.width - 280, 14),
                    orientation="horizontal")
    save(fig, "channel.png")


def time_loop(frames=24, side=320):
    """A wave crossing a square, one Plot per frame, as a GIF."""
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.12))
    u = ngs.GridFunction(ngs.H1(mesh, order=3))
    u.Set(ngs.sin(np.pi * ngs.x) * ngs.sin(np.pi * ngs.y))
    view = ngsplot.frame(ngsplot.render_data(u), fit="box", width=side, height=None)
    fig = figure(view.width, view.height)
    ax = axes_at(fig, 0, 0, view.width, view.height)

    def frame(step):
        t = 2 * np.pi * step / frames
        u.Set(ngs.sin(np.pi * ngs.x) * ngs.sin(np.pi * ngs.y)
              * ngs.cos(3 * np.pi * (ngs.x + ngs.y) - t))
        pyplot.Plot(u, ax=ax, view=view, vmin=-1, vmax=1, cmap="RdBu_r", ncolors=16)

    movie = animation.FuncAnimation(fig, frame, frames=frames, init_func=lambda: None)
    path = os.path.join(OUT, "time-loop.gif")
    movie.save(path, writer=animation.PillowWriter(fps=12), dpi=DPI)
    mpl.close(fig)
    print(f"{path}: {os.path.getsize(path) / 1024:.0f} KB")


def main():
    os.makedirs(OUT, exist_ok=True)
    long_payload = ngsplot.render_data(channel())
    banner(long_payload)
    draw_vs_plot()
    colormaps(ngsplot.render_data(waves()))
    solid()
    long_domain(long_payload)
    time_loop()


if __name__ == "__main__":
    main()
