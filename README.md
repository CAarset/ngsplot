<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/banner-dark.png">
    <img src="docs/images/banner-light.png" alt="ngsplot: NGSolve's Draw, as a matplotlib figure" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/CAarset/ngsplot/actions/workflows/tests.yml"><img src="https://github.com/CAarset/ngsplot/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/NGSolve-6.2-0b7285" alt="NGSolve 6.2">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea043" alt="MIT licence"></a>
</p>

ngsplot draws [NGSolve](https://ngsolve.org) finite element functions as matplotlib images that look
the same as the output of `ngsolve.webgui.Draw`. It reimplements the webgui renderer in NumPy
(subdivision of curved elements, colour bands, lighting, antialiased lines and camera), so a plot
can go into subplots, get a colourbar, be saved with `savefig` or animated, without a browser.

<p align="center">
  <img src="docs/images/draw-vs-plot.png" alt="A quadratic field on a machined block, drawn by ngsolve.webgui.Draw in a browser and by ngsplot in matplotlib" width="100%">
  <br>
  <sub>Left: a screenshot of <code>ngsolve.webgui.Draw</code> in Chromium. Right: <code>ngsplot.Plot</code>
  with the same data. 96.6% of the pixels are identical. Most of the rest lie on the thin mesh
  lines, where one antialiasing sample falls the other way; away from the lines, 99.7% match.</sub>
</p>

## Installation

```bash
pip install "ngsplot[ngsolve] @ git+https://github.com/CAarset/ngsplot"
```

The `ngsolve` extra installs NGSolve. ngsplot itself only needs NumPy and matplotlib, which is
enough to plot data that was built earlier with `ngsplot.render_data`.

## Quick start

```python
import ngsolve as ngs
from ngsplot import Plot, pyplot

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.1))
gfu = ngs.GridFunction(ngs.H1(mesh, order=3))
gfu.Set(ngs.sin(4 * ngs.x) * ngs.cos(5 * ngs.y))

Plot(gfu, colorbar=True)            # what Draw(gfu) shows, as a matplotlib AxesImage
pyplot.savefig("gfu.png")           # saves the rendered pixels unchanged
```

`Plot` takes the same arguments as `Draw`, plus the settings of the viewer's control panel:

```python
Plot(gfu, ax=ax, cmap="viridis", ncolors=16, vmin=-1, vmax=1)   # draw into an existing axis
Plot(gfu, elev=30, azim=-60, wireframe=False, deformation=0.2)   # viewpoint and display options
Plot(gfu, subdomains=False)                                      # no thick lines between subdomains
Plot(gfc, eval="imag")                                           # plot the imaginary part
Plot(gfu, fit="box", width=900, height=None)                     # fit the canvas to a long domain
```

## Features

- Output matches the browser viewer pixel for pixel over most of the picture. The test suite
  compares it with recorded screenshots of the viewer.
- Each plot is a single matplotlib `AxesImage`, so `ax=`, subplots, titles, `colorbar`, `savefig`
  and `FuncAnimation` work as usual.
- Meshes, `GridFunction`s and `CoefficientFunction`s on 1D, 2D and 3D meshes, including vector and
  complex fields, `L2` and `NumberSpace`.
- `fit="box"` fits a long domain to the canvas, wide plots get a horizontal colourbar, and
  `ngsplot.frame` gives the canvas size before anything is drawn.
- `ngsplot.render_data` builds the plot data without creating a notebook widget, so it can run on
  any thread. Painting is plain NumPy and takes about 0.15 s for a 1500-element cubic field on a
  400 × 107 canvas.

## Gallery

<table>
  <tr>
    <td colspan="2" align="center">
      <img src="docs/images/colormaps.png" alt="One field in four colour maps">
      <br><sub>The same field in four matplotlib colour maps, split into the viewer's bands</sub>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="docs/images/channel.png" alt="A long channel framed by its box">
      <br><sub>A long channel with <code>fit="box"</code>; the colourbar goes underneath</sub>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="docs/images/time-loop.gif" alt="An animated wave crossing a square">
      <br><sub>One <code>Plot</code> call per frame of a <code>FuncAnimation</code></sub>
    </td>
  </tr>
</table>

The [demo notebook](examples/ngsplot.ipynb) covers 1D, 2D and 3D spaces, the plotting options and
an animation.

## How it works

`ngsplot.render_data` builds the same data that `Draw` sends to the browser. `ngsplot.scene.paint`
rasterises it in NumPy, following the viewer's shaders, and `Plot` shows the result on a matplotlib
axis. The viewer's control panel is not reproduced, and a function on a 1D mesh is drawn as a plain
black line, as the viewer draws it.

## Development

```bash
git clone https://github.com/CAarset/ngsplot && cd ngsplot
pip install -e ".[test]"
pytest
```

The tests compare ngsplot's output with screenshots of the real viewer stored in `tests/gt/`.
`tools/capture_gt.py` and `tools/capture_dims.py` record them again with headless Chromium
(`pip install -e ".[gt]"`), and `tools/readme_images.py` regenerates the images in this README.

## Licence

MIT, see [LICENSE](LICENSE). ngsplot is an independent reimplementation and is not affiliated
with the NGSolve project. NGSolve and Netgen are licensed under the LGPL-2.1.
