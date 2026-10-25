# ngsplot

Matplotlib plots that look exactly like NGSolve's `Draw`.

NGSolve's own `Draw` renders through a WebGL pipeline behind a live GUI control
panel: correct and familiar, but awkward to drive from a script or a notebook.
This package renders the same picture from the same data with matplotlib, so
colour bars, `savefig`, subplots and animation come for free.

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install -e .            # add [test] for pytest, [gt] for the ground truth harness
```

## Usage

```python
from ngsplot import Plot          # the counterpart of ngsolve.webgui.Draw
from ngsplot import pyplot        # the figure around it: colorbar, savefig, gca, cla, close

Plot(gfu, colorbar=True)                        # a GridFunction, as Draw draws it
Plot(mesh)                                      # the mesh alone, as Draw(mesh) draws it
Plot(cf, mesh=mesh)                             # a CoefficientFunction needs its mesh
Plot(mesh=mesh)                                 # spelled either way, like Draw
Plot(gfu, vmin=0, vmax=1, ncolors=16)           # the panel's own min, max and bands
Plot(gfu, cmap="viridis", wireframe=False)      # any matplotlib name, or a webgui ramp
Plot(gfu, line_width=1, edges=False)            # the lines, thinner or gone
Plot(gfu, elev=30, azim=-60)                    # the view, in matplotlib's own three words
Plot(gfu, deformation=0.7)                      # the deformation slider, by amount
Plot(gfc, eval="imag")                          # which part of a complex value to colour
pyplot.savefig("frame.png")
```

A picture is one `AxesImage` holding the renderer's own array, laid on an axis that covers the
figure, so everything matplotlib does from there still works: `Plot(gfu, ax=ax)` into anybody's
`subplots` or `gridspec`, `pyplot.colorbar` on the last picture or a given one, `cla()` and
`close()`, and a savefig at the figure's own dpi that reproduces the rendered array byte for
byte, which is what makes a loop of frames an animation. A colourbar is measured against the
picture it keys and stands through a repaint, so a panel that redraws every frame keeps the width
it was given.

```python
from matplotlib import pyplot as mpl

fig, axes = mpl.subplots(1, 2, figsize=(9, 3.4))
pyplot.Plot(u, ax=axes[0], colorbar=True)
pyplot.Plot(v, ax=axes[1], cmap="viridis", ncolors=16)
axes[0].set_title("u")
```

Keywords are sorted by who reads them: the ones `ngsplot.scene.paint` and `ngsplot.camera.Camera`
take go to the painter, the rest go to `Draw`, so `order`, `draw_vol` and `settings` keep their
meaning. A payload built earlier by `ngsplot._ngsolve.render_data` may be plotted
directly, and refuses the keywords that would have shaped it instead of ignoring them.

A point of view is `elev`, `azim` and `roll`, in degrees: the three words of
`Axes3D.view_init`, on a camera that stays where the page put it. Left out, they are the view the
page opens with, straight at `xy`, which is what a flat picture wants; `elev=30, azim=-60` is the
angle matplotlib's own 3D plots open with. The `euler_angles` triple a panel drag leaves behind is
still what a payload carries and still what the painter reads, so a saved page replays with the
camera it was saved with — while `Plot` refuses the name, rather than keep two spellings of one
knob.

[examples/ngsplot.ipynb](examples/ngsplot.ipynb) runs all of it against a 1D, a 2D and a 3D
finite element space: the flat line of a function on a mesh of dimension 1 and the same curve as a
circle in three dimensions, the surface case with its colourbar, the knobs side by side, a
`CoefficientFunction` on a named mesh, a vector and a complex field, `L2(order=0)`, a
`NumberSpace`, a solid seen from three angles, and a time loop that animates in the notebook
itself -- an `animation.FuncAnimation` over `Plot` calls, shown with `to_jshtml`, no frames on
disk. It executes end to end; only the meshes need ngsolve.

## Status

Every stage of the pipeline is in place and covered by tests that transcribe the corresponding
GLSL or TypeScript, or compare against recordings of the real page: the colour ramp and band
quantisation and the label formatting of its colourbar (`colormap`, `labels`), the Bezier
geometry and value range a payload carries (`geometry`, `bezier`), the Phong shading
(`lighting`), the camera whose object matrix takes every drag (`camera`), the software raster
with the same 4x MSAA, mitre strip and depth bias as the page's thick lines (`raster`), the
settings and range rules of the panel (`scene`), the payload `Draw` would have sent
(`_ngsolve`) and the matplotlib layer above (`pyplot`).

`docs/SPEC.md` records the reverse-engineered visual, and `tools/capture_gt.py` and
`tools/capture_dims.py` regenerate `tests/gt/` from a headless chromium running the real
renderer. Two deviations from the page are deliberate and documented: the GUI panel and the
notebook widget are not part of the picture, and a function on a 1D mesh is drawn as the page
draws it -- a flat black line along the mesh -- rather than coloured along its values.

