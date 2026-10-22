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

## Status

The renderer is being built module by module against the webgui shaders. The
verified pieces are the colour ramp and band quantisation, the Phong shading,
the colour-bar label formatting, the Bezier triangle evaluation and the decoding
of the render payload (`ngsplot.geometry`: surface, wireframe and boundary
control points, the vector attributes, and the value range ngsolve fits the
colour scale from), each covered by tests that transcribe the corresponding GLSL
or TypeScript, or compare against live `BuildRenderData` output.

`docs/SPEC.md` records the reverse-engineered visual, and `tools/capture_gt.py`
regenerates `tests/gt/` from a headless chromium running the real renderer.
