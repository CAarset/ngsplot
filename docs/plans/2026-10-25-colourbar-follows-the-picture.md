# Colourbar follows the picture — Implementation Plan

**Goal:** Make an ngsplot colourbar keep its geometry across repaints and size itself to the
picture it keys, so a repainted panel stops shrinking and a bar stops standing taller than the
picture.

**Architecture:** All four changes live in `src/ngsplot/pyplot.py`, in the ~40 lines that build
and retire a bar. A bar is anchored to its mappable (so matplotlib's own `remove()` restores the
cell it carved), one bar stands per axis and is repainted in place through
`Colorbar.update_normal`, and the room it is given is derived from the letterboxed picture
instead of the grid cell it shares with it.

**Tech stack:** matplotlib (3.11.2 in `.venv`), NumPy, pytest. No new dependencies.

**Spec:** `docs/SPEC.md` colourbar paragraphs (`:103-117`), amended by this plan; the two defects
as reported by the MRAS agent (shrinking panels, oversized bars) are the requirement.

## Global Constraints

- Output stays bit-identical to `tests/gt/`: `tests/test_parity.py` is the guard and must stay
  green untouched. The raster is the contract; only the figure around it moves.
- The picture is never stretched: it keeps the shape it was painted in and gives room away rather
  than taking any (`src/ngsplot/pyplot.py:1-49`).
- Nothing is registered in `matplotlib.colormaps`; the bar is driven by a private `ScalarMappable`
  over a `ListedColormap` of the bands the picture actually used.
- `Plot` keeps its promise that "titles, labels and colourbars are left to whoever made them" — a
  caller's geometry keywords win over anything derived here.
- Prose is British ("colourbar") in docs and docstrings; the API spelling stays `colorbar`.
  Docstrings say *why* and cite `docs/SPEC.md`, as the existing ones do.
- Suite: `timeout 1500 nice -n 10 .venv/bin/python -m pytest tests/ -q -p no:randomly` — baseline
  **427 passed, 1 skipped** in ~107 s. Never compare timings across runs.
- Measure any figure geometry **after** `figure.canvas.draw()`; positions read before a layout
  pass are wrong (a spike read 0.267 in for a bar that settles at 0.075 in).
- Python floor `>=3.10`; matplotlib has no floor in `pyproject.toml:11`.

## Review Focus

These are the cases no task's tests pin completely; expect a reasonable person to want exactly the
behaviour named.

1. **A figure resized between repaints** — the cell aspect changes, so the room that fitted no
   longer fits. Expect the bar to be given the new room, once, and never to compound across frames.
2. **A caller who removes the bar we handed back** and then repaints with `colorbar=True` — expect
   a fresh bar, the room taken once, and no `KeyError`: `Colorbar.remove()` on an already-removed
   bar raises (`fig.delaxes` → `KeyError: <Axes: label='<colorbar>'>`, measured), so a bar must
   never be removed twice.
3. **A caller who names geometry** (`shrink=`, `fraction=`, `aspect=`, `orientation="horizontal"`,
   `cax=`) — expect their number untouched; the derivation fills only what they left out. A
   horizontal bar gets the derived `shrink` and `fraction` but keeps matplotlib's own `aspect`.
4. **A mesh-only picture in a repaint loop** — still `ValueError` mentioning the mesh, and the
   picture loses no room on the way.
5. **`savefig` of a frame mid-animation** — the pixels of the picture unchanged; a bar is not a
   filter on the raster.

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `src/ngsplot/pyplot.py` | the pyplot layer: what a picture is and what stands beside it | the only code file that changes: `BAR_SHARE`, `_bar_geometry`, `_standing`, `_same_ask`, `colorbar`, `_retire`, `Plot` |
| `tests/test_pyplot.py` | the colourbar contract | six new tests after `:248`, in the file's existing idiom (`gtdata.render_data_of`, `pytest.approx`, docstring states why) |
| `docs/SPEC.md` | what every pixel and every knob is doing | colourbar paragraphs `:103-117` amended |
| `README.md` | the two-minute tour | one clause at `:38-46` that the bar is sized to the picture and stands through a repaint |
| `examples/ngsplot.ipynb` | the demo, outputs committed (26 PNGs) | re-executed once the bar looks right |
| `pyproject.toml` | floors | unchanged: the mechanism is as old as `use_gridspec` colourbars, so no newer matplotlib is asked for (see Task 3) |
| `/home/aarset/.dsh/skills/ngsolve/references/ngsolve.md` | the conventions reference (outside the workspace: needs an approved wider-access write) | the colourbar/repaint convention recorded; mirror re-synced to `.scratch/ngsolve-skill/` |

Bookkeeping today: `_bars` is an `Axes -> [Colorbar]` list written at `pyplot.py:235` and read only
by `_retire` (`:310-318`) — verified by grep, so reshaping its value touches two places.

## Validated numbers (one run, `.venv`, matplotlib 3.11.2)

Reproduced the report exactly, then the rule.  The mpl-defaults column is the old
behaviour, reproduced in the same run by naming the three numbers matplotlib uses when nobody names
them (`fraction=0.15, shrink=1.0, aspect=20`); the picture sizes are the letterbox the axes shows,
read after a draw:

| path | mpl defaults | derived rule |
| --- | --- | --- |
| bare axes, 3:2 raster (our default figure) | bar **1.25×** the picture tall, 4.2% of it wide | bar **1.00×** tall, 3.3% wide; picture 7.20×4.80 → 7.97×5.32 in |
| 2×3 panel, 15.6×8.6 in figure (the MRAS live view) | bar **1.59×** the picture tall, 5.3% wide | bar **1.00×** tall, 3.3% wide; picture 2.84×1.90 → 3.15×2.10 in |
| horizontal bar | 112.5% of the picture wide | 100.1% |
| six repaints into one panel, one axes | anchor width 0.1824 → 0.1459 → 0.1167 → 0.0934 → 0.0747 → 0.0598 (×0.8/frame) | **0.1824 constant**, one `<colorbar>` axes throughout |

The persistent-bar mechanism itself was measured sound: reusing one `ScalarMappable` and calling
`bar.update_normal(mappable)` follows a changed band count (5 → 32 → 16), colormap and range, tick
labels and all.

Two mechanisms the plan leans on, both read out of the installed source: `make_axes_gridspec` sets
`shrink` to exactly the share of the *cell height* the bar spans (measured 0.6 × 2.45 = 1.47 in,
spot on) and leaves the cell's height alone, so `shrink` needs no iteration; and a bar's *width*
comes from `aspect` (`cax.set_box_aspect`) in a plain gridspec and from `fraction` under tight /
constrained layout — measured identical results in both regimes, which is why the rule caps
**both**.

## Task 1 — A removed colourbar gives its room back

`Colorbar.remove()` does `self._ax_remove(self.ax)` (the bar's axes does go away) and then
`try: ax = self.mappable.axes / except AttributeError: return` (`colorbar.py:1065`) before
restoring the parent's `SubplotSpec`. Our `ScalarMappable` has no `axes` attribute and no
`set_axes` in 3.11.2, so the restore is skipped silently and every repaint leaves 0.8 of the
leftover cell behind.

- [ ] Save this plan to `docs/plans/2026-10-25-colourbar-follows-the-picture.md`.
- [ ] Write the failing test in `tests/test_pyplot.py`, after
      `test_the_layer_registers_no_colormap_anywhere` (`:248`):

```python
def test_a_colourbar_removed_gives_the_room_it_took_back():
    """``Colorbar.remove`` restores the axis it carved off of its mappable's ``axes``; a private
    mappable has to answer for the picture it keys or the room is gone for good."""
    payload = gtdata.render_data_of("dims/surface_scalar")
    figure = mpl.figure(figsize=(8.0, 4.0))
    panel = figure.add_subplot(figure.add_gridspec(1, 2)[0, 0])
    figure.canvas.draw()
    room = panel.get_subplotspec().get_position(figure).width
    pyplot.colorbar(pyplot.Plot(payload, ax=panel)).remove()
    figure.canvas.draw()
    assert panel.get_subplotspec().get_position(figure).width == pytest.approx(room)
```

- [ ] Run it, confirm it fails with the width stuck at 0.8 of `room`:
      `timeout 300 nice -n 10 .venv/bin/python -m pytest tests/test_pyplot.py -q -p no:randomly -k removed_gives`.
- [ ] In `colorbar`, one line after `mappable.set_array([])` (`pyplot.py:232-234`):

```python
    mappable.axes = image.axes    # Colorbar.remove reads the axis to restore off the mappable
```

- [ ] Re-run: green. Full file: `.venv/bin/python -m pytest tests/test_pyplot.py -q -p no:randomly`.
- [ ] Commit: `git add -A && git commit` — subject along "A colourbar hands its mappable the axis
      it keys, which is the only way matplotlib will give the room back", body naming
      `colorbar.py:1065`, the measured 0.8-of-cell leftover, and
      `test_a_colourbar_removed_gives_the_room_it_took_back`.

## Task 2 — One colourbar per axis, repainted where it stands

Carving and restoring the cell on every frame is churn even when it works: it reallocates an `Axes`
per frame (the same class of leak as the widget one closed in `37715f6`) and it re-enters
`gridspec` arithmetic every repaint. The bar stands and is retargeted instead.

- [ ] Failing tests, added after Task 1's:

```python
def test_repainting_a_colourbar_keeps_the_room_it_took_the_first_time():
    """A time loop repaints one panel; a bar that carves the cell again each frame is done to
    the picture in eighths, which is what an animation of a run was turning into."""
    payload = gtdata.render_data_of("dims/surface_scalar")
    figure = mpl.figure(figsize=(15.6, 8.6))
    panel = figure.add_subplot(figure.add_gridspec(2, 3)[0, 0])
    room = None
    for _ in range(6):
        pyplot.Plot(payload, ax=panel, colorbar=True)
        figure.canvas.draw()
        width = panel.get_subplotspec().get_position(figure).width
        room = width if room is None else room
        assert width == pytest.approx(room)
        assert len([bar for bar in figure.axes if bar.get_label() == "<colorbar>"]) == 1
    assert len(figure.axes) == 2


def test_a_repainted_colourbar_is_the_bar_of_the_picture_it_was_repainted_with():
    """The bar that stands is repainted rather than replaced, so it has to carry the ramp, the
    band count and the range of the newest picture beside it, not the ones it was built with."""
    payload = gtdata.render_data_of("dims/surface_scalar")
    figure = mpl.figure(figsize=(8.0, 4.0))
    panel = figure.add_subplot(figure.add_gridspec(1, 2)[0, 0])
    bar = pyplot.colorbar(pyplot.Plot(payload, ax=panel, ncolors=5))
    assert bar.cmap.N == 5
    pyplot.Plot(payload, ax=panel, colorbar=True, cmap="viridis", vmin=-0.5, vmax=2.0, ncolors=32)
    assert bar.cmap.N == 32
    assert (bar.norm.vmin, bar.norm.vmax) == pytest.approx((-0.5, 2.0))
    assert len(figure.axes) == 2


def test_a_picture_that_stops_asking_for_a_colourbar_gives_the_room_back():
    """A bar stands while the picture next to it wants one, and no longer than that."""
    payload = gtdata.render_data_of("dims/surface_scalar")
    figure = mpl.figure(figsize=(8.0, 4.0))
    panel = figure.add_subplot(figure.add_gridspec(1, 2)[0, 0])
    figure.canvas.draw()
    room = panel.get_subplotspec().get_position(figure).width
    pyplot.Plot(payload, ax=panel, colorbar=True)
    pyplot.Plot(payload, ax=panel)
    figure.canvas.draw()
    assert panel.get_subplotspec().get_position(figure).width == pytest.approx(room)
    assert not [bar for bar in figure.axes if bar.get_label() == "<colorbar>"]
```

- [ ] Run them: the first two fail (the anchor width walks 0.1824 → 0.0598; `bar.cmap.N` stays 5
      because the object is removed and a new bar built), the third passes already thanks to
      Task 1 — keep it as the guard that a dropped bar is a *true* restore.
- [ ] Reshape the bookkeeping. `_bars` value becomes one record, not a list; the comment at `:85`
      reads `# Axes -> the colourbar standing next to it, and what it was built from`.
- [ ] Two helpers beside `_retire`. `_standing` is needed because `bar.ax.get_figure()` still
      returns the figure after `fig.delaxes` (measured) — only membership in `figure.axes` says a
      bar is still installed; and a removed bar must never be removed again (`KeyError`):

```python
def _standing(bar):
    """Is the bar still installed in a figure?  Someone may have taken it away behind us."""
    figure = bar.ax.get_figure()
    return figure is not None and bar.ax in figure.axes


def _same_ask(had, wants):
    """Did the bar standing get asked for in the same words as the next one would be?  Told by
    what the keywords spell, since a bar built with ticks or a formatter cannot be talked into
    another one after the fact; a differently asked-for bar is built again."""
    return repr(had) == repr(wants)
```

- [ ] `colorbar` becomes: derive the palette, the `ListedColormap` and the `BoundaryNorm` as now;
      ask whether the standing bar will do; if so push the ramp and the scale into *its own*
      mappable and retarget it; otherwise retire it and build a new one. The reuse branch is
      matplotlib's own animation path — `update_normal` re-reads `cmap` and `norm`, calls
      `_reset_locator_formatter_scale()` when the norm differs and redraws (`colorbar.py:505`):

```python
    standing = _bars.get(image.axes)
    if (standing is not None and _standing(standing["bar"])
            and _same_ask(standing["kwargs"], kwargs)):
        mappable = standing["mappable"]
        mappable.set_cmap(cmap)
        mappable.set_norm(norm)
        standing["bar"].update_normal(mappable)
        return standing["bar"]
    if standing is not None and _standing(standing["bar"]):
        standing["bar"].remove()        # gives the room back before the new bar takes it
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    mappable.axes = image.axes          # (Task 1)
    bar = image.axes.get_figure().colorbar(mappable, ax=image.axes, **kwargs)
    _bars[image.axes] = {"bar": bar, "mappable": mappable, "kwargs": kwargs}
    return bar
```

- [ ] `_retire(axes, keep_bar=False)`: pop the record and remove the bar only when `keep_bar` is
      false. `Plot` passes `_retire(axes, keep_bar=bool(colorbar))` and still calls
      `_colorbar(image)` after the new picture is laid down — inside `Plot` the `colorbar` keyword
      shadows the function, which is why the `_colorbar = colorbar` alias exists.
- [ ] Run the three tests, then the file, then the suite. Expect 430 passed, 1 skipped.
- [ ] Commit: subject along "A repaint retargets the colourbar that stands instead of building
      another one", body naming `update_normal`, the 0.1824 → 0.0598 walk against the constant
      now, the `get_figure()` lie that `_standing` works around, and the three test names.

## Task 3 — The bar is sized to the picture, not to the cell

`Plot` passes no geometry keywords, so a bar is sized by the cell: `shrink=1.0` spans the whole
cell while `imshow` letterboxes the raster to its own aspect inside it, and `aspect=20` decides the
width from that height. Hence a bar 1.6× the picture's height. Two rules, both caps, so the common
figure whose picture fills its box keeps looking exactly as it does now: **the bar is as long as
the picture is drawn, and at most 8% of the picture's own width across** (the reporter's number for
the second one; the first is the defect).

- [x] Failing test:

```python
def test_a_colourbar_is_as_tall_as_the_picture_and_no_wider_than_a_share_of_it():
    """A bar sized by the cell stands over a picture that kept the shape it was painted in.
    It is a key to that picture, so it is measured against it, cell aside."""
    payload = gtdata.render_data_of("dims/surface_scalar")
    figure = mpl.figure(figsize=(15.6, 8.6))
    panel = figure.add_subplot(figure.add_gridspec(2, 3)[0, 0])
    image = pyplot.Plot(payload, ax=panel, colorbar=True)
    figure.canvas.draw()
    picture = image.get_window_extent().transformed(figure.transFigure.inverted())
    bar = [axes for axes in figure.axes if axes.get_label() == "<colorbar>"][0].get_position()
    assert bar.height == pytest.approx(picture.height, rel=0.02)
    assert bar.width <= pyplot.BAR_SHARE * picture.width
    assert bar.width >= picture.width / 40      # thinner than that and it is a line, not a key


def test_the_room_named_for_a_colourbar_is_the_room_it_gets():
    """The layer sizes a bar nobody named room for; a caller who names it is not overruled."""
    payload = gtdata.render_data_of("dims/surface_scalar")
    figure = mpl.figure(figsize=(8.0, 4.0))
    panel = figure.add_subplot(figure.add_gridspec(1, 2)[0, 0])
    bar = pyplot.colorbar(pyplot.Plot(payload, ax=panel), shrink=1.0)
    figure.canvas.draw()
    assert bar.ax.get_position().height == pytest.approx(
        panel.get_subplotspec().get_position(figure).height, rel=0.02)
```

Run: the first fails at `bar.height/picture.height == 1.59`; the second passes and stays as the
guard on the derivation.

- [x] Add the constant beside `DPI` (`:66-69`) and the geometry function at the file's foot:

```python
#: the widest a colourbar gets, as a share of the width of the picture it keys
BAR_SHARE = 0.08
```

```python
def _bar_geometry(image, kwargs):
    """Room for a bar beside a picture, taken out of the axis the picture sits in.

    A bar is sized by the cell it shares with the picture, but the picture keeps the shape it was
    painted in and gives the rest of the cell up, so cell and picture disagree whenever their
    shapes do: the bar then stands over the picture and takes room the picture could have used.
    The three numbers matplotlib sizes a bar with are ``shrink``, the share of the cell the bar
    spans, ``fraction``, the share of the axis it gives up, and ``aspect``, the bar's own shape.
    Each is left to whoever named it, and a bar with a ``cax`` of its own is nobody's cell.
    """
    out = dict(kwargs)
    figure = image.axes.get_figure()
    if figure is None or "cax" in out:
        return out
    inches = figure.get_size_inches()
    spec = image.axes.get_subplotspec()
    cell = (spec.get_position(figure) if spec is not None
            else image.axes.get_position(original=True))
    horizontal = out.get("orientation", "vertical") == "horizontal"
    pad = float(out.get("pad", 0.05))
    x0, x1, y0, y1 = image.get_extent()          # the shape imshow letterboxes to, not the array's
    shape = (x1 - x0) / (y1 - y0) if not horizontal else (y1 - y0) / (x1 - x0)
    along = cell.height * inches[1] if not horizontal else cell.width * inches[0]
    across = cell.width * inches[0] if not horizontal else cell.height * inches[1]
    if not (math.isfinite(shape) and shape > 0 and along > 0 and across > 0):
        return out                               # nothing to size the bar against

    def picture(fraction):                       # what the axis shows of the picture
        fit = min(along, across * (1 - fraction - pad) / shape)
        return fit, shape * fit                  # along the bar, across it

    fraction = float(out.setdefault(
        "fraction", min(0.15, BAR_SHARE * picture(0.15)[1] / across)))
    fit_along, fit_across = picture(fraction)
    shrink = float(out.setdefault("shrink", fit_along / along))
    if not horizontal and "aspect" not in out:
        # matplotlib gives a bar a width of its length over ``aspect``; ask for the cap, but never
        # a fatter bar than the 20 it would have used
        out["aspect"] = max(20.0, shrink * along / (BAR_SHARE * fit_across))
    return out
```

`setdefault` is the whole override rule: a caller's number is read back and used to size the rest.
mpl accepts `fraction=0` and a `shrink` of `1e-9` without complaint (measured), so a degenerate
cell yields a degenerate bar rather than an exception, and the guard above covers the non-finite
cases.

- [x] `colorbar` derives once per call and reuses on a fit: hold `"geometry"` in the `_bars`
      record; compute `geometry = _bar_geometry(image, kwargs)`; the reuse branch additionally
      requires `_fits(standing["geometry"], geometry)`, and stores the geometry it settled on; the
      build branch passes `**geometry` to `fig.colorbar` and stores it:

```python
def _fits(had, wants, tolerance=0.01):
    """Does the room a bar was given still fit the picture beside it?  A resized figure wants
    other room, and a bar asked for differently is built again rather than argued with."""
    return all(math.isclose(float(had[name]), float(wants[name]), rel_tol=tolerance, abs_tol=1e-3)
               for name in ("fraction", "shrink", "aspect", "orientation")
               if name in had and name in wants)
```

An added `import math` at the head of the file. A repaint at a fixed size derives the same numbers,
so an animation never rebuilds; a resize re-carves from a cell that `remove()` restored (Task 1),
so it cannot compound.

- [x] Run the two tests, then the file, then the suite;
      `test_a_plot_can_ask_for_a_colourbar_and_makes_room_for_it` must still pass (`fraction`
      0.064 keeps `width < 1.0`).
- [x] Produce the contact sheet for review: a 2×3 figure at 15.6×8.6 in with
      `Plot(payload, ax=panel, colorbar=True)` in every panel, saved before the change (`git
      stash`) and after, to `.scratch/bar-before-after.png` (gitignored). Check the picture in each
      panel got *bigger* as well as the bar matching its height.
- [x] **No floor was raised**: `lib/matplotlib/colorbar.py` at the tags `v3.4.0`, `v3.5.0` and
      `v3.6.0` has `def update_normal(self, mappable)` and the `ax = self.mappable.axes` restore in
      `remove()` in all three, so the mechanism is as old as `use_gridspec` colourbars and nothing
      here needs a newer matplotlib than the layer already assumed.  `pyproject.toml:11` is left
      with no floor; the code was checked to raise no `DeprecationWarning` under matplotlib 3.11.
- [x] Commit: subject along "A colourbar is measured against the picture it keys rather than the
      cell it shares with it", body with the two measured ratios from the table above (both from
      one run), the aspect/fraction dual cap and why (width comes from `aspect` in a plain
      gridspec, from `fraction` under constrained layout — same result either way), and the two
      test names.

## Task 4 — The record, the demo, the reference

- [ ] `docs/SPEC.md:103-117`: keep the "a colourbar cannot be built on the picture" argument;
  replace the closing sentence of the geometry paragraph and the `:116-117` repaint paragraph so
  they state what is now true — the bar is anchored to its mappable, it stands through a repaint
  and is retargeted, the room it takes is `shrink`/`fraction`/`aspect` derived from the letterbox
  with `BAR_SHARE` as the cap, a caller's own numbers win, and a picture that stops asking for a
  bar gets its room back.
- [ ] `README.md:38-46`: one clause — a bar is sized to the picture and survives the repaints of a
  time loop.
- [ ] Re-execute the demo so its committed outputs show the new bar:
  `nice -n 10 .venv/bin/jupyter-execute --inplace examples/ngsplot.ipynb` (nbclient 0.11.0 is in
  `.venv`). If a cell errors or the kernel is missing, stop and report rather than committing
  half-updated outputs.
- [ ] Record the convention in `/home/aarset/.dsh/skills/ngsolve/references/ngsolve.md` (Pictures
  section: one bar per axis, `update_normal` to retarget, geometry derived from the letterbox,
  `mappable.axes` is what makes `remove()` restore) — outside the workspace, so the write needs an
  approved `danger-full-access` retry — and re-sync the mirror: `cp` the skill and reference into
  `.scratch/ngsolve-skill/`, then `diff -rq` the two trees for identical.
- [ ] Full suite one last time (expect 432 passed, 1 skipped) plus `git status` clean; then
  `present` the contact sheet.
- [ ] Commit: subject along "The record says what a colourbar is now measured against", body
  listing SPEC/README/notebook/reference.

## Out of scope

- No figure-resize callback: the geometry is re-derived at paint time, which covers a repaint
  after a resize and nothing between paints.
- No re-execution of the `tests/gt/` capture: parity is a guard here, not a target.
- MRAS's `f9f0dbb` external hand-back must be **dropped**, not left in: with one bar standing and
  the cell carved once, re-setting the panel's original `SubplotSpec` every frame re-widens the
  picture *underneath* the still-carved `cax`, so it overlaps instead of doing nothing. Their
  comment claiming the workaround degrades gracefully is wrong under this design and should be
  corrected when the hand-back goes.
