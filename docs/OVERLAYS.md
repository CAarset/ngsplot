# The overlays — colourbar, tick labels, axes triad

Companion to [SPEC.md](SPEC.md); it supersedes that file's **Overlays** stub and every
number in it (SPEC.md is left untouched here — §0 lists the corrections instead).
Everything here was read out of `webgui@0.2.39` (`scene.ts`, `colormap.ts`,
`axes.ts`, `label.ts`, `render_object.ts`, `utils.ts`, `camera.ts`, `gui.ts`, `styles.css`,
from `https://registry.npmjs.org/webgui/-/webgui-0.2.39.tgz`, line numbers refer to that
tarball's `src/`) and then **checked against the captured ground truth** in `tests/gt/`
(NGSolve 6.2.2608, headless chromium, 900×600 widget, `devicePixelRatio` 1). Citations look
like (`colormap.ts:205`); measurements name the sidecar or PNG they came from.

Scope is the two things that live on top of the picture — the **Colourbar** (one GL quad +
five DOM text nodes) and the **Axes** triad (one GL `LineSegments` + three DOM labels).
`Wireframe` and `Edges` are in-scene geometry, not overlays, and are covered in SPEC.md.

Line-number convention: `colormap.ts` is 272 lines, `axes.ts` 62, `label.ts` 59,
`render_object.ts` 81, `utils.ts` 430, `scene.ts` 830.

---

## 0. Frame, camera, pixel map

The overlay pass draws into the same framebuffer as the mesh, with a **separate camera**
that is rebuilt on every resize:

```ts
const aspect = width / height;                                   // scene.ts:198
this.ortho_camera = new THREE.OrthographicCamera(
  -aspect, aspect, 1.0, -1.0, -100, 100);                        // scene.ts:199-206
```

so overlay object space is isotropic: **1 unit in y spans the full canvas height, and
1 unit in x spans the same number of pixels** (`2·aspect / 2 · w = w/aspect = h`). That
single fact is what makes every overlay formula below collapse to a multiple of `h`.

Canvas size (`square_scalar.json` → `canvas {width: 894, height: 594, pixel_ratio: 1,
offset_width: 900, offset_height: 598}`): the host div is `clientWidth/Height` 900×600 and
`calcSize` subtracts `widthMargin = 6`, `heightMargin = 6` (`scene.ts:88-89`, `739-743`)
→ `w = 894`, `h = 594`, `aspect = 1.505050505050505` — recorded verbatim as
`uniforms.aspect` in every scene. The *same* `w, h` go to `renderer.setSize(w, h)` and to
every `obj.onResize(w, h)` (`scene.ts:216-217`), so GL and DOM agree by construction.
Resize is edge-triggered: `checkResize` only fires when the size actually changed
(`scene.ts:750-753`).

Object space → CSS pixels (`label.ts:45-46` uses exactly this):

```
px_x = (ndc_x + 1) / 2 · w      px_y = (1 − ndc_y) / 2 · h      ndc = (x/aspect, y, z)
     = (x/(2·aspect) + 0.5)·w        = (0.5 − y/2)·h
```

Because `setSize` takes CSS pixels and `setPixelRatio` (`utils.ts:137`) scales only the
drawing buffer, **every formula in this file is in CSS pixels and is `devicePixelRatio`
independent**. Only the PNG pixel measurements assume `pixel_ratio: 1`.

Corrections this file makes to SPEC.md's stub:

* bar ink is columns **31..328**, rows **4..25** inclusive (`square_scalar.png`), not
  31..327 / 4..24 — the float bounds are 31.29..328.29 × 4.455..25.245 (§1.2);
* `dx = 74.25` px exactly (`h/8`), not 74.3 (§2.2);
* a *zero* tick prints `0.00`; `−4.4e-16` is a tick that is not zero (§2.3);
* `Math.abs` does **not** make negative minima print positive (§2.4);
* there is no "hide the bar when `min == max`" rule (§1.5).

---

## 1. The colourbar

### 1.1 Geometry and anchor

Built in the constructor, and **not built at all for a mesh-only function**:

```ts
const mesh_only = this.data.funcdim == 0;                          // colormap.ts:127
if (mesh_only) this.enabled = false;                               // colormap.ts:128
if (!mesh_only) {
  const geo = new THREE.PlaneGeometry(1, 0.07).translate(0.5, 0, 0);// colormap.ts:132
  material = new THREE.MeshBasicMaterial({ depthTest: false,
               side: THREE.DoubleSide, wireframe: false });         // colormap.ts:133-137
  this.three_object = new THREE.Mesh(geo, material);                // colormap.ts:138
  this.three_object.autoUpdateMatrix = false;                       // colormap.ts:139
} else { this.three_object = null; }                                // colormap.ts:140-142
```

The quad is 1 × 0.07 with its **left edge at the local origin** (`.translate(0.5,0,0)`), so
positioning the object positions the left edge. `mesh_only` (a `CF` drawn without a
function, `funcdim == 0`) has `three_object === null` *and* `enabled === false`: nothing is
ever drawn and the label text is set to `''` (`colormap.ts:253`). `mesh_only.json` records
`objects[].object/visible/matrixWorld = null` for Colorbar while `Axes.visible = true` —
the direct proof of that branch.

The anchor is recomputed on resize, from the object's *current* world position:

```ts
onResize(w, h) {
  if (this.mesh_only) return;                                      // colormap.ts:194
  const aspect = w / h;                                            // colormap.ts:195
  this.three_object.getWorldPosition(p);                           // colormap.ts:197
  this.three_object.translateOnAxis(p, -1.0);                      // colormap.ts:198
  this.three_object.translateY(0.95);                              // colormap.ts:199
  this.three_object.translateX(-0.93 * aspect);                     // colormap.ts:200
  this.three_object.updateWorldMatrix(false, false);                // colormap.ts:201
```

The intent is "subtract the old position, add the new one", i.e. the anchor is
**`(-0.93·aspect, 0.95, 0)`** (§1.7 shows where that intent leaks). `camera.json` records
`objects[]` with `matrixWorld` for 8 of the 9 scenes; every one has rotation `identity`,
scale `(1,1,1)` and translation exactly `(-1.3996969696969697, 0.95, 0.0)`, equal to
`-0.93 · 1.505050505050505` to the last digit — **8/8, residual 0.0**. (`scene.ts` sidecars
themselves do not carry `matrixWorld`; `square_scalar.json → render_objects: null`.)

### 1.2 The rect in canvas pixels

Substituting the anchor into the pixel map, with `a = aspect`:

| quantity | formula | at `w=894, h=594` |
|---|---|---|
| left edge | `(1 − 0.93)/2 · w = 0.035·w` | 31.29 |
| right edge | left + `h/2` | 328.29 |
| top edge | `(1 − 0.985)/2 · h = 0.0075·h` | 4.455 |
| bottom edge | `(1 − 0.915)/2 · h = 0.0425·h` | 25.245 |
| width | `1/(2a) · w = h/2` | 297.0 |
| height | `0.07/2 · h = 0.035·h` | 20.79 |

The width being `h/2` — **independent of `w`** — is the isotropy of §0 doing its work; a
wider canvas makes the bar a *smaller fraction* of the picture, never a longer one.

Measured in `square_scalar.png`: the non-white run along row 15 is columns **31..328**,
contiguous, and rows **4..25** (scanning rows 0..31, columns 40..320, so the tick glyphs at
`top = 36` are excluded). Against the float bounds the worst edge error is **0.29 px** —
that is antialiasing plus the inclusive-column convention, not a formula error.

### 1.3 The ramp texture

`makeColormapTexture(n_colors)` (`colormap.ts:61-105`) interpolates
blue → cyan → green → yellow → red over four equal segments, with

```
x = i / (n_colors − 1)                              // colormap.ts:71
hx = 4x, 4x−1, 4x−2, 4x−3                           // colormap.ts:74, 80, 86, 92
color = c_j·(1−hx) + c_{j+1}·hx                     // colormap.ts:75-97
alpha = 1                                           // colormap.ts:101
```

`n_colors` is `gui_status.Colormap.ncolors`, the GUI "Number of colors" slider, integer
2..32 (`gui.ts:76-81` default `ncolors: 8`; `cmap.add(settings,'ncolors',2,32,1)` at
`gui.ts:364`). The ramp is therefore **piecewise-linear in linear light** with knots at
`x = 0, 0.25, 0.5, 0.75, 1`.

`makeTexture` packs it (`colormap.ts:7-24`):

```
n_colors = colors.length / 4                 // :9   ← the float array length, not the GUI value
width    = min(n_colors, 1024)               // :9
height   = floor((n_colors + width − 1)/width)// :10
if (width > 1) n_colors = width * height     // :11   ← pad up to a full row
new THREE.DataTexture(data, width, height, RGBAFormat, FloatType)   // :14-20
magFilter = THREE.NearestFilter              // :21
```

For 2 ≤ n_colors ≤ 32 the texture is `n_colors × 1` and `NearestFilter`
(`minFilter` keeps the `FloatType` default, also `Nearest`, because mipmaps are
unsupported on float textures — recorded `mag/min filter: 1003` = `NearestFilter`). So:
**one texel per colour band, `colormap_size` is irrelevant here (§1.4), and where
`colormap_size` *does* enter is the mesh shader (§1.4)**.

`square_scalar.json → tex_colormap` records 32 floats for `ncolors = 8` and `size: null`
— the probe read `texture.image.size`, which is `undefined` for a `DataTexture` (the real
fields are `image.width/height`). The 32 floats are the ramp above at `x = i/7`, and the
bar's eight bands are those texels **after three's sRGB encoding**:

| i | ramp (linear) | `round(255·sRGB(·))` | measured `square_scalar.png` row 15 |
|---|---|---|---|
| 0 | (0, 0, 1) | (0, 0, 255) | (0, 0, 255) |
| 1 | (0, 0.5714, 1) | (0, 199, 255) | (0, 199, 255) |
| 2 | (0, 1, 0.8571) | (0, 255, 238) | (0, 255, 238) |
| 3 | (0, 1, 0.4286) | (0, 255, 146) | (0, 255, 146) |
| 4 | (0.1429, 1, 0) | (146, 255, 0) | (146, 255, 0) |
| 5 | (0.5714, 1, 0) | (238, 255, 0) | (238, 255, 0) |
| 6 | (1, 0.5714, 0) | (255, 199, 0) | (255, 199, 0) |
| 7 | (1, 0, 0) | (255, 0, 0) | (255, 0, 0) |

Sampled one pixel inside each eighth of the bar (the bands are `297/8 = 37.125` px wide:
`NearestFilter` with the plane's default `uv.x ∈ [0,1]`), **8/8 exact**. The un-encoded
alternative — writing `round(255·ramp)` straight — puts band 1 at (0, 146, 255), 53 levels
off: the bar is colour-managed, `MeshBasicMaterial` runs `sRGBTransferOETF`
(`colorspace_fragment.glsl.js`) and the render target is not linear. Band means over the
box rows 4..26 × cols 31..329, which mix the texels with the antialiased white border:
`square_nc2.png` **(140.6, 26.5, 141.0)** against the ideal (127.5, 0, 127.5) for two
half-width bands; `square_nc32.png` **(123.7, 212.3, 123.9)**; `square_scalar.png`
**(126.5, 185.4, 126.9)** over 6554 non-white of 6877 sampled pixels.

A custom palette (`Draw(ngsolve, colors=[[r,g,b],...])`) goes through
`colorsToFloat32Array` (`colormap.ts:26-35`) and `updateTexture` prefers it
(`colormap.ts:224`), so the ramp above is only the fallback.

### 1.4 `colormap_min` / `colormap_max`, and where `colormap_size` enters

`scene.ts` refreshes the two uniforms from the GUI once per frame, before any pass:

```ts
if (Colormap.ncolors) { this.global_uniforms.colormap_min.value = Colormap.min;
                        ...colormap_max.value = Colormap.max; }   // scene.ts:780-783
```

and the bar's own GL material does not use them at all — `MeshBasicMaterial` samples
`uv.x` along the quad, i.e. the bar is a *palette strip*, and only the mesh shader maps a
**value** into it (`shader/utils.h:401-416`):

```glsl
float x = (value - colormap_min) / (colormap_max - colormap_min);   // utils.h
float y = 0.5;
```

with the two `colormap_size` guards (`utils.h:15` declares it) skipped when the size is
`(1,1)`. Recorded: `colormap_min: -0.9999999999999998`, `colormap_max: 0.9999999999999988`
in `square_scalar.json` — note they are the *autoscaled* function range, **not** `−1` and
`1`, and that `funcmin/funcmax` live separately under `render_data_summary`. For a fixed
range (`Draw(..., min=..., max=...)`) they are the given numbers: `square_fixed_range.json`
→ `−0.5, 2`. Mesh-only: `−0.5, 0.5` (`gui.ts:328-334`, `min = −0.5`,
`max = mesh_regions_2d − 0.5`).

`colormap_size` has exactly **one** writer — `Colorbar.setTexture(tex, isMeshTexture)`
writes `tex.image.width/height` into it only when `isMeshTexture` is true
(`colormap.ts:228-232`), and the only call passing true is `updateTexture(this.mesh_only)`
(`colormap.ts:224`). So for every scalar/vector plot `colormap_size` stays at its default
`new THREE.Vector2(1, 1)` (`scene.ts:281`) — recorded as `[1, 1]` in all 18 colourbar
sidecars — and only a **mesh-only** plot pushes `(n, 1)` into the shader, where `getColor`
then takes the texel-centre branch. `edges.ts:97` writes a *separate*
`edges_colormap_size`; `mesh.ts` never touches either.

### 1.5 The hide rules — all four

There is no "colourbar disappears when the range degenerates" rule, and nothing of the
sort in `colormap.ts`. The complete set of ways the bar goes away:

1. **GUI/`gui_status` switch.** `RenderObject.update` returns
   `this.name === '' || gui_status.Objects[this.name]` (`render_object.ts:42-50`), and
   `Colorbar.render` draws only when that is truthy (`colormap.ts:173, 188`). So a falsy
   `gui_status.Objects['Colorbar']` hides bar *and* tick labels (rule 1 is also what the
   `Objects` section of the control panel toggles).
2. **Mesh-only.** `three_object === null` kills the GL quad (`colormap.ts:188` tests it)
   and `enabled === false` empties the label text (`colormap.ts:242, 253`); the label
   *container* is still `display: block` when visible (`colormap.ts:187`) but holds five
   empty text nodes, so `mesh_only.json` records **0 tick labels**.
3. **Not the overlay pass.** The draw is `if (three_object && visible && mode == 'overlay')`
   (`colormap.ts:188-189`); in `'update'`, `'default'` and `'no_clipping'` the object is
   updated but never rendered. `mode` is the pass name from `scene.ts:803-814`.
4. **Per-tick `display: none`.** The mid-bar labels only, by the stride rule of §2.5.

A degenerate range (`min >= max`) does none of these. It yields `digits = 3` for every
tick (§2.3) and `setStepSize` returning `1e-8` (`gui.ts:667-674`); the bar still draws.

### 1.6 The bar is encoded; the mesh is not

Two independent measurements say the same thing, and a Python port has to reproduce both:

* the bar bands equal `round(255·sRGB(ramp))`, 8/8 (§1.3) — writing the raw ramp is off by
  up to 53 levels;
* the mesh's deepest blue in `square_scalar.png` has mean **R = 32.96** where the raw-ramp
  prediction is 28.05 and an *encoded* prediction is 93.22 — i.e. the mesh's colours come
  from `getColor` in the custom shader, which never runs the sRGB transfer, so they are
  written to the framebuffer as if already encoded.

Same texture, two colour spaces: the bar applies the transfer, the mesh skips it.

### 1.7 Source bug: the bar walks off the canvas on the second resize

`translateOnAxis(axis, d)` (`Object3D`) normalizes `axis`, so `translateOnAxis(p, -1.0)` at
`colormap.ts:198` subtracts `p/|p|`, not `p`. The undo is exact only when `|p| = 1`, i.e.
on the **first** call from the identity position — which is why all 9 captures are exact
(§1.1, 8/8 zero residual). Every further resize with an actual size change adds
`t − p/|p|` with `t = (−0.93·a, 0.95, 0)`, `|t| = 1.69164`, so the anchor becomes
`λ_n·t` with `λ_{n+1} = λ_n + (1 − 1/|t|) = λ_n + 0.40886`:

| resize # | λ | anchor | rect left, top (px) |
|---|---|---|---|
| 1 | 1.00000 | (−1.39970, 0.95000) | **31.29, 4.46** (measured §1.2) |
| 2 | 1.40886 | (−1.97197, 1.33842) | −138.68, −110.90 |
| 3 | 1.81772 | (−2.54425, 1.72683) | −308.64, −226.26 |

i.e. the rect moves **170.0 px left and 115.4 px up per extra resize** at 894×594, and
after two it is entirely off-canvas. Not exercised by the captures, which each resize once
from construction. A matplotlib port should place the bar at `λ = 1` unconditionally —
that is the state the ground truth shows.

---

## 2. Tick labels (DOM)

### 2.1 The DOM

Five `div`s, each holding one empty text node, inside a wrapper `div`, all appended to the
widget container — created once, never recreated (`colormap.ts:144-157`; note the comment
says "Create 5 html div/text elements"). The wrapper's `display` is the only thing
`render()` touches per frame (`colormap.ts:187`). The count is hardcoded to 5:
`onResize` and `updateLabels` both use `n = this.labels.length` (`colormap.ts:203, 239`)
and **never** `ncolors`, so the tick count is 5 no matter how many colour bands the bar has
(`square_nc2.json` and `square_nc32.json` both show 5 ticks over 2- and 32-band bars).

Each label's `style` attribute is rewritten wholesale on resize:

```ts
this.label_style + `transform: translate(-50%, 0%); left: ${x}px; top: ${y}px`
// colormap.ts:215-219
```

with `label_style` (`colormap.ts:161-165`) being

```
-moz-user-select: none; -webkit-user-select: none; -ms-user-select:none; onselectstart="return false;onmousedown="return false; user-select:none;-o-user-select:none;unselectable="on";position: absolute; z-index: 1; display:block;
```

Only the last three declarations are legal CSS; the `onselectstart="..."` and
`onmousedown="..."` fragments are attribute-looking text *inside* a `style` value, so the
browser drops them and no handler is ever attached. The live part is therefore
`position: absolute; z-index: 1; display: block; transform: translate(-50%, 0%)`.

There is **no font rule anywhere**: `styles.css` styles `.label3d` and `.tooltiptext` but
sets no `font`, and `label_style` sets none. The labels inherit the document font, which
the sidecars record verbatim: `font: '16px "Times New Roman"'`,
`color: rgb(0, 0, 0)` (`square_scalar.json → labels[*]`). Recorded `h = 18` px
= the 16 px Times line box.

**Reading the recorded boxes.** `labels[*].x/y/w/h` are `getBoundingClientRect()`, i.e.
*taken after the element's own CSS transform*. With `translate(-50%, 0%)` the recorded
`x` is `left − w/2`, so recovering the formula's `(left, top)` means

```
left = x + w/2      top = y
```

For `.label3d` (triad labels) the stylesheet adds `translate(-50%, -50%)`
(`styles.css`), so there `centre = (x + w/2, y + h/2)`. Getting this backwards is the one
easy way to "miss" by half a text width.

### 2.2 Position

```ts
const y  = Math.round(0.5 * (0.05 + 0.07) * h);       // colormap.ts:204
const dx = (0.5 * w) / ((n - 1) * aspect);            // colormap.ts:205
const x0 = 0.07 * 0.5 * w;                            // colormap.ts:206
const x  = Math.round(x0 + i * dx);                   // colormap.ts:212
```

i.e. `x_i = round(0.035·w + i · 0.5·w/(4·aspect))`, `top = round(0.06·h)`. In the isotropic
units of §0, `0.5·w/(4·aspect) = h/8`: **the five ticks are spaced `h/8` apart and span the
bar edge to edge** (`4·dx = h/2 = bar width`, and both start at `0.035·w`). At 894×594:
`x0 = 31.29`, `dx = 74.25`, `y = 36`.

Verified against all 40 recorded ticks in the 8 sidecars that draw them:

* `top`: every recorded `y` is **36** — residual **0.0 px** in 40/40;
* centre: `round(x0 + i·dx)` equals the recovered centre in **40/40**, residual **0.000 px**;
  the recorded centres are the integers **31, 106, 180, 254, 328**
  (`square_scalar.json`: `x = 14.3359375, 85.3359375, 153.1171875, 240, 314` with
  `w = 33.328125, 41.328125, 53.765625, 28, 28` → centres 31, 106, 180, 254, 328). Before
  the `Math.round` the predicted values are 31.29, 105.54, 179.79, 254.04, 328.29, so the
  largest pre-rounding discrepancy is **0.46 px** (distance to the nearest integer) — the
  rounding is part of the spec, not a detail.

The gap between the bar's bottom edge (25.245) and the text top (36) is
`0.0175·h = 10.395` px, 10.755 px once `y` is rounded.

### 2.3 The text: `digits`, then `toPrecision`

```ts
const inc = (this.max_ - min) / (n - 1);                       // colormap.ts:241
const value = Math.abs(min + inc * i);                         // colormap.ts:244
const digits = Math.max(2, Math.ceil(
    value != 0 && inc != 0 ? Math.log10(value / Math.abs(inc)) + 2 : 3));  // :245-250
this.labels[i].nodeValue = (min + inc * i).toPrecision(digits); // colormap.ts:251
```

`digits` is computed **per tick**, so one bar mixes precisions. `toPrecision(p)` is
JavaScript's, and its fixed/exponential switch is what a port must reproduce:

* fixed notation iff `−6 ≤ floor(log10|v|) < p`, where the decade is that of the **rounded**
  value (so `123456 @ 6` → `123456`, `123456 @ 5` → `1.2346e+5`);
* otherwise `d.ddde±dd`: mantissa of `p` significant digits with a single leading digit,
  exponent `e`, sign always present, exponent at least two digits;
* ties round away from zero on the *exact binary value*, so `0.125 → 0.13`, `2.675 → 2.67`,
  `1.005 → 1.00`; and `p` may exceed 6 (`toPrecision` allows up to 100) without changing
  the fixed/exponential boundary.

All of that is `tests/gt/toprecision.json`: 59 values × precisions 1..8 = **472 strings**
taken from V8, and the Python port under test (`/tmp/ov/tp.py` scratch) matches **472/472**.

The three formats the parent asked about, worked out from the sidecars:

* **`1.80e+308`** (`line_scalar.json`, `line_deformed.json`, where
  `colormap_min = 1.7976931348623157e308`, `colormap_max = 2.2250738585072014e-308`):
  `inc = −4.4942328371557893e+307`, `value/|inc| = 4.0`, `log10 = 0.60206`, `+2 = 2.60206`,
  `ceil = 3` → `digits = 3`. `1.79769…e308 @ 3` rounds to `1.80·10³⁰⁸`, whose decade
  `308 ≥ 3`, so exponential, and `1.797… → 1.80`. (In IEEE 754 `1.80e+308` *is*
  `Infinity`; the string is what the DOM shows, and `toFixed`-style overflow never happens
  because this path is exponent-form.)
* **`0.00`**: an exact zero tick takes the `: 3` branch of the ternary (`colormap.ts:248`),
  `Math.max(2, 3) = 3` → `toPrecision(3)` of `0` → `0.00`. Note the guard is on the tick
  value, not on `min`: in `line_scalar` the zero is the *last* tick and in
  `numberspace.json` the first.
* **`2.23e-308`** (`numberspace.json`, `min = 0`, `max = 2.2250738585072014e-308`):
  `inc = 5.5626846462680035e-309`, `value/|inc| = 4.0` again → `digits = 3` →
  `2.22507…e-308 @ 3` → decade `−308 < 3` → exponential → `2.23e-308`.

The near-zero ticks that look like errors are the `digits = 2` case of the same formula:
`square_scalar` has `min = −0.9999999999999998` so the middle tick is
`−4.440892098500626e-16`, and with `value/|inc| = 4.44e-16`
(`log10 = −15.35`, `+2 = −13.35`, `ceil = −11`) `Math.max(2, −11) = 2` → **`-4.4e-16`**.
Per-tick `digits` for the recorded bars: `square_scalar` 3,3,2,2,3;
`numberspace` 3,2,3,3,3; `volume_scalar` 2,2,3,3,3.

The full check: **18 of 18 sidecars that draw a colourbar reproduce their five tick strings
byte-exactly** from `colormap_min`/`colormap_max` alone (`mesh_only.json` is the 19th file
and records no tick labels at all, §1.5 rule 2). A few rows of that sweep:

| scene | `colormap_min` | `colormap_max` | recorded ticks |
|---|---|---|---|
| `square_scalar` | −0.9999999999999998 | 0.9999999999999988 | `-1.00` `-0.500` `-4.4e-16` `0.50` `1.00` |
| `square_order1` | −0.9972629810400511 | 0.9999999999999998 | `-0.997` `-0.50` `0.0014` `0.501` `1.00` |
| `square_fixed_range` | −0.5 | 2 | `-0.50` `0.13` `0.750` `1.38` `2.00` |
| `line_scalar` | 1.7976931348623157e+308 | 2.2250738585072014e-308 | `1.80e+308` `1.35e+308` `8.99e+307` `4.5e+307` `0.00` |
| `numberspace` | 0 | 2.2250738585072014e-308 | `0.00` `5.6e-309` `1.11e-308` `1.67e-308` `2.23e-308` |
| `numberspace_l2pwc` | 0 | 33 | `0.00` `8.3` `16.5` `24.8` `33.0` |
| `volume_scalar` | 4.3853809472754945e-16 | 6.0000000000000036 | `4.4e-16` `1.5` `3.00` `4.50` `6.00` |
| `complex_h1` | −1.0016502140160575e-16 | 1.0000000000000009 | `-1.0e-16` `0.25` `0.500` `0.750` `1.00` |
| `surface_scalar` | −5.4518437206651516e-17 | 0.999999999999998 | `-5.5e-17` `0.25` `0.500` `0.750` `1.00` |

(`tests/gt/dims/*.json` store `labels` as a flat list of strings in DOM order
`[tooltip, t0..t4, X, Y, Z]`; the `camera.json` scene sidecars store dicts with
`{text,x,y,w,h,color,font}`.)

### 2.4 Why `Math.abs` is there, and why a negative min still prints negative

`Math.abs` at `colormap.ts:244` feeds **only the digit count**. The printed text is the
signed value `(min + inc*i).toPrecision(digits)` (`colormap.ts:251`). So no, a negative min
does not print positive — it prints e.g. `-1.00`, `-0.500`, `-0.50`, `-4.4e-16`,
`-5.5e-17`, `-1.0e-16`, all present in the sidecars.

The `abs` is a `RangeError` guard. `Math.log10` of a negative number is `NaN`;
`Math.ceil(NaN)` is `NaN`; `Math.max(2, NaN)` is `NaN`; and
`(v).toPrecision(NaN)` throws. Confirmed on V8:

```
RangeError: toPrecision() argument must be between 1 and 100
```

Without the `abs`, every bar with a negative tick would crash the render loop on the first
frame. The `value != 0 && inc != 0` test (`colormap.ts:248`) is the other half of the same
protection: `log10(0)` is `−Infinity` and `log10(x/0)` is `+Infinity`, both of which would
also reach `toPrecision`.

### 2.5 Hiding mid-bar labels

```ts
const hide_label = i > 0 && i < n - 1 &&
  ((dx < 50 && i % 2 == 1) || (dx < 30 && i % 4 == 2));            // colormap.ts:208-211
if (hide_label) this.divs[i].setAttribute('style', 'display: none');// :213
```

The two conditions are **not nested**: with `dx < 30` both fire for `i % 4 == 2`
(which is odd), so the rule is a plain disjunction — `i ∈ {1,3}` when `dx < 50`, plus
`i = 2` when `dx < 30`. Since `dx = h/8` (§2.2), the thresholds are canvas heights:

| condition | equivalent | effect |
|---|---|---|
| `dx < 50` | `h < 400` px | hide ticks 1 and 3 |
| `dx < 30` | `h < 240` px | also hide tick 2 (leaving only the ends) |

Both also require `0 < i < n−1`, so the first and last tick never hide. **Never exercised
by the captures**: every one is 900×600 with `dx = 74.25`, so all five ticks are visible in
all 8 sidecars that draw them, and the `display: none` string appears nowhere in the
recorded styles. Note `dx` does not depend on `w` at all, so widening the canvas never
hides a tick.

The hidden labels keep their text and get *no* `left`/`top` at all — the whole style
attribute is replaced by `display: none`, so a hidden tick also loses `position: absolute`
and lands in normal flow (invisible because `display: none`).

### 2.6 Reproducing it in matplotlib

```python
# w, h  : canvas size in CSS px (the figure size in px, not inches)
# vmin, vmax: the colour range actually sent as colormap_min/colormap_max
bar_left   = 0.035 * w          # colormap.ts:206 / §1.2
bar_width  = h / 2              # §1.2
bar_top    = 0.0075 * h         # §1.2
bar_height = 0.035 * h          # §1.2
for i in range(5):
    dx  = h / 8                                   # colormap.ts:205
    x   = round(0.035 * w + i * dx)               # colormap.ts:212
    y   = round(0.06 * h)                         # colormap.ts:204
    inc = (vmax - vmin) / 4                       # colormap.ts:241  (n = 5, fixed)
    v   = vmin + inc * i
    digits = max(2, ceil(log10(abs(v)/abs(inc)) + 2)) if (v != 0 and inc != 0) else 3
    text = to_precision(v, digits)                # colormap.ts:251, see §2.3
    hide = 0 < i < 4 and ((dx < 50 and i % 2) or (dx < 30 and i % 4 == 2))
    # anchor: (x, y) is the horizontal CENTRE and the TOP of the text box,
    # because of transform: translate(-50%, 0%)          # colormap.ts:218
```

For the palette strip use `imshow` with `interpolation='nearest'` on an `(1, n_colors)`
array of `sRGB(ramp(i/(n_colors-1)))` (§1.3), positioned by `axes` bounding boxes in
figure coordinates — and do **not** let matplotlib's colour normal rescale it: the bar is
the palette, not the data. Ticks: `ha='center', va='top'`, `fontsize` 16, family
`Times New Roman`, colour black, at the `(x, y)` above in *inverted* y (canvas px from the
top).

### 2.7 Two ngsplot defects in this code path

Both are in `src/ngsplot/labels.py` (left as-is here; `tests/test_labels.py` already pins
the first):

1. `tick_digits` (`labels.py:53-62`) returns **2** when `value == 0 or inc == 0`, where
   `colormap.ts:248` uses the `: 3` branch — so ngsplot prints `0.0` where the renderer
   prints `0.00`. The fix is `3`, not `max(2, 3)`, because the branch is not clamped: it is
   already the `Math.max(2, ·)` result.
2. `to_precision` (`labels.py:37-43`) round-trips the rounded `Decimal` through `float()`,
   which **raises** `OverflowError: cannot convert float infinity to integer` at
   `labels.py:40` for `to_precision(1.7976931348623157e308, 3)`: the mantissa rounds to
   `180 → 1.80e+308`, whose `float()` is `inf`, and the code then takes `log10(inf)`.
   JavaScript returns the string `"1.80e+308"` (verified on V8) — the decade must be taken
   from the `Decimal` exponent, never from a `float` conversion. `line_scalar` and
   `line_deformed` hit exactly this input.

---

## 3. The axes indicator

### 3.1 Where it is drawn

`Axes` is a `RenderObject` with `render_modes = ['overlay']` (`axes.ts:22`) — the triad is
drawn **only** in the overlay pass, with the orthographic camera (`scene.ts:683`), and
therefore never in the pick/locate or clipping passes. Its object is a
`THREE.AxesHelper(0.15)` with `matrixAutoUpdate = false` (`axes.ts:17-18`), so the matrix
is written by hand once per frame.

### 3.2 Geometry and matrix

```ts
const { THREE } = data;
this.three_object.matrixWorld
  .makeTranslation(-0.8 * data.controls.aspect, -0.8, 0)
  .multiply(data.controls.rotmat);                              // axes.ts:33-35
data.renderer.render(this.three_object, data.camera);           // axes.ts:37
```

`controls.rotmat` is `extractRotation(this.mat)` where `mat` is the *normalisation* matrix
`T(−center)·S(1/mesh_radius)` (`camera.ts:56-58`, `124-128`) — the world→view-space
transform of the model, with the translation and scale stripped off and the perspective
camera deliberately *not* involved. `controls.aspect` is `width/height`
(`camera.ts:121-123`), the same number as the overlay camera's half-width.

So: three arms of world length **0.15**, from the shared origin at
`(−0.8·aspect, −0.8, 0)`, rotated by the model's own orientation, in a coordinate system
where §0's isotropy makes the pixel map exact. That yields:

| quantity | formula | at 894×594 |
|---|---|---|
| origin | `((1−0.8)/2·w, (1+0.8)/2·h)` = `(0.1·w, 0.9·h)` | (89.4, 534.6) |
| arm length | `0.15/2·h = 0.075·h` | 44.55 px |
| X tip | origin + (44.55, 0) | (133.95, 534.6) |
| Y tip | origin + (0, −44.55) | (89.4, 490.05) |
| Z tip | origin (Z is along the view axis) | collapses to a point |

Verified against `camera.json`'s recorded `objects[] → matrixWorld`:
`matrixWorld == T(−0.8·aspect, −0.8, 0) · extractRotation(controls_mat)` with max abs
difference **0.000e+00 in 8 of 9 scenes** and **5.551e-17** in `square_rotated` (worst
element `5.551115123125783e-17`) — one float rounding, i.e. the formula is exact.
`Axes.visible` is `true` in all 9 scenes, including `mesh_only` (which has no function: the
triad is not tied to the colourbar's fate).

In `square_scalar.png`: the X arm's ink ends at column **133** against a predicted tip of
133.95 (the last fully covered pixel is one step short of the tip), and the projected arm
measures **44.6 px** against 44.55. In `square_rotated.png` (Euler 20°, 30°, 0° in the
recorded order) the Z arm *does* project, to a screen length of **25.89 px**, and its tip is
at (111.67, 547.8).

Strong closure check, and the reason the triad can be ported without touching the mesh: in
the box rows 470..570 × cols 50..160, every pixel that is not near-white is explained by
the predicted arms (±2 px) plus the three recorded label boxes — **0 unexplained pixels in
9/9 captures** (308 ink pixels in the 8 axis-aligned scenes, 340 in `square_rotated` where
all three arms are visible). The mesh contributes nothing there.

### 3.3 Colour of each arm

`AxesHelper` is a `LineSegments` with `LineBasicMaterial({vertexColors: true, toneMapped:
false})` and **depth test left on** (§4). The literal colour array is not on disk (three
r0.152.2 is only present inside the minified `webgui.js` bundle), so the colours are pinned
by measurement, and the measurement is unambiguous: along an arm at fraction `t` of its
length,

```
rgb_linear(t) = mix(pure_axis, pastel_axis, t)
  X: (1,0,0) → (1,0.6,0)     Y: (0,1,0) → (0.6,1,0)     Z: (0,0,1) → (0,0.6,1)
```

i.e. the **origin vertex is the saturated axis colour and the tip vertex the same colour
60 % toward white**; the primary channel is 1 at every `t` and the secondary ramps
linearly `0 → 0.6` in linear light. The framebuffer values are
`255·sRGB(·)`, blended over the white clear colour by the MSAA coverage in 8-bit encoded
space.

`square_scalar.png` row 534 along the X arm (coverage recovered from the blue channel,
which is 0 for red):

| px | measured | coverage α | recovered `G_line` | `255·sRGB(0.6·t)` | Δ |
|---|---|---|---|---|---|
| 93 | (255, 114, 64) | 0.749 | 66.8 | 66.5 | +0.0012 |
| 98 | (255, 138, 64) | 0.749 | 98.8 | 98.2 | +0.0025 |
| 105 | (255, 160, 64) | 0.749 | 128.2 | 128.3 | −0.0004 |
| 111 | (255, 175, 64) | 0.749 | 148.2 | 148.3 | −0.0006 |
| 118 | (255, 190, 64) | 0.749 | 168.2 | 168.1 | +0.0006 |
| 125 | (255, 203, 64) | 0.749 | 185.6 | 185.2 | +0.0016 |
| 133 | (255, 216, 64) | 0.749 | 202.9 | 202.5 | +0.0017 |

Δ in bytes: ≤ 0.6. The R channel is **255 at every one of those columns** (`set()` of
row 534, cols 93..133 = {255}), which forces `R_line = 1` and rules out the "white at the
origin, saturated at the tip" hypothesis: that would put `G_line` near 247 at px 93 instead
of 66.8, 180 levels away. The coverage is independently confirmed by the blue channel: the
1-px arm sits between rows 534 and 535 with encoded blends of exactly 64 and 192 —
`0.25·255 = 63.75` and `0.75·255 = 191.25` — which is the 3:1 sample split that ANGLE's
4×MSAA produces there. The Y arm agrees with the same law to within 1.1 bytes (its joint at
the origin is off by 4, where the two arms' antialiasing overlaps), and the Z arm in
`square_rotated.png` too: `R_line = 0`, `B = 255`, `G_line` within 1.3 bytes of
`255·sRGB(0.6·t)` at every on-line sample.

The line width is 1 px: `LineBasicMaterial.linewidth` is ignored by WebGL, and MSAA is what
softens it.

### 3.4 The three labels

```ts
updateLabels(data) {
  const ldata = { pivot: this.three_object, ...data };           // axes.ts:41
  for (let i = 0; i < 3; i++)
    this.labels[i].render(ldata, new THREE.Vector3().fromArray([0,0,0.2, 0,0], 2 - i));
}                                                                // axes.ts:45-61
```

`fromArray([0,0,0.2,0,0], 2-i)` yields `i=0 → (0,0,0.2)`, `i=1 → (0,0.2,0)`,
`i=2 → (0.2,0,0)`: the labels sit at **`s = 0.2`, 33 % beyond the 0.15 arm tips**, on the
same rotated pivot. `Label3D.render` then projects through the *pivot's* world matrix and
the orthographic camera (`label.ts:41`, matrix = `pivot.matrixWorld` since `pivot` is
given) and converts with §0's map:

```ts
const x = Math.round(((vector.x + 1) * rect.width) / 2);         // label.ts:45
const y = Math.round(((-vector.y + 1) * rect.height) / 2);       // label.ts:46
```

and hides any label that lands within 10 px of a canvas edge. Text is `axes_labels`,
default `['X','Y','Z']` (`gui.ts:118`).

Check: predicted centre = recorded `(x + w/2, y + h/2)` (the `.label3d`
`translate(-50%,-50%)`, §2.1) with residual **0.0 px in both coordinates for 27/27 recorded
labels across 9 scenes**. Default view: X **(149, 535)**, Y **(89, 475)**, Z **(89, 535)**
— the Z label lands exactly on the collapsed Z arm's origin (`square_scalar.json`:
X at (143.21875, 526, w 11.5625, h 18), Y at (83.21875, 466), Z at (84.109375, 526)).
`square_rotated`: (141, 524), (89, 479), (119, 552).

Consequence to carry into a port: **in an axis-aligned view the Z arm is invisible** (it
points along the view axis, projecting to zero length) and the `Z` glyph sits on the
origin. Nothing is wrong; the arm is there.

### 3.5 Reproducing it in matplotlib

```python
import numpy as np
# controls_mat: the 4x4 normalisation matrix, column-major 16 floats as recorded in
# tests/gt/camera.json -> controls_mat (reshape(4,4).T to get row-major)
M = np.array(controls_mat).reshape(4, 4).T
R = M[:3, :3].copy()
for j in range(3):                       # extractRotation: normalise the columns
    R[:, j] /= np.linalg.norm(R[:, j])
T = np.array([-0.8 * aspect, -0.8, 0.0])
pure   = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
pastel = [np.array([1,.6,0]), np.array([.6,1,0]), np.array([0,.6,1])]
for i in range(3):
    for t in np.linspace(0, 1, 200):     # the arm, 200 samples for the colour ramp
        p = T + R @ (0.15 * t * np.eye(3)[i])
        px = (p[0]/aspect + 1)/2 * w ;  py = (1 - p[1])/2 * h
        rgb = np.where(rgb >= 0.0031308, 1.055*rgb**(1/2.4) - 0.055, 12.92*rgb)  # §3.3
        # plot one 1-px segment at (px, py) with that colour, no z-sorting needed
    p = T + R @ (0.2 * np.eye(3)[i])     # the label, at s = 0.2
    # place text centre at ((p[0]/aspect+1)/2*w, (1-p[1])/2*h), ha=center va=center
```

Draw it after the mesh, on top, with no z-order interaction (§4), in black
`16px "Times New Roman"` glyphs. Skip matplotlib's `quiver`/`Axes3D` route: the triad is an
**orthographic** 2-D projection of a rotated 3-D object, and any perspective in the port
will visibly skew the arms.

### 3.6 What is not source-pinned here

* `AxesHelper`'s geometry and colour arrays: three r0.152.2's `AxesHelper.js`/`Line.js` are
  minified into `webgui.js`, and the tarball ships no `node_modules`. §3.3's colours are
  *measured* (7 samples ≤ 0.6 byte, plus Y and Z), not quoted. The arm length 0.15 and the
  label offsets 0.2 **are** quoted (`axes.ts:17`, `axes.ts:45-61`).
* The antialiasing pattern (which rows get 3/4 and 1/4 coverage) is ANGLE/driver behaviour;
  reproduce the arms at 1 px and let the backend antialias, or draw at 1.25 px if the port
  has no MSAA.
* `axes_labels` other than the default: `gui.ts:118` is the only place they come from, and
  the captures never change them.

---

## 4. Render order and blending

`scene.ts:682-703` is the pass runner and `scene.ts:769-817` the frame:

```
render()                                                          // scene.ts:~755-817
  if (this.ortho_camera === undefined) return;                     // :769
  line_thickness.value = Misc.line_thickness / h;                  // :775-776  (5/594)
  colormap_min/max.value = Colormap.min/max                        // :780-783
  setRenderTarget(null)                                            // :796
  setClearColor(getBackgroundColor()); <parent div bg>             // :797-799
  renderer.clear(true, true, true)                                 // :801   ← ONE clear
  renderObjects('update')                                          // :803
  renderObjects('default')                                         // :804
  clippingPlanes = []
  renderObjects('no_clipping')                                     // :808
  renderObjects('overlay')                                         // :814
  afterrender callbacks                                            // :816-817
```

and inside a pass, `renderObjects(mode)` picks
`const camera = mode === 'overlay' ? this.ortho_camera : this.camera;` (`scene.ts:683`),
builds the `data` bag `{camera, canvas, context, clipping_plane, controls, gui_status,
mode, renderer}` (`scene.ts:685-694`) and loops the objects **in registration order**
(`scene.ts:700-702`).

* **One clear, no clears between passes.** `autoClear = false` (`utils.ts:124`) and
  `renderer.clear(true, true, true)` at `scene.ts:801` before the first pass; nothing clears
  again. So the overlay pass composites onto whatever the mesh passes left in the buffer,
  and the depth buffer still holds the mesh's depth.
* **Clear colour.** `getBackgroundColor()` reads the *container's computed text `color`*
  (`utils.ts:152-167`); if that colour is bright (`color.r > 0.5`) it is replaced by
  `color.setScalar(0.1)` (`utils.ts:159-162`). Recorded `clear_color: 16777215`,
  `clear_alpha: 1` = white, opaque, matching the light host page. The GL context is created
  with `alpha: false` (`utils.ts:102-105`, `110`) even though the `WebGLRenderer` is told
  `alpha: true, premultipliedAlpha: false` (`utils.ts:117-122`) and `setClearAlpha(1.0)`
  (`utils.ts:123`) — the context flag wins, so the framebuffer is opaque and **no DOM shows
  through the canvas anywhere**; everything that looks transparent is a DOM element beside
  it, not a hole in it.
* **Order within the overlay pass** is object registration order: the `Colorbar` is created
  before `Axes` (`scene.ts:393-396`), and the sidecars' `objects[]` records the creation
  order `Colorbar, Axes, Edges, Wireframe, Surface`. Bar first, then triad. Both would draw
  identically in the other order (see the depth note below), but the port should keep the
  order.
* **Depth.** The bar's material has `depthTest: false` (`colormap.ts:134`) and
  `depthWrite` is off for it in practice (no `depthTest` means the depth *test* is skipped;
  the quad is at ortho `z = 0`, so nothing else can be drawn over it either way). The triad
  keeps `AxesHelper`'s default `depthTest: true` in an ortho window of `near = −100`,
  `far = 100` (`scene.ts:199-206`): a fragment at `z = 0` maps to window depth 0.5, and a
  mesh fragment could only occlude it by being closer than `2·near·far/(near+far) = 1.98`
  world units in front of the anchor — in the unit-ish overlay box, which the perspective
  mesh never reaches. **Never exercised**: §3.2's 0-unexplained-pixel count in all 9
  captures means the box contains no mesh ink at all, so this is arithmetic about the
  projection, not an observed effect.
* **`render_modes = ['overlay', 'update']`** (`colormap.ts:125`) means the bar is visited
  twice per frame: once in `'update'` (where `mode != 'overlay'`, so `colormap.ts:188-189`
  skips the draw) and once in `'overlay'`. All the bookkeeping — the `min/max/ncolors`
  dirty check, `updateTexture`, `updateLabels`, and the wrapper's `display` (`colormap.ts:
  173-190`) — therefore happens in the `'update'` pass, and the `'overlay'` pass only draws
  the quad and re-hides the labels. `Axes` has `render_modes = ['overlay']` only
  (`axes.ts:22`), so its label positions refresh in the overlay pass itself.
* **Shared uniforms.** `RenderObject` does `this.uniforms = {...global_uniforms}`
  (`render_object.ts:26`), a shallow copy of the *Uniform objects*, so the `'update'` pass
  writing `tex_colormap` (`colormap.ts:234`) is what the mesh samples in the `'default'`
  pass that follows — same object, no upload ordering to worry about.
* **Blending** is the only surprise for a port, and it is §1.6: the bar and the triad go
  through built-in materials, which end in `sRGBTransferOETF` (`colorspace_fragment.glsl.js`)
  because `renderer.outputColorSpace` is the r152 default `SRGBColorSpace` and the default
  framebuffer is not marked linear. The mesh's custom shader writes the raw `getColor`
  result and is **not** encoded. So a Python/matplotlib port must apply sRGB to the bar and
  triad and leave the mesh alone. Both were measured; neither is a style choice.
* Unverified but implied, and worth stating because it is a visible porting question: on a
  **dark** host page (a container whose computed text colour is dark), `setScalar(0.1)`
  (`utils.ts:160`) is a *linear* 0.1, which after encoding is ≈ 0.349 → a clear colour of
  about `#595959`, not `#1a1a1a` — because the clear colour is colour-managed while the mesh
  shader is not. No capture exercises this (`clear_color` is 16777215 in all of them).

---

## 5. What we still cannot pin

1. `THREE.AxesHelper`'s literal vertex/colour arrays in r0.152.2 — only the minified bundle
   is on disk; §3.3 is measurement with residuals, not source.
2. The MSAA sample pattern behind the 3:1 coverage split (§3.3): driver-side, and any port
   with different AA settings will land 1 byte away.
3. The depth-test consequence for the triad (§4): arithmetic only. No capture puts mesh ink
   in the triad's box, so "the mesh never covers the triad" is untested rather than tested.
4. Both stride thresholds (`dx < 50`, `dx < 30`) — every capture is 900×600, `dx = 74.25`.
   A < 400-px-tall widget would exercise them; nothing in `tests/gt/` is.
5. `colormap_size ≠ (1,1)` at runtime. Only a mesh-only `Colorbar` writes it
   (`colormap.ts:229-231`), and mesh-only has `three_object === null`; no capture records
   the uniform after such a draw (`mesh_only.json`'s Colorbar entry is `null`), so the
   shader's texel-centre branch (`utils.h`) is unverified end-to-end.
6. `Colorbar.colors` (a custom `colors=` palette) end-to-end: the code path
   (`colormap.ts:224`, `26-35`) is read but no capture sends a palette.
7. `mesh_only` after the first frame: `mesh_only` is fixed at construction
   (`colormap.ts:127`) from `render_data` present at object creation (`scene.ts:393`), so
   switching an existing widget between mesh-only and function display cannot be checked
   without ngsolve's Python source, which is not in this workspace.
8. `.tooltiptext` — the box that appears in every DOM probe as
   `{text: "tooltip", x: -50, y: 10, w: 120, h: 28}` (`styles.css`: `position: absolute`,
   `left: 50%`, `margin-left: -60px`, `width: 120px`) is the *scene* tooltip, not an overlay
   label: its content rule was not studied and it is off-canvas (`x = -50`) in every capture.
9. One pixel run in `square_scalar.png`: rows 5 and 15 carry non-white pixels at columns
   655..899, far right of the bar (the bar's own run is 31..328). Nothing in `colormap.ts`
   or `axes.ts` explains them; they are presumably mesh/wireframe ink. Flagged, unresolved.

---

## Appendix — the overlay formulas in one table

CSS pixels, `a = w/h`. Valid for the first resize after construction (§1.7).

| thing | formula | source | measured |
|---|---|---|---|
| overlay unit → px | `h/2` in both axes | (`scene.ts:199-206` + §0) | bar width 297 = `h/2` |
| bar anchor | `(−0.93·a, 0.95, 0)` | (`colormap.ts:199-200`) | 8/8, residual 0.0 |
| bar rect | `[0.035w, 0.035w + h/2] × [0.0075h, 0.0425h]` | §1.2 | 31..328 × 4..25, ≤ 0.29 px |
| ramp | 5 knots, 4 linear segments in linear light | (`colormap.ts:71-97`) | 8/8 bands after sRGB |
| texture | `n × 1` `RGBA32F`, `NearestFilter` | (`colormap.ts:9-21`) | 32 floats, filter 1003 |
| tick count | 5, always | (`colormap.ts:148`, `203`, `239`) | 5 ticks on 2- and 32-band bars |
| tick `x_i` | `round(0.035·w + i·h/8)` | (`colormap.ts:205-206,212`) | 40/40, 0.0 px |
| tick `top` | `round(0.06·h)` | (`colormap.ts:204`) | 40/40, 0.0 px |
| tick anchor | centre-x, top-y | (`colormap.ts:218`) | §2.1 |
| tick text | `toPrecision(max(2, ceil(log10(|v|/|inc|)+2)))` | (`colormap.ts:241-251`) | 18/18 byte-exact |
| tick hide | `0<i<4 ∧ ((dx<50 ∧ i odd) ∨ (dx<30 ∧ i%4==2))` | (`colormap.ts:208-211`) | never triggered (h ≥ 400) |
| font | 16 px Times New Roman, black | inherited (`styles.css`) | recorded `font`/`color` |
| triad origin | `(0.1·w, 0.9·h)` | (`axes.ts:33-34`) | 27/27 labels, 0.0 px |
| triad arms | `T + R·(0.15·e_i)`, `R` = `extractRotation(controls_mat)` | (`axes.ts:33-35`, `camera.ts:124-128`) | ≤ 5.6e-17 |
| arm length | `0.075·h` px | §3.2 | 44.6 vs 44.55 |
| arm colour | `mix(pure, pastel, t)` then sRGB | §3.3 | ≤ 0.6 byte (X), ≤ 1.3 (Z) |
| triad labels | `T + R·(0.2·e_i)`, centre anchor | (`axes.ts:45-61`, `label.ts:41-46`) | 27/27, 0.0 px |
| clear | one `clear(true,true,true)`, white, opaque | (`scene.ts:801`, `utils.ts:124`, `152-167`) | `16777215`, alpha 1 |
| order | bar, then triad, both in `'overlay'` only | (`scene.ts:393-396`, `700-702`) | recorded `objects[]` order |
