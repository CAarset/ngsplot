# The Draw visual, reverse-engineered

Every number in this file was measured from the live renderer (NGSolve 6.2.2608,
`webgui@0.2.39`), not guessed. Sources of truth:

* `ngsolve/webgui.py` + `netgen/webgui.py` — Python side that builds `render_data`.
* `webgui@0.2.39` TypeScript + GLSL (`scene.ts`, `mesh.ts`, `colormap.ts`, `utils.ts`,
  `shader/*.vert|frag|utils.h`), from `https://registry.npmjs.org/webgui/-/webgui-0.2.39.tgz`.
* Live introspection of the WebGL scene inside headless chromium (see `tools/`).

## Why Draw is clunky (the workspace's "verify?" question)

The hypothesis was "C origins". It is **wrong in the letter, right in the spirit**:
there is no C in the drawing path at all. The friction comes from Draw being a
*live GUI widget with an async GPU transport*:

* `Draw()` builds a `render_data` dict in Python, serialises it to JSON, ships it to a
  three.js/WebGL2 canvas in the browser, and a **lil-gui/dat control panel** is generated
  from a settings tree for live editing (`gui.ts`, 676 lines of widget code).
* Every interaction is an async round-trip through the widget; nothing is a plain object
  you can compose.
* `netgen.webgui.MakeScreenshot` **returns an un-awaited coroutine**, and
  `_MakeScreenshot` launches chromium and takes the shot after a fixed delay without
  waiting for a WebGL frame — so scripted snapshots are a race. It also needs the
  `preserveDrawingBuffer: true` hack (`getScreenshotHTML()`).
* `GenerateHTML(data, filename=None, template=None)` ignores `template` (it substitutes
  `{render}` into `_html_template`), so the documented way of customising output does not
  work; you have to pass `template=getScreenshotHTML()` *and* it works only by accident
  of the replace order.
* Drawing requires importing the right module first: without `import ngsolve.webgui`
  you get `RuntimeError: Cannot draw object of type GridFunction`; and
  `_get_draw_default_args` lives in `netgen.webgui` while `Draw`'s registration lives in
  `ngsolve.webgui`.
* The keyword that picks which part of a value to colour reaches nothing at all. The lookup
  asks `if "eval_" in kwargs` and the next line reads `kwargs["eval"]`
  (`netgen/webgui.py:564-573`), so `Draw(gfu, eval="norm")` is silently dropped and
  `Draw(gfu, eval_="norm")` dies with `KeyError: 'eval'`. The mode is otherwise a widget
  control — which is to say that on the Python side it is unreachable.
* The panel's deformation slider is two payload fields, `deformation` and `deformation_scale`,
  and `Draw`'s `deformation` takes only a displacement field or a bool. The amount belongs in
  `scale`: `Draw(gfu, deformation=0.7)` answers `NgException: Dimensions don't match, op = +
  dims1 = 0: 3, dims2 = ` because the float is read as a CoefficientFunction to add to the
  geometry.

So: the clunkiness is *live-control GUI + racy async transport*, inherited from an
interactive-tool heritage (Netgen's GUI) rather than from C.

## Taking the payload without taking the window

`ngsplot._ngsolve.render_data(obj, *args, **kwargs)` is the only place in the package that
imports ngsolve. It calls `Draw` and reads back what `Draw` would have sent:

```python
scene = Draw(obj, *args, show=False, **kwargs)   # netgen/webgui.py:674
return scene.widget.value                        # the dict WebGLScene handed the widget
```

`Draw` applies `_get_draw_default_args()` and builds a `WebGLScene`, whose constructor is where
the work is: `GetData` (`netgen/webgui.py:480`) picks an `encoding`, dispatches on
`type(obj)` to the draw type registered for it, and finishes the dict with the range, the flags
and the camera. The widget is a plain `anywidget` object and nothing is displayed with
`show=False`, so the payload exists before any browser does. `ngsolve.webgui` has to be
imported for the dispatch to know `Mesh`, `CoefficientFunction` and `GridFunction`, which the
seam does for its caller.

Two of `Draw`'s habits are kept rather than repaired, because the seam's promise is that a
payload from ngsplot is the payload from `Draw`: a `CoefficientFunction` needs its mesh
positionally and raises `Cannot draw CoefficientFunction without mesh`
(`ngsolve/webgui.py:119`) without one, and the dead `eval_` branch above stays dead.
`show` and `filename` are the two keywords the seam does not pass on; they belong to the window
it never opens. `widget.value` is read instead of calling `GetData()` again so that the arrays
are fitted once and not twice — `GenerateHTML` builds them a second time.

## Laying the picture on matplotlib

`ngsplot.pyplot` decides nothing about how a function looks.  It hands what `Plot` was given to
`_ngsolve.render_data`, hands the payload to `scene.paint`, and lays the resulting array on an
`Axes.imshow`; every word after that is matplotlib's.  Three numbers fix the frame: `WIDTH = 900`
and `HEIGHT = 600`, the size the page's canvas is given, and `DPI = 100`, so the figure is 9 by 6
inches and a `savefig` at the figure's own dpi writes the renderer's array byte for byte — the
outermost row and column included, measured as a maximum difference of 0 against
`matplotlib.image.imread`.  That is what makes a loop of frames an animation rather than a blur,
and it needs no cropping step: an image with `interpolation="nearest"` at a 1:1 device mapping
puts its texels exactly where they were painted.  The extent is in the page's own pixels, so the
rulers of an axis would count pixels too, which is why a plot takes the frame away;
`set_axis_on()` puts them back for whoever means them.  An axis made for a picture covers the
figure because the picture *is* the plot; an axis made by someone else keeps what was on it, is
never stretched, and gives room away rather than taking any, as any image does.

The keywords are sorted by who reads them, from the signatures rather than from a list: every
name `scene.paint` and `Camera.__init__` takes is a keyword of `Plot` and goes to the painter,
and the rest goes to `Draw`, so `order`, `draw_vol`, `objects`, `settings` and `nodal_p1` keep
their meaning and a payload that is already built refuses them by name instead of ignoring them
silently.  Two names need a rule of their own.  `deformation` is a flag on the payload side and
an amount on the painter's, so a number is sent in as `deformation=True, scale=<float>` — see the
`NgException` above — while a payload that is already built can only be lifted at the painter.
Either way the payload's flag is what the painters switch on, and it is load-bearing: lifting by
the amount alone paints a measurably different picture, so `Plot` makes sure the flag reaches
`Draw` and passes the amount on to the painter as well.
`eval` is the keyword `Draw` cannot spell; `Plot` takes it under the panel's own names and writes
it as `function_mode`, the same knob.

A colourbar cannot be built on the picture, and matplotlib says so twice over.  The array of an
`AxesImage` of a plotted Draw holds *colours*, not values, and when a mappable is passed,
`Figure.colorbar` reads `cmap` and `norm` off that mappable and ignores the keywords
(`colorbar.py:304-308`) — a bar driven by the picture prints viridis at 256 bands.  So
`pyplot.colorbar` keeps the texture and the two ends each picture was painted with and drives the
bar from a private `ScalarMappable`: a `ListedColormap` of exactly those texels under a
`BoundaryNorm` on their band edges, so the bar shows the bands that are on the picture and ticks
at the boundaries between them, which is the scale the panel's two fields hold.  Nothing is
registered in `matplotlib.colormaps`.  A bar laid on an axis takes room out of it even when it is
given a `cax` of its own; the `ax=` route shrinks a full-bleed axis to `(0, 0.1, 0.8, 0.9)` and
that is the one thing a picture gives up without being asked.  `Axes` has no `colorbar` attribute,
so the function hands the `Colorbar` back.

Plotting twice into one axis replaces the picture, and its bar with it, so a time loop is one
line and does not stack images on the axis.

## Geometry: exact, not sampled

`ngsolve.webgui.BuildRenderData(mesh, func, order=2, ...)` exports Bezier control points
per element — the *exact* FE function, no sampling. Degrees are capped:
`order2d = min(order, 3)`, `order3d = min(order, 2)`.

Order-2 triangle (`shader/utils.h`, `ORDER==2`), barycentric `(u, v)`, `w = 1-u-v`:

```
b00 = u*u    b01 = 2uv   b02 = v*v
b10 = 2uw    b11 = 2vw   b20 = w*w
P(u,v) = b00*p00 + b01*p01 + b02*p02 + b10*p10 + b11*p11 + b20*p20   # vec4 = (xyz, value)
```

Name `p{a}{b}` means barycentric exponents `(u, v, w) = (n-a-b, b, a)`; `index_pairs`
lists the names in the order the renderer packs them into attributes
(`three/mesh.ts:102-131`), `a` outer, `b` inner — the same enumeration as the payload.

Normal (derivative cross product, then normalise), with `B00 = 2u`, `B01 = 2v`, `B10 = 2w`:

```
du = B00*(p00-p10) + B01*(p01-p11) + B10*(p10-p20)
dv = B00*(p01-p10) + B01*(p02-p11) + B10*(p11-p20)
n  = normalize(cross(du.xyz, dv.xyz))
```

Order 1 uses `p0,p1,p2` (lerp) and `n = normalize(cross(p1-p0, p2-p0))`; order 3 uses the
cubic basis (`p00..p30`). The fragment shader never re-evaluates the polynomial:
`trigsplines.vert` computes position, value and normal at the lattice vertices and
`function.frag` interpolates the varyings, so a software rasteriser must do the same.

Sub-triangulation: each element is drawn as `n_segments^2` sub-triangles of the reference
triangle (`mesh.ts`, `MAX_SUBDIVISION = 20`, `geo.setDrawRange(0, 3*sd*sd)`). With lattice
coordinates, `u = j/n`, `v = (i-j)/n`:

```
for i in range(n):
    for j in range(i + 1):   # up-triangle
        (j, i-j), (j+1, i-j), (j, i-j+1)
    for j in range(i):       # down-triangle
        (j+1, i-j-1), (j+1, i-j), (j, i-j)
```

## Payload: the arrays Python sends

`BuildRenderData` samples each element on `get_intrules(2, order2d)` (sub-triangles of the
reference triangle, `_make_trig`), evaluates `CF((xyz, func))` there in **float64**, and
fits a Bezier triangle with the inverse of the forward Bernstein matrix:

```python
B[j, k] = n!/(i! j! (n-i-j)!) * x**i * y**j * (1-x-y)**(n-i-j)   # netgen/webgui.py:86
ctrl   = inv(B) @ samples          # B rows: samples at (ix/n, iy/n); cols: names p{a}{b}
```

with the row and column indices running through the identical enumeration (outer 0..n,
inner 0..n-outer), `inv(B)` cached per element type in `netgen.webgui._bernstein_cache`.
Then `ngsplot.geometry.trig_points` unpacks base64 float32 arrays (`decode_array`, the Python
counterpart of the page's `readB64`, `utils.ts:209`):

| list | rows | arrays | components |
| --- | --- | --- | --- |
| `Bezier_trig_points` | elements | `K` + `K` (`n==1`) or `K//2` (`n>1`) | `vec4` per control point; then `vec2` per control point, or `vec4` packing two |
| `Bezier_points` | `3 * elements` | `n+1` + `n+1` | `vec4` per point; then `vec2` per point |
| `edges` | boundary segments | `n+1` | `vec4` per point |

`K = (n+1)(n+2)/2`. A `vec4` control point is `(x, y, z, value)` — the first function
component rides along in slot 3; the remaining components are the `vec` attributes
(`vec00_01` etc.), which is why a two-component function leaves the *second* vector
component zero (padding only), verified for degrees 1-3. Scalar functions send no `vec`
arrays at all. Row `3*e + k` of `Bezier_points` is edge `k` of element `e`, the edges being
`(t,0)`, `(0,t)`, `(t,1-t)` of the reference triangle; `edges` lists the boundary segments
walked from their second to their first vertex, with the value slot identically zero
(`cf_edge = CF((xyz, 0.0))`).

`k` of the payload and `p{a}{b}` are the same slot, and on a straight triangle its
position is

```
ctrl[slot(a,b), :3] = ( b*V0 + a*V1 + (n-a-b)*V2 ) / n        # V = element vertices
```

i.e. the shader's `(u, v, w)` corresponds to `(V2, V1, V0)`: the parametrization of the
Bezier patch is a reflection of the mesh parametrisation, orientation-reversing. It is
invisible in the image (the control points carry their own positions) but it decides
`ngsplot.bezier`'s reading of the slots, and it flips the sign of the surface normal —
which matters, because shading is one-sided (`dark_backside`). Greville abscissa of the
slot: `(u, v) = ((n-a-b)/n, b/n)`.

Deformation (`have_deformation = mesh_dim == funcdim and not is_complex`) displaces
`p.xyz += deformation * value` (`DEFORMATION`); for a 2D mesh with a scalar or complex
function it is `p.z += GetValue(deformation * value)` (`DEFORMATION_2D`), Draw's
`deform_2d`/surface view. Both are applied in the vertex shader **after** the normal is
built, so a deformed surface is shaded as if it were flat, and under `DEFORMATION_2D`
`GetNormal` applies the complex-phase rewrite to the `w` slots of its own inputs only.
Python also lifts a 2-component vector deformation: `deformation = CF((deformation, 0.0))`
(`ngsolve/webgui.py:269-270`).

## Range: what the colour scale is fitted from

`funcmin/funcmax` come from `getMinMax(pmat[:, 3:].flatten())` over the float64 *samples*
(not the float32 control points), and `ngs.Vector.MinMax` initialises its bounds with
`numeric_limits<double>::max()`/`min()` instead of `+-inf`. Two consequences worth
copying: a function that never exceeds `DBL_MIN` reports `DBL_MIN` as its maximum, and
infinities are excluded (control points are min/maxed with `ignore_inf=True`; NaN never
takes part). `updatePMinMax` does the same per column of the positions, giving
`mesh_center = (pmin+pmax)/2`, `mesh_radius = |pmax-pmin|/2`. The two keys are written
only inside the `if d['draw_vol']` branch, so a payload built with `draw_vol=False` has
no range to read.

What those ends are worth is decided in four steps, and `scene.view_range` follows them in
the same order. `initColormap` seeds the panel minimum and maximum from `funcmin`/`funcmax`,
falling back to 0 and 1 (`gui.ts:336-343`). A payload's `gui_settings` is applied over that:
sections missing from it are filled from the panel defaults, and then its *flat* names
overwrite the sections they belong to — `autoscale`, `colormap_min`, `colormap_max`,
`colormap_ncolors`, `subdivision`, `line_thickness`, `Misc.reduce_subdivision` into
`fast_draw`, and `edges`/`mesh`/`elements` into `Objects.Edges`/`Wireframe`/`Surface`
(`gui.ts:146-175`). Rendering re-reads the function only if the payload's **top-level**
`autoscale` lets it (`scene.ts:497`), and `updateColormapToAutoscale` then does nothing
unless the panel's own `Colormap.autoscale` is on and there is a function to follow,
`draw_surf || draw_vol` (`gui.ts:366-390`); a drawn length never goes negative, so mode 3
stretches the range to `0 .. max(|funcmin|, |funcmax|)`. Since `Draw(min=a, max=b)` writes
`funcmin`/`funcmax` themselves (`netgen/webgui.py:544-547`) it survives both paths, while
`Draw(autoscale=False)` stops the recompute and leaves a vector at its own negative minimum
rather than the mode-3 one — which is what ngsplot's `autoscale=False` means too. An
animated draw mixes the two payloads' `funcmin`/`funcmax` per frame under the same
condition (`scene.ts:508`). One more default, worth its place here because it is a
*default* and not a rule: the page writes a subdivision of 1 into `settings_default` when
`max(order2d, order3d) <= 1` (`gui.ts:229-231`), before the payload's settings go over the
defaults, so a payload naming a subdivision of its own on a linear mesh gets it.

## Colour: 8 discrete bands, nearest-neighbour

Measured live: the colormap is an **8 x 1 `FloatType` RGBA texture**, `magFilter =
minFilter = NearestFilter (1003)`, `wrapS = wrapT = ClampToEdge (1001)`,
`generateMipmaps = false`, and `colormap_size = (1, 1)`, so `getColor` takes the last
branch:

```
x = (value - colormap_min) / (colormap_max - colormap_min)     # then texel fetch
```

i.e. `band = clip(floor(x * n), 0, n-1)` — hard-edged bands, no interpolation.

The palette is `ramp(i/(n-1))` for `i = 0..n-1`, and the live texture contents are exactly:

| i | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| R | 0 | 0 | 0 | 0 | 0.2857142984867096 | 0.8571428656578064 | 1 | 1 |
| G | 0 | 0.5714285969734192 | 1 | 1 | 1 | 1 | 0.5714285969734192 | 0 |
| B | 1 | 1 | 0.8571428656578064 | 0.2857142984867096 | 0 | 0 | 0 | 0 |

`ramp(t)` (`colormap.ts::makeColormapTexture`) is the blue→cyan→green→yellow→red ramp with
`hx = 4t - k` on segment `k`:

```
4t      < 1: (0, hx, 1)          4t   - 1 < 1: (0, 1, 1-hx)
4t      - 2 < 1: (hx, 1, 0)      4t   - 3 < 1: (1, 1-hx, 0)   else (1,0,0)
```

`ncolors` defaults to 8 (`gui.ts`), is adjustable 2..32, and `Colormap.colors`
(a custom palette) replaces the ramp if given. A draw with no function (`colors` in the
payload, `funcdim = 0`) uploads those region colours instead — `makeEdgeColorsTexture`
(`edges.ts:6-21`) packs them as RGBA texels, `(width, height) = (min(n,1024),
ceil(n/width))`, alpha 1 — so the one-region unit square sends a **single green texel**
(`tests/gt/mesh_only.json`) and `colormap_size` stays `(1, 1)`. The colorbar is hidden:
the bar's measured rectangle is untouched white in that scene. Under the fixed light that green
comes out around `(0, 61, 1)` in the ground-truth PNG, so a test that asks whether a mesh-only
draw is green asks for dominance of the green channel, not for the texel's own value.

## Shading: per-fragment Phong in view space

`light_dir = normalize(vec3(0.5, 0.5, 1.5)) = (0.30151134, 0.30151134, 0.90453403)`,
`light_mat = (ambient 0.3, diffuse 0.7, shininess 10, spec 0.3)`, `dark_backside = true`.
The light is fixed in **view** space (it never moves with the camera).

```
n = normalize(norm)                    # flipped when gl_FrontFacing == false
p = modelViewMatrix * vec4(position, 1)
v = normalize(-p.xyz)                  # perspective: view vector varies per fragment
r = reflect(-s, n)
sDotN   = max(dot(s, n), 0)
diffuse = 0.7 * sDotN
spec    = (diffuse == 0 or shininess == 0) ? 0 : pow(max(dot(r, v), 0), 10)
dimm    = (dark_backside and inside) ? 0.5 : 1
rgb     = dimm * (color * (0.3 + diffuse) + spec * 0.3)
```

`modelViewMatrix` in line 191 is the *object* matrix as the camera sees it, not the view
matrix on its own: the renderer never moves the camera, so all of the centring and scaling
lives in that matrix and the light sees the surface where the user put it. Feeding the
eye-space point instead leaves the specular term wrong for every mesh that is not centred
on the origin — a mean of twenty grey levels over the surface of `tests/gt/square_scalar`,
against 0.03 once the object matrix is included.

The `spec` term is why a single colour band is a *smooth gradient* on screen instead of a
flat fill — 700+ distinct RGB values were measured inside one band, while the band
boundaries stay hard. `alpha = color.w = 1` for surfaces; `NO_FUNCTION_VALUES` paints
`(0.7, 0.7, 0.7)` before lighting, and it is defined exactly when `data.draw_surf ===
false` (`mesh.ts:51`) — the `null` a plain `Draw` sends is *not* `false`, so an ordinary
mesh draw is coloured by its region colour, not greyed out.

A shader writes into an 8-bit framebuffer, so the sum above is clamped on the way out; the float
`Canvas.color` of the software rasteriser is not clamped and reaches 1.21 inside
`tests/gt/dims/cf_mesh`. That is why a picture is compared, and handed to matplotlib, through
`Canvas.to_uint8()` and never through the raw float array.

## Camera

The camera never moves. `scene.ts:318-336` builds a `PerspectiveCamera` with `fov = 40°`
(vertical), `near = 1`, `far = 100`, `zoom = 1`, up `(0,1,0)`, sitting at `(0, 0, 3)` with an
identity quaternion, and nothing in the GUI ever touches it again: every change of view —
including dragging with the mouse — is a premultiplication onto the **object** matrix. That
matrix starts out (camera.ts `reset()`) as

    mat = S(1 / mesh_radius) * T(-mesh_center)

so the mesh is centred on the origin and fits the unit sphere, and `loadSettings`
(camera.ts:174-220) then applies, in order: `settings.mat` if present (that is what the GUI's
own save/restore writes, together with `center`; `isPerspectiveCamera` is stored but never
read), then `settings.euler_angles` as degrees `Euler(a[0], a[1], a[2], a[3] || 'XYZ')`, then
`settings.transformations`, whose entries are `move` (`dist || 1.0` times an *unnormalized*
`dir`) and `rotateX|Y|Z` (degrees). Rotations are wrapped about where the mesh centre
currently is — `wrapTransformation(mat, p)` is `T(p) · mat · T(-p)` — which for a freshly
reset matrix is the origin, so the scripted default is a pure rotation about the origin.
Measured: the unit square of `square_rotated` (`euler_angles = [20, 30, 0]`) gives
`Surface`, `Wireframe` and `Edges` the same `matrixWorld` as `controls.mat`; an *invisible*
object is skipped by `updateMatrixWorld` and keeps its identity matrix (`square_no_wireframe`),
`Axes` carries the rotation but not the scale, and the `Colorbar` is untouched.

The Euler orders are intrinsic turns about the fixed axes, so `'XYZ'` is `Rx @ Ry @ Rz` and
`'YXZ'` is `Ry @ Rx @ Rz` — all six verified against the live renderer with the same
`[10, 20, 30]` in `tests/gt/euler_orders.json` (`tools/probe_euler_orders.py`).

A view a *script* asks for is not spelled that way, because nobody writing `Draw` ever wrote
`euler_angles`: a panel drag left it in `settings['camera']` and that is all it has ever been
good for. The three words users know are matplotlib's — `elev`, `azim` and `roll` of
`Axes3D.view_init`, degrees — and since the camera never moves and the screen keeps the world's
axes (right `+x`, up `+y`, the eye at `+z`), they are three more turns of the object about the
mesh centre: `Rz(roll) · Ry(azim) · Rx(-elev)`, tipped first, then turned, then rolled. The
elevation carries the minus sign because the camera looks along `-z`: a positive `elev` takes the
top of the mesh away from the eye and brings the underside into sight, which is what looking down
from above is. A positive `azim` makes `+x` recede, and a positive `roll` turns the picture
counter-clockwise. That is the page's triple `[-elev, azim, roll]` in the order `'ZYX'`, and
`tests/test_camera.py` holds that identity as a known answer, with each single turn checked
against the page's own single-axis triple. Three zeros — three angles left out — are the page's
head-on view, straight at `xy`: page parity is the promise, so the *renderer's* default is not
matplotlib's, and a solid asks for matplotlib's `elev=30, azim=-60` out loud. `Plot` refuses
`euler_angles` by name, so a script has one spelling of a point of view while a saved page still
replays with the camera it was saved with.

The projection is three.js' `PerspectiveCamera.updateProjectionMatrix` with `left = -right`
and `top = -bottom`, so with `t = tan(fov/2)`: `fx = 1/(aspect * t)`, `fy = 1/t`,
`A = -(far+near)/(far-near)`, `B = -2 far near/(far-near)`, and the bottom row `(0, 0, -1, 0)`.
Verified for the 894x594 canvas: `aspect = 1.505050505050505`, `fx = 1.8255051310470312`,
`fy = 2.7474774194546225`, `A = -1.0202020202020203 = -101/99`, `B = -2.0202020202020203 =
-200/99`, and the view matrix is exactly `T(0, 0, -3)`. `aspect` is `uniforms.aspect` too —
the same number the thick-line shader uses to square up screen space — and it is the ratio of
the *canvas buffer*, `renderer.domElement.width / .height`, which for a 900x600 viewport is
894x594 (`calcSize()` subtracts a couple of pixels from `offsetWidth/offsetHeight`: 900x598).
Pixel mapping is the plain GL viewport: `px = (x_ndc + 1) / 2 * width`,
`py = (1 - y_ndc) / 2 * height` (y downwards), depth `z_ndc / 2 + 1/2`.

Verified: the 900x600 ground truth puts the unit square at x 255..639, y 105..489 — 385 px,
matching `1/(3*tan(20°))` exactly: the mesh has radius 1 at distance 3, so it fills ~92% of
the vertical field.

Background: **white** — `setClearColor(0xffffff, 1)`, and `getBackgroundColor()` returns
white for a light theme (dark theme, i.e. light text, gives `0.1` grey). Antialiasing is
MSAA (`antialias: true`), `devicePixelRatio` respected.

Implemented in `src/ngsplot/camera.py`, with `tests/gt/camera.json` (`tools/probe_camera.py`)
holding the object, view and projection matrices of all nine ground-truth scenes as the known
answers.

## Raster: painting it without a GPU

`src/ngsplot/raster.py` reproduces what those two programs do, in numpy, one element at a
time. `tests/test_parity.py` puts the result against all eight recorded pictures that draw
a surface: inside the surface and away from the lines that are not painted yet, the median
difference is zero grey levels out of 255 and the mean is 0.03 to 0.35. What is left is a
quarter of a percent of pixels that sit exactly on a colour band boundary, where the
shader's float32 `floor` and this module's float64 one choose neighbouring texels.

Per element, in the order the GPU uses:

* **Lattice.** `bezier.lattice` evaluates position, value and normal at the lattice
  vertices `(j, i-j)` of the `n_segments` sub-triangulation, which is what
  `trigsplines.vert` does before it hands the three quantities to the fragment stage as
  varyings. The polynomial is not re-evaluated per pixel, and a deformation is added to
  the position *after* the normal is built (`trigsplines.vert:25-30`), so a deformed
  surface is shaded as the flat one.
* **Projection.** The lattice goes through `camera.model_view_projection` to pixel
  centres, carrying `1/w`: `p_`, `value_` and `normal_` are interpolated with perspective
  correction, as GL does, not linearly in screen space.
* **Sub-triangles.** `n_segments^2` of them, each a half-space test on its three edges at
  the centre of every pixel of its bounding box. A NaN in any control point poisons every
  lattice vertex of the element, which is the shader's per-vertex `isNan` guard culling
  all of them at once.
* **Fragments run once per pixel**, at the `gl_FragCoord` centre, not once per sample.
  Measured: shading each sample at its own position makes the interior error twelve times
  worse (mean 0.145 grey levels instead of 0.0119) and puts every disagreement on a band
  edge. The multisampled framebuffer mixes coverage; the colour is the one the centre
  produced.
* **Facing.** `gl_FrontFacing` counts counter-clockwise in window space, which under the
  y-downward pixel convention is a positive cross product of the screen-space edges. The
  back side has its normal flipped and `inside` set, and `dark_backside` then multiplies
  its colour by exactly 0.5 -- the two effects always travel together, which is what
  `test_the_back_side_gets_the_flipped_normal_and_half_the_light` pins. Which real winding
  counts as front is a mirroring convention between ngsolve and three.js, settled by the
  recorded picture rather than argued.
* **Samples.** `Canvas` keeps `samples^2` colour planes and `samples^2` depth planes and
  averages the colours on read, because that is a multisampled framebuffer: one colour
  *and* one depth per sample, averaged on resolve. A framebuffer that blended instead --
  the obvious implementation -- leaves the shared edge of two coplanar sub-triangles
  visibly grey, because two quarters of coverage composite to 0.4375 instead of covering
  the union; the recorded silhouettes show the union.
* **Depth.** `LessEqualDepth`, three.js' default, so coplanar triangles are painted in
  submission order and the later one wins. The surface additionally draws with
  `polygonOffset(1, 1)`, which is the only reason the wireframe and the boundary edges --
  submitted after it, at the same depth -- are visible at all.
* **Colour space.** `round(clamp(c, 0, 1) * 255)`, nothing else. None of these
  hand-written programs contains three.js' `colorspace_fragment` include and the canvas is
  not colour-managed; putting the linear-to-sRGB ramp between the shader and the PNG costs
  a mean of sixty-eight grey levels in the same comparison.
* **Buffer geometry.** The screenshot is 900x600 while the drawing buffer is 894x594 --
  the page keeps a scrollbar gutter -- and the buffer starts at the top left of the
  picture. Shifting the comparison by one pixel costs an order of magnitude, which is what
  makes that offset a measurement and not an assumption.

Left for later, roughly in order of need: the lines (`Wireframe` and `Edges`, whose
`THICK_LINES` expansion is spelled out under Overlays), the overlays, `function_mode == 8`
(`getPositionAsColor`), the clipping plane, the selected-face highlight, `render_depth`,
and the `DEFORMATION` displacement of a mesh whose dimension matches its function.

## Overlays

* **Colorbar**: `PlaneGeometry(1, 0.07)` translated `+0.5` in x, `MeshBasicMaterial`
  (unlit) textured with the *same* nearest-filtered colormap → its bands are the palette
  itself. Placed at world `(-0.93*aspect, 0.95, 0)`, `depthTest: false`, drawn in an
  overlay pass. 5 HTML tick labels at pixel
  `x_i = round(0.035*w + i * 0.5*w/(4*aspect))`, `y = round(0.06*h)`; the middle labels are
  hidden when `dx < 50` (odd `i`) or `dx < 30` (`i % 4 == 2`). Measured in the 900x600
  ground truth: bar spans x 31..327, y 4..24 — `x_0 = 31.3`, `dx = 74.3` predicted.
* **Labels**: `value = min + i*(max-min)/4`; `digits = max(2, ceil(log10(|value|/|inc|) + 2))`
  (3 when `value == 0`), then JS `toPrecision(digits)` — which is why a zero tick prints as
  e.g. `-1.1e-16`.
* **Axes**: `AxesHelper(0.15)` at `(-0.8*aspect, -0.8, 0)` in view space (the RGB→XYZ triad
  in the bottom-left), plus 3 HTML labels.
* **Wireframe** (element edges): `THREE.Line` strips of `n_segments + 1` points along each
  Bézier element edge (`splines.vert`, order-matched), colour `(0,0,0,1)` — 1 px wide, since
  WebGL ignores `linewidth`; MSAA softens it.
* **Edges** (mesh boundary / curve edges): same Bézier evaluation but with `THICK_LINES`,
  which offsets the strip in screen space by `line_thickness/2` in NDC per side, where
  `line_thickness = Misc.line_thickness / canvas_height` (measured `0.0084175 = 5/594`) →
  ~2.5 px total.
* GUI panel (`Open Controls`) and the notebook widget are *not* part of the picture;
  ngsplot renders neither.

## Function modes

`function_mode` selects the plotted component (`GetValue`): 0..2 = component, 3 = length,
5/6 = complex real/imag rotated by `complex_scale`. `>= 4` components are reduced to
`Norm(cf)` on the Python side with a printed warning. Modes 4/7/8 are pick/locate passes.

How the mode is settled, in order of who writes it (`gui.ts:529-571`): `initComplex` seeds the
panel at 5 for a complex value — into `settings_default` as well, so a payload that brings its
own `Colormap` section inherits it — and at 3 for a two- or three-component one, into the live
settings only. A `Colormap` section that names `eval` therefore survives only where nothing
overwrites it, which is a scalar function; and `data.eval` at the top of the payload is written
last of all, over both. A flat `eval` beside the sections is not a name `updateSettings` folds,
so nothing reads it. Since `Draw`'s own spelling is dead (above), a script reaches the mode
through `settings={"Colormap": {"eval": …}}`, or, in ngsplot, straight to the renderer.
