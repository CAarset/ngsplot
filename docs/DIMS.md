# Ground truth for the dimensions `tools/capture_gt.py` does not cover

Captured once per scene by [`tools/capture_dims.py`](../tools/capture_dims.py) into
`tests/gt/dims/`: `Draw(...).GetData()` → payload JSON + standalone `.png.html`, then a
headless-chromium (SwiftShader) screenshot and a live-renderer probe of that HTML.
ngsolve **6.2.2608**, webgui **0.2.39** (jsdelivr CDN), viewport 900×600, canvas and
drawing buffer both **894×594** at `devicePixelRatio` 1 (the 3 px shaved off each side are
the renderer's own margins — see `docs/SPEC.md`, `Camera`).

Re-run with

    PLAYWRIGHT_BROWSERS_PATH=$PWD/.pw-browsers .venv/bin/python tools/capture_dims.py [name ...]

Anatomy of a sidecar `NAME.json` (all of it read from the live page, nothing inferred):

| key | content |
|---|---|
| `render_data_summary` | every payload key; a base64 array list becomes `{b64_items, floats_per_item}` |
| `colormap_min`, `colormap_max`, `colormap_size`, `function_mode`, `n_segments`, `line_thickness`, `dark_backside`, `deformation`, `complex_scale`, `light_dir`, `light_mat` | the GLSL uniforms after render init (`tex_colormap` skipped) |
| `render_objects` | `scene.render_objects[i]` → `name`, `three_object.visible`, `geometry.drawRange.count`, `instance_count`, numeric uniforms |
| `canvas`, `viewport`, `labels` | canvas rect/buffer size, viewport, absolutely-positioned label texts |
| `png_stats` | mean/std of the PNG (`std_all` 32.82–92.87 over the captured scenes, so none is blank) |
| `draw_ok`, `error` | whether `Draw()` succeeded, and its verbatim traceback when it did not |

The **payload itself** is only in `NAME.png.html` (`var render_data = {...}`); read it with
`tests/gtdata.py`, which accepts the subdirectory: `gtdata.render_data_of("dims/line_scalar")`.
Payload array lists are base64 strings, so a length must be decoded before it means anything
(`ngsplot.geometry.decode_array`) — `edges` for a 1D mesh is 3 strings of 88 *characters*,
which is 3 × 16 floats.

## Scene table

Measured from the payload and the sidecar; "arrays" counts are floats per array after decoding.

| scene | mesh | funcdim | `draw_surf` / `draw_vol` | wireframe `Bezier_points` | surface `Bezier_trig_points` | `edges` | `points3d` | `funcmin` / `funcmax` | colormap min/max |
|---|---|---|---|---|---|---|---|---|---|
| `line_scalar` | 1D, 4 segments on [0,1]×{0}, H1(2), `gfu.Set(x*x)` | 1 | `True` / `False` | 3 × **0** | 6 × **0** | 3 × 16 (= 4 × 4) | – | **1.7977e+308 / 2.2251e-308** | same garbage |
| `line_deformed` | idem, `deformation=True` | 1 | `True` / `False` | 3 × 0 | 6 × 0 | 3 × 16 | – | same garbage | same garbage |
| `numberspace` | `unit_square` maxh 0.25 (34 trigs), `GridFunction(NumberSpace)`, never `Set` | 1 | `True` / `False` | 3 × 408 | 6 × 136 | 3 × 64 | – | 0.0 / **2.2251e-308** | 0 / 2.2251e-308 |
| `numberspace_l2pwc` | idem, `L2(order=0)`, dof of element *i* = *i* | 1 | `True` / `False` | 3 × 408 | 6 × 136 | 3 × 64 | – | 0.0 / 33.0 | 0 / 33 |
| `volume_scalar` | `unit_cube` maxh 0.5 (28 tets), `x+2y+3z` | 1 | `True` / `True` | 3 × 432 | 6 × 144 | 3 × 96 | 10 × 112 | 4.385e-16 / 6.0 | same |
| `surface_scalar` | gmsh 6×6 node patch, see *Capture limitation 1* | 1 | `True` / `False` | 3 × 864 | 6 × 288 | 3 × **0** | – | -5.45e-17 / 1.0 | same |
| `cf_mesh` | `unit_square`, `Draw(ngs.x*ngs.y, mesh)` | 1 | `True` / `False` | 3 × 408 | 6 × 136 | 3 × 64 | – | 0.0 / 1.0 | 0 / 1 |
| `cf_vector` | `unit_square`, `Draw(CF((-y, x)), mesh)` | **2** | `True` / `False` | 6 × 408 | 9 × 136 | 3 × 64 | – | -1.0 / 1.0 | **0 / 1** (norm rule) |
| `vec_hcurl` | `unit_square`, HCurl(1) GF set to `CF((-y, x))` | 2 | `True` / `False` | 6 × 408 | 9 × 136 | 3 × 64 | – | -1.0 / 1.0 | 0 / 1 |
| `complex_h1` | `unit_square`, complex H1 GF | 2 (`is_complex`) | `True` / `False` | 6 × 408 | 9 × 136 | 3 × 64 | – | -1.0017e-16 / 1.0000000000000009 | same |
| `cf_nomesh` | `Draw(ngs.x*ngs.y)` | – | – | – | – | – | – | no PNG, see *Failures* | |
| `volume_cuts` | `Draw(gf, objects=[ngs.x, ngs.y])` | – | – | – | – | – | – | no PNG, see *Failures* | |

`ngs.solve`/mesh construction: 1D built by hand with `netgen.libngpy._meshing.Mesh(1)` +
`AddPoints`/`AddElements`; 2D `netgen.geom2d.unit_square.GenerateMesh(maxh=0.25)`; 3D
`ngs.unit_cube.GenerateMesh(maxh=0.5)`; the patch via `netgen.read_gmsh.ReadGmsh`
(`surface_geometry.msh`, with `$PhysicalNames`).

## Payload array layout (from `ngsolve/webgui.py`, counts checked against every scene)

Which elements feed which array is decided by two region lookups, and they are the whole
story of the odd dimensions:

```python
vb = [ngs.VOL, ngs.BND][mesh.dim-2]        # webgui.py:383 (wireframe) and :461 (surface)
vb = [ngs.VOL, ngs.BND, ngs.BBND][mesh.dim-1]   # webgui.py:421 (edges)
```

* `mesh_dim == 1` makes `mesh.dim-2` **−1**, so the wireframe and the surface sample
  `regions[BND]` of a 1D mesh — boundary *points*, which carry no triangles. Both array
  groups come back as `order2d+1` and `ndtrig` **empty** arrays while `draw_surf` stays
  `True` (`draw_surf = func and draw_surf`, `webgui.py:288`). Nothing errors; the surface is
  simply never drawn.
* `mesh_dim == 2` takes `VOL` (the trigs) and `BND` (boundary segments); `mesh_dim == 3`
  takes `BND` (surface triangles) and `BBND` (the 1D edges of the volume).

Layouts, all `[x, y, z, value]` float32 quads:

* `edges` — `order2d+1` arrays of `n_seg × 4`, sampled with a segment rule and Bernstein-
  converted along the segment. The 4th component is **hard-coded 0**
  (`cf_edge = ngs.CF((xyz, 0.0))`, `webgui.py:343`), verified numerically: `max |col 3| == 0`
  in all three scenes that have edges. Edge lines therefore never carry function values.
* `Bezier_points` — `order2d+1` arrays of `n_trigs × 3 × 4`: the Bézier control points of the
  three **edges** of each triangle (the wireframe), sampled at the `3(og+1)` boundary points
  of `ir_trig` (`webgui.py:377-399`). Square: 34·3·4 = 408 ✓. Patch: 72·3·4 = 864 ✓.
* `Bezier_trig_points` — `ndtrig = (og+1)(og+2)/2` arrays of `n_trigs × 4`: the full Bézier
  triangle (og=2 → 6). Scalar → 6 arrays; `funcdim ≥ 2` → 9, the extra three holding the
  remaining components packed pairwise (`webgui.py:498-509`) ✓ measured.
* `points3d` — **only when `draw_vol`, i.e. `func and mesh.dim==3 and draw_vol and mesh.ne>0`**
  (`webgui.py:287`). `np_per_tet` arrays of `n_tets × 4`. These are **not** Bézier data: the
  volume block encodes the sampled points directly, with no Bernstein matrix (`webgui.py:519-544`, the encoding at :538).
  Decoding `points3d[0]` of `volume_scalar` row by row gives `[x, y, z, x+2y+3z]` with
  x, y, z ∈ {0, 0.5, 1}: 10 samples per tet = 4 corners + 6 edge midpoints (`order3d=2`),
  28 tets → 112 floats per array.
* `mesh_center` / `mesh_radius` are computed once, inside the surface block, from the min/max
  of positions accumulated over wireframe + edges + surface samples (`webgui.py:474-480`, stored at `:511-512`). Measured: 1D `[0.5, 1.1125e-308, 1.1125e-308]`, r 0.5; 2D `[0.5, 0.5,
  1.1125e-308]`, r 0.7071067811865476; 3D `[0.5, 0.5, 0.5]`, r 0.8660254037844386.
  The `1.1125369292536007e-308` is the subnormal half of `DBL_MIN`: a coordinate that is
  constantly 0 keeps the sentinel **max** `DBL_MIN = 2.2250738585072014e-308` (0 < DBL_MIN, so
  it never replaces it), and `0.5*(0 + DBL_MIN)` is that value. It is numerically harmless
  (radius changes in the 308th decimal) but it is why a "flat" mesh never reports an exactly 0
  centre.
* Always present: `autoscale=True`, `show_mesh=True`, `show_wireframe=(order2d>0)`,
  `deformation_scale=1.0`, `order2d=min(order,3)`, `order3d=min(order,2)`, `ngsolve_version`,
  empty `gui_settings`/`objects`.

## The 1D case: `webgui.Draw` does not plot a 1D function

`tests/gt/dims/line_scalar.png` (`gfu.Set(x*x)`, a parabola on [0,1]) is a **dead-flat
2-pixel horizontal black line, rows 296–297**, and 97.1 % white. The picture contains no trace
of the function: with `Bezier_trig_points` and `Bezier_points` empty, `Surface` and `Wireframe`
exist with **0 instances** (measured in `render_objects`), and only `Edges` (4 instances) paints.

Because the autoscale accumulator never runs in 1D —
`funcmin, funcmax = getMinMax(pmat[:,3:].flatten())` sits inside the surface block
(`webgui.py:475`) — the payload ships the untouched sentinels of `ngs.Vector(...).MinMax()`:
`funcmin = DBL_MAX = 1.7976931348623157e+308`, `funcmax = DBL_MIN = 2.2250738585072014e-308`.
Those reach the colormap uniforms verbatim and even the colorbar tick labels, which the probe
reads as `['1.80e+308', '1.35e+308', '8.99e+307', '4.5e+307', '0.00']`. The colorbar is
visible: it hides only when min equals max, and here they differ by 308 orders of magnitude.

**Consequence:** there is no visual target to match for `mesh_dim == 1`. ngsplot has to define
1D plotting itself (a curve of the function along the mesh line is the only sensible reading of
"what Draw would have shown"), and the sentinel min/max must be detected rather than trusted.

## Autoscale and degenerate ranges

* `getMinMax` initialises to `(DBL_MAX, DBL_MIN)`; only the surface samples (`webgui.py:475`)
  and, for volumes, the tet samples (`:535`) update it. An **all-zero field** therefore lands on
  `funcmin = 0.0`, `funcmax = 2.2251e-308` — exactly what `numberspace` measures, and its
  colorbar labels are `['0.00', '5.6e-309', '1.11e-308', '1.67e-308', '2.23e-308']`. A constant
  field draws one flat colour and a colorbar of subnormal tick labels.
* Vector norm rule (`webgui.ts` `gui.ts:376-379`): the colormap gets `min = 0`,
  `max = max(|funcmin|, |funcmax|)` — measured `funcmin/funcmax = -1/1` against colormap
  `0 … 1` for both `cf_vector` and `vec_hcurl`.
* `funcdim ≥ 2` draws the **norm** (`function_mode = 3` measured); complex draws
  `function_mode = 5` with the payload split into real/imag (`is_complex` → `cf = func.realimag`,
  `webgui.py:304-305`) and the live `complex_scale = [1, -0.0]`.
* `cf.dim > 3` is silently replaced by `Norm(cf)` with a printed warning (`webgui.py:326-328`, the print at :327).

## What is rendered, per dimension (live `render_objects`)

`scene.objects` does not exist in 0.2.39; visibility is `render_objects[i].three_object.visible`
and the instance count lives on the mesh or its geometry (`scene.objects.map` in
`tools/capture_gt.py`'s older probe is dead code and yields `objects_err`).

| scene | objects (visible, instance count) |
|---|---|
| `line_scalar` | Colorbar✓, Axes✓, **Edges✓(4)**, Wireframe✓(**0**), Surface✓(**0**) |
| `numberspace` | Colorbar✓, Axes✓, Edges✓(16), Wireframe✓(102 = 34·3), Surface✓(34) |
| `surface_scalar` | Colorbar✓, Axes✓, **Edges✓(0)**, Wireframe✓(216 = 72·3), Surface✓(72) |
| `volume_scalar` | Colorbar✓, Axes✓, Edges✓(24), **Clipping Plane✗**, Wireframe✓(108), Surface✓(36) |
| `cf_vector`, `complex_h1` | Colorbar✓, Axes✓, Edges✓(16), Wireframe✓(102), Surface✓(34), **Vectors✗** |

So a 3D draw is by default the **closed coloured surface** of the boundary triangles plus the
black `BBND` outline; the camera looks down −Z. `points3d` feeds the hidden `Clipping Plane`
object behind the "Draw 3d function" slider — it is not part of the default picture, and the
same object appears for `funcdim == 3` as `ClippingVectorsObject`.

## Deformation

`have_deformation = (mesh_dim == funcdim) and not is_complex`, and
`have_z_deformation = (mesh_dim == 2 and funcdim > 0)` — the same two lines appear three times
(`mesh.ts:13-14`, `edges.ts:45-46`, `scene.ts:292-294`), selecting `DEFORMATION` vs
`DEFORMATION_2D` in the shaders. When either is set the `deformation` uniform is
`deformation_scale = 1.0` (`gui.ts:582-593`).

`line_deformed` shows the flag firing and doing nothing: payload `deformation: True`, live
uniform `deformation = 1`, and a PNG **byte-identical** to `line_scalar.png` (sha256
`597828385894c447136e9d029d8d670850e9851a206c509a81b2e4798cbd95f2` for both). The reason is the
layout: `DEFORMATION` displaces along the 4th component of the vertex data, which for edges is
hard-coded 0, and the surface it could displace has no instances. `cf_vector.png` and
`vec_hcurl.png` are likewise byte-identical to each other (sha256 `035ada1e734b97ae2736a7fedfa85787252d773ab620e524e1780a6442d0b632`):
an HCurl(1) interpolant of `CF((-y,x))` rasterizes exactly as the bare vector CF.

## NumberSpace, and how to get a genuinely piecewise-constant field

`ngs.NumberSpace(mesh, order=0|1|2)` has **`ndof == 1`** for every order (measured: 1, 1, 1,
against `L2(order=0)` = 34 and `H1(order=2)` = 85 on the same mesh). A `GridFunction` over it is
a single global constant, its payload an ordinary scalar 2D field, and an unset one is 0
everywhere → the degenerate autoscale above and one flat colour.
**NumberSpace is not the "draw element numbers" mechanism.** The real one is `L2(mesh, order=0)`
(34 dofs = 34 trigs); dofs are indexed per element with `fes.GetDofNrs(ngs.ElementId(ed))`
(`VectorAccess` is absent in this wheel). `numberspace_l2pwc.png` sets dof *i* to *i* and draws
a per-triangle flat-colour element-number map, arriving as a plain order-2 field whose Bézier
control points are constant per triangle, autoscaled 0…33.

## CoefficientFunction handling

* **A bare `CoefficientFunction` is not drawable without a mesh.** `wg.Draw(ngs.x*ngs.y)` raises
  inside `Draw()` itself (the payload is built in `WebGLScene.__init__` → `GetData()`, not
  deferred): `RuntimeError: Cannot draw CoefficientFunction without mesh`
  (`ngsolve/webgui.py:119`, reached from `netgen/webgui.py:784 Draw()` →
  `netgen/webgui.py:451 WebGLScene.__init__` → `:489 GetData()`). Full
  traceback in `cf_nomesh.json`.
* **`Draw(cf, mesh)` is the required form** and produces a payload indistinguishable from the
  equivalent `GridFunction` (`cf_mesh.png`: funcdim 1, min/max 0…1, same object list). Works for
  vector CFs too. `Draw(obj, region)` additionally narrows to that region plus its boundaries
  (`webgui.py:125-131`): the region's `VB()` and each `Boundaries()` level.
* `objects=[...]` passes dicts through verbatim and otherwise calls `obj._GetWebguiData()`
  (`webgui.py:290-295`), which a `CoefficientFunction` does not implement — the documented
  cutplane spelling `objects=[ngs.x, ngs.y]` fails (see *Failures*). Even with hand-built dicts,
  `makeRenderObject` (`scene.ts:57-67`) dispatches only `lines`, `points`, `fieldlines`, `text`:
  **webgui 0.2.39 has no cutplane/slice object type**, and 3D slicing exists solely through the
  hidden clipping-plane objects the GUI toggles.

## Capture limitations (so nobody reads more into these PNGs than they hold)

1. **`surface_scalar` is not a surface in 3D.** `ReadGmsh` of a `$MeshFormat 2.2` file whose
   elements are dim-2 yields a netgen mesh whose `Coordinates()` are **2 doubles per point**
   (49 points → 98 values, measured), i.e. the `z = 0.15·sin(πx)·sin(πy)` bump is dropped at
   import: every decoded Bézier point of `surface_scalar` has `z = 0`. So this scene is a flat
   square with `mesh_dim == 2` and says nothing about how a curved manifold would be centred or
   shaded; a genuine 2-in-3 baseline is still missing.
2. **Its empty `edges` is the capture file's fault, not the renderer's.** the written `.msh`
   declares a `boundary` physical group but contains no 1D elements, so the imported mesh has
   `GetBoundaries() == ()` and `GetBBoundaries() == ()` and `edges` is 3 empty arrays → `Edges`
   instance count 0 → no outline in `surface_scalar.png`. A 2-in-3 mesh with real boundary
   segments is untested. (Building one by hand needs a face descriptor, and
   `netgen.libngpy._meshing.Mesh` exposes only the getters `FaceDescriptor`/`FaceDescriptors`/
   `EdgeDescriptor`/`EdgeDescriptors`/`GetNFaceDescriptors` — no `AddFaceDescriptor` — while a
   hand-built `Mesh(2)` dies with "has no face descriptor: fd.size = 0".)
3. **Its surface colours are the one thing ngsplot does not reproduce, and this capture cannot
   say why.** Painting the recorded payload gives back the outline exactly — same bounding box,
   rows 104–489 and columns 254–639, and the amount of ink inside it within 1 % of the
   recording's — while 3.3 % of the interior (a 2-pixel erosion of the pixels both pictures
   paint) differs by more than 8 of 255, mean 2.1.  The differences run both ways (455 pixels
   brighter, 267 darker), so it is not one shading factor such as a dimmed back side, and what
   is left is how an imported P2 triangle patch fills the six Bézier attribute arrays that
   `trigsplines.vert` reads as `p00 p01 p02 p10 p11 p20`.  Unprovable from a capture whose
   geometry was truncated at import: `tests/test_dims.py` pins the outline of this scene and
   leaves its colours out of the parity list.  A 2-in-3 mesh built by netgen itself, with real
   boundary segments, would settle it and is still owed.

## Failures recorded verbatim

1. `cf_nomesh` → `RuntimeError: Cannot draw CoefficientFunction without mesh`
   (`ngsolve/webgui.py:119`, raised inside `Draw()`). No PNG.
2. `volume_cuts` (`objects=[ngs.x, ngs.y]`) → `AttributeError: 'ngsolve.fem.CoefficientFunction'
   object has no attribute '_GetWebguiData'` (`ngsolve/webgui.py:295`, raised inside `Draw()`).
   No PNG.

## What this changes for ngsplot

* **1D needs its own design**, not a parity target: curve of the function over the mesh line.
* **Guard the autoscale**: treat `(DBL_MAX, DBL_MIN)` and `min == max` as "no range" (matplotlib
  would get a flat colour here, and its `vmin`/`vmax` escape hatch is exactly what the webgui
  lacks).
* `CoefficientFunction` input must be allowed to arrive **with** a mesh and coerced through the
  same path as a `GridFunction`; `Draw(cf)` without one is the user's error to receive.
* Edge/wireframe geometry never carries values — wireframe colour is structural, not data.
* 3D means "closed boundary surface + BBND outline" by default; volume slicing is out of scope.
* Norm and complex ranges follow `min = 0, max = max(|funcmin|, |funcmax|)` and `function_mode`
  3 / 5, so `vmin`/`vmax` overrides must be able to beat that rule.

## Files

`tests/gt/dims/{line_scalar,line_deformed,numberspace,numberspace_l2pwc,volume_scalar,surface_scalar,cf_mesh,cf_vector,vec_hcurl,complex_h1}.{png,json}`
plus a `.png.html` reproducer each, and `surface_geometry.msh`. PNG byte sizes: 8058 (both
lines), 41031, 43678, 29991, 42290, 44774, 45112 (`cf_vector` = `vec_hcurl`), 44885.
