"""Capture ground truth for the dimensions/scene kinds that tools/capture_gt.py does not cover.

Same machinery as capture_gt (headless chromium screenshot of the real webgui renderer plus
a JSON sidecar), but for 1D meshes, NumberSpace/piecewise-constant fields, 3D volumes, a 2D
surface embedded in 3D, bare CoefficientFunction draws, vector and complex fields. Output goes
to tests/gt/dims/ so the existing tests/gt/*.png baselines are untouched.

Every sidecar carries a full inventory of the payload keys with array sizes instead of the
(base64) data, plus the live-renderer probe (objects, uniforms, labels) — the facts we need
to reimplement Draw's visual for non-2D inputs. Scenes that fail are recorded with their
exact traceback and produce no PNG on purpose.

    PLAYWRIGHT_BROWSERS_PATH=$PWD/.pw-browsers .venv/bin/python tools/capture_dims.py [name ...]
"""

import base64
import json
import os
import sys
import traceback

import numpy as np

import ngsolve as ngs
import ngsolve.webgui as wg          # registers the draw types
import netgen.libngpy._meshing as ngm
from netgen.geom2d import unit_square
from netgen.read_gmsh import ReadGmsh

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))
import netgen.webgui as ngwg        # noqa: E402  (GenerateHTML)

OUTDIR = os.path.join(_ROOT, "tests", "gt", "dims")

_UNIFORMS = """light_dir light_mat dark_backside function_mode deformation complex_scale
colormap_min colormap_max colormap_size tex_colormap n_segments line_thickness"""

# Same machinery as capture_gt.PROBE, but the object list walks scene.render_objects —
# the wrapper objects of webgui@0.2.39 (capture_gt's probe used scene.objects, which does
# not exist in 0.2.39 and only produced objects_err). Visibility lives on the wrapped
# three_object; instance counts on either three_object or its geometry.
PROBE = """
() => {
  const u = scene.uniforms, out = {};
  const vec = v => ArrayBuffer.isView(v) ? Array.from(v)
               : (v && v.toArray) ? v.toArray() : v;
  for (const k of __UNIFORMS__) {
    if (!(k in u)) continue;
    const v = u[k].value;
    if (k === 'tex_colormap') continue;
    out[k] = vec(v);
  }
  try {
    out.render_objects = scene.render_objects.map(o => {
      const to = o.three_object, g = to && to.geometry;
      const ic = g && g.instanceCount !== undefined ? g.instanceCount
                 : (to && to.instanceCount !== undefined ? to.instanceCount : null);
      return { name: o.name, visible: to ? to.visible : null,
               draw_range: g && g.drawRange ? g.drawRange.count : null,
               instance_count: ic,
               numeric_uniforms: Object.fromEntries(Object.entries(o.uniforms || {})
                 .filter(([k, v]) => typeof v.value === 'number')
                 .map(([k, v]) => [k, v.value])) };
    });
  } catch (e) { out.render_objects_err = String(e); }
  try {
    const c = document.querySelector('canvas'), r = c.getBoundingClientRect();
    out.canvas = {x: r.x, y: r.y, w: r.width, h: r.height, buf_w: c.width,
                  buf_h: c.height, dpr: window.devicePixelRatio};
  } catch (e) { out.canvas_err = String(e); }
  try {
    out.labels = [...document.querySelectorAll('div')]
      .filter(d => !d.children.length && d.textContent.trim() &&
                   getComputedStyle(d).position === 'absolute')
      .map(d => d.textContent);
  } catch (e) { out.labels_err = String(e); }
  return out;
}
""".replace("__UNIFORMS__", json.dumps(_UNIFORMS.split()))


def shoot(data, png, width=900, height=600):
    """Headless screenshot + PROBE of the rendered scene (as capture_gt.shoot does)."""
    import asyncio

    from playwright.async_api import async_playwright

    html = png + ".html"
    ngwg.GenerateHTML(data, filename=html)
    with open(html) as f:
        text = f.read().replace("preserveDrawingBuffer: false", "preserveDrawingBuffer: true")
    with open(html, "w") as f:
        f.write(text)

    async def main():
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                args=["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"])
            page = await browser.new_page(viewport={"width": width, "height": height})
            await page.goto("file://" + os.path.abspath(html))
            await page.add_style_tag(content=".lil-gui{display:none!important}")
            await page.wait_for_function("typeof scene !== 'undefined'")
            await page.evaluate(
                "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
            await page.wait_for_timeout(1500)
            meta = await page.evaluate(PROBE)
            await page.screenshot(path=png)
            await browser.close()
        return meta

    return asyncio.run(main())


# --------------------------------------------------------------------------- meshes

def mesh_1d(n=4):
    """A straight line on the x-axis: netgen dim-1 mesh, n segments over [0,1].

    netgen.geom2d has no unit_line and netgen.meshing has no UnitInterval in this
    wheel, so the mesh is built by hand; element index (region) 1 becomes material
    'default'.
    """
    nm = ngm.Mesh(1)
    nm.AddPoints(np.array([[i / n, 0.0, 0.0] for i in range(n + 1)]))
    nm.AddElements(1, 1, np.array([[i + 1, i + 2] for i in range(n)], dtype=np.int32), base=1)
    return ngs.Mesh(nm)


def mesh_square():
    return ngs.Mesh(unit_square.GenerateMesh(maxh=0.25))


def mesh_3d():
    # netgen.VolumeTestMesh is not exposed in the Python bindings either; unit_cube is the
    # documented pythonic stand-in.
    return ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))


def mesh_surface():
    """Curved triangulated surface (2D manifold in 3D), imported from a hand-written
    gmsh 2.2 file: 6x6 sine bump z = 0.15 sin(pi x) sin(pi y) over [0,1]^2.

    A hand-rolled netgen.libngpy._meshing.Mesh(2) core-dumps once it references a face
    descriptor (no AddFaceDescriptor in this wheel), so the surface comes in via the
    gmsh reader, which creates face descriptors from the physical groups.
    """
    n = 6
    nodes, tris = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            nodes.append((i / n, j / n, 0.15 * np.sin(np.pi * i / n) * np.sin(np.pi * j / n)))
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i + 1
            tris += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    path = os.path.join(OUTDIR, "surface_geometry.msh")
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write('$PhysicalNames\n2\n2 1 "surface"\n1 2 "boundary"\n$EndPhysicalNames\n')
        f.write(f"$Nodes\n{len(nodes)}\n")
        for k, (x, y, z) in enumerate(nodes):
            f.write(f"{k+1} {x:.10f} {y:.10f} {z:.10f}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(tris)}\n")
        for k, t in enumerate(tris):
            f.write(f"{k+1} 2 2 1 1 {t[0]} {t[1]} {t[2]}\n")
        f.write("$EndElements\n")
    return ngs.Mesh(ReadGmsh(path))


# --------------------------------------------------------------------------- scenes
#
# Each builder returns what to hand to Draw -- (object, positional, keywords) -- and not a
# payload, so that a test can send the very same arguments down another path and compare.

def _line(deformation=False):
    gfu = ngs.GridFunction(ngs.H1(mesh_1d(), order=2))
    gfu.Set(ngs.x * ngs.x)
    return gfu, (), {"deformation": deformation}


def _numberspace():
    return ngs.GridFunction(ngs.NumberSpace(mesh_square())), (), {}


def _numberspace_l2pwc():
    """Supplementary: what a genuinely piecewise-constant-per-element field draws as.

    NumberSpace reports ndof==1 for any order, so a GridFunction over it is a single
    constant; L2(order=0) is the real per-element-constant space. Dofs are set through
    fes.GetDofNrs(ngs.ElementId(element)) since VectorAccess is absent in this build.
    """
    mesh = mesh_square()
    fes = ngs.L2(mesh, order=0)
    gfu = ngs.GridFunction(fes)
    for i, ed in enumerate(mesh.Elements()):
        for dof in fes.GetDofNrs(ngs.ElementId(ed)):
            gfu.vec[dof] = i
    return gfu, (), {}


def _volume(objects=None):
    gfu = ngs.GridFunction(ngs.H1(mesh_3d(), order=2))
    gfu.Set(ngs.x + 2 * ngs.y + 3 * ngs.z)
    return gfu, (), {"objects": objects} if objects else {}


def _surface():
    gfu = ngs.GridFunction(ngs.H1(mesh_surface(), order=2))
    gfu.Set(ngs.x * ngs.y)
    return gfu, (), {}


def _cf_nomesh():
    # expected: RuntimeError from GetData (Draw itself is lazy)
    return ngs.x * ngs.y, (), {}


def _cf_mesh():
    return ngs.x * ngs.y, (mesh_square(),), {}


def _cf_vector():
    return ngs.CF((-ngs.y, ngs.x)), (mesh_square(),), {}


def _vec_hcurl():
    gfu = ngs.GridFunction(ngs.HCurl(mesh_square(), order=1))
    gfu.Set(ngs.CF((-ngs.y, ngs.x)))
    return gfu, (), {}


def _complex_h1():
    gfu = ngs.GridFunction(ngs.H1(mesh_square(), order=2, complex=True))
    gfu.Set(ngs.x + 1j * ngs.y)
    return gfu, (), {}


SCENES = {
    "line_scalar": _line,
    "line_deformed": lambda: _line(deformation=True),
    "numberspace": _numberspace,
    "numberspace_l2pwc": _numberspace_l2pwc,
    "volume_scalar": _volume,
    "volume_cuts": lambda: _volume(objects=[ngs.x, ngs.y]),
    "surface_scalar": _surface,
    "cf_nomesh": _cf_nomesh,
    "cf_mesh": _cf_mesh,
    "cf_vector": _cf_vector,
    "vec_hcurl": _vec_hcurl,
    "complex_h1": _complex_h1,
}


def draw(name):
    """The payload of a scene, built by ``Draw`` exactly as the capture built it."""
    obj, args, kwargs = SCENES[name]()
    return wg.Draw(obj, *args, show=False, **kwargs).GetData()


SCENES = {
    "line_scalar": _line,
    "line_deformed": lambda: _line(deformation=True),
    "numberspace": _numberspace,
    "numberspace_l2pwc": _numberspace_l2pwc,
    "volume_scalar": _volume,
    "volume_cuts": lambda: _volume(objects=[ngs.x, ngs.y]),
    "surface_scalar": _surface,
    "cf_nomesh": _cf_nomesh,
    "cf_mesh": _cf_mesh,
    "cf_vector": _cf_vector,
    "vec_hcurl": _vec_hcurl,
    "complex_h1": _complex_h1,
}


# --------------------------------------------------------------------------- capture

def summarize(data):
    """Payload with big arrays replaced by size inventories."""
    out = {}
    for k in sorted(data):
        v = data[k]
        if isinstance(v, list) and v and all(isinstance(s, str) for s in v):
            out[k] = {"b64_items": len(v),
                      "floats_per_item": [len(base64.b64decode(s)) // 4 for s in v]}
        elif isinstance(v, dict):
            out[k] = v
        elif isinstance(v, list):
            out[k] = {"list_len": len(v), "items": v[:8]}
        else:
            out[k] = v
    return out


def png_stats(png):
    from PIL import Image
    a = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)
    return {"mean_rgb": [round(c, 2) for c in a.reshape(-1, 3).mean(axis=0)],
            "std_rgb": [round(c, 2) for c in a.reshape(-1, 3).std(axis=0)],
            "std_all": round(float(a.std()), 2)}


def capture(name):
    sidecar = os.path.join(OUTDIR, f"{name}.json")
    meta = {"name": name, "viewport": [900, 600]}
    try:
        data = draw(name)
        meta["draw_ok"] = True
        meta["render_data_summary"] = summarize(data)
    except Exception:
        meta["draw_ok"] = False
        meta["error"] = traceback.format_exc()
        with open(sidecar, "w") as f:
            json.dump(meta, f, indent=1, default=str)
        print(f"{name}: FAILED\n{meta['error']}")
        return
    png = os.path.join(OUTDIR, f"{name}.png")
    try:
        probe = shoot(data, png)
        meta.update(probe)
        meta["png_stats"] = png_stats(png)
    except Exception:
        meta["render_error"] = traceback.format_exc()
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(f"{name}: ok, objects={[(o['name'], o['visible'], o['draw_range'], o['instance_count']) for o in meta.get('render_objects', [])]}, "
          f"png std={meta.get('png_stats', {}).get('std_all')}, "
          f"cmap={[round(v, 3) if isinstance(v, float) else v for v in
                    (meta.get('colormap_min', [None]), meta.get('colormap_max', [None]))]}")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    for name in sys.argv[1:] or list(SCENES):
        capture(name)
