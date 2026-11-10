"""Capture ground truth from the real webgui renderer, for the pixel-parity tests.

Renders each named scene with ``ngsolve.webgui.Draw``, screenshots the WebGL canvas with
headless chromium, and stores the PNG plus a JSON sidecar describing everything the
renderer put on screen (canvas rect, uniforms, colorbar tick labels and their pixel boxes).
The sidecar is what lets a parity test compare like with like: the page has a body margin,
a lil-gui panel and HTML overlay labels that are not part of the plot.

    PLAYWRIGHT_BROWSERS_PATH=$PWD/.pw-browsers .venv/bin/python tools/capture_gt.py [name ...]
"""

import asyncio
import json
import os
import sys

import ngsolve as ngs
import ngsolve.webgui as wg          # registers the draw types
import netgen.webgui as ngwg
from netgen.geom2d import unit_square

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "gt")
WIDTH, HEIGHT = 900, 600


def _scene(**kwargs):
    """``x^2 - y^2`` on the unit square, H1 order 2 — the scene the spec was derived from."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=kwargs.pop("order", 2))
    gfu = ngs.GridFunction(fes)
    gfu.Set(ngs.x * ngs.x - ngs.y * ngs.y)
    return wg.Draw(gfu, mesh, show=False, **kwargs).GetData()


def _mesh_only():
    return wg.Draw(ngs.Mesh(unit_square.GenerateMesh(maxh=0.2)), show=False).GetData()


SCENES = {
    # default Draw: order 2, 8 colours, subdivision 5, line thickness 5, white background
    "square_scalar": lambda: _scene(),
    "square_order1": lambda: _scene(order=1),
    # the classic "mesh warped by its own function value" look
    "square_deformed": lambda: _scene(deformation=True),
    "square_nc2": lambda: _scene(settings={"Colormap": {"ncolors": 2}}),
    "square_nc32": lambda: _scene(settings={"Colormap": {"ncolors": 32}}),
    "square_fixed_range": lambda: _scene(
        settings={"Colormap": {"autoscale": False, "min": -0.5, "max": 2.0}}),
    "square_rotated": lambda: _scene(settings={"camera": {"euler_angles": [20.0, 30.0, 0.0]}}),
    "square_no_wireframe": lambda: _scene(
        settings={"Misc": {"line_thickness": 1}, "Objects": {"Wireframe": False}}),
    "mesh_only": _mesh_only,
}

_UNIFORMS = """light_dir light_mat dark_backside function_mode deformation complex_scale
colormap_min colormap_max colormap_size tex_colormap n_segments line_thickness"""

PROBE = """
() => {
  const u = scene.uniforms, out = {};
  const vec = v => ArrayBuffer.isView(v) ? Array.from(v)
               : (v && v.toArray) ? v.toArray() : v;
  for (const k of __UNIFORMS__) {
    if (!(k in u)) continue;
    let v = u[k].value;
    if (k === 'tex_colormap') {
      out.tex_colormap = {size: vec(v.image.size),
                          data: Array.from(v.image.data),
                          mag: v.magFilter, min: v.minFilter};
      continue;
    }
    out[k] = vec(v);
  }
  try {
    out.clear_color = scene.renderer.getClearColor(
      new window.webgui.THREE.Color()).getHex();
    out.clear_alpha = scene.renderer.getClearAlpha();
  } catch (e) { out.clear_color_err = String(e); }
  try {
    const cam = scene.camera;
    out.camera = {fov: cam.fov, near: cam.near, far: cam.far, zoom: cam.zoom,
                  pos: cam.position.toArray(), quat: cam.quaternion.toArray(),
                  proj: Array.from(cam.projectionMatrix.elements)};
  } catch (e) { out.camera_err = String(e); }
  try {
    const c = document.querySelector('canvas'), r = c.getBoundingClientRect();
    out.canvas = {x: r.x, y: r.y, w: r.width, h: r.height,
                  buf_w: c.width, buf_h: c.height, dpr: window.devicePixelRatio};
  } catch (e) { out.canvas_err = String(e); }
  try {
    out.objects = scene.objects.map(o => ({
      name: o.name, visible: o.visible,
      count: o.geometry && o.geometry.drawRange ? o.geometry.drawRange.count : null,
      instances: o.geometry && o.geometry.instanceCount !== undefined
                 ? o.geometry.instanceCount : null,
      uniforms: Object.fromEntries(Object.entries(o.material.uniforms || {})
        .filter(([k, v]) => typeof v.value === 'number')
        .map(([k, v]) => [k, v.value])),
    }));
  } catch (e) { out.objects_err = String(e); }
  try {
    out.labels = [...document.querySelectorAll('div')]
      .filter(d => !d.children.length && d.textContent.trim() &&
                   getComputedStyle(d).position === 'absolute')
      .map(d => { const b = d.getBoundingClientRect();
                  return {text: d.textContent, x: b.x, y: b.y, w: b.width, h: b.height,
                          font: getComputedStyle(d).font, color: getComputedStyle(d).color}; });
  } catch (e) { out.labels_err = String(e); }
  return out;
}
""".replace("__UNIFORMS__", json.dumps(_UNIFORMS.split()))


def shoot(data, png, width=WIDTH, height=HEIGHT):
    """Render ``render_data`` headlessly; return the page metadata probe."""
    from playwright.async_api import async_playwright

    html = png + ".html"
    ngwg.GenerateHTML(data, filename=html)
    # GenerateHTML ignores its `template` argument, so patch the buffer flag in place:
    # without it the canvas may already be cleared when the screenshot is taken.
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


def capture(name):
    data = SCENES[name]()
    png = os.path.join(OUTDIR, f"{name}.png")
    meta = shoot(data, png)
    meta["name"] = name
    meta["viewport"] = [WIDTH, HEIGHT]
    meta["render_data"] = {k: data[k] for k in sorted(data)
                           if k not in ("Bezier_points", "Bezier_trig_points", "edges",
                                        "gui_settings", "vertices", "trigs", "surf_trigs",
                                        "surf_values", "bnd_trigs", "bnd_values", "volumes")}
    with open(png[:-4] + ".json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(f"{name}: {os.path.getsize(png)} bytes, "
          f"labels={[l['text'] for l in meta.get('labels', [])]}, "
          f"cmap=[{meta.get('colormap_min')}, {meta.get('colormap_max')}], "
          f"ncolors={len(meta.get('tex_colormap', {}).get('data', [])) // 4}")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    for name in sys.argv[1:] or list(SCENES):
        capture(name)
