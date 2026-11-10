"""Dump the camera and object transforms of the ground-truth scenes.

``tools/capture_gt.py`` stores the render data of every scene next to its screenshot, so
the whole renderer can be replayed from the ``.png.html`` files alone — no ngsolve
needed. This opens each of them in headless chromium and records what the three.js
camera ends up doing, which is the known answer for ``ngsplot.camera``:

    PLAYWRIGHT_BROWSERS_PATH=$PWD/.pw-browsers .venv/bin/python tools/probe_camera.py [name ...]

The result goes to ``tests/gt/camera.json``.
"""

import asyncio
import glob
import json
import os
import sys

GT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "gt")
WIDTH, HEIGHT = 900, 600

PROBE = """
() => {
  const els = (m) => (m && m.elements ? Array.from(m.elements) : null);
  const s = scene;
  const object3d = (o) => {
    for (const v of Object.values(o)) if (v && v.isObject3D) return v;
    return null;
  };
  const objects = (s.render_objects || []).map((o) => {
    const obj = object3d(o);
    return {
      name: o.name,
      object: obj ? obj.type : null,
      visible: obj ? obj.visible : null,
      matrixWorld: obj ? els(obj.matrixWorld) : null,
      drawRange: obj && obj.geometry ? [obj.geometry.drawRange.start, obj.geometry.drawRange.count] : null,
      instanceCount: obj && obj.geometry && obj.geometry.instanceCount !== undefined
        ? obj.geometry.instanceCount : null,
    };
  });
  return {
    controls_mat: els(s.controls.mat),
    controls_center: s.controls.center.toArray(),
    mesh_center: s.mesh_center.toArray(),
    mesh_radius: s.mesh_radius,
    euler_angles: s.controls.getEulerAngles(),
    camera: {
      type: s.camera.type,
      fov: s.camera.fov,
      near: s.camera.near,
      far: s.camera.far,
      zoom: s.camera.zoom,
      aspect: s.camera.aspect,
      position: s.camera.position.toArray(),
      quaternion: s.camera.quaternion.toArray(),
      up: s.camera.up.toArray(),
      projection: els(s.camera.projectionMatrix),
      view: els(s.camera.matrixWorldInverse),
    },
    uniforms: {
      aspect: s.uniforms.aspect ? s.uniforms.aspect.value : null,
      n_segments: s.uniforms.n_segments ? s.uniforms.n_segments.value : null,
      line_thickness: s.uniforms.line_thickness ? s.uniforms.line_thickness.value : null,
    },
    canvas: {
      width: s.renderer.domElement.width,
      height: s.renderer.domElement.height,
      pixel_ratio: s.renderer.getPixelRatio(),
      offset_width: s.element.offsetWidth,
      offset_height: s.element.offsetHeight,
    },
    objects,
  };
}
"""


def probe(html):
    from playwright.async_api import async_playwright

    async def main():
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                args=["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"])
            page = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
            await page.goto("file://" + os.path.abspath(html))
            await page.add_style_tag(content=".lil-gui{display:none!important}")
            await page.wait_for_function("typeof scene !== 'undefined'")
            await page.evaluate(
                "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
            await page.wait_for_timeout(1500)
            meta = await page.evaluate(PROBE)
            await browser.close()
        return meta

    return asyncio.run(main())


def main(names):
    if not names:
        names = [os.path.basename(f)[: -len(".png.html")] for f in
                 sorted(glob.glob(os.path.join(GT, "*.png.html")))]
    out = {}
    for name in names:
        html = os.path.join(GT, f"{name}.png.html")
        print(name, flush=True)
        out[name] = probe(html)
    path = os.path.join(GT, "camera.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
