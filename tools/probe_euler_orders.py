"""Known answers for the Euler rotation orders of ``ngsplot.camera``.

The renderer takes ``settings.camera.euler_angles = [x, y, z]`` — optionally with a fourth
entry naming the order — and turns it into a matrix with three.js ``Euler`` semantics
(``camera.ts:183-192``). Which product of elementary rotations three.js means by ``'ZYX'``
and friends is exactly the kind of thing to measure rather than remember, so this reads
one of the ground-truth scenes, swaps the angles for every supported order, and records
the resulting object matrix:

    PLAYWRIGHT_BROWSERS_PATH=$PWD/.pw-browsers .venv/bin/python tools/probe_euler_orders.py

which writes ``tests/gt/euler_orders.json``. It only replays an existing ``.png.html``, so
it needs no ngsolve; the object matrix is what ``ngsplot.camera.object_matrix`` has to
reproduce, and the angles are fed back through ``CameraControls.getEulerAngles()`` to show
what the GUI would report for them.
"""

import asyncio
import json
import os
import re

from probe_camera import GT, HEIGHT, WIDTH, PROBE

ANGLES = [10.0, 20.0, 30.0]
ORDERS = ["XYZ", "YXZ", "XZY", "ZXY", "YZX", "ZYX"]
SOURCE = os.path.join(GT, "square_scalar.png.html")


def variant(order):
    """The ground-truth page, with the camera angles replaced by `ANGLES` and `order`."""
    with open(SOURCE) as f:
        text = f.read()
    match = re.search(r"var render_data = (\{[^\n]*\})\n", text)
    data = json.loads(match.group(1))
    data["gui_settings"] = {"camera": {"euler_angles": [*ANGLES, order]}}
    replacement = "var render_data = " + json.dumps(data) + "\n"
    return text[: match.start()] + replacement + text[match.end():]


def probe(html, path):
    from playwright.async_api import async_playwright

    async def main():
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                args=["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader"])
            out = {}
            for order, page_html in html.items():
                with open(path.format(order=order), "w") as f:
                    f.write(page_html)
                page = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
                await page.goto("file://" + os.path.abspath(path.format(order=order)))
                await page.add_style_tag(content=".lil-gui{display:none!important}")
                await page.wait_for_function("typeof scene !== 'undefined'")
                await page.evaluate(
                    "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
                await page.wait_for_timeout(1200)
                out[order] = await page.evaluate(PROBE)
                await page.close()
            await browser.close()
        return out

    return asyncio.run(main())


def main():
    html = {order: variant(order) for order in ORDERS}
    result = probe(html, "/tmp/euler_{order}.html")
    for order in ORDERS:
        result[order]["euler_angles_requested"] = [*ANGLES, order]
    path = os.path.join(GT, "euler_orders.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(path)


if __name__ == "__main__":
    main()
