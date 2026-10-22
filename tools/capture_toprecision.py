"""Capture V8's own Number.prototype.toPrecision outputs as a test fixture.

``labels.to_precision`` is a Python port of that method, so the ground truth is the
engine itself: a sweep over tie cases (0.125, 1.375, 2.5, ...), decade carries (0.999),
notation boundaries (1e-6, 1e21) and the digit counts a colorbar really produces.
"""
import asyncio, json, os
from playwright.async_api import async_playwright

VALUES = [
    0.0, 1.0, -1.0, 0.5, -0.5, 0.125, -0.125, 0.375, -0.375, 1.375, -1.375, 2.5, -2.5,
    0.0625, -0.0625, 0.1875, 1.005, 2.675, 0.999, 0.9995, 9.995, 99.95, 0.0999,
    1e-6, 1.0000001e-6, 1e-7, 1e21, 1e-21, 1.5e-16, -4.440892098500626e-16,
    1234.5678, -1234.5678, 12345678.0, 0.000123456, 1e6, 1e7, 0.001, 1/3, 2/3,
    1.9999999999999998, 0.9999999999999999, 123.456, -0.00049, 49e-5, 0.1, 0.2, 0.3,
    -0.9999999999999998, 0.9999999999999988, -0.5000000000000001, 0.49999999999999933,
    -0.9972629810400511, 0.0013735089505376, 0.75, 0.625, 2.0, 1e-5, 9.9e-6, 1.5e-4,
]
JS = """
(values) => values.map((v, i) => {
  const out = [v];
  for (let p = 1; p <= 8; p++) { try { out.push(v.toPrecision(p)); } catch (e) { out.push('ERR'); } }
  out.push(String(v));
  return out;
})
"""

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--enable-unsafe-swiftshader"])
        pg = await b.new_page()
        rows = await pg.evaluate(JS, VALUES)
        await b.close()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "tests", "gt", "toprecision.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"note": "Number.prototype.toPrecision as reported by the same V8 "
                           "engine that renders the webgui colorbar labels",
                   "precisions": list(range(1, 9)), "rows": rows}, f, indent=1)
    print(out, len(rows), "rows")

asyncio.run(main())
