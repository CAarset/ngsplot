"""Reading the recorded ground truth: what the real renderer drew, and with what.

``tools/capture_gt.py`` writes three files per scene -- the screenshot ``<scene>.png``, the
page that produced it ``<scene>.png.html`` (which embeds the render data the ngsolve side
sent), and ``<scene>.json``, the uniform values read out of the live renderer while it drew
that scene.  ``tools/probe_camera.py`` adds ``camera.json`` with the matrices of every scene.

``tests/gt/dims`` holds the captures taken for ``docs/DIMS.md``, one per mesh dimension and
per kind of input.  They are the same screenshots and pages, but their JSON is a summary of
the payload rather than the uniforms read off a live renderer.
"""

import json
import os
import re

import numpy as np

GT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt")

#: a dims capture is a 900x600 screenshot of a page whose buffer sits at its top left
GUTTER = 6


def render_data_of(scene_name):
    """The render data of a ground-truth scene, read back from its own page."""
    with open(os.path.join(GT, f"{scene_name}.png.html")) as f:
        text = f.read()
    blob = re.search(r"var render_data = (\{[^\n]*\})\n", text).group(1)
    return json.loads(blob)


def settings_of(scene_name):
    """``gui_settings`` of a ground-truth scene, as the renderer would have seen them."""
    return render_data_of(scene_name).get("gui_settings", {})


def uniforms_of(scene_name):
    """The uniform values the renderer held when it drew the scene."""
    with open(os.path.join(GT, f"{scene_name}.json")) as f:
        return json.load(f)


def framebuffer_of(scene_name):
    """``(width, height)`` of the drawing buffer the scene was rendered into.

    The uniform records of ``tools/capture_gt.py`` name the buffer outright.  The summaries
    written for ``tests/gt/dims`` (``docs/DIMS.md``) do not, so those fall back to the
    screenshot, which carries the buffer at its top left plus a 6 px gutter of page."""
    try:
        canvas = uniforms_of(scene_name)["canvas"]
    except KeyError:
        height, width = png_of(scene_name).shape[:2]
        return width - GUTTER, height - GUTTER
    return canvas["buf_w"], canvas["buf_h"]


def png_of(scene_name):
    """The screenshot of a scene as ``(height, width, 4)`` bytes, top row first."""
    from PIL import Image

    with Image.open(os.path.join(GT, f"{scene_name}.png")) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8)
