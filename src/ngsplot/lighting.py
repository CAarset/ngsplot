"""Phong shading identical to the webgui renderer.

The light lives in *view* space and specular uses the per-fragment view vector, so one
colour band shades as a smooth gradient across the canvas.  See docs/SPEC.md, "Shading".
"""

import numpy as np

#: normalize(vec3(0.5, 0.5, 1.5)), as measured from scene.uniforms.light_dir
LIGHT_DIR = np.array(
    [0.30151134457776363, 0.30151134457776363, 0.9045340337332909], dtype=np.float64
)

#: scene.uniforms.light_mat = (ambient, diffuse, shininess, spec)
AMBIENT = 0.3
DIFFUSE = 0.7
SHININESS = 10.0
SPEC = 0.3

#: scene.uniforms.dark_backside
DARK_BACKSIDE = True


def normalize(v, axis=-1):
    """Unit vectors, mapping a zero vector to zero instead of NaN."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def _dot(a, b):
    """Row-wise dot product of two (N, 3) arrays."""
    return np.einsum("ij,ij->i", a, b)


def shade(color, normal, position, *, inside=False, dark_backside=DARK_BACKSIDE,
          light_dir=LIGHT_DIR, ambient=AMBIENT, diffuse_coeff=DIFFUSE,
          shininess=SHININESS, spec_coeff=SPEC):
    """Shade ``color`` exactly as ``calcLight`` in shader/utils.h does.

    ``normal`` and ``position`` must already be in *view* space (``normal`` with the
    back-face flip applied, ``inside`` saying whether the flip happened).  The result is
    *not* clamped, matching the shader: the framebuffer clamps on write, so an overbright
    highlight saturates a channel rather than darkening the others.
    """
    color = np.atleast_2d(np.asarray(color, dtype=np.float64))
    n = normalize(np.atleast_2d(np.asarray(normal, dtype=np.float64)))
    p = np.atleast_2d(np.asarray(position, dtype=np.float64))

    s = np.broadcast_to(np.asarray(light_dir, dtype=np.float64), n.shape)
    v = normalize(-p)
    dot_ns = _dot(n, s)
    sn = np.clip(dot_ns, 0.0, None)
    r = -s + 2.0 * dot_ns[:, None] * n
    diffuse = diffuse_coeff * sn
    spec = np.clip(_dot(r, v), 0.0, None) ** shininess
    # `if (diffuse == 0.0 || light_shininess == 0.0) spec = 0.0;` — a surface turned away
    # from the light never gets a highlight, even when the reflection still points at the
    # eye (which happens on back faces).
    spec = np.where((diffuse == 0.0) | (shininess == 0.0), 0.0, spec)

    inside = np.zeros(len(n), dtype=bool) if inside is False else (
        np.ones(len(n), dtype=bool) if inside is True
        else np.broadcast_to(np.asarray(inside, dtype=bool), (len(n),))
    )
    dimm = np.where(inside & dark_backside, 0.5, 1.0)

    return dimm[:, None] * (color * (ambient + diffuse[:, None])
                            + spec[:, None] * spec_coeff)
