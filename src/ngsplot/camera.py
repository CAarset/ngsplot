"""From mesh coordinates to screen pixels, the way the webgui does it.

The renderer never moves its camera. ``scene.ts`` puts a perspective camera at ``(0, 0, 3)``
with an identity quaternion, a vertical field of view of 40°, ``near = 1`` and ``far = 100``,
and those numbers never change: every picture change -- including dragging with the mouse --
happens in the *object* matrix, which ``camera.ts`` builds in ``reset()`` as

    mat = S(1 / mesh_radius) * T(-mesh_center)

so the mesh lands centred on the origin and inside the unit sphere, and then premultiplies
rotations and pans onto it. Keeping the camera fixed is what makes the projection a constant
and a rotation of the plot a plain 4x4 matrix product.

Matrices here are ordinary mathematical 4x4 matrices acting on column vectors,
``p' = M @ p``. three.js keeps ``Matrix4.elements`` column-major, so
``matrix_from_three(elements) == np.asarray(elements).reshape(4, 4).T``. The renderer stores
and uploads them as doubles but multiplies them in float32 shader uniforms; for pixel work,
cast the product with ``clip_points(points, mvp.astype(np.float32))``.
"""

import numpy as np

__all__ = [
    "CAMERA_POSITION",
    "DEG",
    "FAR",
    "FOV",
    "NEAR",
    "Camera",
    "chain",
    "clip_points",
    "matrix_from_three",
    "object_matrix",
    "perspective_matrix",
    "rotation_about_axis",
    "rotation_from_euler",
    "rotation_from_quaternion",
    "scaling",
    "three_from_matrix",
    "to_pixels",
    "transform_points",
    "translation",
    "view_matrix",
    "wrap_about",
]

# scene.ts:318-336 -- fixed for every scene, not part of the settings
FOV = 40.0
NEAR = 1.0
FAR = 100.0
CAMERA_POSITION = (0.0, 0.0, 3.0)

DEG = np.pi / 180.0


def matrix_from_three(elements):
    """A three.js ``Matrix4.elements`` list (column-major) as a numpy matrix."""
    return np.asarray(elements, dtype=np.float64).reshape(4, 4).T


def three_from_matrix(matrix):
    """The ``Matrix4.elements`` list numpy's row-major ``matrix`` would serialize to."""
    return np.asarray(matrix, dtype=np.float64).T.reshape(-1).copy()


def chain(*matrices):
    """The combined matrix of applying ``matrices`` from left to right."""
    out = np.eye(4)
    for m in matrices:
        out = np.asarray(m, dtype=np.float64) @ out
    return out


def transform_points(points, matrix):
    """``matrix`` applied to ``(n, 3)`` points; the result is not divided by ``w``."""
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    return p @ matrix[:3, :3].T + matrix[:3, 3]


def clip_points(points, matrix):
    """``matrix`` applied in homogeneous coordinates, returning ``(n, 4)`` clip values."""
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    out = np.empty((p.shape[0], 4))
    out[:, :3] = p @ matrix[:3, :3].T + matrix[:3, 3]
    out[:, 3] = p @ matrix[3, :3] + matrix[3, 3]
    return out


def to_pixels(clip, width, height):
    """Divide clip coordinates by ``w`` and map them to pixels measured from the top left.

    Returns ``(x, y, depth, w)``. The GL viewport puts NDC ``(-1, 1)`` on ``[0, width]`` and
    ``[height, 0]``, so ``y`` grows downwards and ``depth`` is the value that ends up in the
    depth buffer: ``0`` at the near plane, ``1`` at the far plane. ``w`` comes back along so
    that callers can discard the points behind the camera (``w <= 0``) before drawing.
    """
    clip = np.atleast_2d(np.asarray(clip, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        w = clip[:, 3]
        x = clip[:, 0] / w
        y = clip[:, 1] / w
        z = clip[:, 2] / w
    return (0.5 * (x + 1.0) * width, 0.5 * (1.0 - y) * height, 0.5 * z + 0.5, w)


def translation(offset):
    """``Matrix4.makeTranslation``."""
    m = np.eye(4)
    m[:3, 3] = np.asarray(offset, dtype=np.float64).reshape(3)
    return m


def scaling(scale):
    """``Matrix4.makeScale``; a scalar gives a uniform scale."""
    s = np.broadcast_to(np.asarray(scale, dtype=np.float64), (3,))
    m = np.eye(4)
    m[np.arange(3), np.arange(3)] = s
    return m


def rotation_about_axis(axis, angle):
    """``Matrix4.makeRotationAxis``: a right-handed turn about ``axis`` by ``angle`` radians."""
    v = np.asarray(axis, dtype=np.float64).reshape(3)
    v = v / np.linalg.norm(v)
    x, y, z = v
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rotation_from_quaternion(quaternion):
    """``Matrix4.makeRotationFromQuaternion``, quaternion as ``(x, y, z, w)``."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0.0],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0.0],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


_AXIS = {"X": 0, "Y": 1, "Z": 2}


def rotation_from_euler(angles, order="XYZ"):
    """``Matrix4.makeRotationFromEuler`` for one of the six static Euler orders.

    ``angles`` are the turns about the fixed axes X, Y and Z, in degrees, and ``order`` is
    the sequence in which they are applied -- ``'XYZ'`` is the matrix ``Rx @ Ry @ Rz``,
    ``'YXZ'`` the matrix ``Ry @ Rx @ Rz``, and so on.
    """
    name = str(order).upper()
    if len(name) != 3 or len(set(name)) != 3 or set(name) - set(_AXIS):
        raise ValueError(f"unknown Euler order {order!r}, expected e.g. 'XYZ'")
    a = np.asarray(angles, dtype=np.float64).reshape(3) * DEG
    out = np.eye(4)
    for axis in name:
        vector = np.zeros(3)
        vector[_AXIS[axis]] = 1.0
        out = out @ rotation_about_axis(vector, a[_AXIS[axis]])
    return out


def wrap_about(matrix, point):
    """``wrapTransformation`` (camera.ts:6-13): do ``matrix`` while ``point`` stays put."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    return translation(p) @ np.asarray(matrix, dtype=np.float64) @ translation(-p)


def object_matrix(mesh_center, mesh_radius, *, euler_angles=None, transformations=None,
                  elev=None, azim=None, roll=None):
    """The matrix the surface, wireframe and edge objects are drawn with.

    ``mesh_center`` and ``mesh_radius`` are the two numbers ``BuildRenderData`` reports;
    ``euler_angles`` and ``transformations`` are the entries of ``settings['camera']``,
    applied in the order the renderer applies them (see ``camera.ts:174-220``).

    ``elev``, ``azim`` and ``roll`` are the three turns of ``Axes3D.view_init``, in degrees,
    put on last and made about the centre of the mesh: ``elev`` looks at the mesh from above,
    ``azim`` turns it about the vertical of the canvas, ``roll`` rolls it about the axis to
    the eye.  Three zeros are the view the page opens with, which looks straight at ``xy``:
    the camera never moves here, so these are turns of the object, and they are the page's
    triple ``[-elev, azim, roll]`` in the order ``ZYX`` -- tipped first, then turned, then
    rolled.
    """
    center = np.asarray(mesh_center, dtype=np.float64).reshape(3)
    radius = float(mesh_radius)
    if not radius > 0:
        raise ValueError(f"mesh_radius must be positive, got {mesh_radius}")

    # reset(): mat = S(1 / mesh_radius) * T(-mesh_center), so the mesh is centred first
    mat = chain(translation(-center), scaling(1.0 / radius))

    if euler_angles is not None:
        angles = list(euler_angles)
        order = angles[3] if len(angles) > 3 else "XYZ"
        rot = rotation_from_euler(angles[:3], order)
        # the turn happens about the centre of the mesh, which reset() has put at the origin
        mat = chain(mat, wrap_about(rot, transform_points(center, mat)[0]))

    for move in transformations or ():
        kind = move.get("type")
        if kind == "move":
            dist = move.get("dist") or 1.0
            mat = chain(mat, translation(np.asarray(move["dir"], float) * dist))
        elif kind in ("rotateX", "rotateY", "rotateZ"):
            vector = np.zeros(3)
            vector[_AXIS[kind[-1]]] = 1.0
            rot = rotation_about_axis(vector, float(move["angle"]) * DEG)
            mat = chain(mat, wrap_about(rot, transform_points(center, mat)[0]))

    # the screen keeps its axes where the world has them, since the camera does not turn; a
    # positive elevation looks down on the mesh, and the camera looks along -z with +y up, so
    # that is a *negative* turn about x
    for name, angle in zip("XYZ", (elev, azim, roll)):
        if angle is None or not float(angle):
            continue
        degrees = -float(angle) if name == "X" else float(angle)
        vector = np.zeros(3)
        vector[_AXIS[name]] = 1.0
        rot = rotation_about_axis(vector, degrees * DEG)
        mat = chain(mat, wrap_about(rot, transform_points(center, mat)[0]))
    return mat


def view_matrix(position=CAMERA_POSITION, quaternion=(0.0, 0.0, 0.0, 1.0)):
    """``camera.matrixWorldInverse``: world coordinates into the camera's view space."""
    rotation = rotation_from_quaternion(quaternion)
    offset = np.asarray(position, dtype=np.float64).reshape(3)
    inverse = np.eye(4)
    inverse[:3, :3] = rotation[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ offset
    return inverse


def perspective_matrix(aspect, fov=FOV, near=NEAR, far=FAR, zoom=1.0):
    """``PerspectiveCamera.updateProjectionMatrix`` with ``left = -right`` and ``top = -bottom``.

    ``fov`` is the *vertical* field of view in degrees and ``aspect`` is the width of the
    canvas divided by its height; ``zoom`` divides the field of view, as in three.js.
    """
    # the arithmetic below follows three.js term by term: a reordered product would not be
    # bit-identical to what the renderer's projection matrix holds
    top = near * np.tan(0.5 * DEG * fov) / zoom
    height = top * 2.0
    width = aspect * height
    m = np.zeros((4, 4))
    m[0, 0] = 2.0 * near / width
    m[1, 1] = 2.0 * near / height
    m[2, 2] = -(far + near) / (far - near)
    m[2, 3] = -2.0 * far * near / (far - near)
    m[3, 2] = -1.0
    return m


class Camera:
    """Everything needed to put a mesh on a canvas of ``width`` x ``height`` pixels."""

    def __init__(self, mesh_center, mesh_radius, *, width, height,
                 fov=FOV, near=NEAR, far=FAR, zoom=1.0,
                 position=CAMERA_POSITION, quaternion=(0.0, 0.0, 0.0, 1.0),
                 euler_angles=None, transformations=None,
                 elev=None, azim=None, roll=None):
        self.mesh_center = np.asarray(mesh_center, dtype=np.float64).reshape(3)
        self.mesh_radius = float(mesh_radius)
        self.width = int(width)
        self.height = int(height)
        self.fov = float(fov)
        self.near = float(near)
        self.far = float(far)
        self.zoom = float(zoom)
        self.position = np.asarray(position, dtype=np.float64).reshape(3)
        self.quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
        self.object = object_matrix(self.mesh_center, self.mesh_radius,
                                    euler_angles=euler_angles,
                                    transformations=transformations,
                                    elev=elev, azim=azim, roll=roll)
        self.view = view_matrix(self.position, self.quaternion)
        self.projection = perspective_matrix(self.aspect, self.fov, self.near, self.far,
                                             self.zoom)
        self.model_view_projection = self.projection @ self.view @ self.object

    @classmethod
    def from_render_data(cls, render_data, *, width, height, **kwargs):
        """A camera for one ``BuildRenderData`` result, honouring its ``gui_settings``.

        A keyword given here wins over the recorded one, which is how a script keeps a
        payload and still picks the angle it wants.
        """
        recorded = render_data.get("gui_settings", {}).get("camera", {}) or {}
        recorded = {key: value for key, value in recorded.items()
                    if key in ("euler_angles", "transformations")}
        return cls(render_data["mesh_center"], render_data["mesh_radius"],
                   width=width, height=height, **recorded, **kwargs)

    @property
    def aspect(self):
        """The canvas aspect, which is what the renderer feeds to both cameras."""
        return self.width / self.height

    @property
    def model_view(self):
        """``modelViewMatrix`` of the shaders: the object matrix as the camera sees it."""
        return self.view @ self.object

    def project(self, points):
        """Mesh points as ``(n, 4)``: pixel ``x``, pixel ``y`` from the top, depth, ``w``.

        Rows with ``w <= 0`` sit behind the camera and must not be drawn; see
        :func:`to_pixels` for what the four columns mean.
        """
        return np.stack(to_pixels(clip_points(points, self.model_view_projection),
                                  self.width, self.height), axis=-1)
