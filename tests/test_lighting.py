"""Phong shading, checked against hand-computed values of ``calcLight``.

The constants are the ones measured from the live renderer (tests/gt/*.json):
``light_dir = normalize(0.5, 0.5, 1.5)``, ``light_mat = (0.3, 0.7, 10, 0.3)``,
``dark_backside = true``.  All expected numbers below are worked out by hand from
``dimm * (color * (ambient + diffuse) + spec * spec_coeff)``.
"""

import numpy as np
import pytest

from ngsplot import colormap, lighting

VIEW_POS = (0.0, 0.0, -3.0)          # camera sits 3 units back along -z in view space
# float64: the palette is uploaded as float32 texels, the shader math is float32 too,
# but the expected values here are worked out in double precision.
GREEN = colormap.palette(8)[4].astype(np.float64)   # (0.2857142984867096, 1, 0)


def test_front_face_diffuse_and_specular():
    rgb = lighting.shade(GREEN, (0, 0, 1), VIEW_POS)[0]
    # s.n = 0.9045340337332909 -> diffuse = 0.7 * s.n = 0.6331738236133037
    # r = -s + 2(s.n)n, v = (0,0,1) -> r.v = 0.9045340337332909 -> spec = 0.36664783205
    # The expected triple is the formula evaluated in exact decimal arithmetic.
    # green saturates above 1: the shader does not clamp, the framebuffer does.
    assert rgb == pytest.approx([0.3766154539957957, 1.043168173229264,
                                 0.1099943496159602], rel=1e-12)
    assert rgb[1] > 1.0
    # RGBA8 conversion: scale, round-to-nearest, clamp (the framebuffer's job)
    assert np.array_equal((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8),
                          [96, 255, 28])


def test_back_face_is_dimmed_by_exactly_a_half():
    front = lighting.shade(GREEN, (0, 0, 1), VIEW_POS)[0]
    back = lighting.shade(GREEN, (0, 0, 1), VIEW_POS, inside=True)[0]
    assert back == pytest.approx(0.5 * front, rel=0, abs=0)   # exact: dimm is 0.5


def test_back_face_facing_away_gets_only_ambient():
    # normal pointing into the screen: diffuse clips to 0, and the shader kills the
    # highlight whenever diffuse is 0 even though the reflection still points at the eye.
    rgb = lighting.shade(GREEN, (0, 0, -1), VIEW_POS)[0]
    assert rgb == pytest.approx(GREEN * 0.3, rel=1e-15)
    rgb_in = lighting.shade(GREEN, (0, 0, -1), VIEW_POS, inside=True)[0]
    assert rgb_in == pytest.approx(GREEN * 0.15, rel=1e-15)


def test_grazing_normal_has_no_highlight():
    rgb = lighting.shade(GREEN, (1, 0, 0), VIEW_POS)[0]
    # s.n = 0.30151134457776363 -> ambient + diffuse = 0.5110579412044346;
    # the reflected ray points away from the eye, so max(r.v, 0) = 0.
    assert rgb == pytest.approx(GREEN * 0.5110579412044346, rel=1e-15)


def test_highlight_is_zero_without_diffuse_or_shininess():
    assert lighting.shade(GREEN, (0, 0, 1), VIEW_POS, diffuse_coeff=0.0)[0][2] == 0.0
    assert lighting.shade(GREEN, (0, 0, 1), VIEW_POS, shininess=0.0)[0][2] == 0.0


def test_light_direction_is_unit_length():
    assert np.linalg.norm(lighting.LIGHT_DIR) == pytest.approx(1.0, rel=1e-15)
    assert lighting.LIGHT_DIR == pytest.approx(lighting.normalize([0.5, 0.5, 1.5]),
                                               rel=1e-15)


def test_normal_pointing_away_from_the_light_is_ambient_only():
    # A surface whose normal is the light direction: pure max diffuse + max-ish specular.
    s = lighting.normalize([0.5, 0.5, 1.5])
    rgb = lighting.shade([1.0, 1.0, 1.0], s * -1, VIEW_POS)[0]
    assert np.all(rgb <= 1.0)          # turned away: ambient only, no highlight
    assert rgb == pytest.approx([0.3, 0.3, 0.3], rel=1e-15)


def test_shade_is_vectorised_and_normalises_its_inputs():
    n = np.array([[0, 0, 2.0], [0, 0, 1.0]])          # unnormalised first row
    rgb = lighting.shade(np.tile(GREEN, (2, 1)), n, np.tile(VIEW_POS, (2, 1)))
    assert rgb.shape == (2, 3)
    assert rgb[0] == pytest.approx(rgb[1], rel=1e-15)


def test_normalize_maps_zero_vector_to_zero():
    assert lighting.normalize([0.0, 0.0, 0.0]).tolist() == [0.0, 0.0, 0.0]
    assert np.allclose(lighting.normalize([[3, 0, 4], [0, -2, 0]]),
                       [[0.6, 0, 0.8], [0, -1, 0]])
