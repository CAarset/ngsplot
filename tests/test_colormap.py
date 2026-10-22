"""Known-answer tests: the palette is the live GPU texture, dumped verbatim."""

import numpy as np
import pytest

from ngsplot import colormap as cm

#: texel contents read out of the live scene (scene.uniforms.tex_colormap.value.image.data)
LIVE_TEXTURE = [
    0, 0, 1, 1,
    0, 0.5714285969734192, 1, 1,
    0, 1, 0.8571428656578064, 1,
    0, 1, 0.2857142984867096, 1,
    0.2857142984867096, 1, 0, 1,
    0.8571428656578064, 1, 0, 1,
    1, 0.5714285969734192, 0, 1,
    1, 0, 0, 1,
]


def test_palette_matches_live_texture_bit_for_bit():
    live = np.array(LIVE_TEXTURE, dtype=np.float32).reshape(8, 4)
    assert cm.palette(8) == pytest.approx(live[:, :3], abs=0.0)


def test_palette_endpoints_and_shape():
    p = cm.palette(8)
    assert p.shape == (8, 3)
    assert list(p[0]) == [0.0, 0.0, 1.0]
    assert list(p[-1]) == [1.0, 0.0, 0.0]


@pytest.mark.parametrize("n", [2, 3, 8, 32])
def test_palette_is_ramp_sampled_at_texel_centres(n):
    expected = cm.ramp(np.arange(n) / (n - 1)).astype(np.float32)
    assert cm.palette(n) == pytest.approx(expected, abs=0.0)


def test_ramp_segments():
    assert cm.ramp(0.0) == pytest.approx([0, 0, 1])
    assert cm.ramp(0.25) == pytest.approx([0, 1, 1])   # blue -> cyan
    assert cm.ramp(0.5) == pytest.approx([0, 1, 0])    # cyan -> green
    assert cm.ramp(0.75) == pytest.approx([1, 1, 0])   # green -> yellow
    assert cm.ramp(1.0) == pytest.approx([1, 0, 0])
    assert cm.ramp(2.0) == pytest.approx([1, 0, 0])    # last segment clamps


def test_band_index_is_nearest_neighbour_step():
    # 8 bands over [-1, 1]: edges at -1 + k*0.25
    edges = -1.0 + 0.25 * np.arange(9)
    below = edges[:-1] + 1e-9
    assert list(cm.band_index(below, -1.0, 1.0, 8)) == list(range(8))
    on_edge = edges[1:-1]
    assert list(cm.band_index(on_edge, -1.0, 1.0, 8)) == list(range(1, 8))


def test_band_index_clamps_outside_range():
    assert cm.band_index([-1e6, -1.0, 1.0, 1e6], -1.0, 1.0, 8) .tolist() == [0, 0, 7, 7]


def test_band_index_degenerate_range_is_first_band():
    assert cm.band_index([5.0, 7.0], 5.0, 5.0, 8).tolist() == [0, 0]


def test_lookup_uses_band_colour():
    rgb = cm.lookup([-1.0, -0.5, 0.999, 1.0], -1.0, 1.0)
    pal = cm.palette(8)
    assert rgb[0] == pytest.approx(pal[0])
    assert rgb[1] == pytest.approx(pal[2])
    assert rgb[2] == pytest.approx(pal[7])
    assert rgb[3] == pytest.approx(pal[7])


def test_to_matplotlib_roundtrip():
    cmap = cm.to_matplotlib()
    assert cmap.N == 8
    # matplotlib normalises [vmin, vmax] -> [0, 1] and quantises at the band centres
    assert cmap(0.0)[:3] == pytest.approx(cm.palette(8)[0], abs=1e-6)
    assert cmap(1.0 - 1e-9)[:3] == pytest.approx(cm.palette(8)[7], abs=1e-6)


def test_palette_rejects_single_band():
    with pytest.raises(ValueError):
        cm.palette(1)
