"""The colourbar tick labels must be the strings the webgui renderer really shows.

Two levels: ``to_precision`` against V8's own ``Number.prototype.toPrecision`` (fixture
captured by tools/capture_toprecision.py), and the assembled label rows against the text
of the ``<div>``s in a live screenshot (tests/gt/*.json).
"""

import json
from pathlib import Path

import pytest

from ngsplot import labels

GT = Path(__file__).parent / "gt"


def test_to_precision_matches_v8_over_the_whole_fixture():
    fixture = json.loads((GT / "toprecision.json").read_text())
    bad = []
    for row in fixture["rows"]:
        value = row[0]
        for p, expected in zip(fixture["precisions"], row[1:-1]):
            got = labels.to_precision(value, p)
            if got != expected:
                bad.append((value, p, got, expected))
    assert not bad, f"{len(bad)} mismatches, e.g. {bad[:5]}"


def test_to_precision_ties_go_away_from_zero():
    # Python's round() would give 0.12 and 1.37 here; the colourbar shows 0.13 and 1.38.
    assert labels.to_precision(0.125, 2) == "0.13"
    assert labels.to_precision(-0.125, 2) == "-0.13"
    assert labels.to_precision(1.375, 3) == "1.38"
    assert labels.to_precision(-1.375, 3) == "-1.38"


def test_to_precision_notation_switches_like_js():
    assert labels.to_precision(1e-6, 2) == "0.0000010"   # e = -6 is still fixed
    assert labels.to_precision(9.99e-6, 2) == "0.000010"  # the carry stays in fixed form
    assert labels.to_precision(1e21, 2) == "1.0e+21"
    assert labels.to_precision(0.999, 2) == "1.0"        # rounding carries a decade
    assert labels.to_precision(0.0, 3) == "0.00"
    assert labels.to_precision(-4.440892098500626e-16, 2) == "-4.4e-16"


def test_tick_digits_matches_live_ranges():
    # bounds of the x^2-y^2 grid function, as measured from the renderer uniforms
    inc = (0.9999999999999988 + 0.9999999999999998) / 4
    assert [labels.tick_digits(-0.9999999999999998 + i * inc, inc) for i in range(5)] \
        == [3, 3, 2, 2, 3]
    assert labels.tick_digits(0.0, 0.5) == 2      # log10(0) = -inf -> the minimum of 2
    assert labels.tick_digits(1.0, 0.0) == 2      # degenerate increment


@pytest.mark.parametrize("scene, expected", [
    ("square_scalar", ["-1.00", "-0.500", "-4.4e-16", "0.50", "1.00"]),
    ("square_fixed_range", ["-0.50", "0.13", "0.750", "1.38", "2.00"]),
    ("square_order1", ["-0.997", "-0.50", "0.0014", "0.501", "1.00"]),
])
def test_colorbar_labels_are_the_live_div_texts(scene, expected):
    sidecar = json.loads((GT / f"{scene}.json").read_text())
    lo, hi = sidecar["colormap_min"], sidecar["colormap_max"]
    assert labels.colorbar_labels(lo, hi) == expected


def test_colorbar_labels_are_five_evenly_spaced_values():
    got = labels.colorbar_labels(-1.0, 3.0)
    assert len(got) == labels.N_LABELS
    assert got == ["-1.0", "0.0", "1.0", "2.00", "3.00"]
