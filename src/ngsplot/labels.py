"""Tick labels identical to the webgui colorbar (JS ``toPrecision`` semantics)."""

import math
from decimal import ROUND_HALF_UP, Decimal

#: number of colorbar tick labels (colormap.ts creates 5 divs)
N_LABELS = 5


def _round_sig(x, p, k):
    """``x`` to ``p`` significant digits in decade ``k``, as a Decimal.

    Ties go away from zero, on the exact decimal value of the double — ``0.125`` becomes
    ``0.13`` and ``-0.125`` becomes ``-0.13``, which is what V8's ``toPrecision`` does and
    not what Python's banker's rounding does.  The tie cases of a colourbar
    (``0.125``, ``1.375``, …) are common enough to be visible in a screenshot.
    """
    exp = k - p + 1
    return Decimal(x).scaleb(-exp).to_integral_value(rounding=ROUND_HALF_UP).scaleb(exp)


def to_precision(x, p):
    """``Number.prototype.toPrecision`` in Python.

    Significant digits are kept *and padded*, and the notation switches to exponential
    outside ``-6 <= exponent < p``, which is what makes a colourbar zero print as
    ``-4.4e-16`` when the autoscale bounds are not exactly symmetric.
    """
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0.0:
        return "0." + "0" * (p - 1) if p > 1 else "0"

    k = math.floor(math.log10(abs(x)))
    while True:  # rounding may carry into the next decade, e.g. 0.999 -> 1.00
        q = float(_round_sig(x, p, k))
        k2 = math.floor(math.log10(abs(q))) if q else k
        if k2 == k or q == 0.0:
            break
        k = k2

    if p > k >= -6:
        return f"{q:.{max(p - 1 - k, 0)}f}"

    mantissa, exponent = f"{q:.{p - 1}e}".split("e")
    exponent = int(exponent)
    return f"{mantissa}e{'+' if exponent >= 0 else '-'}{abs(exponent)}"


def tick_digits(value, inc):
    """Digits rule from ``Colorbar.updateLabels``: ``max(2, ceil(log10(|v|/|inc|) + 2))``.

    A tick at exactly 0 lands on the floor of the rule (``log10 0 = -inf``) and prints as
    ``0.0``, while a tick at ``-4.4e-16`` — the usual result of autoscaling a function
    whose bounds differ in the last bits — prints as ``-4.4e-16``.
    """
    if value == 0 or inc == 0:
        return 2
    return max(2, math.ceil(math.log10(abs(value) / abs(inc)) + 2))


def colorbar_labels(vmin, vmax, n=N_LABELS):
    """The ``n`` tick labels drawn along the colorbar, left to right."""
    inc = (vmax - vmin) / (n - 1)
    return [to_precision(vmin + inc * i, tick_digits(vmin + inc * i, inc)) for i in range(n)]
