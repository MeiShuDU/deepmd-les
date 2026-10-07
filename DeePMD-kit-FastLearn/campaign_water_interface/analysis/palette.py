"""The campaign's figure palette, with the all-pairs separation it was chosen for.

The palette is not a matter of taste here: the figures put four arms on one axis
and a reader has to tell any two of them apart. CIEDE2000 is the yardstick, and
the floor is the MINIMUM over all pairs - the pair that is hardest to separate is
the one that decides whether the palette works, not the average.

The encoding is two hues and two line styles rather than four hues:

  hue      arm topology - blue is short-range-only, red is long-range
  linestyle family      - solid is cace, dashed is sea

Why not four hues: a 4-hue qualitative set usually has one close pair, and the
campaign's original four-hue set had exactly that (amber/orange against red at
dE00 21.4, below any reasonable floor). Two hues with a topology meaning have a
floor of ~30 and, more usefully, put the comparison the campaign exists to make
- what the long-range channel is worth - in the one channel a reader scans first.
The replicates are drawn as thin lines with a bold mean, so they cost no hue.

`report()` prints the floor for each candidate set, so the choice can be re-checked
if an arm is ever added. A 5th arm must reuse a hue and take a new line style.
"""
import itertools

import numpy as np

# hue = arm topology
SR_ONLY = "#0072B2"   # blue
LONG_RANGE = "#C44E52"  # red
TOPOLOGY_COLOR = {"sr": SR_ONLY, "lr": LONG_RANGE}

# linestyle = family
FAMILY_STYLE = {"cace": "-", "sea": "--"}

# neutral ink for references and text
INK = "#333333"
GRID = "#D9D9D9"

# the sets that were considered, kept so the floor can be re-derived
CANDIDATES = {
    "campaign-4hue-as-shipped": ["#4C72B0", "#C44E52", "#55A868", "#DD8452"],
    "seaborn-deep-minus-orange": ["#4C72B0", "#C44E52", "#55A868", "#8172B2"],
    "okabe-ito-4": ["#0072B2", "#D55E00", "#009E73", "#CC79A7"],
    "tab10-blue-red-green-purple": ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"],
    "this-campaign-2hue": [SR_ONLY, LONG_RANGE],
}


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


_XYZ_FROM_RGB = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def lab(hex_color):
    """sRGB hex -> CIELAB (D65)."""
    xyz = _XYZ_FROM_RGB @ _to_linear(_rgb(hex_color))
    t = xyz / _WHITE
    delta = 6.0 / 29.0
    f = np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e00(color_a, color_b):
    """CIEDE2000 between two sRGB hex strings."""
    l1, a1, b1 = lab(color_a)
    l2, a2, b2 = lab(color_b)
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    g = 0.5 * (1 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dlp, dcp = l2 - l1, c2p - c1p
    dh = h2p - h1p
    dh = dh - 360 if dh > 180 else (dh + 360 if dh < -180 else dh)
    dhp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    l_bar, c_barp = (l1 + l2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        h_bar = h1p + h2p
    else:
        h_bar = (h1p + h2p) / 2
        if abs(h1p - h2p) > 180:
            h_bar += 180 if h_bar < 180 else -180
    tt = (1 - 0.17 * np.cos(np.radians(h_bar - 30))
          + 0.24 * np.cos(np.radians(2 * h_bar))
          + 0.32 * np.cos(np.radians(3 * h_bar + 6))
          - 0.20 * np.cos(np.radians(4 * h_bar - 63)))
    dtheta = 30 * np.exp(-((h_bar - 275) / 25) ** 2)
    rc = 2 * np.sqrt(c_barp ** 7 / (c_barp ** 7 + 25 ** 7))
    sl = 1 + 0.015 * (l_bar - 50) ** 2 / np.sqrt(20 + (l_bar - 50) ** 2)
    sc, sh = 1 + 0.045 * c_barp, 1 + 0.015 * c_barp * tt
    rt = -np.sin(np.radians(2 * dtheta)) * rc
    return float(np.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhp / sh) ** 2
                         + rt * (dcp / sc) * (dhp / sh)))


def floor(palette):
    """The minimum pairwise CIEDE2000 over a palette - its worst pair."""
    pairs = list(itertools.combinations(palette, 2))
    values = [delta_e00(a, b) for a, b in pairs]
    worst = pairs[int(np.argmin(values))]
    return min(values), worst


def report():
    for name, palette in CANDIDATES.items():
        value, pair = floor(palette)
        print(f"{name:28s} floor {value:6.2f}  worst pair {pair[0]}/{pair[1]}")


if __name__ == "__main__":
    report()
