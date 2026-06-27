"""Programmatic overlap checker for manuscript figures (imported by make_figN during development).

After a figure is laid out and drawn, collect the rendered bounding boxes (figure-pixel coordinates)
of every text element (titles, axis labels, tick labels, in-axes annotations, legends, panel letters,
figure text) and report pairwise overlaps: text vs text, and legend or annotation vs plotted data.
The report is a list of named pairs with their overlap in pixels. An empty report means nothing
overlaps. Use it in a loop: lay out, draw, check, fix, repeat until the report is empty.
"""
from __future__ import annotations

import sys

try:                                     # tick labels may contain a unicode minus; keep console safe
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _ascii(s):
    return s.encode("ascii", "replace").decode("ascii")


def _overlap_px(b1, b2):
    """Pixel overlap (ox, oy) of two bboxes; both positive means a true 2-D overlap."""
    ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
    oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
    return ox, oy


def _text_bbox(t, renderer):
    """Text-only window extent. For an annotation this excludes the arrow patch, so a callout
    arrow crossing data is not mistaken for the label sitting on data."""
    from matplotlib.text import Annotation, Text
    if isinstance(t, Annotation):
        return Text.get_window_extent(t, renderer)
    return t.get_window_extent(renderer)


def _text_items(fig, renderer):
    """All visible text elements as (name, bbox). Panel letters and in-axes notes are ax.texts."""
    items = []
    for i, ax in enumerate(fig.axes):
        for t, nm in ((ax.title, "title"), (ax.xaxis.label, "xlabel"), (ax.yaxis.label, "ylabel")):
            if t.get_text().strip():
                items.append((f"ax{i}.{nm}[{t.get_text()[:18]}]", t.get_window_extent(renderer)))
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            if lab.get_text().strip():
                items.append((f"ax{i}.tick[{lab.get_text()[:8]}]", lab.get_window_extent(renderer)))
        for t in ax.texts:
            if t.get_text().strip():
                items.append((f"ax{i}.note[{t.get_text()[:14]}]", _text_bbox(t, renderer)))
        leg = ax.get_legend()
        if leg is not None:
            items.append((f"ax{i}.legend", leg.get_window_extent(renderer)))
    for t in fig.texts:
        if t.get_text().strip():
            items.append((f"figtext[{t.get_text()[:18]}]", t.get_window_extent(renderer)))
    return items


def text_text_collisions(fig, min_overlap_px=1.5):
    """Pairwise text overlaps. Two text boxes collide if they overlap by more than
    min_overlap_px in BOTH dimensions (a hair of edge contact is not flagged)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = _text_items(fig, r)
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            n1, b1 = items[i]
            n2, b2 = items[j]
            ox, oy = _overlap_px(b1, b2)
            if ox > min_overlap_px and oy > min_overlap_px:
                out.append((n1, n2, round(ox, 1), round(oy, 1)))
    return out


def legend_annot_vs_data(fig, min_pts=1):
    """Legend boxes and in-axes annotations that sit on top of plotted data (lines, markers, bars).
    For each axes, test the legend bbox and every ax.text bbox against the display-space vertices of
    that axes' line and collection artists and the rectangles of its bar patches."""
    import numpy as np
    from matplotlib.patches import Rectangle
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []

    def data_points(ax):
        pts = []
        for line in ax.get_lines():
            xy = line.get_xydata()
            if len(xy):
                pts.append(ax.transData.transform(xy))
        for coll in ax.collections:
            off = coll.get_offsets()
            if off is not None and len(off):
                pts.append(ax.transData.transform(off))
        return np.vstack(pts) if pts else np.empty((0, 2))

    def bar_boxes(ax):
        boxes = []
        for p in ax.patches:
            if isinstance(p, Rectangle) and p.get_width() and p.get_height():
                boxes.append(p.get_window_extent(r))
        return boxes

    for i, ax in enumerate(fig.axes):
        pts = data_points(ax)
        bars = bar_boxes(ax)
        probes = []
        leg = ax.get_legend()
        if leg is not None:
            probes.append((f"ax{i}.legend", leg.get_window_extent(r)))
        for t in ax.texts:
            if t.get_text().strip():
                probes.append((f"ax{i}.note[{t.get_text()[:14]}]", _text_bbox(t, r)))
        for nm, bb in probes:
            if len(pts):
                inside = ((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1)
                          & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1))
                if int(inside.sum()) >= min_pts:
                    out.append((nm, "data-point(s)", int(inside.sum())))
            for bj, bbar in enumerate(bars):
                ox, oy = _overlap_px(bb, bbar)
                if ox > 2 and oy > 2:
                    out.append((nm, f"bar#{bj}", round(ox, 1)))
    return out


def report(fig, label=""):
    """Print the full collision report. Returns True when clean (no collisions)."""
    tt = text_text_collisions(fig)
    ld = legend_annot_vs_data(fig)
    print(f"--- collision report: {label} ---")
    if not tt and not ld:
        print("  CLEAN: no text-text or legend/annotation-vs-data overlaps")
        return True
    for n1, n2, ox, oy in tt:
        print(_ascii(f"  TEXT-TEXT  {n1}  <->  {n2}   overlap {ox}x{oy} px"))
    for n1, n2, v in ld:
        print(_ascii(f"  OVER-DATA  {n1}  on  {n2}   ({v})"))
    return False
