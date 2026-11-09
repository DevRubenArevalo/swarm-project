"""HTML slide bodies for the Swarm Fidelity video. Rendered at 1920x1080."""

# Validated palette (dataviz reference instance, light mode)
SURFACE, INK, INK2, INK3, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#85837b", "#e4e3de"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
BAD, GOOD = S2, S1          # orange = the wrong/old measurement, blue = corrected

CSS = f"""
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:1920px;height:1080px;overflow:hidden}}
body{{background:{SURFACE};color:{INK};
  font-family:'Inter','Inter Display',sans-serif;-webkit-font-smoothing:antialiased}}
.slide{{position:absolute;inset:0;padding:84px 104px 96px;display:flex;flex-direction:column}}
.kicker{{font-size:21px;font-weight:600;letter-spacing:.15em;text-transform:uppercase;
  color:{INK3};margin-bottom:22px}}
h1{{font-family:'Inter Display','Inter',sans-serif;font-size:62px;font-weight:500;
  line-height:1.08;letter-spacing:-.018em;max-width:1500px}}
h1 em{{font-style:normal;color:{BAD}}}
h1 b{{font-weight:500;color:{GOOD}}}
.lede{{font-size:30px;line-height:1.46;color:{INK2};max-width:1320px;margin-top:26px}}
.body{{flex:1;display:flex;flex-direction:column;justify-content:center;min-height:0}}
.cap{{font-size:20px;color:{INK3};margin-top:18px;line-height:1.45}}
.mono{{font-family:'JetBrainsMono Nerd Font','JetBrains Mono',monospace;
  font-variant-numeric:tabular-nums}}

/* progress rail */
.rail{{position:absolute;left:0;right:0;bottom:0;height:5px;background:{LINE}}}
.rail i{{position:absolute;left:0;top:0;bottom:0;background:{INK};display:block}}
.foot{{position:absolute;left:104px;right:104px;bottom:34px;display:flex;
  justify-content:space-between;font-size:18px;color:{INK3};letter-spacing:.02em}}

/* stat tiles */
.tiles{{display:flex;gap:28px}}
.tile{{flex:1;border:1px solid {LINE};border-radius:14px;padding:30px 32px;background:#fff}}
.tile .v{{display:block;font-size:62px;font-weight:600;line-height:1;
  letter-spacing:-.02em;font-variant-numeric:tabular-nums}}
.tile .l{{display:block;font-size:21px;color:{INK2};margin-top:14px;line-height:1.35}}

/* artifact cards */
.cards{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
.card{{border:1px solid {LINE};border-radius:14px;padding:26px 30px;background:#fff;
  display:flex;gap:22px;align-items:flex-start}}
.card .n{{font-size:34px;font-weight:600;color:{BAD};line-height:1;
  width:46px;flex:none;font-variant-numeric:tabular-nums}}
.card .t{{font-size:27px;font-weight:600;line-height:1.3}}
.card .d{{font-size:21px;color:{INK2};line-height:1.42;margin-top:9px}}
.card .c{{font-size:19px;color:{GOOD};line-height:1.4;margin-top:11px;font-weight:500}}

/* rules list */
.rules{{display:flex;flex-direction:column;gap:19px;margin-top:8px}}
.rule{{display:flex;gap:20px;align-items:baseline;font-size:29px;line-height:1.3}}
.rule span{{color:{GOOD};font-weight:600;flex:none;width:40px}}

figure{{display:flex;flex-direction:column;align-items:center}}
figure img{{display:block;max-width:100%;max-height:640px;object-fit:contain}}
.legend{{display:flex;gap:34px;font-size:22px;color:{INK2};margin-bottom:6px}}
.legend i{{display:inline-block;width:26px;height:4px;border-radius:2px;
  margin-right:10px;vertical-align:middle}}
"""


# ---------------------------------------------------------------- svg helpers
def _sc(d0, d1, r0, r1):
    return lambda v: r0 + (v - d0) / (d1 - d0) * (r1 - r0)


def line_chart(series, xs, *, w=1500, h=540, ylab="", xlab="", ymax=1.0,
               ytick=None, yfmt=lambda v: f"{v:.0%}", end_labels=True,
               ref=None, endfmt=None, endnames=None, mr=200):
    """series: [(name, color, [y...])]; xs: numeric x positions.

    End labels are dodged vertically so they never overlap, and `endnames`
    supplies short names when the legend names are too long for the gutter.
    """
    endfmt = endfmt or yfmt
    ml, mt, mb = 96, 62, 78
    x = _sc(min(xs), max(xs), ml, w - mr)
    y = _sc(0, ymax, h - mb, mt)
    ytick = ytick or [i * ymax / 4 for i in range(5)]
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
         f'font-family="Inter,sans-serif">']
    # y grid + labels (recessive)
    for t in ytick:
        o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y(t):.1f}" y2="{y(t):.1f}" '
                 f'stroke="{LINE}" stroke-width="1"/>')
        o.append(f'<text x="{ml-18}" y="{y(t)+8:.1f}" text-anchor="end" '
                 f'font-size="22" fill="{INK3}">{yfmt(t)}</text>')
    # optional reference line (not a series)
    if ref:
        pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in ref)
        o.append(f'<polyline points="{pts}" fill="none" stroke="{INK3}" '
                 f'stroke-width="2" stroke-dasharray="7 7"/>')
    # x labels
    for v in xs:
        o.append(f'<text x="{x(v):.1f}" y="{h-mb+40}" text-anchor="middle" '
                 f'font-size="23" fill="{INK2}">{v:g}</text>')
    o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y(0):.1f}" y2="{y(0):.1f}" '
             f'stroke="{INK3}" stroke-width="1.5"/>')
    if xlab:
        o.append(f'<text x="{(ml+w-mr)/2:.0f}" y="{h-10}" text-anchor="middle" '
                 f'font-size="22" fill="{INK3}">{xlab}</text>')
    if ylab:
        o.append(f'<text x="{ml}" y="{mt-26}" font-size="22" fill="{INK3}">{ylab}</text>')
    # series: 2px lines, >=8px markers, 2px surface ring
    ends = []
    for i, (name, col, ys) in enumerate(series):
        pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(xs, ys))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                 f'stroke-width="3" stroke-linejoin="round"/>')
        for a, b in zip(xs, ys):
            o.append(f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="6.5" fill="{col}" '
                     f'stroke="{SURFACE}" stroke-width="2.5"/>')
        short = (endnames or [n for n, _, _ in series])[i]
        ends.append([y(ys[-1]), y(ys[-1]), endfmt(ys[-1]), short, col])

    if end_labels and ends:
        GAP = 54                       # value line + name line + breathing room
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):  # push down
            if ends[i][1] - ends[i - 1][1] < GAP:
                ends[i][1] = ends[i - 1][1] + GAP
        over = ends[-1][1] - (h - mb - 10)
        if over > 0:                   # whole stack too low -> lift it
            for e in ends:
                e[1] -= over
        xe = x(xs[-1]) + 24
        for y0, y1, val, name, col in ends:
            # a swatch carries identity; the text stays in ink tokens. No leader
            # line -- a stroke from the last point reads as a data spike.
            o.append(f'<rect x="{xe:.1f}" y="{y1-11:.1f}" width="20" height="4" '
                     f'rx="2" fill="{col}"/>')
            o.append(f'<text x="{xe+30:.1f}" y="{y1:.1f}" font-size="25" '
                     f'font-weight="600" fill="{INK}">{val}</text>')
            o.append(f'<text x="{xe+30:.1f}" y="{y1+26:.1f}" font-size="20" '
                     f'fill="{INK2}">{name}</text>')
    o.append("</svg>")
    return "".join(o)


def bar_chart(labels, values, colors, *, w=1500, h=500, yfmt=lambda v: f"{v:.0%}",
              valfmt=None, ymax=None, xlab="", highlight=None):
    valfmt = valfmt or yfmt
    ml, mr, mt, mb = 96, 40, 60, 86
    ymax = ymax or max(values) * 1.35
    y = _sc(0, ymax, h - mb, mt)
    n = len(labels)
    band = (w - ml - mr) / n
    bw = min(band * 0.56, 96)
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
         f'font-family="Inter,sans-serif">']
    for i in range(5):
        t = i * ymax / 4
        o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y(t):.1f}" y2="{y(t):.1f}" '
                 f'stroke="{LINE}" stroke-width="1"/>')
        o.append(f'<text x="{ml-18}" y="{y(t)+8:.1f}" text-anchor="end" '
                 f'font-size="22" fill="{INK3}">{yfmt(t)}</text>')
    for i, (lb, v, c) in enumerate(zip(labels, values, colors)):
        cx = ml + band * (i + 0.5)
        if v > 0:
            bh = y(0) - y(v)
            o.append(f'<rect x="{cx-bw/2:.1f}" y="{y(v):.1f}" width="{bw:.1f}" '
                     f'height="{bh:.1f}" rx="5" fill="{c}"/>')
            o.append(f'<text x="{cx:.1f}" y="{y(v)-16:.1f}" text-anchor="middle" '
                     f'font-size="26" font-weight="600" fill="{INK}">{valfmt(v)}</text>')
        else:
            o.append(f'<line x1="{cx-bw/2:.1f}" x2="{cx+bw/2:.1f}" y1="{y(0):.1f}" '
                     f'y2="{y(0):.1f}" stroke="{INK3}" stroke-width="3"/>')
            o.append(f'<text x="{cx:.1f}" y="{y(0)-16:.1f}" text-anchor="middle" '
                     f'font-size="21" fill="{INK3}">0</text>')
        bold = ' font-weight="600"' if i == highlight else ""
        fill = INK if i == highlight else INK2
        o.append(f'<text x="{cx:.1f}" y="{h-mb+40}" text-anchor="middle" '
                 f'font-size="23" fill="{fill}"{bold}>{lb}</text>')
    o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y(0):.1f}" y2="{y(0):.1f}" '
             f'stroke="{INK3}" stroke-width="1.5"/>')
    if xlab:
        o.append(f'<text x="{(ml+w-mr)/2:.0f}" y="{h-12}" text-anchor="middle" '
                 f'font-size="22" fill="{INK3}">{xlab}</text>')
    o.append("</svg>")
    return "".join(o)


def legend(items):
    return ('<div class="legend">' + "".join(
        f'<div><i style="background:{c}"></i>{n}</div>' for n, c in items) + "</div>")
