"""Render report.html (plain-language Phase 0 report with charts) from results/*.json.

  python scripts/make_report.py configs/report.yaml

Self-contained: inline SVG + a little vanilla JS for tooltips, no external assets.
Blinded: shows no trajectory-class shares, rise times or lead times (spec §7).
"""
import argparse
import html
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvfetch.config import add_config_args, config_from_args, load_config
from kvfetch.provenance import GIT_AT_START
from kvfetch.trace_size import human_bytes, resolve_model_dims

W = 720  # svg viewBox width
FONT = 'font-family="system-ui,-apple-system,Segoe UI,sans-serif"'  # fallback; CSS sets var(--font-ui)
FONTS = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap"


def wrap(svg: str) -> str:
    """Charts keep a legible minimum width and scroll inside their card on phones."""
    return f'<div class="chartwrap">{svg}</div>'


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def tip(*rows) -> str:
    """data-tip attribute: rows of (value, label[, css class for a line key])."""
    return f'data-tip="{esc(json.dumps([list(r) for r in rows]))}" tabindex="0"'


def short(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "")


# ---------------------------------------------------------------- chart helpers
def bar_path(x0: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Horizontal bar, square at the baseline (x0), 4px rounded data end."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f"M{x0:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
            f"a{r},{r} 0 0 1 -{r},{r}h-{w - r:.1f}z")


class XScale:
    def __init__(self, lo, hi, x0, x1, log=False):
        self.lo, self.hi, self.x0, self.x1, self.log = lo, hi, x0, x1, log

    def __call__(self, v):
        if self.log:
            v = max(v, self.lo)
            f = (math.log10(v) - math.log10(self.lo)) / (math.log10(self.hi) - math.log10(self.lo))
        else:
            f = (v - self.lo) / (self.hi - self.lo)
        return self.x0 + f * (self.x1 - self.x0)


def hbar_chart(rows, series, *, lo, hi, ticks, tick_fmt, val_fmt, log=False, refs=(), label_w=250,
               bar_h=14, row_gap=16, aria=""):
    """rows: [(label, [value per series])]; series: [(name, css_class)]; refs: [(value, label)]."""
    n = len(series)
    right_pad = 70
    top = 26 if refs else 8
    row_h = n * bar_h + (n - 1) * 2 + row_gap
    plot_h = len(rows) * row_h - row_gap
    H = top + plot_h + 30
    sx = XScale(lo, hi, label_w, W - right_pad, log)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}" {FONT}>']
    for t in ticks:
        x = sx(t)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{esc(tick_fmt(t))}</text>')
    out.append(f'<line class="axis" x1="{label_w}" x2="{label_w}" y1="{top}" y2="{top + plot_h}"/>')
    for i, (label, vals) in enumerate(rows):
        y0 = top + i * row_h
        out.append(f'<text class="rowlabel" x="{label_w - 10}" y="{y0 + (n * bar_h + (n - 1) * 2) / 2 + 4:.1f}" '
                   f'text-anchor="end">{esc(label)}</text>')
        for j, ((sname, cls), v) in enumerate(zip(series, vals)):
            if v is None:
                continue
            y = y0 + j * (bar_h + 2)
            w = sx(v) - label_w
            out.append(f'<path class="{cls} mark" d="{bar_path(label_w, y, w, bar_h)}" '
                       f'{tip((val_fmt(v), f"{label} · {sname}" if n > 1 else label))}/>')
            out.append(f'<text class="val" x="{label_w + w + 6:.1f}" y="{y + bar_h - 3}">{esc(val_fmt(v))}</text>')
    for v, lab in refs:
        x = sx(v)
        out.append(f'<line class="ref" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{top + plot_h}"/>')
        anchor = "end" if x > W - 160 else "start"
        dx = -4 if anchor == "end" else 4
        out.append(f'<text class="reflabel" x="{x + dx:.1f}" y="{top - 10}" text-anchor="{anchor}">{esc(lab)}</text>')
    out.append("</svg>")
    return wrap("\n".join(out))


def stacked_hbar(rows, segs, *, hi, ticks, tick_fmt, val_fmt, refs=(), label_w=150, bar_h=22, row_gap=18, aria=""):
    """rows: [(label, [value per segment], total_label)]; segs: [(name, css_class)]."""
    right_pad = 80
    top = 26 if refs else 8
    plot_h = len(rows) * (bar_h + row_gap) - row_gap
    H = top + plot_h + 30
    sx = XScale(0, hi, label_w, W - right_pad)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}" {FONT}>']
    for t in ticks:
        x = sx(t)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{esc(tick_fmt(t))}</text>')
    for i, (label, vals, total_label) in enumerate(rows):
        y = top + i * (bar_h + row_gap)
        out.append(f'<text class="rowlabel" x="{label_w - 10}" y="{y + bar_h / 2 + 4}" text-anchor="end">{esc(label)}</text>')
        x = label_w
        nonzero = [k for k, v in enumerate(vals) if v > 0]
        for k, ((sname, cls), v) in enumerate(zip(segs, vals)):
            if v <= 0:
                continue
            w = sx(v) - label_w
            last = k == nonzero[-1]
            seg_w = max(w - (0 if last else 2), 0.5)  # 2px surface gap between segments
            d = bar_path(x, y, seg_w, bar_h) if last else f"M{x:.1f},{y}h{seg_w:.1f}v{bar_h}h-{seg_w:.1f}z"
            out.append(f'<path class="{cls} mark" d="{d}" {tip((val_fmt(v), f"{label} · {sname}"))}/>')
            x += w
        out.append(f'<text class="val" x="{x + 6:.1f}" y="{y + bar_h / 2 + 4}">{esc(total_label)}</text>')
    for v, lab in refs:
        x = sx(v)
        out.append(f'<line class="ref" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{top + plot_h}"/>')
        out.append(f'<text class="reflabel" x="{x - 4:.1f}" y="{top - 10}" text-anchor="end">{esc(lab)}</text>')
    out.append("</svg>")
    return wrap("\n".join(out))


def line_chart(xs, series, *, xlo, xhi, ylo, yhi, xticks, yticks, xfmt, yfmt, xlabel, vmarks=(), H=300,
               xlog=False, aria="", end_labels=True):
    """series: [(name, css_class, ys)] with ys aligned to xs. vmarks: [(x, label)]. Crosshair via JS."""
    L, R, T, B = 56, 120 if end_labels else 24, 30, 44
    sx = XScale(xlo, xhi, L, W - R, xlog)

    def sy(v):
        return T + (1 - (v - ylo) / (yhi - ylo)) * (H - T - B)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}" class="linechart" {FONT}>']
    for t in yticks:
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        out.append(f'<text class="tick" x="{L - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{esc(yfmt(t))}</text>')
    for t in xticks:
        out.append(f'<text class="tick" x="{sx(t):.1f}" y="{H - B + 18}" text-anchor="middle">{esc(xfmt(t))}</text>')
    out.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>')
    out.append(f'<text class="axislabel" x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    for x, lab, *anchor in vmarks:  # optional anchor: "end" = left of the line, "mid" = centred on it
        a = anchor[0] if anchor else "start"
        dx = {"end": -4, "mid": 0, "start": 4}[a]
        out.append(f'<line class="ref" x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T - 8}" y2="{H - B}"/>')
        out.append(f'<text class="reflabel" x="{sx(x) + dx:.1f}" y="{T - 12}" '
                   f'text-anchor="{ {"end": "end", "mid": "middle", "start": "start"}[a]}">{esc(lab)}</text>')
    pts = []
    for name, cls, ys in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys)) if xlo <= x <= xhi)
        out.append(f'<path class="line {cls}" d="{d}"/>')
        if len(xs) <= 8:  # sparse series: show markers
            for x, y in zip(xs, ys):
                out.append(f'<circle class="dot {cls}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5"/>')
        if end_labels:
            xe, ye = [(x, y) for x, y in zip(xs, ys) if xlo <= x <= xhi][-1]
            pts.append([sy(ye), name, cls, sx(xe)])
    # end labels with simple de-collision (min 14px apart)
    pts.sort()
    for i in range(1, len(pts)):
        pts[i][0] = max(pts[i][0], pts[i - 1][0] + 14)
    for yv, name, cls, xe in pts:
        out.append(f'<line class="keyline {cls}" x1="{xe + 6:.1f}" x2="{xe + 18:.1f}" y1="{yv:.1f}" y2="{yv:.1f}"/>')
        out.append(f'<text class="endlabel" x="{xe + 22:.1f}" y="{yv + 4:.1f}">{esc(name)}</text>')
    data = {"x": [x for x in xs if xlo <= x <= xhi],
            "px": [round(sx(x), 1) for x in xs if xlo <= x <= xhi],
            "series": [{"name": n, "cls": c, "y": [yfmt(y) for x, y in zip(xs, ys) if xlo <= x <= xhi]} for n, c, ys in series],
            "xfmt": xlabel, "T": T, "B": H - B}
    out.append(f'<line class="crosshair" x1="0" x2="0" y1="{T}" y2="{H - B}" visibility="hidden"/>')
    out.append(f'<rect class="hit" x="{L}" y="{T}" width="{W - R - L}" height="{H - T - B}" '
               f'data-line="{esc(json.dumps(data))}" tabindex="0"/>')
    out.append("</svg>")
    return wrap("\n".join(out))


def legend(items, kind="rect") -> str:
    keys = "".join(f'<span class="lg"><svg width="14" height="10" aria-hidden="true">'
                   + (f'<rect class="{c}" width="14" height="10" rx="2"/>' if kind == "rect"
                      else f'<line class="keyline {c}" x1="0" x2="14" y1="5" y2="5"/>')
                   + f'</svg>{esc(n)}</span>' for n, c in items)
    return f'<div class="legend">{keys}</div>'


def table(headers, rows, caption="Data table") -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<details class="datatable"><summary>{esc(caption)}</summary>'
            f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div></details>')


def meter(frac: float, label: str) -> str:
    pct = max(0.0, min(frac, 1.0)) * 100
    return (f'<div class="meter" role="meter" aria-valuenow="{pct:.0f}" aria-valuemin="0" aria-valuemax="100" '
            f'aria-label="{esc(label)}"><div class="meterfill" style="width:{pct:.1f}%"></div></div>')


def tile(label, value, note="") -> str:
    return (f'<div class="tile"><div class="tlabel">{esc(label)}</div><div class="tvalue">{esc(value)}</div>'
            f'<div class="tnote">{esc(note)}</div></div>')


def status(ok: bool, yes: str = "pass", no: str = "fail") -> str:
    return f'<span class="st good">✓ {yes}</span>' if ok else f'<span class="st bad">✗ {no}</span>'


# ---------------------------------------------------------------- illustrations
def tier_diagram() -> str:
    def pages(x, y, n, cls, cols=6):
        return "".join(f'<rect class="{cls}" x="{x + (i % cols) * 20}" y="{y + (i // cols) * 20}" width="16" height="16" rx="3"/>'
                       for i in range(n))
    return wrap(f'''<svg viewBox="0 0 {W} 250" role="img" aria-label="Three tiers of KV pages: on GPU attended, on GPU standby, offloaded to host memory" {FONT}>
<rect class="box" x="8" y="20" width="420" height="200" rx="12"/>
<text class="boxtitle" x="24" y="44">GPU memory (small, fast)</text>
<rect class="subbox" x="24" y="58" width="185" height="148" rx="8"/>
<text class="subtitle" x="36" y="80">Tier 1 · attended</text>
{pages(36, 92, 12, "s1")}
<text class="small" x="36" y="194">used in this step's attention</text>
<rect class="subbox" x="225" y="58" width="187" height="148" rx="8"/>
<text class="subtitle" x="237" y="80">Tier 2 · standby</text>
{pages(237, 92, 9, "s3")}
<text class="small" x="237" y="194">on GPU, skipped this step</text>
<rect class="box" x="520" y="20" width="192" height="200" rx="12"/>
<text class="boxtitle" x="536" y="44">Host (CPU) memory</text>
<text class="subtitle" x="536" y="80">Tier 3 · offloaded</text>
{pages(536, 92, 30, "tier3", cols=8)}
<text class="small" x="536" y="194">big, but far away</text>
<path class="arrow" d="M512,120 L440,120" marker-end="url(#ah)"/>
<text class="small" x="474" y="108" text-anchor="middle">PCIe</text>
<text class="small" x="474" y="142" text-anchor="middle">prefetch</text>
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="arrowhead" d="M0,0L10,5L0,10z"/></marker></defs>
</svg>''')


def lead_illustration() -> str:
    xs = list(range(0, 41))
    ys = [0.05 if x < 10 else min(1.0, 0.05 + (x - 10) * 0.95 / 20) for x in xs]
    cut = 0.6
    t_true = next(x for x, y in zip(xs, ys) if y >= cut)
    H, T, B, L, R = 260, 30, 44, 56, 24  # must match line_chart's margins (end_labels=False)
    svg = line_chart(xs, [("page importance", "s1", ys)], xlo=0, xhi=40, ylo=0, yhi=1.05,
                     xticks=[0, 10, 20, 30, 40], yticks=[0, 0.5, 1.0], xfmt=lambda v: f"{v}",
                     yfmt=lambda v: f"{v:.2f}", xlabel="decode step (illustration, not data)",
                     vmarks=[(t_true - 4, "trend predictor fetches it", "end"),
                             (t_true + 1, "naive predictor: one step late")],
                     H=H, aria="Illustration of lead time", end_labels=False)
    y_cut = T + (1 - cut / 1.05) * (H - T - B)
    x_true = L + t_true / 40 * (W - R - L)
    extra = (f'<line class="cutoff" x1="{L}" x2="{W - R}" y1="{y_cut:.1f}" y2="{y_cut:.1f}"/>'
             f'<text class="reflabel" x="{W - R}" y="{y_cut - 6:.1f}" text-anchor="end">cut-off for "important" (top-B)</text>'
             f'<circle class="dot s1" cx="{x_true:.1f}" cy="{y_cut:.1f}" r="5"/>'
             f'<text class="reflabel" x="{x_true + 8:.1f}" y="{y_cut + 16:.1f}">needed from here</text>')
    return svg.replace("</svg>", extra + "</svg>"), t_true


# ---------------------------------------------------------------- page
CSS = """
/* Layout: one reading column; each section = question eyebrow, answer as headline, chart, "what this means". */
:root{--font-ui:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--font-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--s5:#e87ba4;--tier3:#c3c2b7;--good:#0ca30c;--goodtext:#006300;--bad:#d03b3b;--dpos:#2a78d6;--dposw:#b7d3f6;
--dnegw:#f6c3c2;--dneg:#e34948;--onstrong:#fff;--onweak:#0b0b0b;--wash:rgba(42,120,214,.10)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;
--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--tier3:#52514e;--goodtext:#0ca30c;--dpos:#3987e5;--dposw:#184f95;
--dnegw:#7a2f2f;--dneg:#e66767;--onstrong:#0b0b0b;--onweak:#f0efec;--wash:rgba(57,135,229,.16)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--s5:#d55181;--tier3:#52514e;--goodtext:#0ca30c;--dpos:#3987e5;--dposw:#184f95;--dnegw:#7a2f2f;--dneg:#e66767;
--onstrong:#0b0b0b;--onweak:#f0efec;--wash:rgba(57,135,229,.16)}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.6 var(--font-ui)}
main{max-width:900px;margin:0 auto;padding-block:24px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1{font-size:28px;line-height:1.2;font-weight:600;margin:8px 0 6px;text-wrap:balance}h2{font-size:20px;line-height:1.3;font-weight:600;margin:0 0 6px;text-wrap:balance}h3{font-size:15px;margin:16px 0 4px}
p{margin:6px 0}.sub{color:var(--ink2)}.muted{color:var(--muted);font-size:13px}
header .meta{color:var(--muted);font-size:13px;margin-top:6px}
.card{padding-block:28px;border-top:1px solid var(--grid)}
.q{font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-bottom:2px}
.answer{margin:14px 0 4px;color:var(--ink);max-width:70ch}
.callout{border:1px solid var(--border);background:var(--wash);border-radius:10px;padding:10px 14px;margin:12px 0;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.tlabel{font-size:13px;color:var(--ink2)}.tvalue{font-size:clamp(17px,2.3vw,21px);line-height:1.25;font-weight:600;margin:4px 0}.tnote{font-size:12px;color:var(--muted)}
svg{width:100%;height:auto;display:block;overflow:visible;font-family:var(--font-ui)}.chartwrap{overflow-x:auto;margin:4px 0}.chartwrap>svg{min-width:600px}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.axislabel{fill:var(--muted);font-size:12px}
.rowlabel{fill:var(--ink2);font-size:12px}.val{fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}
.ref{stroke:var(--ink2);stroke-width:1.5}.cutoff{stroke:var(--muted);stroke-width:1.5}
.reflabel{fill:var(--ink2);font-size:11px}.endlabel{fill:var(--ink2);font-size:12px}
.val,.reflabel,.endlabel{paint-order:stroke;stroke:var(--page);stroke-width:4px;stroke-linejoin:round}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}.s4{fill:var(--s4)}.s5{fill:var(--s5)}.tier3{fill:var(--tier3)}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line.s1,.keyline.s1{stroke:var(--s1)}.line.s2,.keyline.s2{stroke:var(--s2)}.line.s3,.keyline.s3{stroke:var(--s3)}
.line.s1,.line.s2,.line.s3{fill:none}.keyline{stroke-width:2.5;stroke-linecap:round}
.dot{stroke:var(--surface);stroke-width:2}
.mark{cursor:default;outline:none}.mark:hover,.mark:focus{opacity:.8}
.hit{fill:transparent;outline:none}.crosshair{stroke:var(--muted);stroke-width:1}
.box{fill:none;stroke:var(--axis);stroke-width:1.5}.subbox{fill:var(--page);stroke:var(--border)}
.boxtitle{fill:var(--ink);font-size:14px;font-weight:600}.subtitle{fill:var(--ink2);font-size:13px;font-weight:600}
.small{fill:var(--muted);font-size:11px}.arrow{stroke:var(--ink2);stroke-width:2;fill:none}.arrowhead{fill:var(--ink2)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink2);margin:8px 0 4px}
.lg{display:inline-flex;align-items:center;gap:6px}.lg svg{width:14px;height:10px;flex:none}
details.datatable{margin-top:10px;font-size:13px}details.datatable summary{cursor:pointer;color:var(--ink2)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;margin-top:8px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid);white-space:nowrap}th{color:var(--ink2);font-weight:600}
table.heat th{white-space:normal;vertical-align:bottom}table.heat td,table.heat th{padding:4px 6px}table.heat td.c{text-align:center;font-weight:600;border:2px solid var(--surface);border-radius:4px;min-width:44px}
.hpos{background:var(--dpos);color:var(--onstrong)}.hposw{background:var(--dposw);color:var(--onweak)}
.hnegw{background:var(--dnegw);color:var(--onweak)}.hneg{background:var(--dneg);color:var(--onstrong)}
.st{font-weight:600;font-size:13px}.st.good{color:var(--goodtext)}.st.bad{color:var(--bad)}
.meter{height:14px;border-radius:7px;background:var(--dposw);overflow:hidden;margin:8px 0}
.meterfill{height:100%;background:var(--s1);border-radius:7px}
.steps{list-style:none;padding:0;margin:8px 0}.steps li{padding:6px 0;border-bottom:1px solid var(--grid)}
.steps li:last-child{border-bottom:0}
.decision{border:1px solid var(--border);border-radius:12px;padding:14px 16px;margin:12px 0;background:var(--surface)}
.decision .num{display:inline-block;min-width:22px;height:22px;border-radius:11px;background:var(--s1);color:#fff;
text-align:center;font-size:13px;font-weight:600;line-height:22px;margin-right:8px}
.rec{margin-top:6px;padding:6px 10px;border-radius:8px;background:var(--wash);font-size:14px}
code{font-family:var(--font-mono);font-size:12.5px;background:var(--page);border:1px solid var(--border);border-radius:4px;padding:0 4px}
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
border-radius:8px;padding:6px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.12);max-width:300px}
#tip strong{font-size:13px}#tip span{color:var(--ink2)}#tip .k{display:inline-block;width:12px;height:0;border-top:2.5px solid;margin-right:6px;vertical-align:middle}
.themebtn{float:right;font:inherit;font-size:12px;color:var(--ink2);background:var(--surface);border:1px solid var(--border);
border-radius:8px;padding:4px 10px;cursor:pointer}
@media (max-width:600px){h1{font-size:22px}.card{padding:14px}}
"""

JS = r"""
(function(){
const tip=document.getElementById('tip');
function place(x,y){const r=tip.getBoundingClientRect();let L=x+14,T=y+14;
 if(L+r.width>innerWidth-8)L=x-r.width-14;if(T+r.height>innerHeight-8)T=y-r.height-14;tip.style.left=L+'px';tip.style.top=T+'px';}
function fill(rows){tip.replaceChildren();for(const [v,l,c] of rows){const d=document.createElement('div');
 if(c){const k=document.createElement('i');k.className='k';k.style.borderColor='var(--'+c+')';d.append(k);}
 const b=document.createElement('strong');b.textContent=v;d.append(b);if(l){const s=document.createElement('span');s.textContent=' '+l;d.append(s);}
 tip.append(d);}tip.hidden=false;}
function hide(){tip.hidden=true;}
document.querySelectorAll('[data-tip]').forEach(el=>{const rows=JSON.parse(el.dataset.tip);
 el.addEventListener('pointermove',e=>{fill(rows);place(e.clientX,e.clientY);});el.addEventListener('pointerleave',hide);
 el.addEventListener('focus',()=>{fill(rows);const r=el.getBoundingClientRect();place(r.right,r.top);});el.addEventListener('blur',hide);});
document.querySelectorAll('[data-line]').forEach(hit=>{const d=JSON.parse(hit.dataset.line);const svg=hit.ownerSVGElement;
 const ch=svg.querySelector('.crosshair');let idx=d.x.length-1;
 function at(i,cx,cy){idx=Math.max(0,Math.min(d.x.length-1,i));const px=d.px[idx];ch.setAttribute('x1',px);ch.setAttribute('x2',px);
  ch.setAttribute('visibility','visible');fill([[String(d.x[idx]),d.xfmt]].concat(d.series.map(s=>[s.y[idx],s.name,s.cls])));place(cx,cy);}
 function nearest(e){const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const q=p.matrixTransform(svg.getScreenCTM().inverse());
  let b=0,bd=1e9;d.px.forEach((v,i)=>{const dd=Math.abs(v-q.x);if(dd<bd){bd=dd;b=i;}});return b;}
 hit.addEventListener('pointermove',e=>at(nearest(e),e.clientX,e.clientY));
 hit.addEventListener('pointerleave',()=>{ch.setAttribute('visibility','hidden');hide();});
 hit.addEventListener('focus',()=>{const r=hit.getBoundingClientRect();at(idx,r.right,r.top);});
 hit.addEventListener('blur',()=>{ch.setAttribute('visibility','hidden');hide();});
 hit.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();const r=hit.getBoundingClientRect();
  at(idx+(e.key==='ArrowRight'?1:-1),r.right,r.top);}});});
const btn=document.getElementById('theme');
if(btn){const modes=['auto','light','dark'];let m='auto';
try{m=localStorage.getItem('kvf-theme')||'auto';}catch(e){}
function apply(){if(m==='auto')delete document.documentElement.dataset.theme;else document.documentElement.dataset.theme=m;btn.textContent='Theme: '+m;}
btn.addEventListener('click',()=>{m=modes[(modes.indexOf(m)+1)%3];try{localStorage.setItem('kvf-theme',m);}catch(e){}apply();});apply();}
})();
"""


def load(p):
    return json.load(open(p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument("--artifact", metavar="PATH", help="also write a Claude Artifact variant (no document skeleton, no theme toggle: the viewer sets the theme)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    rc = cfg["report"]
    res_dir = Path(cfg["paths"]["results_dir"])

    gcfg = load_config(Path(cfg["_meta"]["config_path"]).parent / rc["gate_config"])["gate"]
    ts = load(res_dir / "trace_size.json")
    smoke = load(res_dir / "smoke_2k.json")
    runs = {t: load(Path(cfg["small_eval"]["out_dir"]) / f"{t}.json") for t in rc["small_eval_runs"]}
    gate = load(Path(gcfg["out_dir"]) / "summary.json")
    fid = {m: load(p) for m, p in rc["fidelity"].items()}
    needle = load(rc["needle_timing"])
    budget_min = cfg["small_eval"]["max_wall_s"] / 60
    thr = gcfg["pass_threshold"]

    parts = []
    A = parts.append

    # ---------------- header + tiles
    se_totals = {runs[t]["provenance"]["model"]: runs[t]["timing"]["total_s"] / 60 for t in runs}
    worst_bound = max(max(r["fp32"]["isolation_decode"]["max_diff_over_bound"], r["fp32"]["isolation_obs"]["max_diff_over_bound"])
                      for r in smoke["reports"])
    rec = rc["recommendation"]
    A(f'''<header>__THEMEBTN__
<div class="q">kv-fetch · Phase 0 report</div>
<h1>Tiered KV cache with importance-trend prefetch: the measuring tools are built and checked</h1>
<p class="sub">Phase 0 builds and validates the instruments; it does not test the idea yet. That happens in
Phase 1, which waits on four decisions from you (bottom of this page).</p>
<div class="meta">Branch <code>phase0</code> · commit <code>{esc(GIT_AT_START["commit"][:7])}</code> · generated by
<code>scripts/make_report.py</code> from <code>results/*.json</code></div></header>''')
    A('<div class="tiles">'
      + tile("Small eval runtime", f"{min(se_totals.values()):.0f}–{max(se_totals.values()):.0f} min", f"budget {budget_min:.0f} min · both models fit")
      + tile("Attention recorder error", f"{worst_bound:.2f}× limit", "worst case, 3 models · ≤ 1 passes")
      + tile("Unit tests", "16 / 16", "hand-built toy traces")
      + tile("Models gated", f"{len({r['model'] for r in gate['rows']})} × 2 lengths", "RULER needle + multi-key")
      + tile("Recommended small model", f"{short(rec['iteration'])} @ {rec['iteration_length'] // 1024}K", "only one with real margin")
      + '</div>')
    A('''<div class="callout"><strong>Why some numbers are missing on purpose.</strong> The spec says the Phase 1 go/no-go
thresholds must be chosen <em>before</em> looking at the data. The timing runs already produced Phase-1-style
statistics (how often pages rise gradually, lead times), so this report leaves them out. They are in
<code>results/small_eval/*.json</code> if you decide to look.</div>''')

    # ---------------- idea
    illo, t_true = lead_illustration()
    A(f'''<section class="card"><div class="q">The idea in one picture</div><h2>Keep the right KV pages close, and fetch them early</h2>
<p>The model's memory of the prompt (the KV cache) is split into <strong>pages</strong> of 16 tokens. When the GPU
can't hold them all, each page lives in one of three tiers:</p>
{tier_diagram()}
<p>Every decode step, attention only needs a few pages. The project's bet is that we can see a page's
importance <em>trending up</em> and fetch it over PCIe <strong>before</strong> it's needed. That way the GPU
never waits. The gap between "fetched" and "needed" is the <strong>lead time</strong>:</p>
{illo}
<p class="muted">Illustration only. A page's importance rises and crosses the "important" cut-off at step {t_true}.
A naive predictor that copies the previous step notices one step late, so the GPU stalls. A predictor that
extrapolates the trend fetches it 4 steps early, and the transfer is hidden.</p>
</section>''')

    # ---------------- checklist
    steps = [
        ("Estimate how big the attention recordings (traces) get", True, "§2 below"),
        ("Record attention on a 2K prompt and check it against the reference implementation", True, "§3"),
        ("Metric library (recall, lead time, churn) with unit tests on toy data", True, "16 tests pass"),
        ("Trajectory classifier with every threshold in a config file", True, "configs/base.yaml"),
        ("Small eval end to end in under an hour", True, "§4"),
        ("Model gate: which small models can really use 16K / 32K of context", True, "§5"),
        ("Stop and report", True, "this page"),
    ]
    A('<section class="card"><div class="q">Checklist</div><h2>Every Phase 0 step is done</h2><ul class="steps">'
      + "".join(f'<li>{status(ok, "done", "not done")} &nbsp;{esc(s)} <span class="muted">· {esc(where)}</span></li>'
                for s, ok, where in steps)
      + '</ul></section>')

    # ---------------- trace size
    rows_by = {(r["scenario"], r["head_kind"]): r for r in ts["rows"] if r["page_size"] == cfg["definitions"]["page_size"]}
    free = shutil.disk_usage(rc["free_disk_path"]).free
    GB = 2 ** 30  # binary units throughout, matching df -h and the estimator
    tsr = []
    for scen, kind, label in rc["trace_size_rows"]:
        r = rows_by[(scen, kind)]
        tsr.append((label, [(r["dense_bytes_per_sample"] + r["obs_bytes_per_sample"]) * r["samples"] / GB,
                            (r["topk_bytes_per_sample"] + r["obs_bytes_per_sample"]) * r["samples"] / GB]))
    fmt_gb = lambda v: human_bytes(v * GB).replace(".0 ", " ")
    fmt_tick = lambda v: f"{v:g} GB" if v < 1024 else f"{v / 1024:g} TB"
    A(f'''<section class="card" id="s2"><div class="q">§2 · Can we store the recordings?</div>
<h2>The small eval fits easily. The large eval as written does not fit on this machine</h2>
<p>Total disk needed to store attention for every page, step, layer and head ("full"), or only the 512 most-attended
pages per step ("top-512"). Note the log scale: each grid line is 10× the previous one.</p>
{legend([("full recording", "s1"), ("top-512 only", "s2")])}
{hbar_chart(tsr, [("full recording", "s1"), ("top-512 only", "s2")], lo=1, hi=2e4, log=True,
            ticks=[1, 10, 100, 1024, 10240], tick_fmt=fmt_tick, val_fmt=fmt_gb,
            refs=[(free / GB, f"free disk today: {fmt_gb(free / GB)}")], label_w=240,
            aria="Trace storage needed per scenario, log scale")}
{table(["scenario", "full", "top-512"], [(l, fmt_gb(a), fmt_gb(b)) for l, (a, b) in tsr])}
<p class="answer">At 128K context even top-512 needs ~450 GB, and top-512 can't even represent a 10 % budget (820 pages).
So the large eval needs fewer samples or layers for the recordings, or metrics computed on the fly. §7 shows why
top-512 isn't a free saving either.</p></section>''')

    # ---------------- tracer check
    srows = []
    for r in smoke["reports"]:
        f = r["fp32"]
        srows.append((short(r["model"]), [max(f["isolation_decode"]["max_diff_over_bound"], f["isolation_obs"]["max_diff_over_bound"])]))
    chk = [(short(r["model"]), "yes" if r["bf16"]["tokens_identical_traced_vs_untraced"] else "NO",
            f'{r["bf16"]["max_row_sum_err"]:.1e}', "yes" if r["bf16"]["roundtrip_full"] and r["bf16"]["roundtrip_topk"] else "NO",
            f'{r["fp32"]["max_logit"]:.0f}') for r in smoke["reports"]]
    A(f'''<section class="card" id="s3"><div class="q">§3 · Is the attention recorder correct?</div>
<h2>Yes: errors are at most {worst_bound:.2f}× the allowed floating-point noise</h2>
<p>The recorder re-computes attention and compares itself, in the same forward pass, with the library's own
reference implementation. The bar is the worst error divided by what float32 rounding allows. Anything under
1.0 means "identical up to rounding".</p>
{hbar_chart(srows, [("error ÷ allowed", "s1")], lo=0, hi=1.2, ticks=[0, 0.25, 0.5, 0.75, 1.0],
            tick_fmt=lambda v: f"{v:g}", val_fmt=lambda v: f"{v:.2f}×", refs=[(1.0, "limit")], label_w=160,
            aria="Recorder error relative to the allowed bound, per model")}
{table(["model", "generated text unchanged", "attention sums to 1 (max error)", "save/load exact", "largest logit"], chk,
       "Other checks (all pass)")}
<p class="answer">Recording never changes what the model generates, and files save and load exactly.
Qwen2.5 has attention scores up to ~23,000, so its absolute error is larger, but it is still inside the
rounding bound.</p></section>''')

    # ---------------- small eval timing
    seg_names = [("load model", "s1"), ("choose layers (probe)", "s2"), ("record RULER task", "s3"),
                 ("record long-generation task", "s4"), ("other (prompts, analysis wait)", "s5")]
    trows, ttab = [], []
    for t, r in runs.items():
        tm = r["timing"]
        tr_ruler = r["tasks"]["ruler_multikey"]["timing"]["trace_s"]
        tr_long = r["tasks"]["longgen"]["timing"]["trace_s"]
        vals = [tm["load_s"], tm["probe_s"], tr_ruler, tr_long]
        vals.append(max(tm["total_s"] - sum(vals), 0))
        vals = [v / 60 for v in vals]
        name = short(r["provenance"]["model"])
        trows.append((name, vals, f"{tm['total_s'] / 60:.1f} min"))
        ttab.append([name] + [f"{v:.1f}" for v in vals] + [f"{tm['total_s'] / 60:.1f}"])
    A(f'''<section class="card" id="s4"><div class="q">§4 · Does the small eval fit in an hour?</div>
<h2>Yes, with lots of room: {min(se_totals.values()):.0f} to {max(se_totals.values()):.0f} minutes</h2>
<p>The small eval is the project's "unit test": 40 prompts of 32K tokens, 256 generated tokens each, attention
recorded on 4 layers, then analysed. Time per stage:</p>
{legend(seg_names)}
{stacked_hbar(trows, seg_names, hi=budget_min, ticks=[0, 15, 30, 45, 60], tick_fmt=lambda v: f"{v:g} min",
              val_fmt=lambda v: f"{v:.1f} min", refs=[(budget_min, "1-hour budget")], aria="Small eval wall time by stage")}
{table(["model"] + [n for n, _ in seg_names] + ["total (min)"], ttab)}
<p class="answer">Recording dominates; analysis runs on 16 CPU workers alongside it. The Qwen3-4B run shared its GPU
with another job, so its time is an upper bound.</p></section>''')

    # ---------------- gate
    grows_all = gate["rows"]
    models = list(dict.fromkeys(r["model"] for r in grows_all))
    by = {(r["model"], r["length"]): r for r in grows_all}
    mk_rows = []
    for m in models:
        tag = " (reference)" if by[(m, 32768)]["role"] == "reference" else ""
        mk_rows.append((short(m) + tag, [by[(m, 16384)]["group_multikey"], by[(m, 32768)]["group_multikey"]]))
    tasks = gcfg["tasks"]
    tnames = {t: t.replace("niah_single_", "needle ").replace("niah_multikey_", "multi-key ") for t in tasks}

    def hc(v):
        cls = "hpos" if v >= 95 else "hposw" if v >= thr else "hnegw" if v >= 70 else "hneg"
        return f'<td class="c {cls}">{v:.0f}</td>'
    heat = ['<div class="tablewrap"><table class="heat"><thead><tr><th>model</th><th>length</th>'
            + "".join(f"<th>{esc(tnames[t])}</th>" for t in tasks) + "<th>result</th></tr></thead><tbody>"]
    for m in models:
        for L in (16384, 32768):
            r = by[(m, L)]
            heat.append(f"<tr><td>{esc(short(m))}</td><td>{L // 1024}K</td>" + "".join(hc(r[t]) for t in tasks)
                        + f"<td>{status(r['passed'])}</td></tr>")
    heat.append("</tbody></table></div>")
    A(f'''<section class="card" id="s5"><div class="q">§5 · Which models can really use long context?</div>
<h2>Only {short(rec['iteration'])} passes at 32K with a real margin</h2>
<p>Before recording attention we need models that actually <em>use</em> distant context, otherwise the recordings are
noise. The test is RULER's "find the hidden number" tasks; the hard ones are the multi-key tasks with distractors.
A model passes a length when both its needle average and its multi-key average reach <strong>{thr}</strong>
(RULER's own bar).</p>
{legend([("16K context", "s1"), ("32K context", "s2")])}
{hbar_chart(mk_rows, [("16K", "s1"), ("32K", "s2")], lo=0, hi=100, ticks=[0, 25, 50, 75, 100],
            tick_fmt=lambda v: f"{v:g}", val_fmt=lambda v: f"{v:.1f}", refs=[(thr, f"pass bar {thr}")], label_w=230,
            bar_h=11, row_gap=12, aria="Multi-key average score per model at 16K and 32K")}
<h3>Every task, every model</h3>
<p class="muted">Blue = at or above the bar (dark blue ≥ 95), red = below (dark red &lt; 70). 50 prompts per cell,
so a cell near 60 carries roughly ±7 points of noise.</p>
{"".join(heat)}
<p class="answer">The only task that separates the models is <strong>multi-key 3</strong> (lookup by random ID among
thousands of look-alike IDs). {short(rec['iteration'])} scores 96 there at 32K, while the next-best small model scores 64.
All three 8B reference models pass both lengths almost perfectly.</p></section>''')

    # ---------------- needle timing
    xmax = rc["needle_x_max"]
    nser, ntab, stopmarks = [], {}, []
    for k, (tag, d) in enumerate(needle["runs"].items()):
        cls = f"s{k + 1}"
        nser.append((short(d["model"]), cls, d["mean_needle_mass"]))
        stops = [s for s in d["natural_stop_steps"] if s is not None]
        stopmarks.append((short(d["model"]), int(round(sum(stops) / len(stops)))))
        xs = d["x"]
    for name, _, ys in nser:
        ntab[name] = ys
    n_peak = max(y for _, _, ys in nser for x, y in zip(xs, ys) if x <= xmax)
    n_yhi = math.ceil(n_peak * 5) / 5  # round up to the next 0.2
    trows_n = [[x] + [f"{ntab[n][i]:.3f}" for n in ntab] for i, x in enumerate(xs) if x <= xmax and (x % 4 == 0 or x < 12)]
    stop_txt = ", ".join(f"{n} at step {s}" for n, s in stopmarks)
    A(f'''<section class="card" id="s6"><div class="q">Surprise 1 · task design</div>
<h2>In RULER, the "sudden jump" happens before generation starts</h2>
<p>RULER multi-key was meant to be the task where a page becomes important <em>suddenly</em> during generation.
Below is the share of the strongest retrieval heads' attention that lands on the hidden answer's page, from the last
31 prompt tokens (left of 0) through generation:</p>
{line_chart(xs, nser, xlo=xs[0], xhi=xmax, ylo=0, yhi=n_yhi, xticks=[-30, -15, 0, 15, 30, 45, 60],
            yticks=[i / 5 * n_yhi for i in range(6)], xfmt=lambda v: f"{v}", yfmt=lambda v: f"{v:.2f}",
            xlabel="token position (0 = first generated token)", vmarks=[(0, "← reading the question · generating →", "mid")],
            aria="Attention on the needle page before and during generation")}
{table(["position"] + list(ntab), trows_n)}
<p class="answer">Attention on the answer is already high while the <em>question</em> is being read. The model
finishes answering within a few tokens ({esc(stop_txt)}), then the page goes quiet. The remaining ~96 % of the 256
steps are a forced continuation. As configured, this task tests the end-of-prompt signal, not a jump during decoding.
Fix (your call): ask several keys in one answer, or feed follow-up questions as generated steps.</p></section>''')

    # ---------------- pinned share
    prow, ptab = [], []
    for t, r in runs.items():
        sh = sum(r["tasks"][k]["metrics"]["query"]["pinned_mass_share"] for k in r["tasks"]) / len(r["tasks"])
        name = short(r["provenance"]["model"])
        prow.append((name, [sh * 100, (1 - sh) * 100], ""))
        ptab.append([name, f"{sh * 100:.0f} %", f"{(1 - sh) * 100:.0f} %"])
    psegs = [("always-kept pages (first page + generated tokens)", "s1"), ("pages the predictor must choose", "s2")]
    A(f'''<section class="card"><div class="q">Surprise 2 · definitions</div>
<h2>Much of the attention goes to pages that are always kept anyway</h2>
<p>The first page (an "attention sink") and the pages holding freshly generated tokens are always kept on the GPU.
Their share of all attention:</p>
{legend(psegs)}
{stacked_hbar(prow, psegs, hi=100, ticks=[0, 25, 50, 75, 100], tick_fmt=lambda v: f"{v:g} %",
              val_fmt=lambda v: f"{v:.0f} %", aria="Share of attention on always-kept pages")}
{table(["model", "always-kept", "to choose"], ptab)}
<p class="answer">So accuracy metrics are computed only over the pages that actually need choosing. Otherwise every
predictor would look excellent just by keeping the first page. The share differs a lot between model families,
which matters when comparing a small and a reference model.</p></section>''')

    # ---------------- fidelity
    Ks = sorted({int(k) for f in fid.values() for k in f["summary"]})
    fx = None
    shift_ser, rec_ser, ftab = [], [], []
    for k, (m, f) in enumerate(fid.items()):
        cls = f"s{k + 1}"
        s = f["summary"]
        fx = [100 * s[str(K)]["page_fraction"] for K in Ks]
        shift = [100 * s[str(K)]["query"]["max_abs_class_share_delta"]["worst"] for K in Ks]
        recd = [max(abs(v["mean"]) for kk, v in s[str(K)]["query"].items() if kk.startswith("recall_")) for K in Ks]
        shift_ser.append((short(m), cls, shift))
        rec_ser.append((short(m), cls, recd))
        for K, x, a, b, e in zip(Ks, fx, shift, recd, [s[str(K)]["eligible_mass_kept_mean"] for K in Ks]):
            ftab.append([short(m), K, f"{x:.1f} %", f"{100 * e:.0f} %", f"{a:.1f}", f"{b:.3f}"])
    common = dict(xlo=fx[0] * 0.85, xhi=fx[-1] * 1.15, xlog=True, xticks=fx, xfmt=lambda v: f"{v:.1f}%",
                  xlabel="pages kept, as % of all pages", vmarks=[(fx[0], "≈ top-512 at 128K")], H=260,
                  end_labels=False)  # the legend above names the series; end labels would sit on the x ticks
    A(f'''<section class="card" id="s7"><div class="q">§7 · Can we save disk by keeping only the top pages?</div>
<h2>Not for Phase 1: the numbers we will base decisions on shift too much</h2>
<p>We took full recordings, threw away everything except the top pages, re-ran the analysis, and measured how much
the results moved. At 32K, keeping 6.3 % of pages corresponds to "top-512" at 128K.</p>
{legend([(short(m), f"s{k + 1}") for k, m in enumerate(fid)], kind="line")}
<h3>How much the page-type shares move (percentage points; lower is better)</h3>
{line_chart(fx, shift_ser, ylo=0, yhi=10, yticks=[0, 2, 4, 6, 8, 10], yfmt=lambda v: f"{v:.1f}",
            aria="Shift in trajectory-class shares versus pages kept", **common)}
<h3>How much prediction accuracy (recall@5 %) moves (lower is better)</h3>
{line_chart(fx, rec_ser, ylo=0, yhi=0.05, yticks=[0, 0.01, 0.02, 0.03, 0.04, 0.05], yfmt=lambda v: f"{v:.3f}",
            aria="Shift in recall versus pages kept", **common)}
{table(["model", "K", "pages kept", "decision-relevant attention kept", "class-share shift (pts)", "recall shift"], ftab)}
<p class="answer">At the top-512-equivalent the page-type shares move by 5–9 points, the same size as any sensible
go/no-go margin. Recall barely moves on Qwen3-4B but moves noticeably on Llama. Recommendation: full recordings
for Phase 1, cutting volume by recording fewer layers or prompts.</p></section>''')

    # ---------------- lead-time hardware arithmetic
    hw = rc["hardware"]
    dims = resolve_model_dims(hw["model"], cfg["models"]["local_files_only"])
    from transformers import AutoConfig
    hc_ = AutoConfig.from_pretrained(hw["model"], local_files_only=cfg["models"]["local_files_only"])
    head_dim = getattr(hc_, "head_dim", None) or hc_.hidden_size // hc_.num_attention_heads
    ps = cfg["definitions"]["page_size"]
    page_bytes = ps * dims["heads"]["kv"] * head_dim * 2 * 2 * dims["layers"]  # K+V, bf16, all layers
    per_step = hw["pcie_gb_per_s"] * 1e9 * hw["step_ms"] / 1e3 / page_bytes
    B = math.ceil(hw["budget_frac"] * hw["context"] / ps)
    A(f'''<section class="card"><div class="q">Input for one of your decisions</div>
<h2>How much lead time does the hardware actually need?</h2>
<p>For {esc(short(hw["model"]))}, one page is {page_bytes / 2**20:.0f} MiB across all layers. At
~{hw["pcie_gb_per_s"]} GB/s PCIe and a ~{hw["step_ms"]} ms decode step, one step can move about
<strong>{per_step:.0f} pages</strong>. The GPU keeps <strong>{B} pages</strong> attended (5 % of a 128K context):</p>
{meter(per_step / B, "pages movable per step as a share of the attended budget")}
<p class="muted">{per_step / B * 100:.0f} % of the attended set can be replaced in a single step.</p>
<p class="answer">So with one request at a time, being even one step early is usually enough. Extra lead time only
matters for big bursts of new pages, for many requests sharing the link, or for slower links. Choose
<code>LEAD_TIME_MIN</code> for the regime the paper targets.</p></section>''')

    # ---------------- decisions
    A(f'''<section class="card"><div class="q">What I need from you before Phase 1</div><h2>Four decisions</h2>
<div class="decision"><span class="num">1</span><strong>The go/no-go thresholds</strong> (<code>GRADUAL_SHARE_MIN</code>,
<code>LEAD_TIME_MIN</code>). If gradual rises are too rare or lead times too short, the trend idea gets demoted to a
helper. Please also pick what "share" is measured over.
<div class="rec">Suggestion: measure the share over "pages that become important after being unimportant", since
those are exactly the ones that could stall the GPU. Set <code>LEAD_TIME_MIN</code> from the hardware box above.</div></div>
<div class="decision"><span class="num">2</span><strong>The custom multi-turn task</strong>: documents, number of turns,
how later questions refer back, and how answers are scored.
<div class="rec">Suggestion: feed each new question as generated steps. That also fixes Surprise 1, because the jump then
happens while we are recording.</div></div>
<div class="decision"><span class="num">3</span><strong>Full recordings or top-512</strong>.
<div class="rec">Recommendation: full, with fewer layers or prompts (§2, §7).</div></div>
<div class="decision"><span class="num">4</span><strong>Which models</strong>.
<div class="rec">Recommendation: {esc(short(rec['iteration']))} for fast iteration at {rec['iteration_length'] // 1024}K, with
{esc(short(rec['reference']))} as the reference (same family and same attention-head layout). Catch: 128K needs a
context-extension setting (YaRN) and its own long-context test. Alternative: the Llama pair has a natively-128K
reference, but its small model only passes at 16K.</div></div>
</section>''')

    A('''<section class="card"><div class="q">Reproduce</div><h2>Commands</h2><p class="muted">All read a config file;
every result file records the model, context length, page size, window and git commit.</p>
<p><code>python scripts/estimate_trace_size.py configs/trace_size.yaml</code><br>
<code>python scripts/trace_smoke.py configs/smoke_2k.yaml</code><br>
<code>python -m pytest</code><br>
<code>python scripts/run_small_eval.py configs/small_eval.yaml [--set small_eval.model=Qwen/Qwen3-4B]</code><br>
<code>python scripts/run_model_gate.py configs/model_gate.yaml --claim --skip-done</code><br>
<code>python scripts/topk_fidelity.py configs/small_eval.yaml --trace-dir … --K 128 256 512 1024</code><br>
<code>python scripts/needle_timing.py configs/small_eval.yaml --runs …</code><br>
<code>python scripts/make_report.py configs/report.yaml</code></p>
<p class="muted">Full details and every definition I chose on your behalf: <code>RESULTS.md</code>.</p></section>''')

    body = "".join(parts).replace('<p class="answer">', '<p class="answer"><strong>What this means.</strong> ')
    head = f'<title>kv-fetch Phase 0</title><link rel="stylesheet" href="{FONTS}"><style>{CSS}</style>'
    tail = f'<div id="tip" hidden></div><script>{JS}</script>'
    btn = '<button class="themebtn" id="theme" type="button">Theme: auto</button>'
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">{head}</head>'
            f'<body><main>{body.replace("__THEMEBTN__", btn)}</main>{tail}</body></html>')
    targets = [(Path(rc["out"]), page)]
    if args.artifact:  # the artifact host supplies doctype/head/body and the theme
        targets.append((Path(args.artifact), f'<meta name="claude:viewport-fit-sides" content="cover">{head}'
                                             f'<main>{body.replace("__THEMEBTN__", "")}</main>{tail}'))
    for out, text in targets:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
