"""Visual building blocks shared by every screen: the style sheet and small HTML components.

Streamlit draws the widgets (buttons, inputs); these helpers draw everything that is only read
(headers, stat tiles, status pills, steppers, message lists). All text coming from people or
agents is escaped before it is put in HTML.
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from urllib.parse import quote

import streamlit as st

from foodrescue.app import icons

# Status tones. Each pill always shows its label, so color is never the only cue.
TONES = ("neutral", "info", "good", "warning", "serious", "critical")

CSS = """
<style>
/* Neutral palette: white surfaces, zinc grays, near-black for text and the one accent. */
:root {
  --fr-ink: #18181b; --fr-ink-2: #52525b; --fr-muted: #71717a; --fr-line: #e4e4e7; --fr-line-soft: #f4f4f5;
  --fr-surface: #ffffff; --fr-bg: #fafafa;
  --fr-blue: #2563eb; --fr-green: #16a34a; --fr-amber: #d97706; --fr-orange: #ea580c; --fr-red: #dc2626;
}
[data-testid="stMainBlockContainer"].block-container {
  padding-top: 5.6rem !important; padding-bottom: 3rem; max-width: 1400px;
}
.stAppDeployButton { display: none; }

/* Top bar: the native page navigation across the whole window, fixed, with the app's name on its left. */
header[data-testid="stHeader"] {
  position: fixed !important; top: 0; left: 0 !important; right: 0; width: 100vw !important;
  z-index: 999970 !important; /* below the sidebar, so the sidebar's own button can sit on the bar */
  height: 3.9rem; background: var(--fr-surface); border-bottom: 1px solid var(--fr-line);
}
header[data-testid="stHeader"] > div { height: 100%; align-items: center; }
div:has(> nav[data-testid="stTopNav"]) { display: flex; align-items: center; height: 100%; }
div:has(> nav[data-testid="stTopNav"])::before {
  content: "Food Rescue"; font-size: 1.15rem; font-weight: 650; color: var(--fr-ink); white-space: nowrap;
  margin: 0 2rem 0 4.4rem;
}
/* Streamlit keeps a slot before the links for its "open sidebar" button, 15px wide only while the sidebar is
   closed. That button now sits on its own at the top left, so the slot takes no room: the links never move. */
div:has(> nav[data-testid="stTopNav"]) > div, div:has(> nav[data-testid="stTopNav"]) > div > div {
  flex: 0 0 0 !important; width: 0 !important; min-width: 0 !important; margin: 0 !important; padding: 0 !important;
}

/* The presenter sidebar slides over the page, below the top bar, instead of pushing the page aside. */
section[data-testid="stSidebar"] {
  position: fixed !important; top: 3.9rem !important; left: 0; height: calc(100vh - 3.9rem) !important;
  z-index: 999980 !important; border-right: 1px solid var(--fr-line); box-shadow: 4px 0 16px rgba(0, 0, 0, .06);
  overflow: hidden; /* when closed it has no width: nothing of it may show */
}
[data-testid="stSidebarHeader"] { height: .75rem !important; min-height: 0 !important; padding: 0 !important; }

/* One hamburger button, always visible, at the top left before the app's name: it opens and closes the
   sidebar. Streamlit has two buttons (close, inside the sidebar; open, in the top bar); both are moved here. */
[data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] {
  position: fixed !important; top: calc((3.9rem - 36px) / 2) !important; left: 10px !important;
  z-index: 1000001 !important; visibility: visible !important; opacity: 1 !important; display: block !important;
}
[data-testid="stSidebarCollapseButton"] button, button[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] button {
  width: 36px !important; height: 36px !important; border-radius: 6px !important; display: flex !important;
  align-items: center; justify-content: center; color: var(--fr-ink) !important; visibility: visible !important;
}
[data-testid="stSidebarCollapseButton"] button:hover, button[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stExpandSidebarButton"] button:hover { background: var(--fr-line-soft) !important; }
/* Streamlit's own icon wrapper has its own, smaller hover box at the button's left edge: remove it, so the only
   hover box is the button itself, with the hamburger lines centered in it. */
[data-testid="stSidebarCollapseButton"] button > *, button[data-testid="stExpandSidebarButton"] > *,
[data-testid="stExpandSidebarButton"] button > * { display: none !important; }
[data-testid="stSidebarCollapseButton"] button::after, button[data-testid="stExpandSidebarButton"]::after,
[data-testid="stExpandSidebarButton"] button::after {
  content: ""; width: 18px; height: 14px; display: block; flex: none;
  background: linear-gradient(currentColor 0 0) top / 100% 2px no-repeat,
              linear-gradient(currentColor 0 0) center / 100% 2px no-repeat,
              linear-gradient(currentColor 0 0) bottom / 100% 2px no-repeat;
}
[data-testid="stTopNavLink"] {
  padding: .5rem .8rem !important; border-radius: 0 !important; background: transparent !important;
  height: auto !important;
}
[data-testid="stTopNavLink"] * { color: var(--fr-muted) !important; }
[data-testid="stTopNavLink"] [data-testid="stMarkdownContainer"] p { font-size: .95rem !important; font-weight: 500; }
[data-testid="stTopNavLink"] [data-testid="stIconMaterial"] { font-size: 1.1rem !important; }
[data-testid="stTopNavLink"]:hover * { color: var(--fr-ink) !important; }
[data-testid="stTopNavLink"][aria-current="page"] { box-shadow: inset 0 -2px 0 var(--fr-ink); }
[data-testid="stTopNavLink"][aria-current="page"] * { color: var(--fr-ink) !important; }

[class*="st-key-card"] {
  background: var(--fr-surface); border: 1px solid var(--fr-line); border-radius: 8px; padding: 1rem 1.15rem;
}
[class*="st-key-hero"] { background: var(--fr-ink); border-radius: 8px; padding: 1.15rem 1.3rem; }
.fr-eyebrow { font-size: .8rem; color: var(--fr-muted); font-weight: 500; }
.fr-title { font-size: 1.6rem; font-weight: 650; color: var(--fr-ink); line-height: 1.2; margin: .15rem 0 .2rem;
  letter-spacing: -.015em; }
.fr-sub { color: var(--fr-muted); font-size: .9rem; }
.fr-section { font-size: .95rem; font-weight: 600; color: var(--fr-ink); margin: 0 0 .1rem; }
.fr-section-sub { font-size: .8rem; color: var(--fr-muted); margin: 0 0 .75rem; }
.fr-muted { color: var(--fr-muted); font-size: .85rem; }
.fr-empty { color: var(--fr-muted); font-size: .87rem; padding: .5rem 0; }

.fr-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 4px; }
.fr-tile { background: var(--fr-surface); border: 1px solid var(--fr-line); border-radius: 8px; padding: 14px 16px; }
.fr-tile .label { font-size: .8rem; color: var(--fr-muted); font-weight: 500; }
.fr-tile .value { font-size: 1.55rem; font-weight: 600; color: var(--fr-ink); margin-top: 4px;
  font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.fr-tile .value small { font-size: .85rem; font-weight: 500; color: var(--fr-muted); margin-left: 3px; }
.fr-tile .hint { font-size: .75rem; color: var(--fr-muted); margin-top: 2px; }

/* Status: a colored dot next to the label, so color is never the only cue. */
.fr-pill { display: inline-flex; align-items: center; gap: 6px; padding: 1px 8px; border-radius: 6px;
  border: 1px solid var(--fr-line); background: var(--fr-surface); color: var(--fr-ink-2); font-size: .76rem;
  font-weight: 500; white-space: nowrap; line-height: 1.6; }
.fr-pill::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: #a1a1aa; }
.fr-pill.info::before { background: var(--fr-blue); }
.fr-pill.good::before { background: var(--fr-green); }
.fr-pill.warning::before { background: var(--fr-amber); }
.fr-pill.serious::before { background: var(--fr-orange); }
.fr-pill.critical::before { background: var(--fr-red); }
.fr-tag { display: inline-block; font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: .7rem;
  font-weight: 500; padding: 0 5px; border-radius: 4px; background: var(--fr-line-soft); color: var(--fr-ink-2); }
.fr-tag.serious { color: #c2410c; }

.fr-steps { display: flex; gap: 4px; margin: .6rem 0 .35rem; }
.fr-step { flex: 1; min-width: 0; }
.fr-step .bar { height: 3px; border-radius: 2px; background: var(--fr-line); }
.fr-step .name { font-size: .72rem; color: var(--fr-muted); margin-top: 6px; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.fr-step.done .bar { background: var(--fr-ink); }
.fr-step.done .name { color: var(--fr-ink-2); }
.fr-step.current .bar { background: linear-gradient(90deg, var(--fr-ink) 50%, var(--fr-line) 50%); }
.fr-step.current .name { color: var(--fr-ink); font-weight: 600; }
.fr-step.failed .bar { background: var(--fr-red); }
.fr-step.failed .name { color: var(--fr-red); font-weight: 600; }
.fr-step.waiting .bar { background: var(--fr-amber); }
.fr-step.waiting .name { color: #b45309; font-weight: 600; }

.fr-list-item { display: grid; grid-template-columns: 46px 1fr; gap: 10px; padding: 9px 0;
  border-bottom: 1px solid var(--fr-line-soft); font-size: .86rem; }
.fr-list-item:last-child { border-bottom: none; }
.fr-time { color: var(--fr-muted); font-variant-numeric: tabular-nums; font-size: .8rem; padding-top: 1px; }
.fr-who { color: var(--fr-ink); font-weight: 600; }
.fr-arrow { color: var(--fr-muted); margin: 0 4px; }
.fr-text { color: var(--fr-ink-2); margin-top: 2px; line-height: 1.45; }

.fr-quote { background: var(--fr-bg); border: 1px solid var(--fr-line-soft); border-radius: 6px; padding: 8px 12px;
  color: var(--fr-ink-2); font-size: .87rem; margin: .5rem 0; }
.fr-quote.cite { border-left: 2px solid var(--fr-ink); border-radius: 0 6px 6px 0; }
.fr-quote.ask { border-left: 2px solid var(--fr-amber); border-radius: 0 6px 6px 0; color: var(--fr-ink); }
.fr-kv { display: grid; grid-template-columns: 110px 1fr; gap: 4px 12px; font-size: .86rem; margin-top: .5rem; }
.fr-kv .k { color: var(--fr-muted); }
.fr-kv .v { color: var(--fr-ink); }
.fr-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; }

.fr-table { width: 100%; border-collapse: collapse; font-size: .85rem; }
.fr-table th { text-align: left; font-weight: 500; color: var(--fr-muted); font-size: .76rem; padding: 6px 10px;
  border-bottom: 1px solid var(--fr-line); }
.fr-table td { padding: 9px 10px; border-bottom: 1px solid var(--fr-line-soft); color: var(--fr-ink);
  vertical-align: top; }
.fr-table td.num { font-variant-numeric: tabular-nums; text-align: right; }
.fr-table th.num { text-align: right; }
.fr-table tr:last-child td { border-bottom: none; }

.fr-meter { margin-bottom: 12px; }
.fr-meter .top { display: flex; justify-content: space-between; font-size: .83rem; color: var(--fr-ink-2);
  margin-bottom: 5px; }
.fr-meter .top b { color: var(--fr-ink); font-weight: 500; }
.fr-meter .track { height: 6px; background: var(--fr-line-soft); border-radius: 3px; overflow: hidden; }
.fr-meter .fill { height: 100%; background: var(--fr-ink); border-radius: 3px; }

.fr-tl { position: relative; margin: .3rem 0 0; }
.fr-tl-item { display: grid; grid-template-columns: 52px 18px 1fr; gap: 8px; padding-bottom: 14px; position: relative; }
.fr-tl-item .dot { width: 11px; height: 11px; border-radius: 50%; border: 2px solid #d4d4d8;
  background: var(--fr-surface); margin-top: 4px; position: relative; z-index: 1; }
.fr-tl-item::after { content: ""; position: absolute; left: 65px; top: 18px; bottom: -2px; width: 1px;
  background: var(--fr-line); }
.fr-tl-item:last-child::after { display: none; }
.fr-tl-item.done .dot { background: var(--fr-ink); border-color: var(--fr-ink); }
.fr-tl-item.next .dot { border-color: var(--fr-ink); }
.fr-tl-item.done .what, .fr-tl-item.done .fr-time { color: var(--fr-muted); }
.fr-tl-item .what { font-size: .9rem; color: var(--fr-ink); font-weight: 500; }
.fr-tl-item.next .what { font-weight: 600; }
.fr-tl-item .detail { font-size: .8rem; color: var(--fr-muted); margin-top: 1px; }

.fr-hero-eyebrow { font-size: .8rem; color: #a1a1aa; font-weight: 500; }
.fr-hero-title { font-size: 1.35rem; font-weight: 600; margin: .2rem 0 .2rem; color: #fafafa; }
.fr-hero-sub { color: #d4d4d8; font-size: .88rem; }
.fr-hero-time { font-size: 2rem; font-weight: 600; font-variant-numeric: tabular-nums; color: #fafafa;
  line-height: 1; }

.fr-legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .8rem; color: var(--fr-ink-2); margin-top: 8px; }
.fr-legend > span { display: inline-flex; align-items: center; gap: 6px; }
.fr-swatch { width: 16px; height: 5px; border-radius: 3px; display: inline-block; }
.fr-swatch.dashed { background: repeating-linear-gradient(90deg, #a1a1aa 0 4px, transparent 4px 7px) !important; }

.fr-clock { margin: 4px 0 2px; }
.fr-clock .value { font-size: 2.2rem; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--fr-ink);
  letter-spacing: -.02em; line-height: 1.15; }
.fr-progress { height: 4px; border-radius: 2px; background: var(--fr-line); overflow: hidden; margin: 8px 0 4px; }
.fr-progress div { height: 100%; background: var(--fr-ink); }
</style>
"""


def apply_theme() -> None:
    html(CSS + merchant_tab_icon_css())


def merchant_tab_icon_css() -> str:
    """The Merchant tab's icon: our own fork and spoon (Streamlit's icon set only has them with a knife).
    It is painted over Streamlit's icon, which keeps its place so the tab's spacing does not change."""
    svg = icons.fork_and_spoon(24, color="#000").replace('width="24" height="24" ',
                                                         'xmlns="http://www.w3.org/2000/svg" ')
    mask = f'url("data:image/svg+xml,{quote(svg)}") center / 1.1rem 1.1rem no-repeat'
    link = '[data-testid="stTopNavLink"][href$="/merchant"]'
    return (f"<style>{link} [data-testid=\"stIconMaterial\"] {{ color: transparent !important; "
            f"background-color: var(--fr-muted); -webkit-mask: {mask}; mask: {mask}; }}"
            f"{link}:hover [data-testid=\"stIconMaterial\"], {link}[aria-current=\"page\"] "
            "[data-testid=\"stIconMaterial\"] { background-color: var(--fr-ink); }</style>")


def html(markup: str) -> None:
    """Draw raw HTML. Lines are joined so Markdown never reads indented HTML as a code block."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def e(text: object) -> str:
    return escape(str(text))


def card(key: str):
    """A white card (a keyed container, styled by the `st-key-card...` class)."""
    return st.container(key=f"card-{key}")


def section(title: str, subtitle: str = "") -> None:
    html(f'<div class="fr-section">{e(title)}</div>'
         + (f'<div class="fr-section-sub">{e(subtitle)}</div>' if subtitle else '<div style="height:.45rem"></div>'))


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    html(f'<div class="fr-eyebrow">{e(eyebrow)}</div><div class="fr-title">{e(title)}</div>'
         f'<div class="fr-sub">{e(subtitle)}</div>')


def count(n: int, singular: str, plural: str | None = None) -> str:
    """'1 donation', '3 donations'."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def pill(label: str, tone: str = "neutral") -> str:
    return f'<span class="fr-pill {tone}">{e(label)}</span>'


def tag(label: str, tone: str = "") -> str:
    return f'<span class="fr-tag {tone}">{e(label)}</span>'


def tiles(items: Iterable[tuple]) -> None:
    """Stat tiles: (label, value, hint[, source]). A value like "12.5 kg" shows its unit smaller; a source is
    shown when hovering the tile (its hint is then dotted-underlined)."""
    cells = []
    for label, value, hint, *source in items:
        number, _, unit = str(value).partition(" ")
        shown = f"{e(number)}<small>{e(unit)}</small>" if unit else e(value)
        title = f' title="{e(source[0])}"' if source else ""
        hint_style = ' style="text-decoration:underline dotted;cursor:help"' if source else ""
        cells.append(f'<div class="fr-tile"{title}><div class="label">{e(label)}</div><div class="value">{shown}'
                     f'</div><div class="hint"{hint_style}>{e(hint)}</div></div>')
    html(f'<div class="fr-tiles">{"".join(cells)}</div>')


def stepper(steps: list[str], current: int, state: str = "current") -> str:
    """`state` of the current step: current, done (everything finished), failed or waiting."""
    parts = []
    for i, name in enumerate(steps):
        if state == "done" or i < current:
            cls = "done"
        elif i == current:
            cls = state
        else:
            cls = ""
        parts.append(f'<div class="fr-step {cls}"><div class="bar"></div><div class="name">{e(name)}</div></div>')
    return f'<div class="fr-steps">{"".join(parts)}</div>'


def list_items(rows: Iterable[tuple[str, str]], empty: str = "Nothing yet.") -> None:
    """Rows of (time, inner HTML), newest first is up to the caller."""
    body = "".join(f'<div class="fr-list-item"><div class="fr-time">{e(time)}</div><div>{inner}</div></div>'
                   for time, inner in rows)
    html(body or f'<div class="fr-empty">{e(empty)}</div>')


def table(headers: list[str], rows: list[list[str]], numeric: set[int] | None = None, empty: str = "") -> None:
    """A light HTML table. Cells are HTML: escape plain text with `e` first."""
    if not rows:
        html(f'<div class="fr-empty">{e(empty or "Nothing yet.")}</div>')
        return
    numeric = numeric or set()
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{e(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{cell}</td>'
                                    for i, cell in enumerate(row)) + "</tr>" for row in rows)
    html(f'<table class="fr-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


def meter(label: str, value: float, total: float, text: str) -> str:
    share = 0 if total <= 0 else max(0.0, min(1.0, value / total))
    return (f'<div class="fr-meter"><div class="top"><span>{e(label)}</span><b>{e(text)}</b></div>'
            f'<div class="track"><div class="fill" style="width:{share * 100:.0f}%"></div></div></div>')
