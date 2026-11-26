"""Small inline SVG icons for the map and its legend (no icon font, so they also work offline)."""

from __future__ import annotations

DARK = "#27272a"
HEART_RED = "#dc2626"


def fork_and_spoon(size: int = 16, color: str = "#fff") -> str:
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="{color}" aria-hidden="true">'
            '<ellipse cx="8" cy="7" rx="2.7" ry="3.8"/><rect x="7.15" y="9.5" width="1.7" height="11" rx=".85"/>'
            '<rect x="13.4" y="3" width="1.1" height="5.5" rx=".55"/>'
            '<rect x="15.45" y="3" width="1.1" height="5.5" rx=".55"/>'
            '<rect x="17.5" y="3" width="1.1" height="5.5" rx=".55"/>'
            '<path d="M13.4 7.5h5.2v1.3a2.6 2.6 0 0 1-5.2 0z"/>'
            '<rect x="15.15" y="10" width="1.7" height="10.5" rx=".85"/></svg>')


def heart_in_hand(size: int = 18, hand: str = DARK, heart: str = HEART_RED) -> str:
    """The usual charity symbol: a heart held in an open hand."""
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" aria-hidden="true">'
            f'<g transform="translate(6.2 1.2) scale(.58)"><path fill="{heart}" d="M12 21.35l-1.45-1.32C5.4 15.36 '
            '2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 '
            '22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></g>'
            f'<g fill="none" stroke="{hand}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M2.5 15.6H6l3.2 2h4.3a1.3 1.3 0 0 1 0 2.6H9.6"/>'
            '<path d="M13.6 20.2l4.9-2.6a1.3 1.3 0 0 1 1.3 2.2l-5.6 3H2.5"/></g></svg>')


def truck(size: int = 18, color: str = "#fff", cut: str = "#000") -> str:
    """A delivery truck; `cut` is the badge color, used for the window and the wheel rims."""
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="{color}" aria-hidden="true">'
            '<rect x="1.5" y="5.5" width="12.5" height="10" rx="1.2"/>'
            '<path d="M15 8.8h3.6c.4 0 .8.2 1 .5l2.4 3.2c.1.2.2.4.2.6v2.4H15z"/>'
            f'<path d="M16.3 10.2h2.2l1.7 2.3h-3.9z" fill="{cut}"/>'
            f'<circle cx="6" cy="17.4" r="2.4" stroke="{cut}" stroke-width="1.3"/>'
            f'<circle cx="18" cy="17.4" r="2.4" stroke="{cut}" stroke-width="1.3"/></svg>')
