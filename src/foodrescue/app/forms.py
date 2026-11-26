"""Pieces shared by the account forms: picking a place on the map, and opening hours."""

from __future__ import annotations

import datetime as dt

import folium
import streamlit as st
from streamlit_folium import st_folium

from foodrescue.models import Location


def location_picker(key: str, start: Location, label: str = "Location") -> tuple[float, float]:
    """A map: click to place the account. The choice is kept in the session until the form is saved.

    Kept outside `st.form` (a click must move the marker at once); the form reads the chosen point."""
    store = f"_location_{key}"
    if store not in st.session_state:
        st.session_state[store] = (round(start.lat, 5), round(start.lon, 5))
    lat, lon = st.session_state[store]
    st.markdown(f"**{label}** · click the map to move the marker")
    fmap = folium.Map(location=(lat, lon), zoom_start=15, tiles="OpenStreetMap", scroll_wheel_zoom=False)
    folium.Marker((lat, lon), tooltip="Here").add_to(fmap)
    answer = st_folium(fmap, height=260, use_container_width=True, returned_objects=["last_clicked"],
                       key=f"pick-{key}-{lat}-{lon}")
    clicked = (answer or {}).get("last_clicked")
    if clicked:
        point = (round(clicked["lat"], 5), round(clicked["lng"], 5))
        if point != (lat, lon):
            st.session_state[store] = point
            st.rerun()
    st.caption(f"{lat:.5f}, {lon:.5f}")
    return lat, lon


def forget_location(key: str) -> None:
    st.session_state.pop(f"_location_{key}", None)


def to_time(minutes: int) -> dt.time:
    minutes = max(0, min(minutes, 24 * 60 - 1))
    return dt.time(minutes // 60, minutes % 60)


def to_minutes(value: dt.time) -> int:
    return value.hour * 60 + value.minute
