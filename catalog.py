"""
catalog.py - Star / Constellation / Deep-Sky-Object catalog loading
----------------------------------------------------------------------
Everything about DOWNLOADING and PARSING the sky catalog lives here.
No astronomy math, no networking to the phone - just: get the data,
turn it into clean Python structures.

Data source: d3-celestial (open, BSD-licensed) on GitHub.
"""

import os
import json
import urllib.request

import numpy as np

CACHE_DIR = "catalog_cache"
CATALOG_URLS = {
    "stars":      "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/stars.6.json",
    "starnames":  "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/starnames.json",
    "constlines": "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/constellations.lines.json",
    "messier":    "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/messier.json",
}


def download_catalogs():
    """Downloads each catalog file once, then reuses the cached copy forever."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = {}
    for key, url in CATALOG_URLS.items():
        path = os.path.join(CACHE_DIR, key + ".json")
        if not os.path.exists(path):
            print(f"Downloading {key} catalog...")
            urllib.request.urlretrieve(url, path)
        paths[key] = path
    return paths


def lon_to_ra_deg(lon):
    """The catalog stores positions as GeoJSON longitude (-180..180).
    Astronomy uses right ascension (0..360). This converts one to the other."""
    return lon if lon >= 0 else lon + 360


def load_stars(paths, mag_limit):
    """Returns stars dimmer than mag_limit filtered out (keeps the view clean)."""
    with open(paths["stars"], encoding="utf-8") as f:
        raw = json.load(f)
    with open(paths["starnames"], encoding="utf-8") as f:
        names = json.load(f)

    ra, dec, mag, name = [], [], [], []
    for feat in raw["features"]:
        m = feat["properties"]["mag"]
        if m > mag_limit:
            continue
        lon, lat = feat["geometry"]["coordinates"]
        info = names.get(str(feat["id"]), {})
        label = info.get("name") or info.get("bayer") or info.get("flam") or ""
        ra.append(lon_to_ra_deg(lon) / 15.0)  # convert degrees -> hours
        dec.append(lat)
        mag.append(m)
        name.append(label)

    return {"ra_hours": np.array(ra), "dec_deg": np.array(dec), "mag": np.array(mag), "name": name}


def load_constellation_lines(paths):
    """Returns a list of (ra_hours_array, dec_deg_array) - one per line segment."""
    with open(paths["constlines"], encoding="utf-8") as f:
        raw = json.load(f)

    segments = []
    for feat in raw["features"]:
        for line in feat["geometry"]["coordinates"]:
            ra = np.array([lon_to_ra_deg(lon) / 15.0 for lon, lat in line])
            dec = np.array([lat for lon, lat in line])
            segments.append((ra, dec))
    return segments


def load_messier(paths):
    """Returns all 110 Messier objects (nebulae, galaxies, clusters), with their type code."""
    with open(paths["messier"], encoding="utf-8") as f:
        raw = json.load(f)

    ra, dec, name, mag, otype = [], [], [], [], []
    for feat in raw["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        props = feat["properties"]
        label = props.get("alt") or props.get("name") or feat["id"]
        ra.append(lon_to_ra_deg(lon) / 15.0)
        dec.append(lat)
        name.append(label)
        mag.append(props.get("mag", 99) or 99)
        otype.append(props.get("type", ""))

    return {"ra_hours": np.array(ra), "dec_deg": np.array(dec), "name": name,
            "mag": np.array(mag), "type": otype}
