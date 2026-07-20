"""
StarGazer Web - Backend Server
--------------------------------
Serves the live sky view as a web page instead of a matplotlib window.
Same phone-orientation math and star/planet catalog as before, pushed to
a browser over WebSocket - this gives us a real UI to build on, including
click-an-object -> AI explanation (currently a placeholder, see /api/explain).

SETUP (one-time):
    pip install fastapi "uvicorn[standard]" skyfield numpy --break-system-packages

HOW TO RUN:
    1. Make sure index.html is in the SAME FOLDER as this file.
    2. python server.py
    3. Open http://localhost:8000 in your browser.
    4. On your phone, turn on "Switch Stream" in Sensorstream IMU+GPS.
"""

import socket
import math
import os
import json
import threading
import time
import asyncio
import urllib.request

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from skyfield.api import load, wgs84, Star

# ── CONFIG ──────────────────────────────────────────────────────────
UDP_PORT = 5555
LISTEN_IP = "0.0.0.0"

FALLBACK_LAT = 8.5241
FALLBACK_LON = 76.9366

STAR_MAG_LIMIT = 4.5
CATALOG_REFRESH_SECONDS = 20  # how often to recompute all star/planet alt-az

CACHE_DIR = "catalog_cache"
CATALOG_URLS = {
    "stars":      "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/stars.6.json",
    "starnames":  "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/starnames.json",
    "constlines": "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/constellations.lines.json",
    "messier":    "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/messier.json",
}

SENSOR_FIELD_COUNTS = {1: 3, 8: 1}


# ── CATALOG DOWNLOAD + CACHE (same as sky_view.py, proven working) ──
def download_catalogs():
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
    return lon if lon >= 0 else lon + 360
# ============================================================
# PURPOSE:
# Converts longitude values from the downloaded catalog into
# Right Ascension format expected by Skyfield.
#
# WHY:
# Some catalogs store longitude as negative values.
# Skyfield expects values between 0° and 360°.
#
# INPUT:
# Longitude
#
# OUTPUT:
# Corrected Right Ascension in degrees.
# ============================================================


def load_stars(paths, mag_limit):
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
        ra.append(lon_to_ra_deg(lon) / 15.0)
        dec.append(lat)
        mag.append(m)
        name.append(label)

    return {"ra_hours": np.array(ra), "dec_deg": np.array(dec), "mag": np.array(mag), "name": name}

    # ============================================================
    # PURPOSE:
    # Reads the star catalog and creates a clean list of stars.
    #
    # WHY:
    # The downloaded JSON contains thousands of stars with lots
    # of unnecessary information.
    #
    # This function extracts only:
    # - Right Ascension
    # - Declination
    # - Magnitude
    # - Name
    #
    # Dim stars are ignored to improve performance.
    #
    # OUTPUT:
    # Dictionary containing star positions and names.
    # ============================================================


def load_constellation_lines(paths):
    with open(paths["constlines"], encoding="utf-8") as f:
        raw = json.load(f)

    segments = []
    for feat in raw["features"]:
        for line in feat["geometry"]["coordinates"]:
            ra = np.array([lon_to_ra_deg(lon) / 15.0 for lon, lat in line])
            dec = np.array([lat for lon, lat in line])
            segments.append((ra, dec))
    return segments

    # ============================================================
    # PURPOSE:
    # Loads constellation line data.
    #
    # WHY:
    # Stars are only points.
    #
    # This function loads which stars should be connected together
    # so constellations like Orion can be drawn.
    #
    # OUTPUT:
    # List of constellation line segments.
    # ============================================================


def load_messier(paths):

    with open(paths["messier"], encoding="utf-8") as f:
        raw = json.load(f)

    ra, dec, name, mag = [], [], [], []
    for feat in raw["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        props = feat["properties"]
        label = props.get("alt") or props.get("name") or feat["id"]
        ra.append(lon_to_ra_deg(lon) / 15.0)
        dec.append(lat)
        name.append(label)
        mag.append(props.get("mag", 99) or 99)

    return {"ra_hours": np.array(ra), "dec_deg": np.array(dec), "name": name, "mag": np.array(mag)}
    # ============================================================
    # PURPOSE:
    # Loads deep sky objects such as galaxies,
    # nebulae and star clusters.
    #
    # WHY:
    # StarGazer should display more than just stars and planets.
    #
    # OUTPUT:
    # Dictionary containing Messier object positions.
    # ============================================================


# ── PHONE ORIENTATION MATH (proven working) ─────────────────────────
def quat_to_azimuth_altitude(x, y, z):
    w_sq = 1.0 - (x * x + y * y + z * z)
    w = math.sqrt(w_sq) if w_sq > 0 else 0.0
    world_x = -2 * (x * z + w * y)
    world_y = 2 * (w * x - y * z)
    world_z = 2 * (x * x + y * y) - 1
    azimuth = math.degrees(math.atan2(world_x, world_y)) % 360
    altitude = math.degrees(math.asin(max(-1.0, min(1.0, world_z))))
    return azimuth, altitude

    # ============================================================
    # PURPOSE:
    # Converts the phone's Rotation Vector (Quaternion)
    # into Azimuth and Altitude.
    #
    # WHY:
    # Phones do NOT directly tell us where they are pointing.
    #
    # Sensorstream sends a Quaternion.
    #
    # This function converts that mathematical rotation into
    # normal sky coordinates.
    #
    # INPUT:
    # Quaternion (x, y, z)
    #
    # OUTPUT:
    # Azimuth
    # Altitude
    #
    # NOTE:
    # I only need to understand the INPUT and OUTPUT.
    # The quaternion math itself comes from 3D rotation formulas.
    # ============================================================


def parse_packet(text, state):
    parts = [p.strip() for p in text.split(',')]
    i = 1
    while i < len(parts):
        try:
            sensor_id = int(float(parts[i]))
        except ValueError:
            break
        n_fields = SENSOR_FIELD_COUNTS.get(sensor_id, 3)
        values = parts[i + 1: i + 1 + n_fields]
        if len(values) < n_fields:
            break
        try:
            values = [float(v) for v in values]
        except ValueError:
            break
        if sensor_id == 1:
            state["lat"], state["lon"] = values[0], values[1]
        elif sensor_id == 84:
            state["quat"] = tuple(values)
        i += 1 + n_fields

        # ============================================================
        # PURPOSE:
        # Parses raw Sensorstream UDP packets.
        #
        # WHY:
        # The phone sends sensor values as one long comma-separated
        # string.
        #
        # This function separates the packet and extracts:
        # - GPS
        # - Rotation Vector
        #
        # OUTPUT:
        # Updates the temporary state dictionary.
        # ============================================================


shared_state = {"az": None, "alt": None,
                "lat": FALLBACK_LAT, "lon": FALLBACK_LON}
state_lock = threading.Lock()


def udp_listener_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, UDP_PORT))
    print(f"[listener] Listening on UDP port {UDP_PORT}...")
    while True:
        data, _ = sock.recvfrom(4096)
        text = data.decode("utf-8", errors="replace").strip()
        local = {}
        parse_packet(text, local)
        if not local:
            continue
        with state_lock:
            if "quat" in local:
                x, y, z = local["quat"]
                az, alt = quat_to_azimuth_altitude(x, y, z)
                shared_state["az"] = az
                shared_state["alt"] = alt
            if "lat" in local:
                shared_state["lat"] = local["lat"]
                shared_state["lon"] = local["lon"]
    # ============================================================
    # PURPOSE:
    # Continuously listens for UDP packets sent by the phone.
    #
    # WORKFLOW:
    #
    # Phone
    #   ↓
    # Receive Packet
    #   ↓
    # Parse Packet
    #   ↓
    # Convert Quaternion
    #   ↓
    # Save Latest Phone Direction
    #
    # WHY:
    # Runs forever in a separate thread so the website can keep
    # running while sensor data is constantly updated.
    # ============================================================


# ── LOAD CATALOGS + EPHEMERIS AT STARTUP ────────────────────────────
print("Loading star/constellation/deep-sky catalogs...")
paths = download_catalogs()
stars = load_stars(paths, STAR_MAG_LIMIT)
const_lines = load_constellation_lines(paths)
messier = load_messier(paths)
print(f"Loaded {len(stars['ra_hours'])} stars, {len(const_lines)} constellation "
      f"line segments, {len(messier['ra_hours'])} Messier objects.")

print("Loading planetary data (first run downloads ~17MB, then cached)...")
eph = load('de421.bsp')
ts = load.timescale()
earth = eph['earth']
PLANET_BODIES = {
    'Sun':     eph['sun'],
    'Moon':    eph['moon'],
    'Mercury': eph['mercury'],
    'Venus':   eph['venus'],
    'Mars':    eph['mars'],
    'Jupiter': eph['jupiter barycenter'],
    'Saturn':  eph['saturn barycenter'],
}
print("Planetary data loaded.\n")

star_obj = Star(ra_hours=stars["ra_hours"], dec_degrees=stars["dec_deg"])
messier_obj = Star(ra_hours=messier["ra_hours"],
                   dec_degrees=messier["dec_deg"])

_all_line_ra, _all_line_dec, LINE_LENGTHS = [], [], []
for _ra_arr, _dec_arr in const_lines:
    _all_line_ra.extend(_ra_arr)
    _all_line_dec.extend(_dec_arr)
    LINE_LENGTHS.append(len(_ra_arr))
line_points_obj = Star(ra_hours=np.array(_all_line_ra),
                       dec_degrees=np.array(_all_line_dec))


def compute_catalog_snapshot(lat, lon):
    """Computes current alt/az for every catalog object. Returns a JSON-safe dict."""
    t_now = ts.now()
    observer = earth + wgs84.latlon(lat, lon)

    s_app = observer.at(t_now).observe(star_obj).apparent()
    s_alt, s_az, _ = s_app.altaz()

    m_app = observer.at(t_now).observe(messier_obj).apparent()
    m_alt, m_az, _ = m_app.altaz()

    l_app = observer.at(t_now).observe(line_points_obj).apparent()
    l_alt, l_az, _ = l_app.altaz()

    lines_out = []
    idx = 0
    for length in LINE_LENGTHS:
        seg_az = l_az.degrees[idx:idx + length]
        seg_alt = l_alt.degrees[idx:idx + length]
        lines_out.append([{"az": float(a), "alt": float(b)}
                         for a, b in zip(seg_az, seg_alt)])
        idx += length

    planets_out = []
    for name, body in PLANET_BODIES.items():
        p_app = observer.at(t_now).observe(body).apparent()
        p_alt, p_az, _ = p_app.altaz()
        planets_out.append({"name": name, "az": float(
            p_az.degrees), "alt": float(p_alt.degrees)})

    stars_out = [
        {"az": float(az), "alt": float(alt), "mag": float(mag), "name": name}
        for az, alt, mag, name in zip(s_az.degrees, s_alt.degrees, stars["mag"], stars["name"])
    ]
    messier_out = [
        {"az": float(az), "alt": float(alt), "mag": float(mag), "name": name}
        for az, alt, mag, name in zip(m_az.degrees, m_alt.degrees, messier["mag"], messier["name"])
    ]

    return {
        "type": "catalog",
        "stars": stars_out,
        "messier": messier_out,
        "planets": planets_out,
        "constellation_lines": lines_out,
    }
    # ============================================================
    # PURPOSE:
    # Calculates the current sky.
    #
    # WHY:
    # Given:
    # - Current Time
    # - GPS Location
    #
    # Skyfield calculates where every star,
    # planet, constellation and Messier object currently appears.
    #
    # This is the astronomy engine of StarGazer.
    #
    # OUTPUT:
    # JSON-ready dictionary sent to the frontend.
    # ============================================================


# ── FASTAPI APP ──────────────────────────────────────────────────────
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()
    # ============================================================
    # PURPOSE:
    # Serves the frontend webpage.
    #
    # Browser
    #     ↓
    # localhost:8000
    #     ↓
    # index.html
    # ============================================================


@app.get("/api/explain")
async def explain(name: str, type: str = "object"):
    """
    PLACEHOLDER - not wired up to a real AI yet on purpose (per plan: build
    the plumbing first, pick the LLM later). This proves the full pipeline
    works: browser click -> HTTP request -> backend -> response -> panel.
    Swap the body of this function for a real LLM call when ready.
    """
    return JSONResponse({
        "name": name,
        "type": type,
        "explanation": f"(AI explanation for {name} goes here - not wired up yet.)",
    })
    # ============================================================
    # PURPOSE:
    # Future AI endpoint.
    #
    # WORKFLOW:
    #
    # User clicks an object
    #        ↓
    # Browser sends request
    #        ↓
    # AI generates explanation
    #        ↓
    # Browser displays result
    #
    # NOTE:
    # This is where I will integrate my LLM later.
    # ============================================================


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_sent = None
    last_catalog_time = 0.0
    try:
        while True:
            with state_lock:
                az, alt, lat, lon = (shared_state["az"], shared_state["alt"],
                                     shared_state["lat"], shared_state["lon"])

            if time.time() - last_catalog_time > CATALOG_REFRESH_SECONDS:
                snapshot = compute_catalog_snapshot(lat, lon)
                await websocket.send_json(snapshot)
                last_catalog_time = time.time()

            if az is not None and (az, alt) != last_sent:
                await websocket.send_json({"type": "pointing", "az": az, "alt": alt, "lat": lat, "lon": lon})
                last_sent = (az, alt)

            await asyncio.sleep(0.15)

    except WebSocketDisconnect:
        pass
    # ============================================================
    # PURPOSE:
    # Sends live updates to the browser.
    #
    # WHY:
    # Instead of the browser repeatedly asking for updates,
    # the server PUSHES new data automatically.
    #
    # Sends:
    # - Phone direction
    # - Updated sky catalog
    #
    # This makes the sky move in real time.
    # ============================================================


def main():
    t = threading.Thread(target=udp_listener_thread, daemon=True)
    t.start()
    print("Turn on 'Switch Stream' on your phone, then open http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
