"""
StarGazer - Live Sky Viewfinder (v2)
--------------------------------------
A graphical "point at the sky, see what's there" viewfinder: shows stars,
constellations, and Messier deep-sky objects (nebulae/galaxies/clusters)
plus planets - all live, centered on wherever your phone is pointing.

SETUP (one-time):
    pip install skyfield matplotlib numpy --break-system-packages

    First run also downloads:
      - a small NASA planetary data file (~17MB, for planets)
      - a small star/constellation/deep-sky-object catalog (~2MB)
    Both are cached locally afterward - only downloads once, ever.

HOW TO RUN:
    1. Run this script: python sky_view.py
    2. On your phone, turn on "Switch Stream" in Sensorstream IMU+GPS.
    3. A black window opens showing the patch of sky your phone's back
       camera is pointed at, live - white dots are stars, gray lines are
       constellations, cyan markers are Messier objects, colored labeled
       dots are planets. The red "+" in the center is exactly where
       you're pointing.

NOTES / LIMITATIONS (v1 of this viewer):
    - Stars are filtered to magnitude <= STAR_MAG_LIMIT for a clean,
      responsive display. Lower number = fewer, brighter-only stars.
    - The projection is a simple flat approximation, accurate for most
      pointing directions but gets stretched out near straight overhead
      (zenith). Fine for a working demo; a true gnomonic projection can
      replace it later if that becomes a problem in practice.
    - The view isn't locked to true north being "up" on screen - it just
      centers on wherever you're currently pointing.
"""

from skyfield.api import load, wgs84, Star
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import socket
import math
import os
import json
import threading
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("TkAgg")

# ── CONFIG ──────────────────────────────────────────────────────────
UDP_PORT = 5555
LISTEN_IP = "0.0.0.0"

FALLBACK_LAT = 8.5241
FALLBACK_LON = 76.9366

FOV_RADIUS_DEG = 45.0        # half-width of the visible patch of sky
STAR_MAG_LIMIT = 4.5         # dimmer stars filtered out (lower = fewer stars)
UPDATE_INTERVAL_MS = 700     # how often to recompute + redraw

CACHE_DIR = "catalog_cache"
CATALOG_URLS = {
    "stars":      "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/stars.6.json",
    "starnames":  "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/starnames.json",
    "constlines": "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/constellations.lines.json",
    "messier":    "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/messier.json",
}

SENSOR_FIELD_COUNTS = {1: 3, 8: 1}


# ── CATALOG DOWNLOAD + CACHE ────────────────────────────────────────
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
        ra.append(lon_to_ra_deg(lon) / 15.0)  # hours
        dec.append(lat)
        mag.append(m)
        name.append(label)

    return {
        "ra_hours": np.array(ra),
        "dec_deg": np.array(dec),
        "mag": np.array(mag),
        "name": name,
    }


def load_constellation_lines(paths):
    with open(paths["constlines"], encoding="utf-8") as f:
        raw = json.load(f)

    segments = []  # list of (ra_hours_array, dec_deg_array) per polyline
    for feat in raw["features"]:
        for line in feat["geometry"]["coordinates"]:
            ra = np.array([lon_to_ra_deg(lon) / 15.0 for lon, lat in line])
            dec = np.array([lat for lon, lat in line])
            segments.append((ra, dec))
    return segments


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

    return {
        "ra_hours": np.array(ra),
        "dec_deg": np.array(dec),
        "name": name,
        "mag": np.array(mag),
    }


# ── PHONE ORIENTATION MATH ──────────────────────────────────────────
def quat_to_azimuth_altitude(x, y, z):
    """Back-camera direction (0,0,-1 in device frame) rotated into
    real-world East/North/Up via the phone's orientation quaternion."""
    w_sq = 1.0 - (x * x + y * y + z * z)
    w = math.sqrt(w_sq) if w_sq > 0 else 0.0

    world_x = -2 * (x * z + w * y)
    world_y = 2 * (w * x - y * z)
    world_z = 2 * (x * x + y * y) - 1

    azimuth = math.degrees(math.atan2(world_x, world_y)) % 360
    altitude = math.degrees(math.asin(max(-1.0, min(1.0, world_z))))
    return azimuth, altitude


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


def udp_listener_thread(shared_state, lock):
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
        with lock:
            if "quat" in local:
                x, y, z = local["quat"]
                az, alt = quat_to_azimuth_altitude(x, y, z)
                shared_state["az"] = az
                shared_state["alt"] = alt
            if "lat" in local:
                shared_state["lat"] = local["lat"]
                shared_state["lon"] = local["lon"]


# ── PROJECTION ───────────────────────────────────────────────────────
def project(az_deg, alt_deg, az0, alt0):
    daz = (az_deg - az0 + 180) % 360 - 180
    dalt = alt_deg - alt0
    x = daz * math.cos(math.radians(alt0))
    y = dalt
    return x, y


def angular_distance(az1, alt1, az2, alt2):
    daz = min(abs(az1 - az2), 360 - abs(az1 - az2))
    dalt = alt1 - alt2
    return math.sqrt(daz ** 2 + dalt ** 2)


# ── MAIN ─────────────────────────────────────────────────────────────
def main():
    print("Downloading/loading star catalogs...")
    paths = download_catalogs()
    stars = load_stars(paths, STAR_MAG_LIMIT)
    const_lines = load_constellation_lines(paths)
    messier = load_messier(paths)
    print(f"Loaded {len(stars['ra_hours'])} stars, {len(const_lines)} "
          f"constellation line segments, {len(messier['ra_hours'])} Messier objects.\n")

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
    messier_obj = Star(
        ra_hours=messier["ra_hours"], dec_degrees=messier["dec_deg"])

    # flatten all constellation-line points into one array for one bulk conversion
    all_line_ra, all_line_dec, line_lengths = [], [], []
    for ra_arr, dec_arr in const_lines:
        all_line_ra.extend(ra_arr)
        all_line_dec.extend(dec_arr)
        line_lengths.append(len(ra_arr))
    line_points_obj = Star(ra_hours=np.array(
        all_line_ra), dec_degrees=np.array(all_line_dec))

    shared_state = {"az": None, "alt": None,
                    "lat": FALLBACK_LAT, "lon": FALLBACK_LON}
    lock = threading.Lock()
    t = threading.Thread(target=udp_listener_thread,
                         args=(shared_state, lock), daemon=True)
    t.start()
    print("Turn on 'Switch Stream' on your phone now.\n")

    # ── FIGURE SETUP ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(-FOV_RADIUS_DEG, FOV_RADIUS_DEG)
    ax.set_ylim(-FOV_RADIUS_DEG, FOV_RADIUS_DEG)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    star_scatter = ax.scatter([], [], s=[], c="white")
    messier_scatter = ax.scatter([], [], s=40, c="cyan", marker="s")
    planet_scatter = ax.scatter([], [], s=120, c="orange", marker="o")
    center_marker = ax.scatter(
        [0], [0], s=200, c="red", marker="+", linewidths=2)

    line_collection = matplotlib.collections.LineCollection(
        [], colors="gray", linewidths=0.6, alpha=0.6)
    ax.add_collection(line_collection)

    title_text = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center",
                         color="white", fontsize=10, family="monospace")

    star_labels = []
    messier_labels = []
    planet_labels = []

    def update(frame):
        with lock:
            az0, alt0 = shared_state["az"], shared_state["alt"]
            lat, lon = shared_state["lat"], shared_state["lon"]

        if az0 is None:
            title_text.set_text("Waiting for phone orientation data...")
            return star_scatter, messier_scatter, planet_scatter, line_collection

        t_now = ts.now()
        observer = earth + wgs84.latlon(lat, lon)

        # stars
        s_app = observer.at(t_now).observe(star_obj).apparent()
        s_alt, s_az, _ = s_app.altaz()
        sx, sy = project(s_az.degrees, s_alt.degrees, az0, alt0)
        sizes = np.clip((6.0 - stars["mag"]) * 8, 8, 80)
        star_scatter.set_offsets(np.column_stack([sx, sy]))
        star_scatter.set_sizes(sizes)

        # constellation lines
        l_app = observer.at(t_now).observe(line_points_obj).apparent()
        l_alt, l_az, _ = l_app.altaz()
        lx, ly = project(l_az.degrees, l_alt.degrees, az0, alt0)
        segments = []
        idx = 0
        for length in line_lengths:
            seg_x = lx[idx:idx + length]
            seg_y = ly[idx:idx + length]
            segments.append(np.column_stack([seg_x, seg_y]))
            idx += length
        line_collection.set_segments(segments)

        # messier
        m_app = observer.at(t_now).observe(messier_obj).apparent()
        m_alt, m_az, _ = m_app.altaz()
        mx, my = project(m_az.degrees, m_alt.degrees, az0, alt0)
        messier_scatter.set_offsets(np.column_stack([mx, my]))

        for lbl in messier_labels:
            lbl.remove()
        messier_labels.clear()
        for i in range(len(mx)):
            if abs(mx[i]) < FOV_RADIUS_DEG and abs(my[i]) < FOV_RADIUS_DEG:
                lbl = ax.text(mx[i] + 1, my[i] + 1, messier["name"][i],
                              color="cyan", fontsize=7, clip_on=True)
                messier_labels.append(lbl)

        # planets
        px_list, py_list, planet_names = [], [], []
        for name, body in PLANET_BODIES.items():
            p_app = observer.at(t_now).observe(body).apparent()
            p_alt, p_az, _ = p_app.altaz()
            px, py = project(p_az.degrees, p_alt.degrees, az0, alt0)
            if angular_distance(az0, alt0, p_az.degrees, p_alt.degrees) < FOV_RADIUS_DEG * 1.5:
                px_list.append(px)
                py_list.append(py)
                planet_names.append(name)
        planet_scatter.set_offsets(np.column_stack(
            [px_list, py_list]) if px_list else np.empty((0, 2)))

        for lbl in planet_labels:
            lbl.remove()
        planet_labels.clear()
        for i, name in enumerate(planet_names):
            lbl = ax.text(px_list[i] + 1.5, py_list[i] + 1.5, name,
                          color="orange", fontsize=9, weight="bold", clip_on=True)
            planet_labels.append(lbl)

        # bright star labels
        for lbl in star_labels:
            lbl.remove()
        star_labels.clear()
        bright_mask = stars["mag"] < 2.0
        for i in np.where(bright_mask)[0]:
            if abs(sx[i]) < FOV_RADIUS_DEG and abs(sy[i]) < FOV_RADIUS_DEG and stars["name"][i]:
                lbl = ax.text(sx[i] + 1, sy[i] + 1, stars["name"][i],
                              color="lightyellow", fontsize=8, clip_on=True)
                star_labels.append(lbl)

        title_text.set_text(f"Pointing: az={az0:6.1f}  alt={alt0:6.1f}   "
                            f"lat={lat:.4f} lon={lon:.4f}")

        in_view_stars = int(
            np.sum((np.abs(sx) < FOV_RADIUS_DEG) & (np.abs(sy) < FOV_RADIUS_DEG)))
        print(f"[frame] az={az0:.1f} alt={alt0:.1f}  stars_in_view={in_view_stars}  "
              f"messier_offsets={len(mx)}  planets_in_view={len(planet_names)}  "
              f"line_segments={len(segments)}")

        fig.canvas.draw_idle()

        return star_scatter, messier_scatter, planet_scatter, line_collection, title_text

    ani = animation.FuncAnimation(
        fig, update, interval=UPDATE_INTERVAL_MS, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
