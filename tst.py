"""
StarGazer - Demo v1
--------------------
Reads phone orientation (Rotation Vector) + GPS over UDP from the
Sensorstream IMU+GPS Android app, and tells you which planet you're
pointing your phone's back camera at.

SETUP (one-time):
    pip install skyfield --break-system-packages

HOW TO RUN:
    1. Start this script FIRST: python stargazer.py
    2. It will download a small planetary data file the first time
       (needs internet, ~17MB, one-time only, then cached forever).
    3. On your phone, open Sensorstream IMU+GPS -> Toggle Sensors,
       and make sure "Rotation Vect." is checked ON (it's not on by
       default in some setups - that's the #1 cause of no output).
    4. Turn on "Switch Stream".
    5. Hold your phone with the BACK CAMERA pointing at the sky,
       like you're taking a photo of the stars.
    6. Watch the terminal for live planet detection.

HOLDING THE PHONE:
    Hold it like a camera - top edge tilted up toward the sky,
    back camera facing up and outward. If detection feels off by
    a consistent amount, we may need to add a calibration offset -
    tell me what you see and we'll adjust.
"""

import socket
import math
from skyfield.api import load, wgs84

# ── CONFIG ──────────────────────────────────────────────────────────
UDP_PORT = 5555
LISTEN_IP = "0.0.0.0"

# Fallback location if GPS hasn't given us a fix yet (Trivandrum, Kerala)
# This gets overwritten automatically the moment real GPS data arrives.
FALLBACK_LAT = 8.5241
FALLBACK_LON = 76.9366

MATCH_TOLERANCE_DEGREES = 10.0  # how close counts as "pointing at it"

# Sensor IDs this app streams, and how many float values follow each one.
# Confirmed against a known-working parser for this exact app (Sensorstream
# IMU+GPS by Axel Lorenz) - not a guess. Most sensors send 3 values, but
# GPS time / pressure / battery temp send just 1.
SENSOR_FIELD_COUNTS = {
    1: 3,    # GPS (lat, lon, alt)
    3: 3,    # Accelerometer (x, y, z) - m/s^2
    4: 3,    # Gyroscope (x, y, z) - rad/s
    5: 3,    # Magnetometer (x, y, z) - microTesla
    6: 3,    # GPS Cartesian/ECEF (x, y, z) - meters
    7: 3,    # GPS velocity (x, y, z) - m/s
    8: 1,    # GPS time - ms
    81: 3,   # Orientation (x, y, z) - degrees
    82: 3,   # Linear acceleration (x, y, z)
    83: 3,   # Gravity (x, y, z) - m/s^2
    84: 3,   # Rotation Vector (x, y, z) - what we actually need
    85: 1,   # Pressure
    86: 1,   # Battery temperature
}
DEFAULT_FIELD_COUNT = 3  # fallback for any sensor ID not listed above


# ── SKYFIELD SETUP ──────────────────────────────────────────────────
print("Loading planetary data (first run downloads ~17MB, then cached)...")
eph = load('de421.bsp')
ts = load.timescale()
earth = eph['earth']

BODIES = {
    'Sun':     eph['sun'],
    'Moon':    eph['moon'],
    'Mercury': eph['mercury'],
    'Venus':   eph['venus'],
    'Mars':    eph['mars'],
    'Jupiter': eph['jupiter barycenter'],
    'Saturn':  eph['saturn barycenter'],
}
print("Planetary data loaded.\n")


# ── MATH: quaternion (from phone) -> azimuth/altitude ──────────────
def quat_to_azimuth_altitude(x, y, z):
    """
    Android's Rotation Vector sensor gives x, y, z of a unit quaternion
    (w is omitted since it's always positive and can be reconstructed).

    Returns (azimuth_degrees, altitude_degrees) representing where the
    phone's BACK is pointing - i.e. where the back camera looks.
    """
    w_sq = 1.0 - (x * x + y * y + z * z)
    w = math.sqrt(w_sq) if w_sq > 0 else 0.0

    # yaw = compass heading (azimuth), 0=North, 90=East, etc.
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    azimuth = yaw % 360

    # pitch = how far tilted from flat -> used as altitude above horizon
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))

    # When phone is held with back camera pointing at sky (screen down,
    # top edge tilted up), altitude above horizon relates to pitch.
    # NOTE: this mapping is our best first guess - if real-world testing
    # shows a consistent offset, we adjust this line.
    altitude = 90 - abs(pitch) if pitch < 0 else pitch

    return azimuth, altitude


def angular_distance(az1, alt1, az2, alt2):
    """Rough angular distance between two az/alt points, in degrees."""
    daz = min(abs(az1 - az2), 360 - abs(az1 - az2))
    dalt = alt1 - alt2
    return math.sqrt(daz ** 2 + dalt ** 2)


def find_closest_body(azimuth, altitude, lat, lon):
    t = ts.now()
    observer = earth + wgs84.latlon(lat, lon)

    best_name, best_diff, best_alt = None, 999, None
    for name, body in BODIES.items():
        astrometric = observer.at(t).observe(body).apparent()
        body_alt, body_az, _ = astrometric.altaz()
        diff = angular_distance(
            azimuth, altitude, body_az.degrees, body_alt.degrees)
        if diff < best_diff:
            best_name, best_diff, best_alt = name, diff, body_alt.degrees

    return best_name, best_diff, best_alt


# ── PARSE ONE UDP PACKET ────────────────────────────────────────────
def parse_packet(text, state):
    """
    Updates `state` dict in place with any GPS / rotation data found.
    Packet format: timestamp, id, v1[, v2, v3...], id, v1[, v2, v3...], ...
    """
    parts = [p.strip() for p in text.split(',')]
    i = 1  # skip leading timestamp

    while i < len(parts):
        try:
            sensor_id = int(float(parts[i]))
        except ValueError:
            break

        count = SENSOR_FIELD_COUNTS.get(sensor_id, DEFAULT_FIELD_COUNT)

        # Bail out cleanly on a truncated/garbled packet instead of
        # raising an IndexError that would kill the whole script.
        if i + count >= len(parts):
            break

        try:
            values = [float(v) for v in parts[i + 1: i + 1 + count]]
        except ValueError:
            break

        if sensor_id == 1:  # GPS
            state['lat'], state['lon'] = values[0], values[1]
        elif sensor_id == 84:  # Rotation Vector
            state['quat'] = tuple(values)
        # else: known/unused or unmapped sensor - already consumed above

        i += 1 + count


# ── MAIN LOOP ────────────────────────────────────────────────────────
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, UDP_PORT))
    print(
        f"Listening on port {UDP_PORT}. Turn on 'Switch Stream' on your phone now.")
    print("Press Ctrl+C to stop.\n")

    state = {'lat': FALLBACK_LAT, 'lon': FALLBACK_LON, 'quat': None}
    frame_count = 0
    hint_shown = False

    try:
        while True:
            data, _ = sock.recvfrom(4096)
            text = data.decode('utf-8', errors='replace').strip()

            try:
                parse_packet(text, state)

                frame_count += 1
                # only process every 5th packet (throttle output)
                if frame_count % 5 != 0:
                    continue

                if state['quat'] is None:
                    if not hint_shown and frame_count >= 300:
                        print("\n[hint] No Rotation Vector data (ID 84) has "
                              "arrived yet. On the phone, open the app's "
                              "'Toggle Sensors' screen and make sure "
                              "'Rotation Vect.' is checked on.")
                        hint_shown = True
                    continue

                az, alt = quat_to_azimuth_altitude(*state['quat'])
                name, diff, body_alt = find_closest_body(
                    az, alt, state['lat'], state['lon'])

                pointer = f"Pointing: az={az:6.1f} deg, alt={alt:6.1f} deg"
                if diff < MATCH_TOLERANCE_DEGREES:
                    line = f"{pointer}  ->  ✨ {name}!  ({diff:.1f} deg off)"
                else:
                    line = f"{pointer}  ->  closest: {name} ({diff:.1f} deg away, alt={body_alt:.1f})"
                # \r returns to the start of the line, \033[K erases any
                # leftover characters from a longer previous line - this
                # makes it update in place like a live readout instead of
                # scrolling a new line every time.
                print(f"\r{line}\033[K", end='', flush=True)

            except Exception as e:
                # A single bad/short packet shouldn't take down the whole
                # session - report it and keep listening.
                print(f"\n[!] Skipped a bad packet ({e})")
                continue

    except KeyboardInterrupt:
        print("\nStopped.")
        sock.close()


main()
