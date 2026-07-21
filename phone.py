"""
phone.py - Reading the phone's orientation over the network
----------------------------------------------------------------------
Everything about TALKING TO THE PHONE lives here: the UDP listener that
receives sensor packets, the parser for the phone app's data format, and
the quaternion math that turns "phone orientation" into "compass
direction + angle above horizon". No astronomy, no web server - just
"what direction is the phone pointing right now".
"""

import socket
import math
import threading

UDP_PORT = 5555
LISTEN_IP = "0.0.0.0"

FALLBACK_LAT = 8.5241
FALLBACK_LON = 76.9366

# The phone app sends multiple sensor readings in one packet, each tagged
# with an ID. Most carry 3 values (x,y,z) but ID 8 (GPS timestamp) only
# carries 1 - this table is how the parser knows how many fields to read.
SENSOR_FIELD_COUNTS = {1: 3, 8: 1}

# The current pointing direction + location, shared between the UDP
# listener thread and the web server. Always read/write this under
# state_lock, since two different threads touch it.
shared_state = {"az": None, "alt": None,
                "lat": FALLBACK_LAT, "lon": FALLBACK_LON}
state_lock = threading.Lock()


def quat_to_azimuth_altitude(x, y, z):
    """
    The phone's Rotation Vector sensor gives x,y,z of a unit quaternion
    (w is omitted since it's always positive and can be reconstructed).

    We rotate the phone's BACK CAMERA direction - (0,0,-1) in the
    device's own coordinate frame - through the device-to-world rotation
    this quaternion represents, giving a real East/North/Up vector, which
    converts to compass azimuth and altitude above the horizon.
    """
    w_sq = 1.0 - (x * x + y * y + z * z)
    w = math.sqrt(w_sq) if w_sq > 0 else 0.0

    world_x = -2 * (x * z + w * y)      # East component
    world_y = 2 * (w * x - y * z)       # North component
    world_z = 2 * (x * x + y * y) - 1   # Up component

    azimuth = math.degrees(math.atan2(world_x, world_y)) % 360
    altitude = math.degrees(math.asin(max(-1.0, min(1.0, world_z))))
    return azimuth, altitude


def parse_packet(text, state):
    """Parses one line of the phone app's CSV-ish sensor format into `state`."""
    parts = [p.strip() for p in text.split(',')]
    i = 1  # skip the leading timestamp
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
        if sensor_id == 1:      # GPS
            state["lat"], state["lon"] = values[0], values[1]
        elif sensor_id == 84:   # Rotation Vector
            state["quat"] = tuple(values)
        i += 1 + n_fields


def udp_listener_thread():
    """Runs forever, receiving phone sensor packets and updating shared_state."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, UDP_PORT))
    print(f"[phone] Listening for phone data on UDP port {UDP_PORT}...")
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


def start_listener():
    """Starts the UDP listener in a background thread and returns immediately."""
    t = threading.Thread(target=udp_listener_thread, daemon=True)
    t.start()
    return t


def get_current_state():
    """Thread-safe read of the current pointing direction + location."""
    with state_lock:
        return dict(shared_state)
