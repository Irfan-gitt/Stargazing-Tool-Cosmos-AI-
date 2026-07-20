"""
StarGazer - Step A: Coordinates Only
--------------------------------------
Reads phone data over UDP and prints ONLY the coordinates:
azimuth, altitude, latitude, longitude.

No planet matching here - this is just to confirm the phone -> laptop
data pipeline works and the numbers look sane.

Run:
    python coords_only.py
Then toggle "Switch Stream" on the phone.
"""

import socket
import math

UDP_PORT = 5555
LISTEN_IP = "0.0.0.0"

SENSOR_FIELD_COUNTS = {
    1: 3,   # GPS: lat, lon, alt
    8: 1,   # GPS timestamp: single value
}


def quat_to_azimuth_altitude(x, y, z):
    """
    Rotates the phone's BACK CAMERA direction (0,0,-1 in device frame,
    opposite the screen) through the device-to-world rotation this
    quaternion represents, giving a real East/North/Up vector, which we
    convert to compass azimuth and altitude above the horizon.
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
            state['lat'], state['lon'] = values[0], values[1]
        elif sensor_id == 84:
            state['quat'] = tuple(values)

        i += 1 + n_fields


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, UDP_PORT))
    print(
        f"Listening on port {UDP_PORT}. Turn on 'Switch Stream' on your phone now.")
    print("Press Ctrl+C to stop.\n")

    state = {'lat': None, 'lon': None, 'quat': None}
    count = 0

    try:
        while True:
            data, _ = sock.recvfrom(4096)
            text = data.decode('utf-8', errors='replace').strip()
            parse_packet(text, state)

            count += 1
            if count % 5 != 0:
                continue

            if state['quat'] is None:
                print("(waiting for rotation vector data...)")
                continue

            az, alt = quat_to_azimuth_altitude(*state['quat'])
            lat = state['lat'] if state['lat'] is not None else "no GPS fix yet"
            lon = state['lon'] if state['lon'] is not None else ""
            print(f"az={az:7.2f}  alt={alt:7.2f}   lat={lat}  lon={lon}")

    except KeyboardInterrupt:
        print("\nStopped.")
        sock.close()


if __name__ == "__main__":
    main()
