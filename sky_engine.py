"""
sky_engine.py - The astronomy calculations
----------------------------------------------------------------------
Everything about TURNING CATALOG DATA INTO "WHERE IS IT RIGHT NOW" lives
here. Loads the planetary ephemeris and star/constellation/Messier
catalogs once at startup, then computes live azimuth/altitude for every
object on demand. No phone, no web server - just "given a place and a
time, where is everything in the sky".
"""

import time

import numpy as np
import requests
from skyfield.api import EarthSatellite, load, wgs84, Star

import catalog

STAR_MAG_LIMIT = 4.5
SATELLITE_TLE_REFRESH_SECONDS = 6 * 60 * 60
CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"

# A focused on-screen set rather than thousands of objects: the major crewed/
# science satellites plus a small Starlink sample. Their real-time sky position
# is still calculated from live TLE orbital elements.
SATELLITE_TARGETS = {
    "ISS (ZARYA)": "stations",
    "CSS (TIANHE)": "stations",
    "HST": "science",
}
# Fetch individual Starlink objects instead of CelesTrak's enormous full
# constellation feed, which may correctly reject automated bulk downloads.
# These are individual active Starlink catalog IDs. Keeping the sample small
# avoids CelesTrak's blocked bulk Starlink endpoint.
STARLINK_CATALOG_IDS = (44714, 44718)


class SkyEngine:
    def __init__(self):
        print("Loading star/constellation/deep-sky catalogs...")
        paths = catalog.download_catalogs()
        self.stars = catalog.load_stars(paths, STAR_MAG_LIMIT)
        self.const_lines = catalog.load_constellation_lines(paths)
        self.messier = catalog.load_messier(paths)
        print(f"Loaded {len(self.stars['ra_hours'])} stars, {len(self.const_lines)} "
              f"constellation line segments, {len(self.messier['ra_hours'])} Messier objects.")

        print("Loading planetary data (first run downloads ~17MB, then cached)...")
        self.eph = load('de421.bsp')
        self.ts = load.timescale()
        self.earth = self.eph['earth']
        self.planet_bodies = {
            'Sun':     self.eph['sun'],
            'Moon':    self.eph['moon'],
            'Mercury': self.eph['mercury'],
            'Venus':   self.eph['venus'],
            'Mars':    self.eph['mars'],
            'Jupiter': self.eph['jupiter barycenter'],
            'Saturn':  self.eph['saturn barycenter'],
        }
        print("Planetary data loaded.\n")

        self.star_obj = Star(
            ra_hours=self.stars["ra_hours"], dec_degrees=self.stars["dec_deg"])
        self.messier_obj = Star(
            ra_hours=self.messier["ra_hours"], dec_degrees=self.messier["dec_deg"])

        all_ra, all_dec, self.line_lengths = [], [], []
        for ra_arr, dec_arr in self.const_lines:
            all_ra.extend(ra_arr)
            all_dec.extend(dec_arr)
            self.line_lengths.append(len(ra_arr))
        self.line_points_obj = Star(ra_hours=np.array(
            all_ra), dec_degrees=np.array(all_dec))
        self.satellites = []
        self._satellite_last_refresh = 0.0

    @staticmethod
    def _parse_tle_records(text):
        """Turn a CelesTrak three-line TLE response into (name, line1, line2)."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return [(lines[index], lines[index + 1], lines[index + 2])
                for index in range(0, len(lines) - 2, 3)
                if lines[index + 1].startswith("1 ") and lines[index + 2].startswith("2 ")]

    def _refresh_satellites(self):
        """Refresh orbital data from CelesTrak; retain the last good set if offline."""
        now = time.time()
        if self.satellites and now - self._satellite_last_refresh < SATELLITE_TLE_REFRESH_SECONDS:
            return
        records = []
        self._satellite_last_refresh = now  # back off even if a source rejects us
        try:
            groups = {}
            for name, group in SATELLITE_TARGETS.items():
                groups.setdefault(group, []).append(name)
            for group, wanted_names in groups.items():
                response = requests.get(CELESTRAK_URL, params={"GROUP": group, "FORMAT": "tle"}, timeout=12)
                response.raise_for_status()
                for name, line1, line2 in self._parse_tle_records(response.text):
                    if name in wanted_names:
                        records.append((name, line1, line2, "satellite"))

            for catalog_id in STARLINK_CATALOG_IDS:
                response = requests.get(CELESTRAK_URL, params={"CATNR": catalog_id, "FORMAT": "tle"}, timeout=12)
                response.raise_for_status()
                for name, line1, line2 in self._parse_tle_records(response.text):
                    records.append((name, line1, line2, "starlink"))
        except requests.RequestException as exc:
            # A partial set (for example ISS/Tiangong/Hubble) is still useful.
            # Most importantly, do not hammer the endpoint after a 403.
            print(f"Satellite TLE update partially unavailable: {exc}")

        if records:
            self.satellites = [
                {"name": name, "kind": kind,
                 "satellite": EarthSatellite(line1, line2, name, self.ts)}
                for name, line1, line2, kind in records
            ]
            print(f"Loaded {len(self.satellites)} live satellite TLEs.")

    def compute_satellite_snapshot(self, lat, lon):
        """Current alt/az positions for the moving satellite overlay."""
        self._refresh_satellites()
        observer = wgs84.latlon(lat, lon)
        now = self.ts.now()
        output = []
        for record in self.satellites:
            alt, az, distance = (record["satellite"] - observer).at(now).altaz()
            output.append({
                "name": record["name"], "kind": record["kind"],
                "az": float(az.degrees), "alt": float(alt.degrees),
                "range_km": round(float(distance.km), 1),
            })
        return {"type": "satellites", "satellites": output}

    def compute_catalog_snapshot(self, lat, lon):
        """Computes current alt/az for every catalog object. Returns a JSON-safe dict."""
        t_now = self.ts.now()
        observer = self.earth + wgs84.latlon(lat, lon)

        s_app = observer.at(t_now).observe(self.star_obj).apparent()
        s_alt, s_az, _ = s_app.altaz()

        m_app = observer.at(t_now).observe(self.messier_obj).apparent()
        m_alt, m_az, _ = m_app.altaz()

        l_app = observer.at(t_now).observe(self.line_points_obj).apparent()
        l_alt, l_az, _ = l_app.altaz()

        lines_out = []
        idx = 0
        for length in self.line_lengths:
            seg_az = l_az.degrees[idx:idx + length]
            seg_alt = l_alt.degrees[idx:idx + length]
            lines_out.append([{"az": float(a), "alt": float(b)}
                             for a, b in zip(seg_az, seg_alt)])
            idx += length

        planets_out = []
        for name, body in self.planet_bodies.items():
            p_app = observer.at(t_now).observe(body).apparent()
            p_alt, p_az, _ = p_app.altaz()
            planets_out.append({"name": name, "az": float(
                p_az.degrees), "alt": float(p_alt.degrees)})

        stars_out = [
            {"az": float(az), "alt": float(alt),
             "mag": float(mag), "name": name}
            for az, alt, mag, name in zip(s_az.degrees, s_alt.degrees, self.stars["mag"], self.stars["name"])
        ]
        messier_out = [
            {"az": float(az), "alt": float(alt), "mag": float(
                mag), "name": name, "otype": otype}
            for az, alt, mag, name, otype in zip(
                m_az.degrees, m_alt.degrees, self.messier["mag"], self.messier["name"], self.messier["type"])
        ]

        return {
            "type": "catalog",
            "stars": stars_out,
            "messier": messier_out,
            "planets": planets_out,
            "constellation_lines": lines_out,
        }
