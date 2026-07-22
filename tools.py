"""Live astronomy tools used by the StarGazer chat assistant.

External services are deliberately optional: weather uses Open-Meteo, research
uses Serper when configured, NASA headlines use NASA's public RSS feed, and
satellites use CelesTrak's public TLE feed.  Every function returns structured
data or a clear error suitable for an AI tool response.
"""

import difflib
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from skyfield import almanac
from skyfield.api import EarthSatellite, Star, load, wgs84

load_dotenv()
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
ROOT = os.path.dirname(os.path.abspath(__file__))
NASA_RSS_URL = "https://www.nasa.gov/rss/dyn/breaking_news.rss"
CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"


def _error(message):
    return {"error": message}


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------
def web_search(query: str, num_results: int = 5):
    """Search the web with Serper. Use for current astronomy news or sources."""
    if not SERPER_API_KEY:
        return _error("SERPER_API_KEY is not configured, so web search is unavailable.")
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": max(1, min(int(num_results), 10))}, timeout=12,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return _error(f"Web search failed: {exc}")
    return {
        "query": query,
        "quick_answer": data.get("answerBox", {}).get("snippet"),
        "results": [{"title": item.get("title", ""), "snippet": item.get("snippet", ""),
                     "link": item.get("link", "")} for item in data.get("organic", [])[:num_results]],
    }


def nasa_articles(topic: str = "astronomy", num_results: int = 5):
    """Return recent official NASA articles, optionally filtered by a topic."""
    try:
        response = requests.get(NASA_RSS_URL, timeout=12)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError) as exc:
        return _error(f"NASA news feed could not be read: {exc}")

    topic_words = set(re.findall(r"[a-z0-9]+", topic.lower()))
    articles = []
    for item in root.findall(".//item"):
        title = item.findtext("title", default="")
        description = re.sub(r"<[^>]+>", "", item.findtext("description", default="")).strip()
        haystack = f"{title} {description}".lower()
        if topic_words and topic.lower() not in {"astronomy", "space", "latest"} and not any(word in haystack for word in topic_words):
            continue
        articles.append({"title": title, "summary": description, "link": item.findtext("link", default=""),
                         "published": item.findtext("pubDate", default="")})
        if len(articles) >= max(1, min(int(num_results), 10)):
            break
    return {"source": "NASA", "topic": topic, "articles": articles}


def latest_discoveries(num_results: int = 5):
    """Find recent astronomy and space-science discoveries from official NASA news."""
    result = nasa_articles("latest", num_results)
    if "error" not in result:
        result["note"] = "Recent NASA headlines; ask web_search for a narrower current topic."
    return result


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
def weather_conditions(latitude: float, longitude: float):
    """Get current observing weather for latitude/longitude using Open-Meteo."""
    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": latitude, "longitude": longitude,
            "current": "cloud_cover,temperature_2m,precipitation,wind_speed_10m,weather_code",
            "hourly": "cloud_cover,precipitation_probability",
            "forecast_days": 2, "timezone": "auto",
        }, timeout=12)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return _error(f"Weather request failed: {exc}")
    current = data.get("current", {})
    return {"location": {"latitude": latitude, "longitude": longitude}, "time": current.get("time"),
            "cloud_cover_percent": current.get("cloud_cover"), "temperature_c": current.get("temperature_2m"),
            "precipitation_mm": current.get("precipitation"), "wind_kmh": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code")}


def local_sky_time(latitude: float, longitude: float):
    """Get the correct local time and astronomical day/twilight/night state for a location."""
    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": latitude, "longitude": longitude, "current": "temperature_2m", "timezone": "auto",
        }, timeout=12)
        response.raise_for_status()
        time_data = response.json()
        timezone_name = time_data.get("timezone", "UTC")
        offset_seconds = int(time_data.get("utc_offset_seconds", 0))
        local_now = datetime.now(timezone(timedelta(seconds=offset_seconds)))
    except (requests.RequestException, TypeError, ValueError) as exc:
        return _error(f"Local time lookup failed: {exc}")

    try:
        ts = load.timescale()
        eph = load(os.path.join(ROOT, "de421.bsp"))
        observer = eph["earth"] + wgs84.latlon(latitude, longitude)
        sun_altitude, _, _ = observer.at(ts.now()).observe(eph["sun"]).apparent().altaz()
        sun_alt = float(sun_altitude.degrees)
    except Exception as exc:
        return _error(f"Daylight calculation failed: {exc}")

    if sun_alt >= 0:
        state = "day"
    elif sun_alt >= -6:
        state = "civil twilight"
    elif sun_alt >= -12:
        state = "nautical twilight"
    elif sun_alt >= -18:
        state = "astronomical twilight"
    else:
        state = "night"
    return {"location": {"latitude": latitude, "longitude": longitude}, "timezone": timezone_name,
            "local_time": local_now.strftime("%Y-%m-%d %H:%M:%S"), "utc_offset": local_now.strftime("%z"),
            "sun_altitude_degrees": round(sun_alt, 1), "sky_state": state,
            "note": "Sky state is based on the Sun's actual altitude at the observing location."}


def moon_phase():
    """Estimate the current lunar phase and illumination. Good for observing planning."""
    reference = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
    cycle = 29.53058867
    age = ((datetime.now(timezone.utc) - reference).total_seconds() / 86400) % cycle
    illumination = (1 - math.cos(2 * math.pi * age / cycle)) / 2 * 100
    phases = [(1.85, "New Moon"), (7.38, "Waxing Crescent"), (9.23, "First Quarter"),
              (14.77, "Waxing Gibbous"), (16.61, "Full Moon"), (22.15, "Waning Gibbous"),
              (24.0, "Last Quarter"), (27.68, "Waning Crescent"), (cycle, "New Moon")]
    name = next(label for limit, label in phases if age < limit)
    return {"phase_name": name, "illumination_percent": round(illumination, 1),
            "lunar_age_days": round(age, 1), "calculation": "Approximate, suitable for planning."}


def photography_advice(name: str, altitude_degrees: float, magnitude: float | None = None,
                       moon_illumination_percent: float | None = None, cloud_cover_percent: float | None = None):
    """Give practical astrophotography advice for an object currently in view."""
    issues = []
    if altitude_degrees < 0:
        issues.append("It is below the horizon.")
    elif altitude_degrees < 15:
        issues.append("It is very low; haze and atmospheric distortion will soften it. Wait until it is above 25°.")
    elif altitude_degrees < 25:
        issues.append("It is usable, but higher altitude will improve sharpness and contrast.")
    if cloud_cover_percent is not None and cloud_cover_percent > 50:
        issues.append(f"Cloud cover is high ({cloud_cover_percent:.0f}%).")
    if magnitude is not None and magnitude > 5 and moon_illumination_percent is not None and moon_illumination_percent > 60:
        issues.append("The bright Moon will reduce contrast for this faint target.")
    if not issues:
        issues.append("Conditions are promising. Use a tripod, manual focus, and the lowest ISO that gives a usable exposure.")
    return {"object": name, "altitude_degrees": altitude_degrees, "advice": " ".join(issues)}


def night_planner(latitude: float, longitude: float, visible_objects: list | None = None):
    """Combine weather, moon phase, and visible targets into a stargazing plan."""
    weather = weather_conditions(latitude, longitude)
    moon = moon_phase()
    if "error" in weather:
        return {"weather": weather, "moon": moon, "recommendation": "Moon data is available, but weather could not be checked."}
    cloud = weather.get("cloud_cover_percent")
    objects = visible_objects or []
    targets = [str(item.get("name", item)) if isinstance(item, dict) else str(item) for item in objects]
    if cloud is not None and cloud >= 70:
        recommendation = f"Poor observing conditions: {cloud:.0f}% cloud cover. Consider another night."
    elif cloud is not None and cloud >= 35:
        recommendation = f"Mixed conditions: {cloud:.0f}% cloud cover. Plan short observing windows and favour bright targets."
    else:
        recommendation = "Good conditions for observing, assuming your local horizon is clear."
    if moon["illumination_percent"] > 65:
        recommendation += " The bright Moon favours planets, Moon photography, and double stars over faint nebulae and galaxies."
    if targets:
        recommendation += f" Targets supplied: {', '.join(targets[:8])}."
    return {"weather": weather, "moon": moon, "targets": targets, "recommendation": recommendation}


# ---------------------------------------------------------------------------
# Object information
# ---------------------------------------------------------------------------
PLANETS = {
    "mercury": {"type": "planet", "description": "The smallest and innermost planet; it is always seen near the Sun."},
    "venus": {"type": "planet", "description": "Earth's cloud-covered neighbour and often the brightest planet in the sky."},
    "mars": {"type": "planet", "description": "The red planet, a cold desert world with polar ice caps."},
    "jupiter": {"type": "planet", "description": "The largest planet, a gas giant with bright cloud bands and four easy-to-see Galilean moons."},
    "saturn": {"type": "planet", "description": "A gas giant famous for its bright ring system."},
    "uranus": {"type": "planet", "description": "An ice giant with a sideways rotation axis; usually needs binoculars or a telescope."},
    "neptune": {"type": "planet", "description": "The most distant major planet, a faint blue ice giant requiring optical aid."},
    "sun": {"type": "star", "description": "Our nearest star. Never view or photograph it without certified solar filters."},
    "moon": {"type": "natural satellite", "description": "Earth's natural satellite and the easiest detailed target for binoculars or a telescope."},
}
MESSIER_TYPES = {
    "s": "spiral galaxy", "e": "elliptical galaxy", "i": "irregular galaxy",
    "n": "nebula", "pn": "planetary nebula", "oc": "open cluster",
    "gc": "globular cluster", "snr": "supernova remnant",
}


@lru_cache(maxsize=1)
def _catalog_entries():
    entries = []
    try:
        with open(os.path.join(ROOT, "catalog_cache", "messier.json"), encoding="utf-8") as file:
            for feature in json.load(file).get("features", []):
                props = feature.get("properties", {})
                raw_type = str(props.get("type", "deep-sky object")).lower()
                entries.append({"name": props.get("alt") or props.get("name") or str(feature.get("id", "")),
                                "catalog_id": str(feature.get("id", "")), "type": MESSIER_TYPES.get(raw_type, raw_type),
                                "magnitude": props.get("mag"), "description": props.get("name", "")})
        with open(os.path.join(ROOT, "catalog_cache", "starnames.json"), encoding="utf-8") as file:
            for identifier, props in json.load(file).items():
                name = props.get("name") or props.get("bayer") or props.get("flam")
                if name:
                    entries.append({"name": name, "catalog_id": f"HIP {identifier}", "type": "star"})
    except (OSError, json.JSONDecodeError):
        pass
    return entries


def object_information(name: str):
    """Look up a star, planet, galaxy, nebula, or Messier/catalog designation."""
    normalized = name.strip().lower()
    if normalized in PLANETS:
        return {"name": name, **PLANETS[normalized]}
    entries = _catalog_entries()
    exact = next((entry for entry in entries if entry["name"].lower() == normalized or entry["catalog_id"].lower() == normalized), None)
    if exact:
        return exact
    choices = difflib.get_close_matches(name, [entry["name"] for entry in entries], n=5, cutoff=0.45)
    return {"name": name, "found": False, "suggestions": choices,
            "note": "The local catalog contains named bright stars and Messier objects."}


@lru_cache(maxsize=1)
def _observable_catalog():
    """Coordinates for locally catalogued named stars and Messier objects."""
    objects = {}
    try:
        with open(os.path.join(ROOT, "catalog_cache", "messier.json"), encoding="utf-8") as file:
            for feature in json.load(file).get("features", []):
                props = feature.get("properties", {})
                longitude, declination = feature["geometry"]["coordinates"]
                right_ascension = longitude if longitude >= 0 else longitude + 360
                for label in (props.get("alt"), props.get("name"), str(feature.get("id", ""))):
                    if label:
                        objects[str(label).lower()] = (right_ascension / 15, declination)
        with open(os.path.join(ROOT, "catalog_cache", "stars.json"), encoding="utf-8") as star_file, \
             open(os.path.join(ROOT, "catalog_cache", "starnames.json"), encoding="utf-8") as name_file:
            names = json.load(name_file)
            for feature in json.load(star_file).get("features", []):
                props = names.get(str(feature.get("id")), {})
                longitude, declination = feature["geometry"]["coordinates"]
                right_ascension = longitude if longitude >= 0 else longitude + 360
                for label in (props.get("name"), props.get("bayer"), props.get("flam")):
                    if label:
                        objects[str(label).lower()] = (right_ascension / 15, declination)
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    return objects


def _observable_target(name, eph):
    normalized = name.strip().lower()
    planetary_bodies = {
        "sun": "sun", "moon": "moon", "mercury": "mercury", "venus": "venus",
        "mars": "mars", "jupiter": "jupiter barycenter", "saturn": "saturn barycenter",
        "uranus": "uranus barycenter", "neptune": "neptune barycenter",
    }
    if normalized in planetary_bodies:
        return eph[planetary_bodies[normalized]]
    coordinates = _observable_catalog().get(normalized)
    if coordinates:
        return Star(ra_hours=coordinates[0], dec_degrees=coordinates[1])
    return None


def rise_set_times(object_name: str, latitude: float, longitude: float, hours: int = 48):
    """Calculate upcoming rise and set times for a planet, named star, or Messier object."""
    try:
        ts = load.timescale()
        eph = load(os.path.join(ROOT, "de421.bsp"))
        target = _observable_target(object_name, eph)
        if target is None:
            return _error("Object not found in the local planet, named-star, or Messier catalog.")
        observer = wgs84.latlon(latitude, longitude)
        start = datetime.now(timezone.utc)
        end = start + timedelta(hours=max(1, min(int(hours), 72)))
        predicate = almanac.risings_and_settings(eph, target, observer)
        times, events = almanac.find_discrete(ts.from_datetime(start), ts.from_datetime(end), predicate)
    except Exception as exc:
        return _error(f"Rise/set calculation failed: {exc}")
    event_list = [{"event": "rises" if bool(event) else "sets", "time_utc": time.utc_datetime().replace(tzinfo=timezone.utc).isoformat()}
                  for time, event in zip(times, events)]
    return {"object": object_name, "location": {"latitude": latitude, "longitude": longitude},
            "window_utc": {"from": start.isoformat(), "to": end.isoformat()}, "events": event_list,
            "note": "Times use the geometric horizon and do not account for local buildings or hills."}


def light_pollution_report(sky_quality_mag: float | None = None, bortle_class: int | None = None):
    """Assess observing conditions from a supplied SQM sky brightness or Bortle class."""
    if sky_quality_mag is not None:
        sqm = float(sky_quality_mag)
        if sqm >= 21.5:
            bortle, assessment = 2, "very dark; excellent for faint nebulae and galaxies"
        elif sqm >= 20.5:
            bortle, assessment = 4, "reasonably dark; good for many deep-sky objects"
        elif sqm >= 19.5:
            bortle, assessment = 6, "bright suburban sky; favour planets, clusters, and bright nebulae"
        else:
            bortle, assessment = 8, "bright urban sky; focus on the Moon, planets, and double stars"
        return {"source": "provided SQM reading", "sqm_mag_per_arcsec2": sqm, "estimated_bortle_class": bortle, "assessment": assessment}
    if bortle_class is not None:
        bortle = max(1, min(int(bortle_class), 9))
        assessment = "excellent deep-sky observing" if bortle <= 3 else "mixed deep-sky observing" if bortle <= 5 else "bright-sky observing; favour bright targets"
        return {"source": "provided Bortle class", "estimated_bortle_class": bortle, "assessment": assessment}
    return {"needs_input": True, "message": "Light pollution needs a local SQM reading or Bortle class. Ask the observer for either value rather than guessing from coordinates."}


METEOR_SHOWERS = [
    ("Quadrantids", 1, 3, 120), ("Lyrids", 4, 22, 18), ("Eta Aquariids", 5, 5, 50),
    ("Southern Delta Aquariids", 7, 30, 25), ("Perseids", 8, 12, 100),
    ("Orionids", 10, 21, 20), ("Leonids", 11, 17, 15), ("Geminids", 12, 14, 120),
]


def meteor_showers(days_ahead: int = 90):
    """List active and upcoming major annual meteor showers and their peak dates."""
    now = datetime.now(timezone.utc).date()
    limit = max(1, min(int(days_ahead), 365))
    upcoming = []
    for name, month, day, zhr in METEOR_SHOWERS:
        peak = datetime(now.year, month, day, tzinfo=timezone.utc).date()
        if peak < now:
            peak = datetime(now.year + 1, month, day, tzinfo=timezone.utc).date()
        days_until = (peak - now).days
        if days_until <= limit:
            upcoming.append({"name": name, "peak_date_utc": peak.isoformat(), "days_until_peak": days_until,
                             "ideal_zhr": zhr, "advice": "Watch after midnight from a dark site; actual rates depend on sky darkness and Moon."})
    return {"generated_on_utc": now.isoformat(), "showers": sorted(upcoming, key=lambda item: item["days_until_peak"]),
            "note": "Peak dates and ideal ZHR are annual planning values, not a local weather forecast."}


def equipment_advice(object_name: str, object_type: str = "object", magnitude: float | None = None,
                     altitude_degrees: float | None = None, equipment: str = "phone"):
    """Recommend practical observing or imaging equipment for a target's type, brightness, and altitude."""
    kind = object_type.lower()
    gear = equipment.lower()
    if "planet" in kind or object_name.strip().lower() in PLANETS:
        recommendation = "Use binoculars for a quick view; a 90–150 mm telescope at medium magnification reveals more detail."
    elif any(word in kind for word in ("galaxy", "nebula", "cluster", "deep")):
        recommendation = "Use binoculars for bright objects or a 150 mm+ telescope under dark skies; a tripod and longer exposures help for imaging."
    elif "satellite" in kind:
        recommendation = "Unaided eyes are usually best. Use a wide field and follow it smoothly; do not use high magnification."
    else:
        recommendation = "Start with unaided eyes or 7×50/10×50 binoculars; use a small telescope for more detail."
    adjustments = []
    if altitude_degrees is not None and altitude_degrees < 20:
        adjustments.append("Wait for it to rise above 20° if possible; low altitude reduces sharpness.")
    if magnitude is not None and magnitude > 6:
        adjustments.append("This is faint; dark skies and a larger aperture matter more than magnification.")
    if gear == "phone":
        adjustments.append("For phone imaging, use a tripod, night mode/manual exposure, and tap-focus on a bright target.")
    return {"object": object_name, "equipment": equipment, "recommendation": recommendation, "adjustments": adjustments}


# ---------------------------------------------------------------------------
# Satellites
# ---------------------------------------------------------------------------
SATELLITE_GROUPS = {"iss": "stations", "tiangong": "stations", "hubble": "science", "starlink": "starlink"}
SATELLITE_NAMES = {"iss": "ISS (ZARYA)", "tiangong": "CSS (TIANHE)", "hubble": "HST"}
STARLINK_SAMPLE_CATALOG_ID = 44714


def _satellite_key(satellite: str):
    key = satellite.strip().lower()
    aliases = {"international space station": "iss", "iss (zarya)": "iss", "css": "tiangong", "hubble space telescope": "hubble"}
    return aliases.get(key, key)


def tle_updates(satellite: str = "iss", limit: int = 3):
    """Fetch current TLE orbital elements for ISS, Tiangong, Hubble, or Starlink."""
    key = _satellite_key(satellite)
    group = SATELLITE_GROUPS.get(key)
    if not group:
        return _error("Supported satellites are ISS, Tiangong, Hubble, and Starlink.")
    try:
        params = {"GROUP": group, "FORMAT": "tle"}
        if key == "starlink":
            # CelesTrak may reject the enormous full Starlink group. A single
            # real Starlink TLE is enough for status and pass prediction.
            params = {"CATNR": STARLINK_SAMPLE_CATALOG_ID, "FORMAT": "tle"}
        response = requests.get(CELESTRAK_URL, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        return _error(f"TLE update failed: {exc}")
    lines = [line.strip() for line in response.text.splitlines() if line.strip()]
    records = []
    wanted = SATELLITE_NAMES.get(key, "")
    for index in range(0, len(lines) - 2, 3):
        if key == "starlink" or wanted.lower() in lines[index].lower():
            records.append({"name": lines[index], "line1": lines[index + 1], "line2": lines[index + 2]})
            if len(records) >= max(1, min(int(limit), 10)):
                break
    return {"source": "CelesTrak", "satellite": satellite, "fetched_at_utc": datetime.now(timezone.utc).isoformat(), "tle": records}


def satellite_status(satellite: str, latitude: float, longitude: float):
    """Give the current altitude, azimuth, and range of a tracked satellite."""
    tle = tle_updates(satellite, 1)
    if "error" in tle or not tle["tle"]:
        return tle if "error" in tle else _error("No matching satellite TLE was found.")
    record = tle["tle"][0]
    ts = load.timescale()
    craft = EarthSatellite(record["line1"], record["line2"], record["name"], ts)
    observer = wgs84.latlon(latitude, longitude)
    topocentric = (craft - observer).at(ts.now())
    altitude, azimuth, distance = topocentric.altaz()
    return {"satellite": record["name"], "time_utc": datetime.now(timezone.utc).isoformat(),
            "altitude_degrees": round(float(altitude.degrees), 1), "azimuth_degrees": round(float(azimuth.degrees), 1),
            "range_km": round(float(distance.km), 1), "above_horizon": bool(altitude.degrees > 0)}


def pass_predictions(satellite: str, latitude: float, longitude: float, hours: int = 24):
    """Estimate visible-above-horizon passes over the next 1–72 hours from current TLE data."""
    tle = tle_updates(satellite, 1)
    if "error" in tle or not tle["tle"]:
        return tle if "error" in tle else _error("No matching satellite TLE was found.")
    record = tle["tle"][0]
    ts = load.timescale()
    craft = EarthSatellite(record["line1"], record["line2"], record["name"], ts)
    observer = wgs84.latlon(latitude, longitude)
    horizon = 0.0
    start = datetime.now(timezone.utc)
    end = start + timedelta(hours=max(1, min(int(hours), 72)))
    times, events = craft.find_events(observer, ts.from_datetime(start), ts.from_datetime(end), altitude_degrees=horizon)
    labels = {0: "rise", 1: "culminate", 2: "set"}
    passes, current = [], {}
    for time, event in zip(times, events):
        when = time.utc_datetime().replace(tzinfo=timezone.utc).isoformat()
        if event == 0:
            current = {"rise_utc": when}
        elif event == 1 and current:
            alt, az, _ = (craft - observer).at(time).altaz()
            current["peak_utc"] = when
            current["max_altitude_degrees"] = round(float(alt.degrees), 1)
            current["peak_azimuth_degrees"] = round(float(az.degrees), 1)
        elif event == 2 and current:
            current["set_utc"] = when
            passes.append(current)
            current = {}
    return {"satellite": record["name"], "location": {"latitude": latitude, "longitude": longitude},
            "window_utc": {"from": start.isoformat(), "to": end.isoformat()}, "passes": passes,
            "note": "Passes are above the geometric horizon; brightness and daylight are not evaluated."}


def satellite_visibility(satellite: str, latitude: float, longitude: float, hours: int = 24):
    """Find upcoming satellite passes that are sunlit while the observer's sky is dark enough to see them."""
    tle = tle_updates(satellite, 1)
    if "error" in tle or not tle["tle"]:
        return tle if "error" in tle else _error("No matching satellite TLE was found.")
    try:
        record = tle["tle"][0]
        ts = load.timescale()
        eph = load(os.path.join(ROOT, "de421.bsp"))
        craft = EarthSatellite(record["line1"], record["line2"], record["name"], ts)
        observer = wgs84.latlon(latitude, longitude)
        earth_observer = eph["earth"] + observer
        start = datetime.now(timezone.utc)
        end = start + timedelta(hours=max(1, min(int(hours), 72)))
        times, events = craft.find_events(observer, ts.from_datetime(start), ts.from_datetime(end), altitude_degrees=10.0)
    except Exception as exc:
        return _error(f"Satellite visibility calculation failed: {exc}")

    visible, current = [], {}
    for time, event in zip(times, events):
        when = time.utc_datetime().replace(tzinfo=timezone.utc).isoformat()
        if event == 0:
            current = {"rise_above_10deg_utc": when}
        elif event == 1 and current:
            alt, az, _ = (craft - observer).at(time).altaz()
            sun_alt, _, _ = earth_observer.at(time).observe(eph["sun"]).apparent().altaz()
            sunlit = bool(craft.at(time).is_sunlit(eph))
            current.update({"peak_utc": when, "max_altitude_degrees": round(float(alt.degrees), 1),
                            "peak_azimuth_degrees": round(float(az.degrees), 1),
                            "sun_altitude_degrees": round(float(sun_alt.degrees), 1), "satellite_sunlit": sunlit})
        elif event == 2 and current:
            current["set_below_10deg_utc"] = when
            if current.get("satellite_sunlit") and current.get("sun_altitude_degrees", 90) <= -6:
                current["visibility"] = "likely visible"
                visible.append(current)
            current = {}
    return {"satellite": record["name"], "location": {"latitude": latitude, "longitude": longitude},
            "visible_passes": visible, "criteria": "Satellite sunlit, observer's Sun below -6°, and pass above 10°.",
            "note": "Cloud, local light pollution, and satellite brightness can still affect visibility."}


# Aliases retained for code that used the initial prototype names.
ai_teacher_search = web_search
photographer_advice = photography_advice
_get_weather = weather_conditions
_get_moon_phase = moon_phase


# LangChain tool definitions consumed by ai_chat.py. Keep this list in one
# place so the assistant's advertised capabilities exactly match the backend.
STARGAZER_TOOLS = [
    tool(web_search), tool(nasa_articles), tool(latest_discoveries),
    tool(weather_conditions), tool(local_sky_time), tool(moon_phase), tool(night_planner),
    tool(photography_advice), tool(rise_set_times), tool(light_pollution_report),
    tool(meteor_showers), tool(equipment_advice), tool(object_information),
    tool(tle_updates), tool(satellite_status), tool(pass_predictions), tool(satellite_visibility),
]
