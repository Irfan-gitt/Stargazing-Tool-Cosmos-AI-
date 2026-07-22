"""
StarGazer Web - Server
--------------------------------
This file is deliberately thin: it just wires together the other
modules and defines the web routes. The actual logic lives in:

    phone.py       - reading the phone's orientation over UDP
    catalog.py     - downloading/parsing the star/constellation/DSO data
    sky_engine.py  - turning catalog data into live azimuth/altitude
    ai_chat.py     - conversation memory + the AI chat itself

SETUP (one-time):
    pip install fastapi "uvicorn[standard]" skyfield numpy --break-system-packages

HOW TO RUN:
    1. Make sure index.html is in the SAME FOLDER as this file.
    2. python server.py
    3. Open http://localhost:8000 in your browser.
    4. On your phone, turn on "Switch Stream" in Sensorstream IMU+GPS.
"""

import os
import time
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

import phone
import ai_chat
from sky_engine import SkyEngine

CATALOG_REFRESH_SECONDS = 20  # how often to recompute all star/planet alt-az
# Orbital positions are recalculated independently from the static sky catalog.
# Half-second updates make ISS and satellites visibly move across the view.
SATELLITE_REFRESH_SECONDS = 0.5

sky = SkyEngine()  # loads catalogs + ephemeris once at startup

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat")
async def chat(request: Request):
    """
    Receives: {"message": "...", "context": {"name", "type", "mag"} or null}
    Returns:  {"reply": "..."}
    All the actual logic lives in ai_chat.handle_chat_message().
    """
    body = await request.json()
    reply = ai_chat.handle_chat_message(
        body.get("message", ""), body.get("context"))
    return JSONResponse({"reply": reply})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_sent = None
    last_catalog_time = 0.0
    last_satellite_time = 0.0
    try:
        while True:
            state = phone.get_current_state()
            az, alt, lat, lon = state["az"], state["alt"], state["lat"], state["lon"]

            if time.time() - last_catalog_time > CATALOG_REFRESH_SECONDS:
                try:
                    snapshot = sky.compute_catalog_snapshot(lat, lon)
                    await websocket.send_json(snapshot)
                except Exception as exc:
                    print(f"Catalog update unavailable: {exc}")
                last_catalog_time = time.time()

            if time.time() - last_satellite_time > SATELLITE_REFRESH_SECONDS:
                # Satellite data is optional: a transient TLE/provider issue
                # must never stop the main star-map WebSocket.
                try:
                    await websocket.send_json(sky.compute_satellite_snapshot(lat, lon))
                except Exception as exc:
                    print(f"Satellite position update unavailable: {exc}")
                last_satellite_time = time.time()

            if az is not None and (az, alt) != last_sent:
                await websocket.send_json({"type": "pointing", "az": az, "alt": alt, "lat": lat, "lon": lon})
                last_sent = (az, alt)

            await asyncio.sleep(0.15)

    except WebSocketDisconnect:
        pass


def main():
    phone.start_listener()
    print("Turn on 'Switch Stream' on your phone, then open http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
