# 🔭 StarGazer - With AI integration 

Point your phone at the night sky and see it live in your browser — stars,
constellations, planets, and deep-sky objects, rendered in real time from
your actual location and orientation. Then just ask: *"why is it red?"*,
*"is tonight good for observing?"*, *"any satellites overhead?"* — the AI
already knows what you're looking at, because your backend tells it.

This is a **curiosity-driven personal project**, not a polished product.
It started from a curiosity : real phone sensor data → real astronomy math → a live sky
renderer → a context-aware AI assistant with real tools.

**Built with the help of AI coding assistants** to actually implement the
idea — the vision, direction, and every design decision were mine, worked
out interactively and iteratively (get the phone data flowing first, then
get the astronomy math right, then render it, then make the AI actually
know what you're pointing at). I'm sharing it because I learned a lot
building it and think it's a fun proof of concept, not because it's
finished software.

> ⚠️ **This is a prototype.** Expect rough edges. See [Known Limitations](#known-limitations--bugs) below before you dive in.

---

## What it actually does

- Streams your phone's orientation + GPS sensors to a small local server
- Computes real, live positions (azimuth/altitude) for ~900 stars, all 89
  constellations, all 110 Messier objects, and the visible planets, using
  actual ephemeris data (Skyfield + NASA JPL)
- Renders all of it as a live "viewfinder" in your browser, matching
  wherever your phone is physically pointed
- Auto-tracks whatever's nearest your crosshair as AI chat context — no
  clicking required, just point and ask
- Answers questions with a real LLM (Groq), backed by live tools:

| Category | What it can do |
|---|---|
| 🔬 **Research** | NASA articles, latest discoveries, general web search |
| 🌙 **Observe** | Local time, weather, Moon phase, night planning, rise/set times, light pollution |
| 📖 **Explore** | Object catalog lookups, meteor shower guide, photography & equipment advice |
| 🛰️ **Satellites** | Live status, TLE data, visible pass predictions — ISS, Tiangong, Hubble, Starlink |

---

## How it works (architecture)

```
Phone (Sensorstream IMU+GPS app)
   │  UDP packets: rotation vector + GPS
   ▼
phone.py            — parses packets, converts orientation → azimuth/altitude
   │
   ▼
sky_engine.py        — Skyfield: turns "where + when" into real positions
   │  (stars/constellations/Messier from catalog.py, planets from JPL ephemeris)
   ▼
server.py             — FastAPI: pushes live positions to the browser over WebSocket
   │
   ▼
index.html            — Canvas rendering: the live sky view + chat UI
   │  auto-tracks nearest object to crosshair → sends as AI context
   ▼
ai_chat.py             — LangGraph + Groq: context-aware conversation with memory
   │
   ▼
tools.py               — real tools the AI can call (research, weather, satellites, etc.)
```

Each piece is a separate, independently testable module — see the comments
at the top of each file for what it's responsible for.

---

## Setup

### 1. Clone and install

```bash
git clone <https://github.com/Irfan-gitt/Stargazing-Tool-Cosmos-AI->
cd stargazer
pip install -r requirements.txt
```

### 2. API keys

Copy `.env.example` to `.env` and fill in your own keys:

```dotenv
GROQ_API_KEY=your_groq_key_here,add another key if possible
SERPER_API_KEY=your_serper_key_here
```

- **Groq** (required, powers the AI chat) — free signup at
  [console.groq.com](https://console.groq.com), generous free tier.
  You can also supply multiple comma-separated keys
  (`GROQ_API_KEYS=key1,key2,key3`) for automatic rotation if you hit rate limits.
- **Serper** (optional, powers web search) — free signup at
  [serper.dev](https://serper.dev), ~2,500 free searches.

Your `.env` file is **never committed** (see `.gitignore`) — it's your own
local secret, not something that ships with the repo.

### 3. Connect your phone

You need an app that streams raw sensor data (rotation vector + GPS) to
your laptop over your local network. This project was built and tested
using **Sensorstream IMU+GPS**:

📲 **[Download: Sensorstream IMU+GPS](https://imu-gps-stream.en.aptoide.com/app)**

**Setup steps:**

1. **Get your laptop's local IP address.** Open Command Prompt (Windows) and run:
   ```
   ipconfig
   ```
   Note the **IPv4 Address** under your active network adapter (e.g. `192.168.1.42`).

2. **Connect your phone to the same WiFi network** as your laptop. This is
   required — the phone talks directly to your laptop over the local
   network, not the internet.

3. **Install and open the app**, go to the **"Toggle Sensors"** tab:
   - Enable **"Rotation Vect."**
   - Enable **"GPS Position"**
   - Check **"Include User-Checked Sensor Data in Stream"** at the bottom

4. Go to the **"Preferences"** tab:
   - Paste your laptop's IPv4 address into **"Edit Target IP Address"**
   - Set **"Edit Target Port"** to `5555`
   - Select **"UDP Stream"** (not "UDP & SD-Card Stream")
   - Check **"Run in Background"**

5. Toggle **"Switch Stream"** to **ON** — the app should show "Stream established".

### 4. Run it

```bash
python server.py
```

Once the phone is streaming, open **http://localhost:8000** in your browser.

---

## Known limitations & bugs

Being upfront, since this is genuinely a prototype:

- **Phone sensor accuracy varies a lot by device.** Cheaper phones and
  certain Android skins (MIUI in particular) have noisier rotation
  sensors and aggressively kill background streaming apps unless you
  manually disable battery optimization / enable autostart for the
  sensor app.
- **Requires the same local network.** Some routers isolate WiFi clients
  from each other (AP/client isolation), which blocks the phone→laptop
  connection entirely. A mobile hotspot from the phone itself is a
  reliable fallback.
- **Projection distortion near zenith.** The sky view uses a proper
  gnomonic projection, but pointing very close to straight overhead can
  still look slightly stretched.
- **Single-session by design.** This is built for one person on their
  own machine — there's no multi-user isolation. Don't expose this
  publicly as-is; see the architecture notes if you want to extend it.
- **AI tool results aren't guaranteed accurate.** Satellite passes,
  weather, and research results depend on third-party APIs and
  approximate calculations — treat them as estimates, not precision
  data.
- **Not yet extensively field-tested.** Most development happened
  through iterative debugging rather than long real-world observing
  sessions — accuracy under real night-sky conditions is still being
  validated.

If you run into something broken, that's expected at this stage — feel
free to open an issue or just fix it and send a PR.

---

## Data sources & credits

- Star / constellation / Messier catalog: [d3-celestial](https://github.com/ofrohn/d3-celestial) (open source, BSD license)
- Planetary positions: [Skyfield](https://rhodesmill.org/skyfield/) + NASA JPL DE421 ephemeris
- Satellite tracking: [CelesTrak](https://celestrak.org/) TLE data
- Weather: [Open-Meteo](https://open-meteo.com/) (free, no API key)
- Web search: [Serper.dev](https://serper.dev)
- LLM: [Groq](https://groq.com/)
