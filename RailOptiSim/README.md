# RailOptimusSim

Railway traffic simulator built with Dash + Plotly.
Single-station network, slot scheduling, accidents, rerouting, dashboards. Mostly demo / sandbox.
(because haha trains)

---

## Features

* Topology: 6 tracks × 4 sections, 7 platforms (configurable)
* ~10 fixed trains (Express / Passenger / Freight)
* Slot engine: headways, dwell times, platform capacity
* Accident handling: block track/section or platform, reroute + impact stats
* Dashboards:

  * Track usage timeline (main, with incident overlays)
  * Journey Gantt (start/stop markers, current time)
  * Network map (paths + incidents)
  * Platform × Train scatter
  * Now board + train overview
  * Run history (last 10 runs, wall-clock timestamps)
* KPIs: throughput, delays (total/avg), completion counts
* CSV exports: per-run log + bulk last-10
* UI: simple HSL theme in `assets/theme.css`, no emojis

---

## Stack

* Python 3.10+ (tested on 3.13)
* Dash, Plotly, dash-bootstrap-components
* pandas, numpy, networkx

`requirements.txt` lists the versions used.

---

## Repo layout

```
RailOptimusSim/
  app.py             # Dash app: layout, callbacks, presets
  simulation.py      # core sim + slot logic
  accident_manager.py  # accident handling
  visualization.py   # all plots
  data.py            # train/network generation
  utils.py           # helpers
  assets/
    theme.css        # HSL tokens
    plotly.css       # plotly tweaks
```

---

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open → [http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## Controls & Views

* Run / Pause / Step / Reset
* Speed control (slot rate)
* Accident trigger (type, location, duration)
* Presets: smooth run, accident, rush, recovery

Visuals:

* Track timeline
* Journey Gantt
* Network map
* Platform scatter
* Now board + train table
* Run history (with CSVs)

---

## Config

Defaults in `app.py`:

```python
NUM_TRACKS = 6
SECTIONS = 4
NUM_STATIONS = 1
PLATFORMS_PER_STATION = 7
HORIZON_MINUTES = 20
```

Trains generated deterministically (`data.py`).

---

## Dev notes

* Plots → `visualization.py`
* Accident manager reroutes dynamically
* Theme → `assets/theme.css` (`.dark` tokens included)
* Keeping UI text clean

---

## Troubleshooting

* No styles? → check `assets/` next to `app.py`
* macOS install issues:

  ```bash
  pip install --upgrade pip wheel
  ```
* App won’t start? → check Python version + traceback

---

## License

MIT (add LICENSE file if distributing).

---

```
      ___
   _|__|  |_____  
  |   __   __  o|  
  |__/  \______/  
   (o)      (o)   
```

---
