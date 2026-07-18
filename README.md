# Smart Traffic Management and Decision Support System

A live traffic signal controller for a real intersection that decides,
tick by tick, whether to hold or switch each light — combining a trained
ML model, a retrieval system over historical traffic records, and a
transparent decision formula, with an AI layer that explains each
decision in plain English.

**Live demo:** [smart-traffic-management-system-cj3juajxxjz6fnbmttefyd.streamlit.app](https://smart-traffic-management-system-cj3juajxxjz6fnbmttefyd.streamlit.app/)

## Overview

The intersection modeled is a real one, Atlanta Intersection 84, using
an actual historical traffic dataset. Rather than one model choosing
signal timings directly, the decision comes from a weighted formula that
combines four inputs: live queue length, live wait time, a trained
congestion classifier's prediction, and similar historical cases
retrieved by search. Every switch decision can be traced back to the
numbers that produced it.

## Key features

- **12 real movements, not 4 directions** — each of North/South/East/West
  is broken into Left/Straight/Right turns, with a conflict-safe 4-phase
  signal plan.
- **A trained ML classifier** predicts congestion severity (Low / Moderate
  / High / Severe) per approach, evaluated with a time-based train/test
  split to avoid data leakage.
- **An IR (information retrieval) system** searches historical traffic
  records with TF-IDF + cosine similarity to surface precedent for
  current conditions.
- **A weighted priority formula** (45% live queue, 20% live wait, 15%
  ML prediction, 10% IR evidence, 10% starvation prevention) decides
  every phase switch.
- **A hard fairness safeguard** guarantees no direction waits indefinitely
  under sustained heavy competing demand.
- **Gemini-powered explanations** describe each approach's current
  situation in plain English, grounded in the ML/IR evidence.
- **A live Streamlit dashboard** shows the intersection, queues, signal
  state, and AI explanations updating in real time.

## How it works

```
Real historical dataset (data/raw/train.csv)
  -> real arrival profiles + real turning proportions
  -> dynamic 12-movement vehicle simulation
  -> live queues and waiting times, tick by tick
  -> ML predicts approach-level congestion severity (NB/SB/EB/WB)
  -> IR retrieves similar historical movement cases (12 movements)
  -> weighted formula scores every movement's priority
  -> controller holds or switches the signal, bounded by min/max green time
  -> starvation safeguard force-switches if a direction has waited too long
  -> Gemini explains each approach's decision, grounded in the ML + IR evidence
  -> all of the above rendered live in the Streamlit dashboard
```

## Model performance

The classifier is trained two ways — on data from every city in the
dataset ("global"), and on only this intersection's own history
("local") — and whichever scores better on a held-out, unseen month is
the one used. Run `python3 -m ml.evaluate` to reproduce this table from
the already-trained model:

| Model | Accuracy | Balanced Accuracy | Precision | Recall | Macro F1 |
|---|---|---|---|---|---|
| Global (all cities) | 44.7% | 40.1% | 0.353 | 0.401 | 0.335 |
| Global on Intersection 84 | 55.0% | 35.3% | 0.459 | 0.353 | 0.314 |
| **Local Intersection 84 (selected)** | **63.3%** | **51.5%** | **0.513** | **0.515** | **0.508** |

The local, intersection-specific model scores higher across every
metric, so it's the one the dashboard actually uses.

## Getting started

**Prerequisites:** Python 3.10+, a [Gemini API key](https://aistudio.google.com/apikey) (free tier).

```bash
# 1. Clone and enter the project
git clone https://github.com/bhavyasingh-sa/smart-traffic-management-system.git
cd smart-traffic-management-system

# 2. Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt

# 3. Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 4. Run the dashboard
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`) and
hit **Play**.

### Running the tests

```bash
pip install pytest
pytest tests/ -v
```

35 tests covering the controller's decision logic, the simulation
engine, and the starvation safeguard.

## Project structure

```
├── app.py                # Streamlit entry point
├── ui/                    # Dashboard rendering (intersection view, panels, toolbar)
├── simulation/
│   ├── movement_definitions.py   # 12 movement IDs, 4-phase signal plan
│   ├── controller_core.py        # The decision formula and switching logic
│   ├── adaptive_simulator.py     # Tick-by-tick simulation engine
│   └── build_profiles.py         # Real arrival-rate profiles from historical data
├── ml/                    # Trains, evaluates, and serves the congestion classifier
├── ir/                    # TF-IDF + cosine similarity retrieval over historical records
├── rag/                   # Gemini-based grounded explanation layer
├── analysis/              # Movement audit, starvation-threshold sensitivity analysis
├── tests/                 # pytest suite
├── data/
│   ├── raw/               # Original dataset (not committed - see below)
│   └── processed/         # Generated arrival-rate and turning-proportion tables
└── models/                # Trained ML classifier + IR index
```

`data/raw/` (the ~426MB source dataset) isn't committed — the live app
never reads it directly, only the offline scripts that build `models/`
and `data/processed/` do. To regenerate those from scratch, place the
dataset at `data/raw/train.csv` and run `simulation/build_profiles.py`,
`analysis/movement_audit.py`, `ml/train.py`, and `ir/build_index.py`.

## Notes on the design

- The signal decision comes from the weighted formula above, not the ML
  model directly — the classifier is one of its five inputs (15%
  weight). See `should_switch_phase()` in `simulation/controller_core.py`.
- Green time varies between a 20-tick floor and a 60-tick ceiling based
  on real-time conditions. Red isn't a separately tracked state — a
  movement is red whenever a different phase is active, so its duration
  follows from however long the other phases take. The 3-tick yellow
  clearance interval is the only fixed timing value.
- `EntryHeading`/`ExitHeading` in the dataset describe direction of
  travel, not the physical side a vehicle arrives from — a northbound
  (NB) vehicle physically queues on the intersection's south leg.
- ML predicts congestion at the approach level (NB/SB/EB/WB, 4
  predictions); IR retrieves evidence at the movement level (12 cases).
  The difference in granularity matches what the underlying historical
  data reliably supports at each level.

## Tech stack

Python, scikit-learn (RandomForestClassifier), pandas/numpy, a from-scratch
TF-IDF + cosine similarity IR implementation, Google Gemini (`google-genai`),
and Streamlit.

## Deployment

Running live on [Streamlit Community Cloud](https://smart-traffic-management-system-cj3juajxxjz6fnbmttefyd.streamlit.app/),
deployed straight from this repository with `app.py` as the entry point.
The Gemini API key is supplied through Streamlit Cloud's Secrets manager
rather than committed to the repo, the same way `python-dotenv` supplies
it locally from `.env`.
