# Smart Traffic Management and Decision Support System

A live traffic signal controller for a real intersection that decides, tick
by tick, whether to hold or switch each light — using a trained ML model,
a retrieval system over real historical traffic records, and a transparent
decision formula, with an AI layer that explains every decision in plain
English.

## What this is

Most "smart traffic light" projects either hardcode a fixed timer, or
wrap a black-box model around the whole decision with no way to explain
why it did what it did. This project instead treats it as a **decision
support system**: real signals (live queue length, live wait time) are
combined with real predictions (a congestion classifier trained on
historical data) and real precedent (similar past traffic conditions,
retrieved from historical records) through a weighted formula that stays
fully inspectable — you can always ask "why did the light just change?"
and get a real, traceable answer.

The intersection modeled is a real one — Atlanta Intersection 84 — using
an actual historical traffic dataset, not synthetic numbers.

## Key features

- **12 real movements, not 4 directions** — each of North/South/East/West
  is broken into Left/Straight/Right turns, matching how an intersection
  actually behaves, with a conflict-safe 4-phase signal plan.
- **A trained ML classifier** predicts congestion severity (Low / Moderate
  / High / Severe) per approach, evaluated with a proper time-based
  train/test split (not a random shuffle) to avoid data leakage.
- **An IR (information retrieval) system** searches historical traffic
  records with TF-IDF + cosine similarity to surface real precedent for
  the current conditions.
- **A transparent priority formula** (45% live queue, 20% live wait, 15%
  ML prediction, 10% IR evidence, 10% starvation prevention) decides
  every phase switch — auditable at any point in time, not a black box.
- **A hard fairness safeguard** guarantees no direction waits indefinitely,
  even under extreme, adversarial traffic conditions.
- **Gemini-powered explanations** describe each approach's current
  situation in plain English, grounded only in the real ML/IR evidence
  above (never invented).
- **A live Streamlit dashboard** shows the intersection, queues, signal
  state, and AI explanations updating in real time as the simulation runs.

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

The ML classifier is trained two ways — once on data from every city in
the dataset ("global"), once on only this intersection's own history
("local") — and whichever scores better on a held-out, unseen month is
the one actually used. Run `python3 -m ml.evaluate` to reproduce this
table instantly from the already-trained model (no retraining needed):

| Model | Accuracy | Balanced Accuracy | Precision | Recall | Macro F1 |
|---|---|---|---|---|---|
| Global (all cities) | 44.7% | 40.1% | 0.353 | 0.401 | 0.335 |
| Global on Intersection 84 | 55.0% | 35.3% | 0.459 | 0.353 | 0.314 |
| **Local Intersection 84 (selected)** | **63.3%** | **51.5%** | **0.513** | **0.515** | **0.508** |

The local, intersection-specific model wins clearly — its traffic patterns
are distinct enough that a specialist model outperforms a generalist one.

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

Open the URL Streamlit prints (typically `http://localhost:8501`), hit
**Play**, and watch the signal adapt to live traffic in real time.

### Running the tests

```bash
pip install pytest
pytest tests/ -v
```

35 tests covering the controller's decision logic, the simulation engine,
and the starvation safeguard.

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

## Design decisions worth knowing

- The signal decision is a **transparent weighted formula**, not the ML
  model directly — the classifier is one input (15% weight) alongside
  live queue/wait data and IR evidence, so every decision stays
  explainable. See the note on `should_switch_phase()` in
  `simulation/controller_core.py`.
- Green time is **not fixed** — it flexes between a 20-tick floor and a
  60-tick ceiling based on real-time conditions, which is what makes this
  genuinely adaptive rather than a pretimed signal.
- `EntryHeading`/`ExitHeading` in the raw dataset describe the
  **direction of travel**, not the physical side a vehicle arrives from —
  a northbound (NB) vehicle physically queues on the intersection's
  *south* leg. This is verified against the real data, not assumed, and
  has been a real source of bugs in earlier iterations of this project.
- The ML model predicts congestion at the **approach level** (NB/SB/EB/WB,
  4 predictions); IR operates at the **movement level** (12 real,
  independently retrieved cases) — this is intentional, not an
  inconsistency, since the historical data only reliably supports
  per-approach ML labels.

## Tech stack

Python, scikit-learn (RandomForestClassifier), pandas/numpy, a from-scratch
TF-IDF + cosine similarity IR implementation, Google Gemini (`google-genai`),
and Streamlit.

## Deployment

This app is deployable on [Streamlit Community Cloud](https://share.streamlit.io)
for free: point it at this repo with `app.py` as the entry file, add
`GEMINI_API_KEY` under the app's Secrets settings, and deploy.
