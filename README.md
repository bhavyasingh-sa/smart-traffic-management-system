# Smart Traffic Management and Decision Support System

A movement-aware adaptive traffic signal controller for a real intersection
(Atlanta Intersection 84), combining a trained ML classifier, an IR
retrieval system over historical traffic records, a transparent weighted
decision formula, and a Gemini-based grounded explanation layer, all
presented through a live Streamlit dashboard.

## Pipeline

```
Real historical dataset (data/raw/train.csv)
  -> real arrival profiles (simulation/build_profiles.py)
  -> real turning proportions (analysis/movement_audit.py)
  -> dynamic 12-movement vehicle simulation (simulation/adaptive_simulator.py)
  -> live queues and waiting times
  -> ML predicts approach-level congestion severity (NB/SB/EB/WB)
  -> IR retrieves similar historical movement cases (12 movements)
  -> weighted formula calculates movement priorities (simulation/controller_core.py)
  -> movement priorities -> phase priorities
  -> controller dynamically holds or switches phases (bounded by MIN/MAX green)
  -> starvation safeguard guarantees fairness under extreme conditions
  -> Gemini (RAG) explains each approach's decision, grounded in ML + IR evidence
  -> all of the above rendered live in the Streamlit dashboard (app.py)
```

## Important, deliberately-documented design notes

- `EntryHeading`/`ExitHeading` in the raw dataset describe the
  **direction of travel**, not the physical side a vehicle arrives
  from (verified against the real data, not assumed). A northbound
  (NB) vehicle physically queues on the *south* leg of the
  intersection.
- ML remains **approach-level only** (NB/SB/EB/WB, 4 predictions) —
  never claim 12 independent ML predictions. IR is genuinely
  movement-level (12 real, retrieved historical cases).
- The starvation safeguard is an **additional fairness constraint**,
  not a replacement for the 45/20/15/10/10 weighted formula, which is
  unchanged.
- The signal decision itself (`controller_core.should_switch_phase`) is
  a transparent, auditable weighted formula, not a black-box model — the
  ML classifier is one input into it, not the thing driving the light
  directly. See `Smart_Traffic_AI_Project_Explanation_Guide.pdf` for the
  full design rationale.

## Local setup

```bash
# 1. Create a fresh virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux

# 2. Install requirements
pip install -r requirements.txt

# 3. Add your Gemini API key (get one at https://aistudio.google.com/apikey)
echo "GEMINI_API_KEY=your_key_here" > .env

# 4. Run the dashboard
streamlit run app.py
```

## Running the tests

```bash
pip install pytest
pytest tests/test_adaptive_simulator.py tests/test_controller_core.py tests/test_starvation_safeguard.py -v
```

## Checking the trained model's scores

```bash
python3 -m ml.evaluate
```

Prints accuracy, balanced accuracy, precision, recall, and F1 (macro +
weighted) for the global model, the global model evaluated on
Intersection 84, and the local Intersection-84-only model (the one
actually selected and used) — instantly, without retraining.

## Regenerating build artifacts (only needed if you change the pipeline)

`data/raw/` (the original ~426MB multi-city Kaggle-style dataset) is not
committed to this repository — it's never read by the live app, only by
these offline build scripts. If you need to regenerate `models/` or
`data/processed/`, download the dataset and place `train.csv` at
`data/raw/train.csv`, then run:

```bash
PYTHONPATH=. python3 simulation/build_profiles.py   # data/processed/traffic_profiles.csv
PYTHONPATH=. python3 analysis/movement_audit.py      # data/processed/movement_matrix.csv, turning_proportions.csv
PYTHONPATH=. python3 -m ml.train                     # models/ml/congestion_classifier.pkl
PYTHONPATH=. python3 -m ir.build_index               # models/ir/traffic_ir.pkl
```

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and
   click **New app**.
3. Select this repo, branch `main`, and main file path `app.py`.
4. Before deploying, open **Advanced settings -> Secrets** and paste:
   ```
   GEMINI_API_KEY = "your_key_here"
   ```
   (Never commit `.env` or paste the key anywhere else — Streamlit Cloud
   exposes this secret to the app the same way a local `.env` does.)
5. Click **Deploy**. First boot can take a minute or two while it
   installs `requirements.txt` and loads the ~75MB of model files.

## Project structure

```
smart_traffic_movement_v5/
├── app.py                                 # Streamlit entry point
├── ui/                                    # Dashboard rendering (intersection view, panels, toolbar, styles)
├── analysis/
│   ├── movement_audit.py                  # Real movement matrix + turning proportions
│   ├── starvation_sensitivity_analysis.py # Starvation threshold sensitivity sweep
│   └── aggregate_sensitivity_results.py   # Sensitivity report generator
├── ml/
│   ├── train.py                           # Trains + evaluates the RandomForest classifier
│   ├── evaluate.py                        # Prints saved accuracy/F1 scores, no retraining
│   ├── predict.py                         # Interactive single-prediction CLI
│   └── features.py                        # Feature prep, label engineering
├── ir/                                    # TF-IDF + cosine similarity retrieval over historical cases
├── rag/                                   # Gemini-based grounded explanation layer
├── simulation/
│   ├── movement_definitions.py            # 12 movement IDs, 4-phase signal plan
│   ├── controller_core.py                 # Stateless controller intelligence (priority formula, decisions)
│   ├── adaptive_simulator.py              # Tick-by-tick stateful simulation engine
│   └── build_profiles.py                  # Real arrival-rate profile generator
├── tests/                                 # pytest suite (35 tests)
├── data/
│   ├── raw/                               # Original dataset (gitignored, not required to run the app)
│   └── processed/                         # Generated: movement_matrix.csv, turning_proportions.csv, traffic_profiles.csv
├── models/
│   ├── ml/congestion_classifier.pkl
│   └── ir/traffic_ir.pkl
├── requirements.txt
└── README.md
```
