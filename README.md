# ISC-2025 Scavenger Satellite AI

Modular AI system for autonomous satellite navigation, debris detection, resource management, and orbit prediction for our satellite for the ISC 2025 challenge.

## Quick Start

```bash
pip install numpy matplotlib scikit-learn pandas pillow
python run_demo.py
```

## Modules

| Module | Purpose |
|--------|---------|
| `navigation/nav.py` | Path planning with collision avoidance |
| `detection/detect.py` | Debris decay classification |
| `manipulation/manipulate.py` | Q-learning debris triage agent |
| `predictive/predictive.py` | Orbit propagation and collision risk |
| `damage_assessment/damage_scan.py` | Satellite damage assessment |

Run individually:

```bash
python ai_modules/navigation/nav.py
python ai_modules/detection/detect.py
python ai_modules/manipulation/manipulate.py
python ai_modules/predictive/predictive.py
python ai_modules/damage_assessment/damage_scan.py
```

Or run the full integrated demo:

```bash
python tests/test_all_integrated.py
```

## Output Files

- `~/Downloads/navigation_3rd_person.gif` – animated trajectory
- `~/Downloads/predictive_orbit_ai_analysis.png` – mission analysis visualisation
- `collision_model.pkl` – trained collision predictor

## Dependencies

numpy
matplotlib
scikit-learn
pandas
pillow