# Car Evolution with Genetic Algorithms

A 2D top-down simulation where a fleet of cars learns to drive a polygon track using **feed-forward neural networks** and a **genetic algorithm**. Each car’s behaviour is encoded as a flat weight vector (DNA); fitness comes from checkpoint progress and lap completion.

**Code, comments, and docstrings are in English.** This README is the full user and architecture guide.

---

## Table of contents

1. [Requirements](#requirements)  
2. [Install and run](#install-and-run)  
3. [Controls](#controls)  
4. [What you see on screen](#what-you-see-on-screen)  
5. [How the simulation works](#how-the-simulation-works)  
6. [Fitness and rewards](#fitness-and-rewards)  
7. [Neural network and car physics](#neural-network-and-car-physics)  
8. [Genetic algorithm defaults](#genetic-algorithm-defaults)  
9. [Sequential parameter runs](#sequential-parameter-runs)  
10. [When a generation ends](#when-a-generation-ends)  
11. [Directory structure](#directory-structure)  
12. [Paths and CSV logging](#paths-and-csv-logging)  
13. [Configuration](#configuration)  
14. [Custom tracks](#custom-tracks)  
15. [Documentation in code](#documentation-in-code)  
16. [Troubleshooting](#troubleshooting)  
17. [License](#license)

---

## Requirements

- **Python 3.10+** recommended  
- **Dependencies:** `numpy`, `pygame` (pinned in `requirements.txt`)

---

## Install and run

1. Open a terminal in the **project root** (folder that contains `main.py` and the `car_evolution/` package).

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   ```

   **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`  
   **Linux / macOS:** `source venv/bin/activate`

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run from the **same project root** so Python resolves `car_evolution` as a package:

   ```bash
   python main.py
   ```

   or:

   ```bash
   python -m car_evolution
   ```

The window shows the **track on the left** and the **GA dashboard on the right** (widths set in `config/settings.py`).

---

## Controls

| Input | Action |
|--------|--------|
| **R** | Restart the **current** parameter run: same **RNG seed**, new population, reset run stats and history (hyperparameters stay fixed for that run). |
| **N** | New random seed (`1` … `999_999`), then same as **R** for the current run. |
| **Esc** | Quit (same as closing the window) |

GA rates and methods are **fixed for each parameter run** until convergence; they are not changed with hotkeys. CSV files already written **remain on disk**; only RAM state is cleared on **R** / **N**.

---

## What you see on screen

- **Cars:** Green while driving, red when dead (collision or stall), cyan when the lap is finished.  
- **Progress marker:** Purple cross / rings highlight the car that currently leads on “best progress” (most gates, then distance to next gate).  
- **Dashboard:** Generation counters, driving/finished counts, time left for the current generation, best gates, leader line, **HISTORY** of generations where at least one car finished the lap, **PARAMETERS**, and **CONTROLS**. The footer (parameters + controls) stays anchored to the bottom so it is not clipped when history grows.

---

## How the simulation works

1. **Track** – Outer and inner polygons define the drivable ring. Waypoints define **checkpoint gates** (perpendicular segments). Collision and sensors use the same segments as physics (`track/layout.py`, `track/geometry.py`).

2. **Each frame** – Every car reads five ray sensors, runs its neural net, updates speed/heading/position, may clear the next gate, updates fitness, and can die (wall proximity, too long without a gate, or wrong-way rule on the first straight).

3. **End of generation** – If all cars are inactive **or** the frame budget is reached, one CSV row is appended, then either the run **converges** (fitness plateau or generation cap) and the app advances to the next fixed-parameter preset, or `Population.evolve()` builds the next generation with the **same** hyperparameters.

4. **Rendering** – The grassy/asphalt/curb backdrop is **baked once** (`rendering/track_background.py`) and blitted each frame; it does not affect physics.

---

## Fitness and rewards

| Situation | Fitness (conceptually) |
|-----------|-------------------------|
| While racing | `checkpoints_cleared × 1000` + small tie-breaker in `[0, 1)` from distance to the **next** gate (closer is better) |
| Lap complete | `num_gates × 1000 + 500_000` |
| Wrong-way on lap 0 (heuristic) | Large penalty if triggered (`car.py`) |

`best_fitness` per car is the maximum fitness seen in its lifetime; the GA sorts by that when evolving.

Constants live on `Car`: `SEGMENT_BONUS`, `TRACK_COMPLETE_BONUS`, `MAX_FRAMES_WITHOUT_CHECKPOINT` (stall timeout in frames).

---

## Neural network and car physics

- **Topology:** `[5, 5, 2]` — five inputs, one hidden layer of five, two outputs (`core/neural_network.py`).  
- **Activation:** `tanh` after each affine layer.  
- **Inputs:** Normalized ray lengths in `[0, 1]` (max ray length 200 px, five angles around the heading).  
- **Outputs:** Interpreted as throttle/brake and steering; integrated into speed (capped), heading, and position (`car.py`).  
- **Collision:** Car “radius” ~10 px to segment distance for death check.

---

## Genetic algorithm defaults

Defined in `Population.__init__` (`core/population.py`):

| Setting | Default | Notes |
|---------|---------|--------|
| `mutation_rate` | `0.05` | Per-gene Bernoulli then Gaussian noise |
| `crossover_rate` | `0.80` | Else clone parent1 DNA |
| `elitism` | `2` | Top genomes copied unchanged |
| `selection_method` | `Tournament` | 3-way sample; alternative `Roulette` |
| `crossover_method` | `Uniform` | Alternative `One-Point` |

---

## Sequential parameter runs

`evolution/run_params.py` defines an ordered list of `EvolutionRunParams` presets (`default_run_presets()`). The game runs them **one after another**:

- Within a preset, **mutation rate**, **crossover rate**, **selection**, and **crossover type** stay constant.
- A run ends when **either** the session peak fitness does not improve for `convergence_plateau_generations` **or** the generation index reaches `max_generations_per_run` (both in `SimulationConfig` in `config/settings.py`).
- Then a **new** CSV file is opened, the RNG is re-seeded with the same base seed (fair comparison across presets), and a fresh population starts with the next preset.

To use your own list, pass `run_presets=` into `EvolutionGame(...)`.

---

## When a generation ends

Either:

- **All cars** are dead or have **finished** the lap, or  
- **`max_frames_per_generation`** frames have elapsed (default `600` at 60 FPS ≈ 10 s).

Then: CSV append → if the run has converged, switch preset (new CSV + new population) **else** `evolve()` → frame counter reset.

---

## Directory structure

```text
project root/
  main.py                 # Entry: car_evolution.run()
  requirements.txt
  README.md
  logs/                   # Created automatically; CSV evolution logs
  car_evolution/
    __init__.py           # run, EvolutionGame
    __main__.py           # python -m car_evolution
    config/               # DisplayConfig, SimulationConfig, Colors, …
    core/                 # Car, NeuralNetwork, Population, RNG
    track/                # geometry.py, layout.py (RaceTrack)
    rendering/            # track_background.py, ui.py (dashboard)
    evolution/            # logger.py, run_params.py
    io/                   # paths.py (PROJECT_ROOT, LOGS_DIR, …)
    app/                  # game.py (EvolutionGame loop)
```

---

## Paths and CSV logging

- **`PROJECT_ROOT`** – Parent folder of the `car_evolution` package (same level as `main.py`). Resolved in `car_evolution/io/paths.py` from `__file__`, so it is stable even if you change the shell’s current working directory (as long as the installed/checked-out tree is unchanged).

- **`LOGS_DIR`** – `PROJECT_ROOT / "logs"`. Created on demand via `ensure_logs_dir()`.

- **One CSV per parameter run** – `evolution_log_YYYYMMDD_HHMMSS_runNN.csv` from `evolution_run_log_path(session_timestamp, run_index)` (shared timestamp for all runs in one window session). `evolution_log_path()` remains available for single-file logging if you build a custom entry point.

- **One row per generation** with columns:

  | Column | Meaning |
  |--------|---------|
  | `Generation` | Index when the row was written (generation that just finished) |
  | `Mutation_Rate` | Current mutation rate |
  | `Crossover_Rate` | Current crossover rate |
  | `Selection_Method` | `Tournament` or `Roulette` |
  | `Max_Fitness_Session` | Best fitness seen since this run started |
  | `Finished_Cars` | How many cars completed the lap that generation |
  | `Leader_Gates` | Gate progress of the fittest car that generation |

---

## Configuration

Edit **`car_evolution/config/settings.py`**:

- **`DisplayConfig`** – `track_width`, `ui_width`, `height`, `fps`  
- **`SimulationConfig`** – `default_seed`, `population_size`, `max_frames_per_generation`  
- **`DebugConfig`** – Reserved flags for future debug tooling  
- **`Colors`** – UI, track theme, car accents  

The game imports the module-level singletons `DISPLAY`, `SIMULATION`, and `DEBUG`.

---

## Custom tracks

1. Build a **`RaceTrack`** instance (outer/inner tuples, waypoints, `start_position`, `start_angle`, optional `checkpoint_half_len`).  
2. Pass it into **`EvolutionGame(track=...)`** in `app/game.py` (or your own launcher).  

Keep polygons simple and non–self-intersecting; waypoints should follow the intended lap order.

---

## Documentation in code

Public modules and most functions include **Google-style docstrings** (Args / Returns where helpful). Browse from:

- `car_evolution/__init__.py` – package map  
- `car_evolution/io/paths.py` – path resolution rules  

No separate Sphinx site is required for development; IDEs and `help()` can consume the docstrings directly.

---

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| `ModuleNotFoundError: car_evolution` | Run from project root or set `PYTHONPATH` to the root. |
| Blank / missing fonts | Dashboard uses **Courier New**; install it or change fonts in `app/game.py`. |
| Logs not appearing | Check write permissions on `PROJECT_ROOT/logs`. |
| Window too large / small | Adjust `DisplayConfig` in `config/settings.py`. |

---

## License

Educational project: clone, study, adapt, and extend for your own experiments with neural networks and evolutionary algorithms.


## Authors
- [Ankier José](https://github.com/AnkierJ)
- [Célio Felipe](https://github.com/DIGAOZX)
- [Gabriel Gonzaga](https://github.com/GabrielFGonzaga)
- [Pedro Henrique](https://github.com/pedrohnq)