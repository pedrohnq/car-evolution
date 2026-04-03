# Car Evolution with Genetic Algorithms

A 2D simulation where a fleet of cars learns to drive around a track using **Artificial Neural Networks** and **Genetic Algorithms**. Each car's performance on the track defines its *fitness*; the best individuals are selected, crossed over, and mutated to form the next generation.

---

## Requirements

- **Python 3.10+** (recommended)
- **Dependencies:** `numpy`, `pygame` (listed in `requirements.txt`)

## How to Install and Run

1. Open your terminal in the root folder of the project (where `main.py` is located).

2. Create and activate a virtual environment (recommended):

   ```bash
   # Create the environment
   python -m venv venv

   # Activation on Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   
   # Activation on Linux/macOS
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the project. The working directory must be the project root for Python to locate the package correctly. You can use either of the options below:

   ```bash
   python main.py
   ```
   or
   ```bash
   python -m car_evolution
   ```

Upon startup, the interface will display the race track on the left and the AI control panel on the right.

---

## Simulation Controls

You can adjust the evolution hyperparameters in real-time using your keyboard:

| Key | Action |
|:---:|---|
| **Up / Down Arrow** | Increase or decrease the **mutation** rate. |
| **Left / Right Arrow** | Adjust the **crossover** rate. |
| **S** | Toggle the **selection** method (Tournament / Roulette Wheel). |
| **C** | Toggle the **crossover** type (Uniform / Single-Point). |
| **R** | Restart the simulation with the **same seed**. |
| **N** | Restart the simulation with a **new random seed**. |
| **Esc / Close Window** | Exit the program safely and save the session logs. |

---

## Architecture Overview

### 1. Track and Physics
- **Geometry:** The track is defined by polygons (inner and outer boundaries) in `track/layout.py`. 
- **Collisions and Sensors:** The cars use raycasting to calculate the distance to the track boundaries. Geometric intersection logic and corridor validation are isolated in `track/geometry.py`.
- **Rendering:** The track visuals (asphalt, grass, curbs) are generated once as a background (`rendering/track_background.py`) to optimize physics processing.

### 2. Control and Neural Network
- Each car is controlled by a *feed-forward* network with *tanh* activation (`core/neural_network.py`).
- **Inputs:** Distances captured by five radial sensors at the front of the vehicle.
- **Outputs:** Acceleration/braking and steering control.
- **Genome:** The weights and biases of each network are flattened into a single vector (DNA), which is manipulated by the genetic algorithm.

### 3. Genetic Algorithm and Fitness
- **Cycle:** Managed in `core/population.py`. Cars earn points by crossing checkpoints (gates) along the track (`core/car.py`).
- **Evolution:** At the end of each generation (when all cars crash or the time limit expires), the best genomes are selected. Elitism ensures the top performers of the previous generation survive intact. The rest of the population is generated via crossover and mutation.
- **Auto-adjustment:** The `evolution/schedule.py` module allows hyperparameters to be adjusted automatically at specific generations.

---

## Directory Structure

The core code is encapsulated within the `car_evolution` package to keep the project root clean:

```text
car_evolution/
 ├── config/       # Global settings (FPS, resolutions, population size)
 ├── core/         # Main classes (Car, Neural Network, Population)
 ├── track/        # Geometry logic and track layouts
 ├── rendering/    # Visual engine (Pygame) and dashboard UI
 ├── evolution/    # Log generation (CSV) and parameter scheduling
 ├── io/           # Dynamic resolution of relative/absolute paths
 └── app/          # Main simulation loop (EvolutionGame)
```

*The `main.py` file serves only as an entry point for the package's `run()` function.*

---

## Logging and Paths

The `io/paths.py` module ensures that files are read and saved in the correct locations, regardless of where the script was executed from.

- **Storage:** All logs go into the `logs/` folder at the project root (created automatically by `ensure_logs_dir()`).
- **Format:** At the end of each generation, a new row is appended to a timestamped CSV file (e.g., `evolution_log_YYYYMMDD_HHMMSS.csv`).
- **Logged Metrics:** `Generation`, `Mutation_Rate`, `Crossover_Rate`, `Selection_Method`, `Max_Fitness_Session`, `Finished_Cars`, `Leader_Gates`.

---

## Customization

- **General Settings:** Edit `car_evolution/config/settings.py` to change parameters such as population size, maximum generation time, and FPS.
- **New Tracks:** To create a different track layout, instantiate a new `RaceTrack` object and pass it as an argument to `EvolutionGame(track=...)` in the `car_evolution/app/game.py` file.

---

## License

Educational project. Feel free to clone, study, adapt, and use it as a baseline for your own experiments with Artificial Intelligence and evolutionary algorithms.