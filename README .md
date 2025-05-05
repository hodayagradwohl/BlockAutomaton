# Block Automaton Simulator

This project provides a visual simulator for a 2D block-based cellular automaton, implemented in Python. The automaton operates on an NxN grid, where updates are applied to non-overlapping 2x2 blocks in alternating red-blue phases. The simulation supports different initial configurations, including randomly generated states and specific patterns such as gliders, traffic lights, and blinkers.

## Features

- Graphical user interface (GUI) for task selection and parameter configuration
- Three experiment types: random state, glider patterns, and special patterns
- Visualization of grid evolution in real-time
- Metric tracking: stability, alive cell ratio, block diversity, and oscillation
- CSV export of all metric data
- Plot generation for metric trends
- Wraparound mode toggle
- Full automation of environment setup and dependency installation

## Requirements

- Python 3.6 or higher
- Internet connection for installing dependencies (first-time run)

## Installation & Execution

Follow these steps to run the simulator:

### 1. Clone the repository
```bash
git clone https://github.com/hodayagradwohl/BlockAutomaton.git
cd BlockAutomaton
```

### 2. Ensure pip is installed
If pip is not installed on your system, run:
```bash
sudo apt update
sudo apt install python3-pip
```

### 3. Run the launcher script
This script automatically installs the required Python libraries and opens the GUI.

```bash
python3 run_all.py
```

If you encounter any permission errors, try:

```bash
python3 -m pip install --user numpy matplotlib pandas
```

> **Note:** On first run, required libraries will be installed: `numpy`, `matplotlib`, `pandas`, `tkinter`.

## Directory Structure

- `BlockAutomatonGUI.py` — Main GUI interface
- `task1.py` — Random initial state simulation
- `task2.py` — Glider pattern simulation
- `task3.py` — Special pattern experiment (traffic light, blinker)
- `run_all.py` — Script for launching the GUI and installing dependencies
- `README.md` — Instructions and documentation

## Output

- Metrics CSV files are saved to: `data_task1/CSV/`
- Graphs are saved to: `data_task1/plots/`
- Glider plots are saved to: `results/` (for task2)

## Troubleshooting

- If the GUI doesn't launch, verify that `tkinter` is installed.
- If `pip` is missing, run `sudo apt install python3-pip`.
- For WSL users: use `sudo apt install python3-tk` to enable GUI support.