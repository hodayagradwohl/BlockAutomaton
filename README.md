# Block Automaton Simulator

This project provides a visual simulator for a 2D block-based cellular automaton, implemented in Python. The automaton operates on an N×N grid, where updates are applied to non-overlapping 2×2 blocks in alternating red-blue phases. The simulation supports different initial configurations, including randomly generated states and specific patterns such as gliders, traffic lights, and blinkers.

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
- Required libraries: `numpy`, `matplotlib`, `pandas`, and `tkinter`

> Note: On some Linux/WSL systems, `tkinter` may need to be installed manually with:
> ```
> sudo apt update
> sudo apt install python3-tk
> ```

> If pip is not installed, use:
> ```
> sudo apt install python3-pip
> ```

## Installation and Usage

1. Clone or download the repository:
   ```
   git clone https://github.com/<your-username>/BlockAutomaton.git
   cd BlockAutomaton
   ```

2. Run the project using:
   ```
   python3 run_all.py
   ```

   This script will automatically:
   - Check for missing libraries
   - Attempt to install any missing dependencies
   - Launch the graphical user interface

3. From the GUI, select a task and configure simulation parameters such as:
   - Grid size (must be even)
   - Number of simulation phases
   - Initial alive cell probability
   - Wraparound toggle
   - Animation speed

4. After the simulation, access your results:
   - Graphs: `data_task1/plots/`
   - CSV files: `data_task1/CSV/`

## Output

- Graphs: Stability, alive ratio, diversity, and oscillation trends over phases
- CSV: All metrics stored by phase, saved per experiment configuration

## Authors

This simulator was developed as part of a computational biology course assignment. It is designed for experimentation and extension.
