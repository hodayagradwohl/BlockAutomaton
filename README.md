# Block Automaton Simulator

This project provides a visual simulator for a 2D block-based cellular automaton implemented in Python. It includes three main tasks:
- Task 1: Random Initial State
- Task 2: Glider Patterns
- Task 3: Special Patterns (Traffic Light & Blinker)

The GUI interface allows users to configure simulation parameters (grid size, number of phases, probability, wraparound mode, etc.) and view or export analysis metrics.

---

## 📦 Requirements & Setup Instructions

This simulator requires **Python 3.6+** and the following Python packages:
- `numpy`
- `matplotlib`
- `tkinter` (usually pre-installed with Python)

---

## 🛠️ Installation Instructions

### 1. Install Python 3 (if not already installed)
You can install Python using your system’s package manager or download it from [python.org](https://www.python.org/).

For Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv python3-tk
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the required Python packages
```bash
pip install numpy matplotlib
```

---

## ▶️ Running the Application

Use the following command to launch the main graphical simulator:
```bash
python3 run_all.py
```

Make sure you are inside the project root directory where `run_all.py` is located.

---

## 📁 Output Data

Simulation outputs are saved in:
- `data_task1/plots/` → metric graphs
- `data_task1/CSV/` → raw CSV metric data
- `results/` → images from Task 2 path tracking

---

## 🧪 Notes

- If you encounter `externally-managed-environment` errors, consider using a virtual environment or pass `--break-system-packages` to `pip install`.
- GUI should work on Linux, macOS, and Windows. Automatic window maximization is only supported on Windows.
- Ensure that `tkinter` is properly installed if you face GUI launch issues.