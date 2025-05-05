import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd

# Constants
SIZE = 100
PHASES = 20
PROBS = [0.1, 0.2, 0.3, 0.4, 0.5]
WRAPAROUND_OPTIONS = [True, False]

# Utility Functions
def init_grid(p):
    return np.random.choice([0, 1], size=(SIZE, SIZE), p=[1 - p, p])

def extract_block(grid, i, j, wraparound):
    if wraparound:
        return np.array([
            [grid[i % SIZE, j % SIZE], grid[i % SIZE, (j + 1) % SIZE]],
            [grid[(i + 1) % SIZE, j % SIZE], grid[(i + 1) % SIZE, (j + 1) % SIZE]]
        ])
    else:
        return grid[i:i + 2, j:j + 2]

def apply_rules_to_block(block):
    num_alive = np.sum(block)
    if num_alive in [0, 1, 4]:
        return 1 - block
    elif num_alive == 2:
        return block
    elif num_alive == 3:
        flipped = 1 - block
        return np.rot90(flipped, 2)

def get_block_coords(phase, wraparound):
    coords = []
    start = 0 if phase % 2 == 1 else 1
    for i in range(start, SIZE, 2):
        for j in range(start, SIZE, 2):
            if wraparound or (i + 1 < SIZE and j + 1 < SIZE):
                coords.append((i, j))
    return coords

def update_grid(grid, phase, wraparound):
    coords = get_block_coords(phase, wraparound)
    new_grid = grid.copy()
    for i, j in coords:
        block = extract_block(grid, i, j, wraparound)
        new_block = apply_rules_to_block(block)
        for di in range(2):
            for dj in range(2):
                ni = (i + di) % SIZE if wraparound else i + di
                nj = (j + dj) % SIZE if wraparound else j + dj
                if ni < SIZE and nj < SIZE:
                    new_grid[ni, nj] = new_block[di, dj]
    return new_grid

def calculate_metrics(prev_grid, current_grid):
    stability = np.mean(prev_grid == current_grid) * 100
    alive_ratio = np.mean(current_grid) * 100
    return stability, alive_ratio

# Run experiments
results = []

for p in PROBS:
    for wrap in WRAPAROUND_OPTIONS:
        grid = init_grid(p)
        stabilities = []
        alive_ratios = []

        for phase in range(1, PHASES + 1):
            new_grid = update_grid(grid, phase, wrap)
            stability, alive = calculate_metrics(grid, new_grid)
            stabilities.append(stability)
            alive_ratios.append(alive)
            grid = new_grid

        results.append({
            "p": p,
            "wraparound": wrap,
            "avg_stability": mean(stabilities),
            "avg_alive": mean(alive_ratios)
        })

# Convert to DataFrame and save
df = pd.DataFrame(results)
df.to_csv("wraparound_vs_nowrap_results.csv", index=False)
print("Results saved to wraparound_vs_nowrap_results.csv")
print(df)
