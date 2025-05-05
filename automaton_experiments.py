import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

DEFAULT_SIZE = 100
DEFAULT_PHASES = 100
DEFAULT_PROBABILITIES = [0.1, 0.2, 0.3, 0.4, 0.5]
RESULTS_DIR = "results"

def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def init_grid(size, p):
    return np.random.choice([0, 1], size=(size, size), p=[1 - p, p])

def extract_block(grid, i, j, size, wraparound):
    if wraparound:
        return np.array([
            [grid[i % size, j % size], grid[i % size, (j + 1) % size]],
            [grid[(i + 1) % size, j % size], grid[(i + 1) % size, (j + 1) % size]]
        ])
    else:
        if i + 1 >= size or j + 1 >= size:
            return None
        return grid[i:i+2, j:j+2]

def apply_rules_to_block(block):
    num_alive = np.sum(block)
    match num_alive:
        case 0 | 1 | 4:
            return 1 - block
        case 2:
            return block
        case 3:
            flipped = 1 - block
            return np.rot90(flipped, 2)

def get_block_coords(size, phase, wraparound):
    coords = []
    start = 0 if phase % 2 == 1 else 1
    for i in range(start, size, 2):
        for j in range(start, size, 2):
            if wraparound or (i + 1 < size and j + 1 < size):
                coords.append((i, j))
    return coords

def update_grid(grid, size, phase, wraparound):
    coords = get_block_coords(size, phase, wraparound)
    for i, j in coords:
        block = extract_block(grid, i, j, size, wraparound)
        if block is None:
            continue
        new_block = apply_rules_to_block(block)
        for di in range(2):
            for dj in range(2):
                ni = (i + di) % size if wraparound else i + di
                nj = (j + dj) % size if wraparound else j + dj
                if ni < size and nj < size:
                    grid[ni, nj] = new_block[di, dj]
    return grid

def calculate_stability(prev_grid, new_grid):
    return np.mean(prev_grid == new_grid) * 100

def calculate_alive_ratio(grid):
    return np.mean(grid) * 100

def calculate_block_diversity(grid, size, phase, wraparound):
    coords = get_block_coords(size, phase, wraparound)
    seen = set()
    for i, j in coords:
        block = extract_block(grid, i, j, size, wraparound)
        if block is None:
            continue
        seen.add(tuple(block.flatten()))
    return len(seen)

def run_experiment(size, p, wraparound=True, phases=DEFAULT_PHASES):
    grid = init_grid(size, p)
    total_stability, total_alive, total_diversity = 0, 0, 0
    for phase in range(1, phases + 1):
        prev_grid = grid.copy()
        grid = update_grid(grid, size, phase, wraparound)
        stability = calculate_stability(prev_grid, grid)
        alive = calculate_alive_ratio(grid)
        diversity = calculate_block_diversity(grid, size, phase, wraparound)
        total_stability += stability
        total_alive += alive
        total_diversity += diversity
    return (
        total_stability / phases,
        total_alive / phases,
        total_diversity / phases
    )

def batch_run():
    ensure_results_dir()
    size = DEFAULT_SIZE
    wraparound = True
    phases = DEFAULT_PHASES

    results = []

    for p in DEFAULT_PROBABILITIES:
        print(f"\nRunning experiment with p = {p:.1f}")
        avg_stability, avg_alive, avg_diversity = run_experiment(size, p, wraparound, phases)
        print(f"→ Stability: {avg_stability:.2f}%, Alive: {avg_alive:.2f}%, Diversity: {avg_diversity:.2f}")
        results.append((p, avg_stability, avg_alive, avg_diversity))

    df = pd.DataFrame(results, columns=["p", "avg_stability", "avg_alive", "avg_diversity"])
    df.to_csv(os.path.join(RESULTS_DIR, "experiment_results.csv"), index=False)

    # גרף השוואה
    plt.figure(figsize=(10, 6))
    plt.plot(df["p"], df["avg_stability"], label="Stability (%)", marker='o')
    plt.plot(df["p"], df["avg_alive"], label="Alive Cells (%)", marker='s')
    plt.plot(df["p"], df["avg_diversity"], label="Diversity", marker='^')
    plt.xlabel("Alive Cell Probability (p)")
    plt.ylabel("Average Metric Value")
    plt.title("Simulation Metrics vs Alive Cell Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "alive_vs_probability.png"))
    print("\nSaved results to 'results/experiment_results.csv' and graph to 'results/alive_vs_probability.png'.")

if __name__ == "__main__":
    batch_run()
