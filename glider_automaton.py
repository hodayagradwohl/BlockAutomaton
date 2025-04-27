import numpy as np
import matplotlib.pyplot as plt
import os

# --- Constants ---
DEFAULT_SIZE = 100  # Default grid size
DEFAULT_PHASES = 100  # Default number of simulation phases
RESULTS_DIR = "results"  # Directory to store result images

# --- Helper Functions ---
def create_initial_grid(size=DEFAULT_SIZE):
    """
    Create the initial grid with a glider pattern positioned at the center.
    """
    grid = np.zeros((size, size), dtype=int)
    mid = size // 2
    glider_pattern = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    grid[mid-2:mid+3, mid-2:mid+3] = glider_pattern
    return grid

def extract_block(grid, i, j, size, wraparound):
    """
    Extract a 2x2 block from the grid, considering wraparound if enabled.
    """
    if wraparound:
        return np.array([
            [grid[i % size, j % size], grid[i % size, (j + 1) % size]],
            [grid[(i + 1) % size, j % size], grid[(i + 1) % size, (j + 1) % size]]
        ])
    else:
        return grid[i:i+2, j:j+2]

def apply_rules_to_block(block):
    """
    Apply the automaton rules to a given 2x2 block.
    """
    num_alive = np.sum(block)
    match num_alive:
        case 0 | 1 | 4:
            return 1 - block  # Flip all cells
        case 2:
            return block  # No change
        case 3:
            flipped = 1 - block
            return np.rot90(flipped, 2)  # Rotate 180 degrees

def get_block_coords(size, phase, wraparound):
    """
    Determine the starting coordinates for the blocks to update in the current phase.
    """
    coords = []
    start = 0 if phase % 2 == 1 else 1
    for i in range(start, size, 2):
        for j in range(start, size, 2):
            if wraparound or (i + 1 < size and j + 1 < size):
                coords.append((i, j))
    return coords

def update_grid(grid, size, phase, wraparound):
    """
    Update the entire grid by applying rules to each block.
    """
    coords = get_block_coords(size, phase, wraparound)
    for i, j in coords:
        block = extract_block(grid, i, j, size, wraparound)
        new_block = apply_rules_to_block(block)
        for di in range(2):
            for dj in range(2):
                ni = (i + di) % size if wraparound else i + di
                nj = (j + dj) % size if wraparound else j + dj
                if ni < size and nj < size:
                    grid[ni, nj] = new_block[di, dj]
    return grid

def calculate_center_of_mass(grid):
    """
    Calculate the center of mass of all live cells in the grid.
    """
    live_cells = np.argwhere(grid == 1)
    if len(live_cells) == 0:
        return (0, 0)
    return np.mean(live_cells, axis=0)

def ensure_results_dir():
    """
    Ensure that the results directory exists.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

# --- Simulation Runner ---
def run_glider_simulation(size=DEFAULT_SIZE, phases=DEFAULT_PHASES, wraparound=True, save_prefix="glider"):
    """
    Run the glider simulation for a given number of phases and save the result plot.
    """
    ensure_results_dir()
    grid = create_initial_grid(size)
    centers = []

    for phase in range(1, phases + 1):
        grid = update_grid(grid.copy(), size, phase, wraparound)
        c_mass = calculate_center_of_mass(grid)
        centers.append(c_mass)
        print(f"Phase {phase}: Center of Mass at ({c_mass[0]:.2f}, {c_mass[1]:.2f})")

    centers = np.array(centers)

    if centers.ndim == 2:
        plt.figure()
        plt.plot(centers[:, 1], centers[:, 0], marker='o')
        plt.title(f"Glider Path (Wraparound={wraparound})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        file_path = os.path.join(RESULTS_DIR, f"{save_prefix}_path_wrap-{wraparound}.png")
        plt.savefig(file_path)
        print(f"Saved glider path plot to {file_path}")
    else:
        print("No movement detected, skipping plot.")

# --- Main Menu ---
def main():
    """
    Main function to interact with the user and run the simulation.
    """
    print("\n=== Glider Automaton Simulator (Assignment 2) ===")
    try:
        phases = int(input("Enter number of phases to simulate (e.g., 100): ") or DEFAULT_PHASES)
        wrap_input = input("Wraparound grid? (y/n) [default: y]: ").lower()
        wraparound = True if wrap_input in ('', 'y', 'yes') else False

        run_glider_simulation(phases=phases, wraparound=wraparound)
    except Exception as e:
        print(f"Error: {e}\nPlease try again.")

if __name__ == "__main__":
    main()
