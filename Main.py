import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SIZE = 100 # Size of the grid
IS_WRAPAROUND  = True # If True, the grid wraps around at the edges

def init_grid(p):
    """
    This function sets the initial state of the automaton.
    param p: The probability of a cell being alive.
    return: A 2D numpy array representing the grid.
    """
    return np.random.choice([0, 1], size=(SIZE, SIZE), p=[p, 1-p]) # 0 for dead, 1 for alive

def get_block_coords(phase):
    """
    This function calculates the coordinates of the blocks in the grid.
    param phase: The phase of the block.
    return: A list of tuples representing the coordinates of the blocks.
    """
    coords = []
    start = 0 if phase % 2 == 1 else 1  # Blue or red
    for i in range(start, SIZE, 2):
        for j in range(start, SIZE, 2):
            if IS_WRAPAROUND or (i + 1 < SIZE and j + 1 < SIZE): 
                coords.append((i, j))
    return coords
    
def extract_block(grid, i, j):
    """
    This function extracts a block from the grid.
    param grid: The grid from which to  extract the block.
    param i: The row index of the block.
    param j: The column index of the block.
    return: A 2D numpy array representing the block.
    """
    if IS_WRAPAROUND:
        return np.array([
            [grid[i % SIZE, j % SIZE],     grid[i % SIZE, (j + 1) % SIZE]],
            [grid[(i + 1) % SIZE, j % SIZE], grid[(i + 1) % SIZE, (j + 1) % SIZE]]
        ])
    else:
        return grid[i:i+2, j:j+2]

def apply_rules_to_block(block):
    """
    This function applies the rules of the automaton to a block.
    param block: The block to which to apply the rules.
    return: A 2D numpy array representing the new state of the block.
    """
    num_alive = np.sum(block)
    
    match num_alive:
        case 0 | 1 | 4:
            return 1 - block  # flip the state of the cells
        case 2:
            return block  # no change
        case 3:
            flipped = 1 - block  # flip the state of the cells
            return np.rot90(flipped, 2)  # rotate the block 180 degrees
        case _:
            raise ValueError(f"Invalid number of alive cells: {num_alive}")

def update_grid(grid, phase):
    """
    This function updates the grid based on the rules of the automaton.
    param grid: The grid to update.
    param phase: The phase of the block.
    return: A 2D numpy array representing the updated grid.
    """
    coords = get_block_coords(phase)
    
    for i, j in coords:
        block = extract_block(grid, i, j)
        new_block = apply_rules_to_block(block)

        for di in range(2):
            for dj in range(2):
                ni = (i + di) % SIZE if IS_WRAPAROUND else i + di
                nj = (j + dj) % SIZE if IS_WRAPAROUND else j + dj
                if ni < SIZE and nj < SIZE:  # check bounds if not wraparound
                    grid[ni, nj] = new_block[di, dj]

    return grid

def calculate_metrics(prev_grid, current_grid):
    """
    Calculate metrics for the current simulation step.

    Returns:
        - stability: Percentage of cells that stayed the same.
        - alive_ratio: Percentage of alive cells (value == 1).
    """
    stability = np.mean(prev_grid == current_grid) * 100
    alive_ratio = np.mean(current_grid) * 100
    return stability, alive_ratio

def calculate_block_diversity(grid, phase):
    """
    Calculate how many unique 2x2 block patterns exist in the current grid.
    :param grid: The current grid (numpy array).
    :param phase: The current phase (determines which blocks to use).
    :return: Number of unique 2x2 block patterns.
    """
    coords = get_block_coords(phase)
    seen_patterns = set()

    for i, j in coords:
        block = extract_block(grid, i, j)
        pattern = tuple(block.flatten())  # Convert block to hashable tuple
        seen_patterns.add(pattern)

    return len(seen_patterns)

def calculate_oscillation_score(prev_prev_grid, prev_grid, current_grid):
    """
    Calculate the oscillation score based on the last three states.
    A cell is considered oscillating if its value changed from prev_prev to prev and then back in current.
    """
    if prev_prev_grid is None:
        return 0  # Not enough history

    oscillating = (prev_prev_grid == current_grid) & (prev_prev_grid != prev_grid)
    return int(np.sum(oscillating))

def run_gui_simulation(phases, prob, interval=1000):
    """
    Run the GUI simulation of the automaton.
    param phases: The number of phases to simulate.
    param prob: The probability of a cell being alive.
    param interval: The interval between frames in milliseconds.
    """
    grid = init_grid(prob)
    prev_grid = grid.copy()
    prev_prev_grid = None

    fig, ax = plt.subplots()
    img = ax.imshow(grid, cmap='Greys', vmin=0, vmax=1)
    title = ax.set_title("")

    wrap_text = "Wraparound: ON" if IS_WRAPAROUND else "Wraparound: OFF"
    size_text = f"Grid size: {SIZE}x{SIZE}"

    # Show initial state as Phase 0
    img.set_data(grid)
    title.set_text(
    f"Phase 0 – Initial State\n"
    f"{wrap_text} | {size_text}"
    )
    plt.pause(1.5)  # pause briefly to show initial state
    plt.get_current_fig_manager().window.state('zoomed')  # maximize the window (Windows only)

    def update(phase):
        """
        Update the grid for each phase.
        param phase: The current phase of the simulation.
        return: A list of artists to update in the animation.
        """
        nonlocal grid, prev_grid, prev_prev_grid

        new_grid = update_grid(grid.copy(), phase)

        stability, alive_ratio = calculate_metrics(prev_grid, new_grid)
        diversity = calculate_block_diversity(new_grid, phase)
        oscillation = calculate_oscillation_score(prev_prev_grid, prev_grid, new_grid)

        img.set_data(new_grid)
        block_color = 'Red' if phase % 2 == 0 else 'Blue'
        title.set_text(
            f"Phase {phase} – {block_color} blocks\n"
            f"{wrap_text} | {size_text}\n"
            f"Stability: {stability:.1f}% | Alive: {alive_ratio:.1f}% | "
            f"Diversity: {diversity} | Oscillation: {oscillation}"
        )

        prev_prev_grid = prev_grid
        prev_grid = grid
        grid = new_grid

        return [img, title]

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, phases + 1), interval=interval, blit=False, repeat=False
    )
    plt.show()

if __name__ == "__main__":
    """
    Main function to run the simulation.
    """
    print("Starting simulation...")
    phases = 250
    prob = 0.5  # probability of cell being alive

    run_gui_simulation(phases, prob, interval=1000)
