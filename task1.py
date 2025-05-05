import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import os
import argparse

# Variables for storing metrics
stabilities = []
alive_ratios = []
diversities = []
oscillations = []

def init_grid(size, p):
    """
    Initialize the grid with random values of 0 and 1.
    size: Size of the grid
    p: Probability of a cell being alive.
    return: A 2D numpy array representing the grid.
    """
    return np.random.choice([0, 1], size=(size, size), p=[p, 1-p])  # 0 for dead, 1 for alive

def get_block_coords(size, phase, is_wraparound):
    """
    This function calculates the coordinates of the blocks in the grid.
    param size: Size of the grid
    param phase: The phase of the block.
    param is_wraparound: If True, the grid wraps around at the edges.
    return: A list of tuples representing the coordinates of the blocks.
    """
    coords = []
    start = 0 if phase % 2 == 1 else 1  # Blue or red
    for i in range(start, size, 2):
        for j in range(start, size, 2):
            if is_wraparound or (i + 1 < size and j + 1 < size):
                coords.append((i, j))
    return coords

def extract_block(grid, i, j, size, is_wraparound):
    """
    This function extracts a block from the grid.
    param grid: The grid from which to  extract the block.
    param i: The row index of the block.
    param j: The column index of the block.
    param size: Size of the grid
    param is_wraparound: If True, the grid wraps around at the edges.
    return: A 2D numpy array representing the block.
    """
    if is_wraparound:
        return np.array([
            [grid[i % size, j % size], grid[i % size, (j + 1) % size]],
            [grid[(i + 1) % size, j % size], grid[(i + 1) % size, (j + 1) % size]]
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

def update_grid(grid, phase, size, is_wraparound):
    """
    This function updates the grid based on the rules of the automaton.
    param grid: The grid to update.
    param phase: The phase of the block.
    param size: Size of the grid
    param is_wraparound: If True, the grid wraps around at the edges.
    return: A 2D numpy array representing the updated grid.
    """
    coords = get_block_coords(size, phase, is_wraparound)
    
    for i, j in coords:
        block = extract_block(grid, i, j, size, is_wraparound)
        new_block = apply_rules_to_block(block)

        # Update the grid with the new block state
        for di in range(2):
            for dj in range(2):
                ni = (i + di) % size if is_wraparound else i + di
                nj = (j + dj) % size if is_wraparound else j + dj
                if ni < size and nj < size:  # Check bounds if not wraparound
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

def calculate_block_diversity(grid, phase, size, is_wraparound):
    """
    Calculate how many unique 2x2 block patterns exist in the current grid.
    :param grid: The current grid (numpy array).
    :param phase: The current phase (determines which blocks to use).
    :param size: Size of the grid
    :param is_wraparound: If True, the grid wraps around at the edges.
    :return: Number of unique 2x2 block patterns.
    """
    coords = get_block_coords(size, phase, is_wraparound)
    seen_patterns = set()

    for i, j in coords:
        block = extract_block(grid, i, j, size, is_wraparound)
        pattern = tuple(block.flatten())  # Convert block to hashable tuple
        seen_patterns.add(pattern)

    return len(seen_patterns)

def calculate_oscillation_score(prev_prev_grid, prev_grid, current_grid):
    """
    Calculate the oscillation score based on the last three states.
    prev_prev_grid: The state of the grid two steps ago.
    prev_grid: The state of the grid one step ago.
    current_grid: The current state of the grid.
    return: Oscillation score.
    """
    if prev_prev_grid is None:
        return 0  # Not enough history

    oscillating = (prev_prev_grid == current_grid) & (prev_prev_grid != prev_grid)
    return int(np.sum(oscillating))

def run_gui_simulation(phases, prob, size, is_wraparound, interval=1000):
    """
    Run the GUI simulation of the automaton.
    param phases: The number of phases to simulate.
    param prob: The probability of a cell being alive.
    param size: Size of the grid
    param is_wraparound: If True, the grid wraps around at the edges.
    param interval: The interval between frames in milliseconds.
    """
    grid = init_grid(size, prob)
    prev_grid = grid.copy()
    prev_prev_grid = None

    fig, ax = plt.subplots()
    img = ax.imshow(grid, cmap='Greys', vmin=0, vmax=1)
    title = ax.set_title("")

    wrap_text = "Wraparound: ON" if is_wraparound else "Wraparound: OFF"
    size_text = f"Grid size: {size}x{size}"

    # Show the initial state as Phase 0
    img.set_data(grid)
    title.set_text(
        f"Phase 0 – Initial State\n"
        f"{wrap_text} | {size_text}"
    )
    plt.pause(1.5)  # Pause briefly to show initial state
    plt.get_current_fig_manager().window.state('zoomed')  # Maximize the window (Windows only)

    def update(phase):
        """
        Update the grid for each phase.
        phase: The current phase of the simulation.
        return: A list of artists to update in the animation.
        """
        nonlocal grid, prev_grid, prev_prev_grid


        new_grid = update_grid(grid.copy(), phase, size, is_wraparound)

        # Calculate the metrics for the current phase
        stability, alive_ratio = calculate_metrics(prev_grid, new_grid)
        diversity = calculate_block_diversity(new_grid, phase, size, is_wraparound)
        oscillation = calculate_oscillation_score(prev_prev_grid, prev_grid, new_grid)

        # Store the metrics for later analysis
        stabilities.append(stability)
        alive_ratios.append(alive_ratio)
        diversities.append(diversity)
        oscillations.append(oscillation)

        img.set_data(new_grid)
        block_color = 'Red' if phase % 2 == 0 else 'Blue'
        title.set_text(
            f"Phase {phase} – {block_color} blocks\n"
            f"{wrap_text} | {size_text}\n"
            f"Stability: {stability:.1f}% | Alive: {alive_ratio:.1f}% | "
            f"Diversity: {diversity} | Oscillation: {oscillation}"
        )

        # Update grid history
        prev_prev_grid = prev_grid
        prev_grid = grid
        grid = new_grid

        if phase == phases:
            fig.canvas.draw()
            plt.pause(5)  
            generate_metrics_graphs(prob, is_wraparound, size)  
            export_metrics_to_csv(prob, is_wraparound, size)    
            plt.close(fig)     

        return [img, title]

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, phases + 1), interval=interval, blit=False, repeat=False
    )

    plt.show()

    # Once the animation finishes, generate the graphs
    generate_metrics_graphs(prob, is_wraparound, size)  # Now generate and save the graphs
    export_metrics_to_csv(prob, is_wraparound, size) # Export the metrics to CSV


def generate_metrics_graphs(prob, is_wraparound, size):
    """
    Generate and save graphs in data_task1/plots/{prob}_{is_wraparound}/.
    """
    plot_dir = f"data_task1/plots/{prob}_{is_wraparound}_{size}"
    os.makedirs(plot_dir, exist_ok=True)

    # Stability
    plt.figure()
    plt.plot(stabilities)
    plt.title("Stability over Phases")
    plt.xlabel("Phase")
    plt.ylabel("Stability (%)")
    path = os.path.join(plot_dir, f"stability_graph_{prob}_{is_wraparound}_{size}.png")
    plt.savefig(path)
    plt.close()

    # Alive Ratio
    plt.figure()
    plt.plot(alive_ratios)
    plt.title("Alive Ratio over Phases")
    plt.xlabel("Phase")
    plt.ylabel("Alive Ratio (%)")
    path = os.path.join(plot_dir, f"alive_ratio_graph_{prob}_{is_wraparound}_{size}.png")
    plt.savefig(path)
    plt.close()

    # Diversity
    plt.figure()
    plt.plot(diversities)
    plt.title("Block Diversity over Phases")
    plt.xlabel("Phase")
    plt.ylabel("Diversity")
    path = os.path.join(plot_dir, f"diversity_graph_{prob}_{is_wraparound}_{size}.png")
    plt.savefig(path)
    plt.close()

    # Oscillation
    plt.figure()
    plt.plot(oscillations)
    plt.title("Oscillation Score over Phases")
    plt.xlabel("Phase")
    plt.ylabel("Oscillation Score")
    path = os.path.join(plot_dir, f"oscillation_graph_{prob}_{is_wraparound}_{size}.png")
    plt.savefig(path)
    plt.close()

    print(f"[Graphs] Saved to: {plot_dir}")

def export_metrics_to_csv(prob, is_wraparound, size):
    """
    Export the collected metric data to a CSV file inside data_task1/CSV.
    """
    csv_dir = "data_task1/CSV"
    os.makedirs(csv_dir, exist_ok=True)

    filename = f"simulation_metrics_{prob}_{is_wraparound}_{size}.csv"
    filepath = os.path.join(csv_dir, filename)

    with open(filepath, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Phase", "Stability", "Alive Ratio", "Diversity", "Oscillation"])
        for i in range(len(stabilities)):
            writer.writerow([
                i + 1,
                stabilities[i],
                alive_ratios[i],
                diversities[i],
                oscillations[i]
            ])
    print(f"[CSV] Saved to: {filepath}")

def reset_metrics():
    """
    Reset the global metrics lists.
    """
    global stabilities, alive_ratios, diversities, oscillations
    stabilities = []
    alive_ratios = []
    diversities = []
    oscillations = []

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run the Block Automaton simulation with custom parameters.')
    parser.add_argument('--size', type=int, default=100, help='Size of the grid (must be even)')
    parser.add_argument('--phases', type=int, default=250, help='Number of phases to simulate')
    parser.add_argument('--prob', type=float, default=0.5, help='Probability of a cell being alive')
    parser.add_argument('--wrap', type=str, default='true', help='Enable wraparound (true/false)')
    parser.add_argument('--interval', type=int, default=100, help='Animation interval in milliseconds')
    
    args = parser.parse_args()
    
    # Convert wrap string to boolean
    if args.wrap.lower() in ['true', 't', '1', 'yes', 'y']:
        args.wrap = True
    else:
        args.wrap = False
    
    # Ensure size is even
    if args.size % 2 != 0:
        print("Warning: Grid size must be even. Adjusting to the nearest even number.")
        args.size = args.size + 1 if args.size % 2 == 1 else args.size
    
    return args

if __name__ == "__main__":
    """
    Main function to run the simulation.
    """
    print("Starting simulation...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Reset metrics before starting simulation
    reset_metrics()
    
    # Run the simulation with the specified parameters
    run_gui_simulation(args.phases, args.prob, args.size, args.wrap, interval=args.interval)

    print("Simulation completed!")