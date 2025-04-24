import numpy as np

SIZE = 100 # Size of the grid
IS_WRAPAROUND  = False # If True, the grid wraps around at the edges

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
        grid[i:i+2, j:j+2] = new_block
    
    return grid

def run_simulation(phases, prob):
    """
    This function runs the simulation for a given number of steps.
    param phases: The number of phases to run the simulation.
    param prob: The probability of a cell being alive.
    """
    grid = init_grid(prob)
    
    for phase in range(phases):
        grid = update_grid(grid, phase)
        print(grid)  # Uncomment to see the grid at each phase
    
    return grid


if __name__ == "__main__":
    """
    Main function to run the simulation.
    """
    print("Starting simulation...")
    phases = 250
    prob = 0.5  # probability of cell being alive

    final_grid = run_simulation(phases, prob)    