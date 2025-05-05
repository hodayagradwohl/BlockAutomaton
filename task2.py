import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- Constants ---
DEFAULT_SIZE = 100
DEFAULT_PHASES = 100
RESULTS_DIR = "results"

# --- Helper Functions ---
def create_initial_grid(size=DEFAULT_SIZE):
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
    if wraparound:
        return np.array([
            [grid[i % size, j % size], grid[i % size, (j + 1) % size]],
            [grid[(i + 1) % size, j % size], grid[(i + 1) % size, (j + 1) % size]]
        ])
    else:
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
        new_block = apply_rules_to_block(block)
        for di in range(2):
            for dj in range(2):
                ni = (i + di) % size if wraparound else i + di
                nj = (j + dj) % size if wraparound else j + dj
                if ni < size and nj < size:
                    grid[ni, nj] = new_block[di, dj]
    return grid

def calculate_center_of_mass(grid):
    live_cells = np.argwhere(grid == 1)
    if len(live_cells) == 0:
        return (0, 0)
    return np.mean(live_cells, axis=0)

def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

# --- GUI Simulation Class ---
class GliderSimulation:
    def __init__(self, root=None, size=DEFAULT_SIZE, phases=DEFAULT_PHASES, 
                 wraparound=True, interval=100):
        self.size = size
        self.phases = phases
        self.wraparound = wraparound
        self.interval = interval
        self.current_phase = 0
        self.grid = create_initial_grid(size)
        self.centers = []

        self.root = root
        if root:
            self.setup_gui()

    def setup_gui(self):
        self.root.title(f"Glider Automaton Simulation - Size: {self.size}, Wraparound: {self.wraparound}")
        try:
            if self.root.tk.call('tk', 'windowingsystem') == 'win32':
                self.root.state('zoomed')
        except:
            pass

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.path_fig = Figure(figsize=(6, 3))
        self.path_ax = self.path_fig.add_subplot(111)
        self.path_canvas = FigureCanvasTkAgg(self.path_fig, master=main_frame)
        self.path_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.start_button = tk.Button(control_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.pause_button = tk.Button(control_frame, text="Pause", command=self.pause_simulation, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.reset_button = tk.Button(control_frame, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_button = tk.Button(control_frame, text="Save Results", command=self.save_results)
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.phase_var = tk.StringVar(value="Phase: 0")
        self.phase_label = tk.Label(control_frame, textvariable=self.phase_var)
        self.phase_label.pack(side=tk.LEFT, padx=20, pady=5)

        self.update_plots()

        self.running = False
        self.job_id = None

    def update_plots(self):
        self.ax.clear()
        self.ax.imshow(self.grid, cmap='binary', interpolation='nearest')
        self.ax.set_title(f"Glider Simulation - Phase {self.current_phase}")
        self.ax.axis('off')
        self.canvas.draw()

        if self.centers:
            centers_array = np.array(self.centers)
            self.path_ax.clear()
            self.path_ax.plot(centers_array[:, 1], centers_array[:, 0], 'bo-', markersize=3)
            self.path_ax.set_title(f"Glider Path (Wraparound={self.wraparound})")
            self.path_ax.set_xlabel("X")
            self.path_ax.set_ylabel("Y")
            self.path_ax.grid(True)
            self.path_fig.tight_layout()
            self.path_canvas.draw()

    def step_simulation(self):
        if self.current_phase < self.phases:
            self.current_phase += 1
            self.grid = update_grid(self.grid.copy(), self.size, self.current_phase, self.wraparound)
            c_mass = calculate_center_of_mass(self.grid)
            self.centers.append(c_mass)
            self.phase_var.set(f"Phase: {self.current_phase}")
            self.update_plots()

            if self.running and self.current_phase < self.phases:
                self.job_id = self.root.after(self.interval, self.step_simulation)
            elif self.current_phase >= self.phases:
                self.running = False
                self.start_button.config(state=tk.DISABLED)
                self.pause_button.config(state=tk.DISABLED)

    def start_simulation(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.step_simulation()

    def pause_simulation(self):
        self.running = False
        if self.job_id:
            self.root.after_cancel(self.job_id)
            self.job_id = None
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)

    def reset_simulation(self):
        self.pause_simulation()
        self.current_phase = 0
        self.grid = create_initial_grid(self.size)
        self.centers = []
        self.phase_var.set(f"Phase: {self.current_phase}")
        self.update_plots()
        self.start_button.config(state=tk.NORMAL)

    def save_results(self):
        ensure_results_dir()

        plt.figure()
        plt.imshow(self.grid, cmap='binary', interpolation='nearest')
        plt.title(f"Glider Simulation - Phase {self.current_phase}")
        plt.axis('off')
        grid_path = os.path.join(RESULTS_DIR, f"glider_grid_phase-{self.current_phase}_wrap-{self.wraparound}.png")
        plt.savefig(grid_path)
        plt.close()

        if self.centers:
            centers_array = np.array(self.centers)
            plt.figure()
            plt.plot(centers_array[:, 1], centers_array[:, 0], 'bo-', markersize=3)
            plt.title(f"Glider Path (Wraparound={self.wraparound})")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            path_path = os.path.join(RESULTS_DIR, f"glider_path_wrap-{self.wraparound}.png")
            plt.savefig(path_path)
            plt.close()

        tk.messagebox.showinfo("Save Complete", f"Results saved to {RESULTS_DIR} directory.")

def run_glider_simulation(size=DEFAULT_SIZE, phases=DEFAULT_PHASES, wraparound=True, save_prefix="glider"):
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

def main():
    parser = argparse.ArgumentParser(description='Glider Automaton Simulator (Task 2)')
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE, help='Grid size (default: 100)')
    parser.add_argument('--phases', type=int, default=DEFAULT_PHASES, help='Number of phases to simulate (default: 100)')
    parser.add_argument('--wrap', type=str, default='true', choices=['true', 'false'], help='Enable grid wraparound (default: true)')
    parser.add_argument('--interval', type=int, default=100, help='Animation interval in milliseconds (default: 100)')
    parser.add_argument('--nogui', action='store_true', help='Run without GUI')
    args = parser.parse_args()

    wraparound = (args.wrap.lower() == 'true')

    if args.nogui:
        run_glider_simulation(size=args.size, phases=args.phases, wraparound=wraparound)
    else:
        root = tk.Tk()
        sim = GliderSimulation(
            root=root,
            size=args.size,
            phases=args.phases,
            wraparound=wraparound,
            interval=args.interval
        )
        root.mainloop()

if __name__ == "__main__":
    main()
