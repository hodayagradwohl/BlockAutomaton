import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import sys
import os

class BlockAutomatonGUI:
    """
    Main GUI menu for the Block Automaton project.
    Allows navigation between different tasks and setting simulation parameters.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI.
        
        Parameters:
        - root: The tkinter root window
        """
        self.root = root
        self.root.title("Block Automaton Simulator")
        self.root.geometry("600x550")
        
        # Set default values
        self.size_var = tk.IntVar(value=100)
        self.phases_var = tk.IntVar(value=250)
        self.prob_var = tk.DoubleVar(value=0.5)
        self.wrap_var = tk.BooleanVar(value=True)
        self.interval_var = tk.IntVar(value=1000)
        
        # Add validation for preventing errors
        self.root.report_callback_exception = self.show_error
        
        # Create output directories if they don't exist
        os.makedirs("data_task1/plots", exist_ok=True)
        os.makedirs("data_task1/CSV", exist_ok=True)
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create all GUI widgets and arrange them in the window"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Block Automaton Simulator", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Task selection frame
        task_frame = ttk.LabelFrame(main_frame, text="Select Task")
        task_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(task_frame, text="Task 1: Random Initial State", 
                  command=self.run_task1).pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(task_frame, text="Task 2: Glider Patterns", 
                  command=self.run_task2).pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(task_frame, text="Task 3: Special Patterns (Traffic Light & Blinker)", 
                  command=self.run_task3).pack(fill=tk.X, padx=5, pady=5)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Simulation Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Grid size
        size_frame = ttk.Frame(params_frame)
        size_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(size_frame, text="Grid Size:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(size_frame, from_=8, to=200, increment=2, textvariable=self.size_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(size_frame, text="(Must be even)").pack(side=tk.LEFT, padx=5)
        
        # Number of phases
        phases_frame = ttk.Frame(params_frame)
        phases_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(phases_frame, text="Number of Phases:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(phases_frame, from_=10, to=1000, increment=10, textvariable=self.phases_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Initial probability
        prob_frame = ttk.Frame(params_frame)
        prob_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(prob_frame, text="Initial Probability (0-1):").pack(side=tk.LEFT, padx=5)
        prob_entry = ttk.Entry(prob_frame, textvariable=self.prob_var, width=5)
        prob_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(prob_frame, text="(e.g., 0.5)").pack(side=tk.LEFT, padx=5)
        
        # Wraparound option
        wrap_frame = ttk.Frame(params_frame)
        wrap_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(wrap_frame, text="Enable Wraparound", variable=self.wrap_var).pack(side=tk.LEFT, padx=5)
        
        # Animation interval
        interval_frame = ttk.Frame(params_frame)
        interval_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(interval_frame, text="Animation Interval (ms):").pack(side=tk.LEFT, padx=5)
        ttk.Scale(interval_frame, from_=1, to=2000, orient=tk.HORIZONTAL, 
                 variable=self.interval_var, length=200).pack(side=tk.LEFT, padx=5)
        ttk.Label(interval_frame, textvariable=self.interval_var).pack(side=tk.LEFT, padx=5)
        
        # Data Analysis frame
        analysis_frame = ttk.LabelFrame(main_frame, text="Data Analysis")
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(analysis_frame, text="View Metrics Graphs", 
                  command=self.open_metrics_folder).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(analysis_frame, text="Export CSV Data", 
                  command=self.open_csv_folder).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(button_frame, text="About", command=self.show_about).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)
    
    def validate_parameters(self):
        """
        Validate the parameters before running a simulation.
        
        Returns:
            bool: True if parameters are valid, False otherwise
        """
        # Check grid size is even
        if self.size_var.get() % 2 != 0:
            messagebox.showerror("Invalid Parameters", "Grid size must be an even number.")
            return False
        
        # Check probability is a float between 0 and 1
        try:
            # Get the value and handle potential formatting issues
            prob_str = str(self.prob_var.get()).strip()
            # Replace possible comma with dot (for locales using comma as decimal separator)
            prob_str = prob_str.replace(',', '.')
            # Split by dot and take only the first part plus at most one decimal part
            parts = prob_str.split('.')
            if len(parts) > 1:
                prob_str = parts[0] + '.' + parts[1]
            
            prob = float(prob_str)
            if not (0.0 <= prob <= 1.0):
                messagebox.showerror("Invalid Parameters", 
                                    f"Probability value {prob} must be between 0 and 1 (e.g., 0.25, 0.5, 0.75).")
                return False
                
            # Update the variable with the sanitized value
            self.prob_var.set(prob)
        except Exception as e:
            messagebox.showerror("Invalid Parameters", 
                                f"Invalid probability format: '{self.prob_var.get()}'\nPlease enter a number between 0 and 1 (e.g., 0.25, 0.5, 0.75).")
            return False
            
        return True
    
    def run_task1(self):
        """Run the random initial state simulation (Task 1)"""
        if not self.validate_parameters():
            return
            
        # Pass parameters to the task1 script
        try:
            # Create command to run task1.py with parameters
            cmd = [
                sys.executable, 
                "task1.py",
                "--size", str(self.size_var.get()),
                "--phases", str(self.phases_var.get()),
                "--prob", str(self.prob_var.get()),
                "--wrap", str(self.wrap_var.get()).lower(),  # Convert to lowercase "true" or "false"
                "--interval", str(self.interval_var.get())
            ]
            
            subprocess.Popen(cmd)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run Task 1: {str(e)}")
    
    def run_task2(self):
        """Run the glider patterns simulation (Task 2)"""
        if not self.validate_parameters():
            return
            
        try:
            # Create command to run task2.py with parameters
            cmd = [
                sys.executable, 
                "task2.py",
                "--size", str(self.size_var.get()),
                "--phases", str(self.phases_var.get()),
                "--wrap", str(self.wrap_var.get()).lower(),  # Convert to lowercase "true" or "false"
                "--interval", str(self.interval_var.get())
            ]
            
            subprocess.Popen(cmd)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run Task 2: {str(e)}")
    
    def run_task3(self):
        """Run the special patterns simulation (Task 3)"""
        if not self.validate_parameters():
            return
            
        try:
            # Create command to run task3.py with parameters
            cmd = [
                sys.executable, 
                "task3.py",
                "--size", str(self.size_var.get()),
                "--phases", str(self.phases_var.get()),
                "--wrap", str(self.wrap_var.get()).lower(),  # Convert to lowercase "true" or "false"
                "--interval", str(self.interval_var.get())
            ]
            
            subprocess.Popen(cmd)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run Task 3: {str(e)}")
    
    def open_metrics_folder(self):
        """Open the folder containing metrics graphs"""
        try:
            path = os.path.abspath("data_task1/plots")
            if os.path.exists(path):
                if sys.platform == 'win32':
                    os.startfile(path)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.Popen(['open', path])
                else:  # Linux
                    subprocess.Popen(['xdg-open', path])
            else:
                messagebox.showinfo("Info", "No metrics graphs available yet. Run a simulation first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open metrics folder: {str(e)}")
    
    def open_csv_folder(self):
        """Open the folder containing CSV data"""
        try:
            path = os.path.abspath("data_task1/CSV")
            if os.path.exists(path):
                if sys.platform == 'win32':
                    os.startfile(path)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.Popen(['open', path])
                else:  # Linux
                    subprocess.Popen(['xdg-open', path])
            else:
                messagebox.showinfo("Info", "No CSV data available yet. Run a simulation first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open CSV folder: {str(e)}")
    
    def show_error(self, exc_type, exc_value, exc_traceback):
        """Handle exceptions globally and show user-friendly error messages"""
        # Format a user-friendly error message
        error_message = f"An error occurred: {exc_value}"
        
        # Special handling for common errors
        if "TclError" in str(exc_type) and "expected floating-point number" in str(exc_value):
            error_message = "Invalid number format in one of the input fields. Please check all numeric inputs."
        
        # Show the error message
        messagebox.showerror("Error", error_message)
        
        # Print detailed traceback to console for debugging
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_traceback)
    
    def show_about(self):
        """Show information about the application"""
        about_text = """Block Automaton Simulator
This application simulates a cellular automaton with blocks of 2x2 cells.
The rules change depending on the number of alive cells in each block.

Features:
- Metrics tracking: stability, alive ratio, diversity, and oscillation
- Data export to CSV
- Performance visualization with graphs

Created for Computational Biology Course (80-512/89-512)
"""
        messagebox.showinfo("About", about_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = BlockAutomatonGUI(root)
    root.mainloop()