
#!/usr/bin/env python3
import subprocess
import sys
import os

REQUIRED_LIBRARIES = [
    'numpy',
    'matplotlib',
    'pandas'
]

def install_library(lib):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print(f"[OK] {lib} installed successfully.")
    except subprocess.CalledProcessError:
        print(f"[Error] Failed to install {lib}")

def install_dependencies():
    print("Checking and installing required libraries...")
    for lib in REQUIRED_LIBRARIES:
        try:
            __import__(lib)
            print(f"[OK] {lib} is already installed.")
        except ImportError:
            print(f"[Installing] {lib} is missing. Installing...")
            install_library(lib)

def check_tkinter():
    try:
        import tkinter
        print("[OK] tkinter is available.")
    except ImportError:
        print("[Warning] tkinter is not available. GUI may not work.")
        print("Please install tkinter manually if you want to use the GUI.")
        print("For Ubuntu/Debian use: sudo apt install python3-tk")

def launch_gui():
    try:
        subprocess.check_call([sys.executable, "BlockAutomatonGUI.py"])
    except subprocess.CalledProcessError as e:
        print(f"[Error] Failed to launch GUI: {e}")

def main():
    install_dependencies()
    check_tkinter()
    print("Launching the Block Automaton GUI...")
    launch_gui()

if __name__ == "__main__":
    main()
