import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def compute_C(d):
    """Compute the C(d) value given dimension d."""
    return 1 + 0.75 * (d - 1) / 99

def compute_I(s):
    """Compute the I(s) value given a dataframe s."""
    return len(s)

def compute_F(S):
    """Compute the F(S) value given a list of dataframes S."""
    total = 0
    for s in S:
        # Extract dimension from filename based on its structure
        if "from" in s.name:
            d = int(s.name.split("from")[0].split("_")[-1])
        else:
            d = int(s.name.split("_")[3])
        total += compute_I(s) * compute_C(d)
    return total

def select_files_and_evaluate():
    # Create a simple tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog to select one or multiple files
    file_paths = filedialog.askopenfilenames(title="Select files")

    # Read the selected files into dataframes and store in a list
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, header=None)
        df.name = os.path.basename(file_path)
        dfs.append(df)

    # Compute F(S) from the list of dataframes
    result = compute_F(dfs)
    return result

# Example usage:
print(select_files_and_evaluate())
