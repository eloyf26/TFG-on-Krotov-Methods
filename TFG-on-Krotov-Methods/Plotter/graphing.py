import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


directory = "C:\\Users\\eloyfernandez\\Documents\\Eloy\\Uni\\TFG\\TFG\\TFG-on-Krotov-Methods\\Analisis"  # replace this with your directory

# list all .csv files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.csv') and f.startswith('functional_valuesd_dim_')]

for file in files:
    # read data from csv file
    df = pd.read_csv(os.path.join(directory, file), header=None)

    # calculate relative change of J_T
    df['relative_change_J_T'] = df[4] / df[1]

    # plot relative change of J_T with respect to the iteration number
    plt.figure()
    plt.plot(df[0], df['relative_change_J_T'])
    plt.xlabel('Iteration number')
    plt.ylabel('Relative change of J_T')
    
    # Extract Problem Dimension and Previous Sequence from filename

    substring = file.replace("functional_valuesd_dim_", "").replace("_0to1.csv", "")
    
    # save the figure
    graph_path = "C:\\Users\\eloyfernandez\\Documents\\Eloy\\Uni\\TFG\\TFG\\TFG-on-Krotov-Methods\\Analisis\\Graphs\\"
    out_path = f'{graph_path}{substring}.png'
    plt.savefig(out_path)
    
