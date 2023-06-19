import os
import pandas as pd
import matplotlib.pyplot as plt

directory = '/path/to/your/csv/files'  # replace this with your directory

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
    _, problem_dim, prev_seq, *_ = file.split('_')
    
    # save the figure
    plt.savefig(f'{problem_dim}{prev_seq}.png')
    
    # show the figure
    plt.show()
