import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

ToleranceDict = {1e-3:"1e-3Tolerance", 1e-2:"1e-2Tolerance"}

root_directory = f"C:\\Users\\eloyfernandez\\Documents\\Eloy\\Uni\\TFG\\TFG\\TFG-on-Krotov-Methods\\"
analysis_directory = os.path.join(root_directory, "Analisis")

tolerance_folders = glob.glob(os.path.join(analysis_directory, "1e-*"))  # get all folders that match the pattern "1e-*"

for tolerance_folder in tolerance_folders:
    ToleranceString = os.path.basename(tolerance_folder)
    Tolerance = 1 / (10 ** int(ToleranceString.replace("Tolerance","").replace("1e-",""))) # extract tolerance value from folder name

    # list all .csv files in the tolerance folder
    files = [f for f in os.listdir(tolerance_folder) if f.endswith('.csv') and f.startswith('functional_valuesd_dim_')]

    for idx, file in enumerate(files):
        # read data from csv file
        df = pd.read_csv(os.path.join(tolerance_folder, file), header=None)

        # calculate relative change of J_T
        df['relative_change_J_T'] = abs(df[4] / df[1])

        # plot relative change of J_T with respect to the iteration number
        plt.figure()
        plt.plot(df[0], df['relative_change_J_T'], label='Relative Change J_T', color='blue')
        plt.plot(df[0], df[1], label='Convergence of J_T', color='red')
        plt.yscale('log')
        plt.xlabel('Iteration number')

        # Check if the last element is smaller than or equal to Tolerance
        if df[1].iloc[-1] <= Tolerance:
            # If convergence is reached, plot a solid horizontal line at the level of Tolerance
            plt.axhline(y=Tolerance, color='green', linestyle='-', label='Convergence Level')
        else:
            # If convergence is not reached, plot a dotted horizontal line at the level of Tolerance
            plt.axhline(y=Tolerance, color='green', linestyle='dotted', label='Expected Convergence Level')

        # Create a legend
        plt.legend(loc="best") 

        # Extract Problem Dimension and Previous Sequence from filename
        substring = file.replace("functional_valuesd_dim_", "").replace("_0to1.csv", "")

        # save the figure
        out_path = os.path.join(root_directory, "Plotter", "Graphs", ToleranceString,substring +".png")
        plt.savefig(out_path)
        plt.close()
        print(f"Graph number {idx + 1} plotted for tolerance {str(Tolerance)    }\n")
