import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import seaborn as sns
import os


def main():
    #
    # load the "results.csv" file from the mia-results directory
    try:
        #file_path = "/Users/sophie/Desktop/Medical Image Analysis Lab/copy/MIALab_Lukas_Studer/bin/mia-result/2023-11-03-12-27-54/results.csv"
        ##df = pd.read_csv(r"results.csv", delimiter=';')
        #df = pd.read_csv(file_path, delimiter=';')
        #folder_path = "/Users/sophie/Desktop/Medical Image Analysis Lab/copy/MIALab_Lukas_Studer/bin/mia-result/results"
        folder_path_ML = r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer" \
                      r"\bin\mia-result\2023-12-06-10-12-32_B0.5"
        all_files_ML = os.listdir(folder_path_ML)

        folder_path_A = r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer" \
                      r"\bin\mia-result\2023-12-06-10-12-32_B0.5"
        all_files_A = os.listdir(folder_path_A)

    except FileNotFoundError as e: #added an exit if directory wrong
        print(f"File 'results.csv' not found: {e}")
        print("Please verify the file path.")
        return

    csv_files = [file for file in all_files_ML if file.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    print("CSV files found in the folder:")
    for csv_file in csv_files:
        print(os.path.join(folder_path_ML, csv_file))
        csv_file_path = os.path.join(folder_path_ML, csv_file)

        df = pd.read_csv(csv_file_path, delimiter=';')


        labels = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']
        filtered_df = df[df['LABEL'].isin(labels)]

        """#  in a boxplot
        # Create box plot for Dice coefficients
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='LABEL', y='DICE', data=filtered_df, color='white', linewidth=1.5)
        plt.title('Dice Coefficients Comparison')
        plt.ylabel('Dice Coefficient')
        plt.xlabel('Label')
        plt.ylim(bottom=0, top=1)
        plt.show()

        # Create box plot for Hausdorff distances
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='LABEL', y='HDRFDST', data=filtered_df, color='white', linewidth=1.5)
        plt.title('Hausdorff Distances Comparison')
        plt.ylabel('Hausdorff Distance')
        plt.xlabel('Label')
        plt.ylim(bottom=0)
        plt.show()"""

        # Set up the figure and subplots
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

        # Create box plots for Dice coefficients
        sns.boxplot(x='LABEL', y='DICE', data=filtered_df, ax=axes[0], hue="alive", fill=False, gap=.1)
        axes[0].set_title('Dice Coefficients Comparison')
        axes[0].set_ylabel('Dice Coefficient')
        axes[0].set_xlabel('Label')
        axes[0].set_ylim(bottom=0, top=1)  # Set lower and upper limit of y-axis

        # Create box plots for Hausdorff distances
        sns.boxplot(x='LABEL', y='HDRFDST', data=filtered_df, ax=axes[1], color='white', linewidth=1.5)
        axes[1].set_title('Hausdorff Distances Comparison')
        axes[1].set_ylabel('Hausdorff Distance')
        axes[1].set_xlabel('Label')
        axes[1].set_ylim(bottom=0)  # Set lower limit of y-axis

        # Adjust layout and display
        plt.tight_layout()
        plt.show()


    

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    #pass  # pass is just a placeholder if there is no other code


if __name__ == '__main__':
    main()
