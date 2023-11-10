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
        folder_path = "/Users/sophie/Desktop/Medical Image Analysis Lab/copy/MIALab_Lukas_Studer/bin/mia-result/results"
        all_files = os.listdir(folder_path)

    except FileNotFoundError as e: #added an exit if directory wrong
        print(f"File 'results.csv' not found: {e}")
        print("Please verify the file path.")
        return

    csv_files = [file for file in all_files if file.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    print("CSV files found in the folder:")
    for csv_file in csv_files:
        print(os.path.join(folder_path, csv_file))
        csv_file_path = os.path.join(folder_path, csv_file)

        df = pd.read_csv(csv_file_path, delimiter=';')


        labels = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']
        filtered_df = df[df['LABEL'].isin(labels)]

        # Plot the Dice coefficients per label
        plt.figure(figsize=(10,6))
        plt.boxplot([filtered_df[filtered_df['LABEL'] == label]['DICE'] for label in labels], labels=labels)
        plt.title('Dice Coefficients per Label')
        plt.ylabel('Dice Coefficient')
        plt.xlabel('Label')
        plt.grid(axis='y')
        plt.show()
    #  in a boxplot

    # Comparison between HDRFDST and the DICE
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['HDRFDST'])
    plt.title('Hausdorff distance Boxplot')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['DICE'])
    plt.title('Dice Coefficient Boxplot')

    plt.tight_layout()
    plt.show()

    # Plotting both
    labels = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']
    filtered_df = df[df['LABEL'].isin(labels)]

    # Set up the figure and subplots
    fig, axes = plt.subplots(nrows=2, ncols=len(labels), figsize=(15, 8))

    # Create box plots for DICE for each label
    for idx, label in enumerate(labels):
        data_label = filtered_df[filtered_df['LABEL'] == label]
        sns.boxplot(x='DICE', data=data_label, ax=axes[0, idx])
        axes[0, idx].set_title(f'{label} - Dice Coefficient')

    # Create box plots for HDRFDST for each label
    for idx, label in enumerate(labels):
        data_label = filtered_df[filtered_df['LABEL'] == label]
        sns.boxplot(x='HDRFDST', data=data_label, ax=axes[1, idx])
        axes[1, idx].set_title(f'{label} - Hausdorff Distance')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    #pass  # pass is just a placeholder if there is no other code


if __name__ == '__main__':
    main()
