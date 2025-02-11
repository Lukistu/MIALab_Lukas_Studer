import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def extract_and_plot_values(folder_paths):
    all_csv_files = []

    for folder_path in folder_paths:
        all_files = os.listdir(folder_path)
        csv_files = [file for file in all_files if file.endswith('results_summary.csv')]
        all_csv_files.extend([os.path.join(folder_path, file) for file in csv_files])

    combined_data = pd.DataFrame()

    for i, csv_file in enumerate(all_csv_files):
        df = pd.read_csv(csv_file, delimiter=';')

        # Extracting 'DICE;MEAN' and 'HDRFDST;MEAN' columns
        required_columns = ['LABEL', 'METRIC', 'STATISTIC', 'VALUE']
        filtered_df = df.loc[df['STATISTIC'] == 'MEAN', required_columns]

        # Pivot the table to have 'LABEL' as rows and 'METRIC' as columns
        filtered_df = filtered_df.pivot(index='LABEL', columns='METRIC', values='VALUE').reset_index()

        combined_data = pd.concat([combined_data, filtered_df], ignore_index=True)
        # Handle missing values or NaNs if necessary
        combined_data.fillna(0, inplace=True)  # Replace NaNs with 0s

    # Set up the figure and subplots
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 8))
    axes = axes.flatten()

    plt.rc('axes', titlesize=20)  # Title font size

    # Only for ML
    x_position = [1,2,5,7,10,30,50,70]


    # Loop through each label and plot them on separate subplots
    selected_labels = ['GreyMatter', 'Amygdala', 'Hippocampus', 'Thalamus', 'WhiteMatter']
    for i, label in enumerate(selected_labels):
        label_data = combined_data[combined_data['LABEL'] == label]

        # DICE Coefficients plot
        sns.pointplot(x=x_position, y='DICE', data=label_data, color='blue', ax=axes[i], native_scale=True)
        axes[i].set_title(f'{label}')
        axes[i].set_xlabel('')
        axes[0].set_ylabel('DICE Values',fontsize=20)
        axes[i].set_ylabel('')
        axes[i].set_ylim(0, 1)  # Set the y-axis limits for DICE to 0 to 1
        axes[i].set_xlim(0, max(x_position))

        # Hausdorff values plot
        sns.pointplot(x=x_position, y='HDRFDST', data=label_data, color='red', ax=axes[i + 5], native_scale=True)
        axes[i + 5].set_xlabel('')
        axes[5].set_ylabel('Hausdorff Values', fontsize=20)
        axes[i + 5].set_ylabel('')
        axes[i + 5].set_ylim(0, 80)  # Set the y-axis limits for Hausdorff to 0 to 100
        axes[i + 5].set_xlim(0, max(x_position))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder_paths = [
        r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer\bin\mia-result"
        r"\2023-12-15-16-08-25_forest-7-1",
        r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer\bin\mia-result"
        r"\2023-12-15-16-17-15_forest-7-2",
        r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer\bin\mia-result"
        r"\2023-12-15-16-26-33_forest-7-5",
        r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer\bin\mia-result"
        r"\2023-12-15-16-34-52_forest-7-7",
        r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer\bin\mia-result"
        r"\2023-12-15-16-41-57_forest-7-10",
        r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer\bin\mia-result"
        r"\2023-12-15-16-51-07_forest-7-30",
        r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer\bin\mia-result"
        r"\2023-12-15-16-58-36_forest-7-50",
        r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer\bin\mia-result"
        r"\2023-12-15-17-06-45_forest-7-70"

    ]

    extract_and_plot_values(folder_paths)
