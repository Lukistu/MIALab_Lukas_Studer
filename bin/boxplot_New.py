import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def main():
    folder_path_ML = r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer" \
                     r"\bin\mia-result\2023-12-13-10-25-05_forest-300-30"
    folder_path_A = r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer" \
                    r"\bin\mia-result\2023-12-06-10-35-03_B0.4"

    all_files_ML = os.listdir(folder_path_ML)
    all_files_A = os.listdir(folder_path_A)

    csv_files_ML = [file for file in all_files_ML if file.endswith('.csv')]
    csv_files_A = [file for file in all_files_A if file.endswith('.csv')]

    if not csv_files_ML or not csv_files_A:
        print("No CSV files found in one or both of the specified folders.")
        return

    for csv_file_ML, csv_file_A in zip(csv_files_ML, csv_files_A):
        csv_file_path_ML = os.path.join(folder_path_ML, csv_file_ML)
        csv_file_path_A = os.path.join(folder_path_A, csv_file_A)

        df_ML = pd.read_csv(csv_file_path_ML, delimiter=';')
        df_A = pd.read_csv(csv_file_path_A, delimiter=';')

        # Assuming both CSV files have the same columns
        labels = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']

        filtered_df_ML = df_ML[df_ML['LABEL'].isin(labels)]
        filtered_df_A = df_A[df_A['LABEL'].isin(labels)]

        # Combine both datasets for visualization
        filtered_df_ML['Type'] = 'ML'
        filtered_df_A['Type'] = 'Atlas'
        combined_data = pd.concat([filtered_df_ML, filtered_df_A])

        # Set up the figure and subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        # Create box plots for Dice coefficients comparing _ML and _A data
        sns.boxplot(x='LABEL', y='DICE', hue='Type', data=combined_data, ax=axes[0], gap=0)
        axes[0].set_title('Dice Coefficients Comparison')
        axes[0].set_ylabel('Dice Coefficient')
        axes[0].set_xlabel('Label')
        axes[0].set_ylim(bottom=0, top=1)  # Set lower and upper limit of y-axis
        axes[0].legend(title='Type')

        # Create box plots for Hausdorff distances comparing _ML and _A data
        sns.boxplot(x='LABEL', y='HDRFDST', hue='Type', data=combined_data, ax=axes[1], gap=0)
        axes[1].set_title('Hausdorff Distances Comparison')
        axes[1].set_ylabel('Hausdorff Distance')
        axes[1].set_xlabel('Label')
        axes[1].set_ylim(bottom=0)  # Set lower limit of y-axis
        axes[1].legend(title='Type')

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
