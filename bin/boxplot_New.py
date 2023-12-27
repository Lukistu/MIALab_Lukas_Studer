import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def main():
    folder_path_ML = r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer" \
                     r"\bin\mia-result\2023-12-15-22-02-18_forest-70-30"
    folder_path_A = r"C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer" \
                    r"\bin\mia-result\2023-12-15-13-40-46_B0.35"

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
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 12))

        # Create box plots for Dice coefficients comparing _ML and _A data
        sns.boxplot(x='LABEL', y='DICE', hue='Type', data=combined_data, ax=axes[0], gap=0)
        axes[0].set_title('Dice Coefficients Comparison')
        axes[0].set_ylabel('Dice Coefficient')
        axes[0].set_xlabel('')
        axes[0].set_ylim(bottom=0, top=1)  # Set lower and upper limit of y-axis
        axes[0].legend(title='Type')

        # Adding horizontal lines to the first plot (Dice Coefficients Comparison)
        for y_value in [i / 10 for i in range(1, 11)]:
            axes[0].axhline(y=y_value, color='gray', linestyle='-', alpha=0.1, label=f'Threshold: {y_value}')

        # Adding vertical lines to separate labels in the first plot
        for i in range(len(labels) - 1):
            axes[0].axvline(x=i + 0.5, color='black', linestyle='--', alpha=0.1)

        # Create box plots for Hausdorff distances comparing _ML and _A data
        sns.boxplot(x='LABEL', y='HDRFDST', hue='Type', data=combined_data, ax=axes[1], gap=0)
        axes[1].set_title('Hausdorff Distances Comparison')
        axes[1].set_ylabel('Hausdorff Distance')
        axes[1].set_xlabel('')
        axes[1].set_ylim(bottom=0)  # Set lower limit of y-axis
        axes[1].legend(title='Type')

        # Adding horizontal lines at intervals of 10 from y = 10 to y = 70 in the second plot
        for y_value in range(10, 71, 10):
            axes[1].axhline(y=y_value, color='gray', linestyle='-', alpha=0.1, label=f'Threshold: {y_value}')

        # Adding vertical lines to separate labels in the second plot
        for i in range(len(labels) - 1):
            axes[1].axvline(x=i + 0.5, color='black', linestyle='--', alpha=0.1)

        plt.subplots_adjust(hspace=20)

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
