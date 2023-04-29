import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


if __name__ == "__main__":
    # paths:
    path_data_folder = Path(__file__).parent.parent / "data"
    path_results_folder = Path(__file__).parent / "results"

    # Load the data
    data = pd.read_csv(path_data_folder / "BreastCancer.csv")

    # Convert diagnosis to numeric values
    data['diagnosis_numeric'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Data exploration
    head = data.head().to_string()

    with open(path_results_folder / "data_head.txt", "w") as f:
        f.write(head)

    with open(path_results_folder / "data_types.txt", "w") as f:
        f.write(data.dtypes.to_string())

    # Save descriptive statistics to a file
    description_str = data.describe()
    description_str = description_str.to_string()
    with open(path_results_folder / "data_description.txt", "w") as f:
        f.write(description_str)

    missing_values_str = data.isna().sum().to_string()
    with open(path_results_folder / "data_missing.txt", "w") as f:
        f.write(missing_values_str)

    # Visualization
    # Histograms or box plots
    data.hist(figsize=(20, 20))
    plt.show()

    # # Scatter plots or pair plots
    # sns.pairplot(data, hue='diagnosis')
    # plt.show()

    # Correlation matrix or heatmap
    corr_matrix = data.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    # Save the plot to a file
    plt.savefig(path_results_folder / "correlation_matrix.png")
    plt.show()

    # # Feature-target relationship analysis
    # # Distribution of the target variable (diagnosis)
    # # Convert diagnosis to numeric values
    # data['diagnosis_numeric'] = data['diagnosis'].map({'M': 1, 'B': 0})
    # sns.countplot(data['diagnosis_numeric'])
    # plt.show()

    # Relationships between individual features and the target variable (diagnosis)
    # Using violin plots
    plt.figure(figsize=(20, 20))
    for i, column in enumerate(data.columns[2:31], 1):
        plt.subplot(6, 5, i)
        sns.violinplot(x='diagnosis', y=column, data=data)

    plt.savefig(path_results_folder / "feature_target_violin.png")
    plt.show()

    # from ydata_profiling import ProfileReport
    # def allInOneEda(data, pathToFigures: Path, name="EDA"):
    #     # EDA using ydata_profiling
    #     profile = ProfileReport(data, explorative=True)
    #     profile.to_notebook_iframe()
    #     # Save the report as an html file
    #     # profile.to_file(pathToFigures/f"{name}.html")

    # Distribution of the target variable (diagnosis)
    data['diagnosis'].hist()
    # sns.countplot(diagnosis_numeric)
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.savefig(path_results_folder / "diagnosis_distribution.png")
    plt.show()
