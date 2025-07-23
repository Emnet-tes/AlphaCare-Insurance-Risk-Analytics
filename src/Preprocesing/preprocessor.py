import pandas as pd

class Preprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Load the data from the specified file path."""
        self.df = pd.read_csv(self.file_path, sep="|", engine="python")

    def initial_exploration(self):
        """Perform initial exploration of the dataset."""
        print("Shape:", self.df.shape)
        print("Columns:", self.df.columns.tolist())
        print(self.df.head())

    def data_quality_checks(self):
        """Check for missing values and data types."""
        print("\nMissing values:")
        print(self.df.isnull().sum())

        print("\nData types:")
        print(self.df.dtypes)

        # Drop duplicate rows (if any)
        self.df.drop_duplicates(inplace=True)

    def descriptive_stats(self):
        """Get descriptive statistics for numerical columns."""
        num_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
        print(self.df[num_cols].describe())