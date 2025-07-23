import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, df):
        self.df = df

    def univariate_analysis(self, num_cols):
        """Perform univariate analysis on numerical columns."""
        for col in num_cols:
            plt.figure(figsize=(8,4))
            sns.histplot(self.df[col], bins=50, kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

    def categorical_analysis(self, cat_cols):
        """Perform analysis on categorical columns."""
        for col in cat_cols:
            plt.figure(figsize=(8, 4))
            sns.countplot(x=col, data=self.df, order=self.df[col].value_counts().index)
            plt.title(f"Count of {col}")
            plt.xticks(rotation=45)
            plt.show()

    def bivariate_analysis(self):
        """Perform bivariate analysis on Loss Ratio."""
        self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
        loss_by_province = self.df.groupby('Province')['LossRatio'].mean().sort_values()

        plt.figure(figsize=(10, 5))
        loss_by_province.plot(kind='barh')
        plt.title("Average Loss Ratio by Province")
        plt.xlabel("Loss Ratio")
        plt.ylabel("Province")
        plt.show()

    def outlier_detection(self, num_cols):
        """Detect outliers in numerical columns."""
        for col in num_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.df[col])
            plt.title(f"Outliers in {col}")
            plt.show()
    
    def gender_comparison(self):
        sns.boxplot(x='Gender', y='LossRatio', data=self.df)
        plt.title("Loss Ratio by Gender")
        plt.show()
    

    def temporal_analysis(self):
        """Perform temporal analysis on transaction month."""
        self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'], errors='coerce')
        monthly_stats = self.df.groupby(self.df['TransactionMonth'].dt.to_period('M')).agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        })

        monthly_stats.plot(kind='line', figsize=(10, 5))
        plt.title("Monthly Total Premiums vs Total Claims")
        plt.ylabel("Amount")
        plt.xlabel("Month")
        plt.show()
    
    def top_insights(self):
        """Generate top 3 insightful visualizations."""
        # 1: Loss Ratio by VehicleType
        loss_by_vehicle = self.df.groupby('VehicleType')['LossRatio'].mean().sort_values(ascending=False).head(10)

        plt.figure(figsize=(10, 5))
        loss_by_vehicle.plot(kind='bar')
        plt.title("Top 10 Vehicle Types by Loss Ratio")
        plt.ylabel("Loss Ratio")
        plt.xlabel("Vehicle Type")
        plt.xticks(rotation=45)
        plt.show()

        # 2: Claim Severity Distribution
        claims_only = self.df[self.df['TotalClaims'] > 0]
        sns.histplot(claims_only['TotalClaims'], bins=50, kde=True)
        plt.title("Distribution of Claim Amounts (for claimed policies only)")
        plt.show()

        # 3: High Risk Vehicle Models
        high_risk_models = self.df.groupby('Model')['LossRatio'].mean().sort_values(ascending=False).head(10)

        plt.figure(figsize=(10, 5))
        high_risk_models.plot(kind='bar')
        plt.title("Top 10 Vehicle Models by Loss Ratio")
        plt.ylabel("Loss Ratio")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.show()