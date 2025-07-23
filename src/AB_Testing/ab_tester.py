import pandas as pd
import numpy as np
import scipy.stats as stats

class ABTester:
    def __init__(self, file_path):
        self.file_path = file_path
        pd.set_option('display.max_columns', None)
        self.df = pd.read_csv(self.file_path, sep="|", engine="python")

    def load_calculation(self):
        """Calculate derived columns."""
        self.df['HadClaim'] = self.df['TotalClaims'] > 0

        # ClaimSeverity: only where a claim occurred
        self.df['ClaimSeverity'] = np.where(self.df['HadClaim'], self.df['TotalClaims'], np.nan)

        # Margin = TotalPremium - TotalClaims
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']

        return self.df

    def province_counts(self):
        province_counts = self.df['Province'].value_counts().reset_index()
        province_counts.columns = ['Province', 'RecordCount']
        return province_counts.sort_values(by='RecordCount', ascending=False).head(5)

    def zipcode_counts(self):
        zipcode_counts = self.df['PostalCode'].value_counts().reset_index()
        zipcode_counts.columns = ['PostalCode', 'RecordCount']
        return zipcode_counts.sort_values(by='RecordCount', ascending=False).head(5)

    def ab_test_province(self, prov_a, prov_b):
        group_a = self.df[self.df['Province'] == prov_a]
        group_b = self.df[self.df['Province'] == prov_b]

        if group_a.empty or group_b.empty:
            print("One of the province groups is empty.")
            return None

        # Claim Frequency
        t_freq, p_freq = stats.ttest_ind(group_a['HadClaim'].dropna(), group_b['HadClaim'].dropna(), equal_var=False)

        # Claim Severity
        sev_a = group_a[group_a['HadClaim']]['ClaimSeverity'].dropna()
        sev_b = group_b[group_b['HadClaim']]['ClaimSeverity'].dropna()

        t_sev, p_sev = stats.ttest_ind(sev_a, sev_b, equal_var=False)

        return {
            'ClaimFrequency_p_value': p_freq,
            'ClaimSeverity_p_value': p_sev,
            'Hypothesis_Rejected': p_sev < 0.05
        }

    def ab_test_zipcode_risk(self, zip_a, zip_b):
        group_a = self.df[self.df['PostalCode'] == zip_a]
        group_b = self.df[self.df['PostalCode'] == zip_b]

        if group_a.empty or group_b.empty:
            print("One of the zip code groups is empty.")
            return None

        t_freq, p_freq = stats.ttest_ind(group_a['HadClaim'].dropna(), group_b['HadClaim'].dropna(), equal_var=False)

        sev_a = group_a[group_a['HadClaim']]['ClaimSeverity'].dropna()
        sev_b = group_b[group_b['HadClaim']]['ClaimSeverity'].dropna()

        t_sev, p_sev = stats.ttest_ind(sev_a, sev_b, equal_var=False)

        return {
            'ClaimFrequency_p_value': p_freq,
            'ClaimSeverity_p_value': p_sev,
            'Hypothesis_Rejected': p_sev < 0.05
        }

    def ab_test_zipcode_margin(self, zip_a, zip_b):
        group_a = self.df[self.df['PostalCode'] == zip_a]
        group_b = self.df[self.df['PostalCode'] == zip_b]

        if group_a.empty or group_b.empty:
            print("One of the zip code groups is empty.")
            return None

        t_margin, p_margin = stats.ttest_ind(group_a['Margin'].dropna(), group_b['Margin'].dropna(), equal_var=False)

        return {
            'Margin_p_value': p_margin,
            'Hypothesis_Rejected': p_margin < 0.05
        }

    def ab_test_gender(self):
        group_m = self.df[self.df['Gender'] == 'Male']
        group_f = self.df[self.df['Gender'] == 'Female']

        if group_m.empty or group_f.empty:
            print("One of the gender groups is empty.")
            return None

        t_freq, p_freq = stats.ttest_ind(group_m['HadClaim'].dropna(), group_f['HadClaim'].dropna(), equal_var=False)

        sev_m = group_m[group_m['HadClaim']]['ClaimSeverity'].dropna()
        sev_f = group_f[group_f['HadClaim']]['ClaimSeverity'].dropna()

        t_sev, p_sev = stats.ttest_ind(sev_m, sev_f, equal_var=False)

        return {
            'ClaimFrequency_p_value': p_freq,
            'ClaimSeverity_p_value': p_sev,
            'Hypothesis_Rejected': p_sev < 0.05
        }