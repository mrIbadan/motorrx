import pandas as pd
import numpy as np
import uuid

class PolicyDataCreator:
    def __init__(self, quote_df: pd.DataFrame, seed: int = 42):
        self.quote_df = quote_df
        self.seed = seed
        np.random.seed(self.seed)

    def generate_policy_data(self, customer_ids: list):
        # Filter the quote DataFrame for the specified customer IDs
        policy_candidates = self.quote_df[self.quote_df['customer_id'].isin(customer_ids)]

        # Generate the Policy_IDs without a loop
        policy_ids = [str(uuid.uuid4()) for _ in range(len(policy_candidates))]

        # Generate the policy DataFrame
        policy_data = pd.DataFrame({
            "Customer_ID": policy_candidates["customer_id"].values,
            "Policy_ID": policy_ids,  # Use pre-generated UUIDs
            "Age": policy_candidates["age"].values,
            "Occupation": policy_candidates["occupation"].values,
            "Credit_Score": policy_candidates["credit_score"].values,
            "Policy_Start_Date": self.random_policy_start_dates(policy_candidates["quote_date"]),
            "Policy_End_Date": pd.to_datetime(self.random_policy_start_dates(policy_candidates["quote_date"])) + pd.DateOffset(years=1),
            "Coverage": policy_candidates["coverage"].values,
            "Premium_Paid": self.calculate_premium(policy_candidates),
            "Policy_Type": policy_candidates["policy_type"].values,
            "New_Business_Flag": policy_candidates["policy_type"].apply(lambda x: 1 if x == "New" else 0)  # New Business Flag
        })

        return policy_data

    def random_policy_start_dates(self, quote_dates):
        return quote_dates + pd.to_timedelta(np.random.randint(0, 365, len(quote_dates)), unit='D')

    def calculate_premium(self, policy_candidates):
        # Calculate the premium based on age, occupation, and credit score
        base_premiums = np.full(len(policy_candidates), 300)  # Base premium

        # Adjustments based on age
        age_adjustments = np.where(policy_candidates['age'].values < 25, 200, 0) + \
                          np.where((policy_candidates['age'].values >= 25) & (policy_candidates['age'].values < 40), 100, 0)
        base_premiums += age_adjustments

        # Adjustments based on credit score
        score_adjustments = np.where(policy_candidates['credit_score'].values < 600, 100, 0) - \
                            np.where(policy_candidates['credit_score'].values >= 800, 50, 0)
        base_premiums += score_adjustments

        premiums = np.clip(np.random.normal(base_premiums, 50), 100, None)  # Ensure a minimum premium
        return premiums


