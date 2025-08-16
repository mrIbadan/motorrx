"""
  - Version 0.0.1
  - Dependencies:  generate_policies_df.py
  - Exaple Use:
    - 

  %%time
  # Create the ClaimsDataCreator instance
  claims_creator = ClaimsDataCreator(policy_df)
  
  # Generate a specified number of claims
  n_claims = 7000  # Specify the number of claims you want to generate
  claims_df = claims_creator.generate_claims(n_claims)
  
  claims_df.head(5)

"""

import pandas as pd
import numpy as np
import uuid
from datetime import timedelta

class ClaimsDataCreator:
    def __init__(self, policy_df: pd.DataFrame, seed: int = 42):
        self.policy_df = policy_df
        self.seed = seed
        np.random.seed(self.seed)

    def generate_claims(self, n_claims: int):
        """Generate a specified number of claims across customer policies."""
        claims_data = []

        # Select unique customer IDs from policy_df
        customer_ids = self.policy_df['Customer_ID'].unique()

        for _ in range(n_claims):
            # Randomly select a customer ID
            customer_id = np.random.choice(customer_ids)

            # Select policies for the customer
            customer_policies = self.policy_df[self.policy_df['Customer_ID'] == customer_id]

            # Randomly select a policy for the claim
            if not customer_policies.empty:
                policy = customer_policies.sample(n=1).iloc[0]

                # Determine the number of claims (1, 2, or 3) based on distribution
                num_claims = self.determine_claims_count()

                for _ in range(num_claims):
                    # Generate claim details
                    claim_date = self.generate_claim_date(policy['Policy_Start_Date'], policy['Policy_End_Date'])
                    claim_amount = np.random.randint(100, 5000)  # Random claim amount
                    claim_cost = claim_amount * np.random.uniform(0.5, 1.5)  # Actual cost can vary
                    exposure = (policy['Policy_End_Date'] - claim_date).days

                    claims_data.append({
                        "claim_id": str(uuid.uuid4()),
                        "Customer_ID": customer_id,
                        "Policy_ID": policy['Policy_ID'],
                        "claim_date": claim_date,
                        "claim_amount": claim_amount,
                        "claim_cost": claim_cost,
                        "claim_description": "Claim for accident or damage",
                        "claim_status": np.random.choice(['Pending', 'Approved', 'Rejected']),
                        "exposure": exposure,
                        "claim_type": np.random.choice(['Accident', 'Theft', 'Weather']),
                        "claim_resolution_date": claim_date + timedelta(days=np.random.randint(1, 30)),
                        "claim_notes": "Additional notes regarding the claim."
                    })

        claims_df = pd.DataFrame(claims_data)
        return claims_df

    def determine_claims_count(self):
        """Determine the number of claims (1, 2, or 3) based on distribution."""
        rand_value = np.random.rand()
        if rand_value < 0.6:
            return 1  # 60% of customers will have 1 claim
        elif rand_value < 0.9:
            return 2  # 30% of customers will have 2 claims
        else:
            return 3  # 10% of customers will have 3 claims

    def generate_claim_date(self, start_date, end_date):
        """Generate a claim date within the policy period."""
        delta_days = (end_date - start_date).days
        claim_date = start_date + timedelta(days=np.random.randint(1, delta_days))
        return claim_date

# Example usage
"""if __name__ == "__main__":
    # Sample policy_df
    policy_df = pd.DataFrame({
        "Policy_ID": ["p1", "p2"],
        "Customer_ID": ["c1", "c1"],
        "Policy_Start_Date": pd.to_datetime(['2024-01-01', '2024-02-01']),
        "Policy_End_Date": pd.to_datetime(['2025-01-01', '2025-02-01']),
        "Coverage": ["TPFT", "Comprehensive"]
    })
"""
