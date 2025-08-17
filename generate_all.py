"""
  - Version 0.0.1
  - Dependencies correct versions of quotes, policies and claims generation data (will fix this once testing is complete
  - Set the number of quotes you want, the % of policies as conversion rate (5% of quote data = 5% conversion rate etc) & set number of claims you want
  - Download the UK Sample Postcode file seperately IF it fails from: 
    - https://drive.google.com/file/d/1YMWPJUlFDCm-EkjivaafcZNUUNyeIfsR/view?usp=drive_link
    - Run 

      cols = ['pcd', 'country', 'region', 'admin_district']
      geo_df = pd.read_csv('path/to/postcodes_api.csv', usecols = cols)
      geo_df = geo_df.dropna(subset=['country'])
    
  - Useage

n_quotes = 10000

# Quote Data
%%time
creator = RawDataCreator(geo_df)
quote_df = creator.generate(n_rows=100000)
print(quote_df.shape)
quote_df.head()

# Policy Data
# Determine 5% of the DataFrame
sample_size = int(len(quote_df) * 0.05)  # Calculate 5% of the DataFrame
sampled_quote_df = quote_df.sample(n=sample_size, random_state=42)  # Sample 5%

# Get unique customer IDs from the sampled DataFrame
customer_ids = sampled_quote_df["customer_id"].unique()

# Create policy data
policy_creator = PolicyDataCreator(sampled_quote_df)
%time policy_df = policy_creator.generate_policy_data(customer_ids)

(policy_df.head())

# Claims data

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
import hashlib
import gdown
import io
from datetime import timedelta

def google_drive_read(url, **kwargs):
    """
    Download a CSV file from Google Drive and read into pandas.
    
    Parameters:
    - url: str, Google Drive shareable link
    - kwargs: any additional args to pass to pd.read_csv
    
    Returns:
    - pandas DataFrame
    """
    # Extract file ID from URL
    if "/d/" in url:
        file_id = url.split("/d/")[1].split("/")[0]
    else:
        raise ValueError("Cannot extract file ID from URL. Check the format.")
    
    # Create direct download URL
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    # Use a temporary file to store the downloaded content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = temp_file.name
    
    # Download content to the temporary file
    try:
        gdown.download(download_url, output=file_path, quiet=False)
        
        # Read the temporary file into a pandas DataFrame
        df = pd.read_csv(file_path, **kwargs)
        
    finally:
        # Clean up by removing the temporary file
        os.remove(file_path)
    
    return df

# Define columns to read
cols = ['pcd', 'country', 'region', 'admin_district']

# Usage
try:
    geo_df = google_drive_read(
        'https://drive.google.com/file/d/1YMWPJUlFDCm-EkjivaafcZNUUNyeIfsR/view?usp=drive_link',
        usecols=cols
    )
    # Display the DataFrame
    geo_df = geo_df.dropna(subset=['country'])
    print(geo_df.head())
    
except ValueError as e:
    print(f"An error occurred: {e}")
class RawDataCreator:
    def __init__(self, geo_df: pd.DataFrame, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)
        # Use provided geo_df (already cleaned by user)
        self.postcodes = geo_df.reset_index(drop=True)
        print(f"🚦 Initialized RawDataCreator with seed {self.seed} | {len(self.postcodes)} postcodes loaded")

    def generate(self, n_rows: int = 1000000):
        # ------------------------
        # DRIVER INFORMATION
        # ------------------------
        ages = np.random.triangular(left=18, mode=40, right=80, size=n_rows).astype(int)
        dob = pd.to_datetime('2024-01-01') - pd.to_timedelta(ages * 365, unit='D')
        license_years = np.random.randint(0, ages - 18 + 1)

        # ------------------------
        # VEHICLE & BUILDING TYPES
        # ------------------------
        vehicle_types = ["Private Car", "Taxi", "Fleet", "Van"]
        vehicle_type = np.random.choice(vehicle_types, size=n_rows)

        building_type = []
        business_type = []
        for vt in vehicle_type:
            if vt in ["Taxi", "Fleet", "Van"]:
                building_type.append("Commercial")
                if vt == "Taxi":
                    business_type.append("Taxi")
                elif vt == "Fleet":
                    business_type.append(np.random.choice(["Taxi", "Private Cab", "Van"]))
                elif vt == "Van":
                    business_type.append("Van")
            else:
                building_type.append("Residential")  # all private vehicles parked in residential
                business_type.append(None)

        # ------------------------
        # OCCUPATION & CREDIT SCORE
        # ------------------------
        occupations = ['Professional', 'Manager', 'Skilled Worker', 'Unskilled Worker', 'Self-Employed']
        occupation_scores = {
            'Professional': (800, 999),
            'Manager': (700, 799),
            'Skilled Worker': (600, 699),
            'Unskilled Worker': (500, 599),
            'Self-Employed': (600, 750)
        }
        random_occupations = np.random.choice(occupations, size=n_rows)
        credit_scores = self.generate_credit_scores(ages, random_occupations, occupation_scores)

        # ------------------------
        # COVERAGE / POLICY / SOURCES
        # ------------------------
        coverage_types = ['TPFT', 'Comprehensive', 'TPO', 'WS']
        coverage = np.random.choice(coverage_types, size=n_rows)

        quote_sources = ['Direct', 'Aggregator', 'Broker', 'Indirect']
        quote_source = np.random.choice(quote_sources, size=n_rows)

        policy_types = ['New', 'Renewal']
        policy_type = np.random.choice(policy_types, size=n_rows)

        # ------------------------
        # CUSTOMER BEHAVIOUR
        # ------------------------
        churn_prob = np.random.beta(2, 5, size=n_rows)
        conversion_flag = np.random.binomial(1, 1 - churn_prob)

        # ------------------------
        # POSTCODE SAMPLING (consistent row from geo_df)
        # ------------------------
        postcode_sample = self.postcodes.sample(
            n=n_rows, replace=True, random_state=self.seed
        ).reset_index(drop=True)

        # ------------------------
        # FINAL DATAFRAME
        # ------------------------
        df = pd.DataFrame({
            "quote_id": [str(uuid.uuid4()) for _ in range(n_rows)],  # Generate UUIDs
            "customer_id": [self.generate_customer_id() for _ in range(n_rows)],  # Generate customer IDs
            "quote_date": pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, n_rows), unit='D'),
            "age": ages,
            "dob": dob,
            "license_years": license_years,
            "vehicle_type": vehicle_type,
            "building_type": building_type,
            "business_type": business_type,
            "occupation": random_occupations,
            "credit_score": credit_scores,
            "coverage": coverage,
            "quote_source": quote_source,
            "policy_type": policy_type,
            "churn_probability": churn_prob,
            "conversion_flag": conversion_flag,
            # New Personal Details
            "number_of_named_drivers": np.random.randint(0, 4, n_rows),
            "marital_status": np.random.choice(['Single', 'Married', 'Divorced'], size=n_rows),
            "occupation_status": np.random.choice(['Employed', 'Unemployed', 'Student'], size=n_rows),
            "homeownership_status": np.random.choice(['Own', 'Rent'], size=n_rows),
            "education_level": np.random.choice(['High School', 'Bachelor', 'Master'], size=n_rows),
            "income_bracket": np.random.choice(['Low', 'Medium', 'High'], size=n_rows),
            "driving_experience": np.random.randint(0, 40, n_rows),
            "previous_insurance_provider": np.random.choice(['Insurer A', 'Insurer B', 'Insurer C'], size=n_rows),
            "credit_level": np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], size=n_rows),
            "health_status": np.random.choice(['Good', 'Fair', 'Poor'], size=n_rows),

            # Parking Information
            "parking_location": np.random.choice(['Garage', 'Driveway', 'Street', 'Commercial'], size=n_rows),
            "parking_security": np.random.choice(['CCTV', 'Security Patrol', 'None'], size=n_rows),
            "vehicle_condition": np.random.choice(['New', 'Good', 'Fair', 'Poor'], size=n_rows),
            "annual_mileage": np.random.randint(5000, 30000, size=n_rows),
            "vehicle_use": np.random.choice(['Personal', 'Business', 'Commercial'], size=n_rows),
            "previous_claims": np.random.poisson(0.5, n_rows),
            "maintenance_history": np.random.choice([0, 1], size=n_rows),
            "usage_of_telematics": np.random.choice([0, 1], size=n_rows),
            "vehicle_history": np.random.choice(['Accident', 'No Accident'], size=n_rows),
            "insurance_claims": np.random.poisson(0.3, n_rows),

            # Taxi, Van, Uber Driver Features
            "driver_rating": np.random.uniform(1, 5, n_rows),
            "vehicle_age": np.random.randint(0, 10, n_rows),
            "passenger_capacity": np.random.randint(1, 8, n_rows),
            "commercial_license": np.random.choice([0, 1], size=n_rows),
            "job_duration": np.random.randint(0, 20, n_rows),
            "daily_hours_driven": np.random.uniform(1, 12, n_rows),
            "app_usage": np.random.randint(0, 100, size=n_rows),
            "fleet_vehicle_id": np.random.choice(['F1', 'F2', 'F3'], size=n_rows),
            "surge_pricing_awareness": np.random.choice([0, 1], size=n_rows),
            "ride_history": np.random.randint(0, 1000, size=n_rows),

            # Security Features
            "immobiliser": np.random.choice([0, 1], size=n_rows),
            "alarm_system": np.random.choice([0, 1], size=n_rows),
            "gps_tracking": np.random.choice([0, 1], size=n_rows),
            "vehicle_recovery_system": np.random.choice([0, 1], size=n_rows),
            "adas": np.random.choice([0, 1], size=n_rows),
            "anti_theft_device": np.random.choice([0, 1], size=n_rows),
            "vehicle_warranty": np.random.randint(0, 5, size=n_rows),  # Years remaining
            "safety_rating": np.random.choice([1, 2, 3, 4, 5], size=n_rows),
            "security_training": np.random.choice([0, 1], size=n_rows),
            "insurance_history": np.random.choice(['Good', 'Average', 'Poor'], size=n_rows),

            # Additional Features
            "customer_satisfaction_score": np.random.uniform(1, 10, n_rows),
            "referral_source": np.random.choice(['Friend', 'Online', 'Advertisement'], size=n_rows),
            "last_policy_update": pd.to_datetime('2024-01-01') - pd.to_timedelta(np.random.randint(0, 365, n_rows), unit='D'),
            "claims_history": np.random.poisson(0.3, n_rows),
            "policy_changes": np.random.poisson(0.5, n_rows),
            "loyalty_status": np.random.randint(0, 10, n_rows),
            "discounts_applied": np.random.choice([0, 1], size=n_rows),
            "payment_frequency": np.random.choice(['Monthly', 'Annually'], size=n_rows),
            "coverage_limits": np.random.randint(10000, 100000, size=n_rows),
            "policy_exclusions": np.random.choice(['None', 'Specific Exclusions'], size=n_rows),

            # Postcode info (all consistent from sampled row)
            "postcode": postcode_sample["pcd"],
            "country": postcode_sample["country"],
            "region": postcode_sample["region"],
            "admin_district": postcode_sample["admin_district"]
        })

        # Remove duplicates
        df = df.drop_duplicates()

        print(f"✅ Generated {n_rows} synthetic records with {df.shape[1]} columns")
        return df

    def generate_credit_scores(self, ages, occupations, occupation_scores):
        """Generate credit scores influenced by occupation and age."""
        credit_scores = []
        for i in range(len(ages)):
            occupation = occupations[i]
            base_score_range = occupation_scores[occupation]
            age_influence = min(999, 100 + (ages[i] - 18) * 2)
            credit_score = np.random.randint(base_score_range[0], base_score_range[1] + 1)
            credit_score = min(credit_score + age_influence - 100, 999)
            credit_scores.append(credit_score)
        return credit_scores

    def generate_customer_id(self):
        """Generate a unique customer ID."""
        return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()  # Hash of a UUID

    def save(self, df: pd.DataFrame, file_path: str = "synthetic_data.csv"):
        df.to_csv(file_path, index=False)
        print(f"📁 Saved dataset to {file_path}")

#---------------------------------------

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
      
# --------------------------------------------------
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
