'''
  - Version 0.01
  - Example run
    -
%%time
creator = RawDataCreator(geo_df)
quote_df = creator.generate(n_rows=100000)
print(quote_df.shape)
quote_df.head()

  - Dependencies = path to geo_df API dataset (Google Drive)
  - Parent to - generate_policies_df, generate_claims_df
'''
import pandas as pd
import gdown
import io

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
    print(geo_df.head())
    
except ValueError as e:
    print(f"An error occurred: {e}")
  
# Generate Data

import numpy as np
import uuid
import hashlib

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

