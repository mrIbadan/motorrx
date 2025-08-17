# motorrx
Motor_Repo Data - Travel & Home may be added here

Version - 0.01

3 files
quote_df,
policy_df,
claims_df,

The common key for all three files is the "Customer_ID" key.

quote_df can have (not in this case) duplicated customer_id's where we can identify a customer searching for a quote multiple times which may occur when searching with different inputs or when the quote comes from multiple sources (shopping around)
Duplicated customer_ids are not yet a feature in this excercise. But many to one quote_ids to customer_ids does exist in reality. (Many quotes from a single customer)
policy_df cannot be created without (left_join) quote_df
claims_df cannot be created without (left_join) policy_df 

There should be no duplicates in policy_df
Multiple claims per user (claim_id is independent) is allowed

Files are not 100% for several reasons - one is in order to encourage best practice. Data does not always come in 100% clean.
Due Dilligence, Schemas, and General Sense Checks are encouraged. 

This is the first version to be published for use in Physar

Data dictionary to be built to allow users to understand what features are useful for the creation of models, analysis etc. 
