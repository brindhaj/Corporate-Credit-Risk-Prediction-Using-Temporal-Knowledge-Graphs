# Extract Entities
# Extract Relations
# Create triplets
# Create Knowledge Graph
import pandas as pd

# Load the dataset
df = df_rating

# Function to create triplets
def create_triplet(subject, predicate, obj):
    return (subject, predicate, obj)

# Function to extract Entities and Relationships for a specific year
def extract_entities_and_relations(df, year):
    triplets = []
    # Filter the data for the specific year
    df_year = df[df['Date'].dt.year == year]

    # Iterate through the rows of the filtered dataframe
    for index, row in df_year.iterrows():
        company = row['Name']

        # Company to sector relationship
        if pd.notnull(row['Sector']):
            triplets.append(create_triplet(company, 'operatesInSector', row['Sector']))

        # Company to credit rating relationship
        if pd.notnull(row['Rating']):
            triplets.append(create_triplet(company, 'hasRating', row['Rating']))
            if pd.notnull(row['Rating Agency Name']):
                triplets.append(create_triplet(row['Rating'], 'ratedBy', row['Rating Agency Name']))

        # Financial metrics relationships
        financial_metrics = ['currentRatio', 'quickRatio', 'cashRatio', 'debtEquityRatio',
                             'debtRatio', 'netProfitMargin', 'operatingProfitMargin',
                             'returnOnAssets', 'returnOnEquity',
                             'freeCashFlowOperatingCashFlowRatio', 'grossProfitMargin']

        for metric in financial_metrics:
            if pd.notnull(row[metric]):
                metric_value = f"{metric}: {row[metric]}"
                #triplets.append(create_triplet(company, metric, metric_value))
                triplets.append(create_triplet(company, metric, row[metric]))

        # Temporal relationship (date)
        if pd.notnull(row['Date']):
            triplets.append(create_triplet(company, 'reportedIn', row['Date']))

    return triplets

# Extract triplets for each year and save to CSV files
for year in [2011, 2012, 2013, 2014, 2015, 2016]:
    triplets = extract_entities_and_relations(df, year)
    triplets_df = pd.DataFrame(triplets, columns=['Subject', 'Predicate', 'Object'])
    filename = f'credit_risk_triplets_{year}.csv'
    triplets_df.to_csv(filename, index=False)
    print(f"Saved triplets for {year} to {filename}")




# Total number of unique entities
unique_entities = pd.concat([triplets_df['Subject'], triplets_df['Object']]).nunique()

# Relationship distribution
relationship_counts = triplets_df['Predicate'].value_counts()

# Unique values for categorical attributes
sectors = triplets_df[triplets_df['Predicate'] == 'operatesInSector']['Object'].unique()
ratings = triplets_df[triplets_df['Predicate'] == 'hasRating']['Object'].unique()

# Extract metrics and calculate summary statistics
metric_triplets = triplets_df[triplets_df['Predicate'] == 'hasMetric']
metrics_df = metric_triplets['Object'].str.extract(r'(?P<metric>[\w]+): (?P<value>[-+]?[0-9]*\.?[0-9]+)')
metrics_df['value'] = metrics_df['value'].astype(float)
metric_stats = metrics_df.groupby('metric')['value'].describe()

# Temporal distribution (if date-related entries exist)
temporal_triplets = triplets_df[triplets_df['Predicate'] == 'reportedIn']
temporal_triplets['Object'] = pd.to_datetime(temporal_triplets['Object'], errors='coerce')
earliest_date = temporal_triplets['Object'].min()
latest_date = temporal_triplets['Object'].max()
date_distribution = temporal_triplets['Object'].dt.year.value_counts().sort_index()

# Summary
print(f"Total Unique Entities: {unique_entities}")
print("\nRelationship Counts:\n", relationship_counts)
print("\nSectors in Dataset:", sectors)
print("Credit Ratings in Dataset:", ratings)
print("\nMetric Statistics:\n", metric_stats)
print(f"\nEarliest Date: {earliest_date}")
print(f"Latest Date: {latest_date}")
print("\nYearly Distribution of Dates:\n", date_distribution)
