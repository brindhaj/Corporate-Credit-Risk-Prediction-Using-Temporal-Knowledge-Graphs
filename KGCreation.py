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
