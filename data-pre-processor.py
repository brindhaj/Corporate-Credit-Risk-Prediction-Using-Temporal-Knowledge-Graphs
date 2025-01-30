# prompt: list all columns
%cd /content/drive/My Drive/Colab Notebooks/Corporate Credit Risk Prediction/

print(df_rating.columns)
import pandas as pd
import os

# Read the CSV file
df = df_rating
# Convert the 'date_column' to datetime format
df['date_column'] = pd.to_datetime(df['Date'])

# Extract the year and create a new column 'year'
df['Year'] = df['date_column'].dt.year

# Extract relevant columns
df = df[['Year', 'Name', 'Symbol', 'Rating']]

# Ensure the 'Year' column is of integer type
df['Year'] = df['Year'].astype(int)

# Filter the DataFrame for years 2011 to 2016
df_filtered = df[df['Year'].between(2011, 2016)]

# Group by 'Year' and save each group to a separate CSV file
for year, group in df_filtered.groupby('Year'):
    # Define the filename
    filename = 'labelled_data_'+str(year)+'.csv'

    # Save the group to a CSV file
    group.to_csv(filename, index=False)
    print(f'Saved data for year {year} to {filename}')
