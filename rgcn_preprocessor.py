#Create entities and relations dictionary dictionary
import os
import pandas as pd

# Function to prepare data for testing
def prepareTest(dfTest, current_year, path):
    dfTest = df[0:test]
    file_path = path + "test_" + str(current_year) + ".txt"
    with open(file_path, "w") as fl:
        print(f"Processing {len(dfTest)} for Test")
        for index, row in dfTest.iterrows():
            strText = f"{row['Source'].strip()}\t{row['Edge'].strip()}\t{row['Dest'].strip()}"
            if index > 0:
                strText = "\n" + strText
            fl.write(strText)

# Function to prepare data for training
def prepareTrain(dfTrain, current_year, path):
    dfTrain = df[test + valid:len(dfTrain)]
    file_path = path + "train_" + str(current_year) + ".txt"
    with open(file_path, "w") as fl:
        print(f"Processing {len(dfTrain)} for Train")
        for index, row in dfTrain.iterrows():
            strText = f"{row['Source'].strip()}\t{row['Edge'].strip()}\t{row['Dest'].strip()}"
            if index > test + valid:
                strText = "\n" + strText
            fl.write(strText)

# Function to prepare data for validation
def prepareValidation(dfValidation, current_year, path):
    dfValidation = df[test:test + valid]
    file_path = path + "valid_" + str(current_year) + ".txt"
    with open(file_path, "w") as fl:
        print(f"Processing {len(dfValidation)} for Validation")
        for index, row in dfValidation.iterrows():
            strText = f"{row['Source'].strip()}\t{row['Edge'].strip()}\t{row['Dest'].strip()}"
            if index > test:
                strText = "\n" + strText
            fl.write(strText)

# Function to preprocess for RGCN
def rgcn_pre_process(df, current_year, path):
    entity_file = path + "entities_" + str(current_year) + ".dict"
    relation_file = path + "relations_" + str(current_year) + ".dict"

    # Delete existing files if they exist
    for file in [entity_file, relation_file]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted existing file: {file}")

    lstEntities = []
    lstRelations = []

    for index, row in df.iterrows():
        lstEntities.append(row["Source"].strip())
        lstEntities.append(row["Dest"].strip())
        lstRelations.append(row["Edge"].strip())

    distinct_entities = set(lstEntities)
    distinct_relations = set(lstRelations)

    with open(entity_file, "w") as f:
        for i, entity in enumerate(distinct_entities):
            strText = f"{i}\t{entity}\n"
            f.write(strText)

    with open(relation_file, "w") as fl:
        for j, rel in enumerate(distinct_relations):
            strText = f"{j}\t{rel}\n"
            fl.write(strText)

    print(f"Files written for year {current_year}:\n - {entity_file}\n - {relation_file}")

# Iterate over years 2011 to 2016
for year in range(2011, 2017):
    current_year = str(year)
    print(f"Processing for year {current_year}")

    # Change directory and load data
    base_path = "/content/drive/My Drive/Colab Notebooks/Corporate Credit Risk Prediction/RGCN-master/data/"
    path = base_path + current_year + "/"
    os.makedirs(path, exist_ok=True)

    # Load data for the current year
    file_name = f"credit_risk_triplets_{current_year}.csv"
    df = pd.read_csv(file_name, delimiter=',')
    print(f"Total rows: {len(df)} for {current_year}")

    # Rename columns
    df = df.rename(columns={
        "Subject": "Source",
        "Predicate": "Edge",
        "Object": "Dest"
    })

    print(df.head())

    # Define splits
    test = 250
    valid = 250
    train = len(df) - test - valid
    print(f"Train: {train}, Test: {test}, Valid: {valid}")

    # Prepare files
    prepareTest(df, current_year, path)
    prepareTrain(df, current_year, path)
    prepareValidation(df, current_year, path)

    # Preprocess for RGCN
    rgcn_pre_process(df, current_year, path)
