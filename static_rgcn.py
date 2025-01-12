import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load RGCN Embeddings for 2015
def load_rgcn_embeddings(directory_path, year):
    file_path = os.path.join(directory_path, f'{year}/{year}_processed_combined_embeddings.csv')
    if os.path.exists(file_path):
        print(f"Loading RGCN embeddings for {year} from {file_path}")
        df = pd.read_csv(file_path)
        return df
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

# Prepare Data for Prediction
def prepare_static_data(dataframe):
    """
    Converts the RGCN embeddings and labels into numpy arrays for training/testing.
    """
    # Assuming 'embedding_numpy' contains RGCN embeddings and 'Label' contains the target class
    embeddings = np.stack(dataframe['embedding_list'].apply(eval).values)  # Convert string to numpy array
    labels = dataframe['Label'].values  # Extract labels
    return embeddings, labels

# Path to your data directory
directory_path = '/content/drive/My Drive/Colab Notebooks/Corporate Credit Risk Prediction/RGCN-master/data'

# Load and prepare data
year = '2014'
rgcn_data = load_rgcn_embeddings(directory_path, year)

rgcn_embeddings, rgcn_labels = prepare_static_data(rgcn_data)

# Standardize embeddings (optional, depends on downstream model)
scaler = StandardScaler()
rgcn_embeddings_scaled = scaler.fit_transform(rgcn_embeddings)

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rgcn_embeddings_scaled, rgcn_labels, test_size=0.2, random_state=42)

# Train Logistic Regression on RGCN embeddings
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Predict and evaluate
y_pred = logreg.predict(X_test)
print("Accuracy on test data:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example: Predict on the entire 2015 dataset
all_predictions = logreg.predict(rgcn_embeddings_scaled)
print("\nPredicted Risk Classes for 2015 Data:\n", all_predictions)
